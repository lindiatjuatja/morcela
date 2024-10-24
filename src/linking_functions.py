from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Optional, Iterable, List, Union

import numpy as np
import pandas as pd
import seaborn as sns
import sklearn.metrics as metrics
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_validate

""" Parameters """

SlorParams = namedtuple(
    "SlorParams",
    ["logprob_coef", "unigram_coef", "norm_intercept", "intercept", "length_exp"])

""" sklearn stuff """

def discretize_score(metric, threshold: float = 0., **kwargs):
    def _score(y_true, y_pred, **score_kwargs):
        for k, v in kwargs.items():
            if k not in score_kwargs:
                score_kwargs[k] = v
        return metric(y_true > threshold, y_pred > threshold, **kwargs)

    return _score


class FeatureWeightedLinearRegression(LinearRegression):
    """
    A linear regression model that allows you to specify fixed ratios
    ("feature weights") between coefficients.
    """

    def __init__(self, feature_weight: Iterable[Optional[float]], **kwargs):
        super().__init__(**kwargs)
        self._feature_weight = np.array(feature_weight, dtype=float)

    @property
    def feature_weight(self) -> np.ndarray:
        return self._feature_weight

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight=None):
        n_weights = len(self.feature_weight)
        n_cols = X.shape[1]
        assert n_cols >= n_weights, \
            f"X must have at least {n_weights} features"

        if n_weights > 0:
            weighted_cols = np.zeros(n_cols, dtype=bool)
            weighted_cols[:n_weights] = ~np.isnan(self.feature_weight)
            weights = self.feature_weight[weighted_cols[:n_weights]]
            weighted_x = X[:, weighted_cols] @ weights[:, None]
            X = np.concatenate((weighted_x, X[:, ~weighted_cols]), axis=1)
            super().fit(X, y, sample_weight=sample_weight)

            coefs = np.empty(n_cols, dtype=float)
            coefs[weighted_cols] = self.coef_[0] * weights
            coefs[~weighted_cols] = self.coef_[1:]
            self.coef_ = coefs
            if hasattr(self, "n_features_in_"):
                self.n_features_in_ = len(coefs)
        else:
            super().fit(X, y, sample_weight=sample_weight)

        return self


class InputTransformMixin(ABC):

    @abstractmethod
    def _transform_inputs(self, X):
        return NotImplemented

    def fit(self, X, y):
        return super().fit(self._transform_inputs(X), y)

    def predict(self, X):
        return super().predict(self._transform_inputs(X))


class SlorFxn(InputTransformMixin, FeatureWeightedLinearRegression):
    """
    A regression model based on Slor.
    """

    def __init__(self, logprob_weight: Optional[float] = None,
                 unigram_weight: Optional[float] = None,
                 norm_intercept_weight: Optional[float] = None,
                 length_exp: float = 1.):
        super(FeatureWeightedLinearRegression, self).__init__()
        self.logprob_weight = logprob_weight
        self.unigram_weight = unigram_weight
        self.norm_intercept_weight = norm_intercept_weight
        self.length_exp = length_exp

    @property
    def feature_weight(self) -> np.ndarray:
        return np.array([self.logprob_weight, self.unigram_weight,
                         self.norm_intercept_weight], dtype=float)

    def _transform_inputs(self, X: np.ndarray) -> np.ndarray:
        x = np.ones(X.shape, dtype=float)
        x[:, :-1] = X[:, :-1]
        return x / X[:, -1:] ** self.length_exp

    @property
    def params(self) -> SlorParams:
        return SlorParams(*self.coef_, self.intercept_, self.length_exp)


""" Acceptability Linking Function Object """


class LinkingFunction:
    """
    A container for managing linking functions between LLM log
    probability scores and grammatical acceptability scores elicited
    from humans.
    """

    _scores = dict(
        corr=lambda y_true, y_pred: stats.pearsonr(y_true, y_pred).statistic,
        acc=discretize_score(metrics.accuracy_score),
        precision=discretize_score(metrics.precision_score, pos_label=0),
        recall=discretize_score(metrics.recall_score, pos_label=0),
        f1=discretize_score(metrics.f1_score, pos_label=0))

    @property
    def _scorers(self):
        return {k: metrics.make_scorer(v) for k, v in self._scores.items()}

    _grid = np.arange(0, 1.01, .1)

    @property
    def _cv(self):
        return KFold(shuffle=True)

    def __init__(self, df: pd.DataFrame):
        """
        Must contain an underlying DataFrame, df, which
        must have the following columns:
        - Acceptability
        - Raw Logprob
        - Raw Unigram
        - Length
        """
        self.df = df
        self.functions = dict()
        self._cv_evals = dict()

    def __len__(self):
        return len(self.df)

    def params(self) -> List[SlorParams]:
        return [l_.params for l_ in self.functions.values()]

    @property
    def has_pairs(self) -> bool:
        return "Pair ID" in self.df.columns and \
            "Grammaticality" in self.df.columns

    @property
    def df_good(self) -> pd.DataFrame:
        return self.df.loc[self.df["Grammaticality"] == "Good"]

    @property
    def df_bad(self) -> pd.DataFrame:
        return self.df.loc[self.df["Grammaticality"] == "Bad"]

    def add(self, column_name: str, *args, **kwargs):
        """
        Fit a new linking function and add its predicted acceptability scores to the
        underlying DataFrame.

        :param column_name: The name of the column to add the predicted
            acceptability scores to

        :param args: Positional arguments for SlorFxn
        :param kwargs: Keyword arguments for SlorFxn
        """
        x = self.df[["Raw Logprob", "Raw Unigram", "Length"]].to_numpy()
        y = self.df["Acceptability"].to_numpy()

        # Fit models
        cv_results, function = self._fit_slor(x, y, *args, **kwargs)
        self._cv_evals[column_name] = cv_results

        # Fit a linking function on the full dataset
        function.fit(x, y)
        if hasattr(function, "best_estimator_"):
            function = function.best_estimator_
        y_hat = function.predict(x)
        self.df[column_name] = y_hat
        self.functions[column_name] = function

    def _fit_slor(self, X, y, logprob_coef: Optional[float] = None,
                  unigram_coef: Optional[float] = None,
                  norm_intercept: Optional[float] = None,
                  length_exp: Optional[float] = None):
        """
        Fits a linking function of the form:
            acceptability =
                (alpha * llm_logprob + beta * unigram_logprob + gamma) /
                length^delta + epsilon.

        :param X: Data matrix containing LLM logprobs, unigram logprobs,
            and lengths, in that order. Shape: (n_sentences, 3)
        :param y: Vector of acceptability scores. Shape: (n_sentences,)

        :param logprob_coef: A pre-set value for alpha
        :param unigram_coef: A pre-set value for beta
        :param norm_intercept: A pre-set value for gamma
        :param length_exp: A pre-set value for delta
        """
        do_cross_validate = False
        if logprob_coef is None or unigram_coef is None or length_exp is None:
            do_cross_validate = True
        
        params = dict(
            logprob_weight=[logprob_coef], unigram_weight=[unigram_coef],
            norm_intercept_weight=[norm_intercept],
            length_exp=self._grid if length_exp is None else length_exp)

        if length_exp is None:
            function = GridSearchCV(SlorFxn(), params, cv=self._cv)
        else:
            function = SlorFxn(logprob_coef, unigram_coef, norm_intercept, length_exp)
        
        if do_cross_validate:
            cv_results = cross_validate(
                function, X, y, cv=self._cv, scoring=self._scorers)
            cv_results = {
                k[5:]: v for k, v in cv_results.items() if k.startswith("test_")}
            function.refit = True
            return cv_results, function
        else:
            function.fit(X, y)
            results = {name: scorer(function, X, y) for name, scorer in self._scorers.items()}
            return results, function

    def evaluate(self) -> pd.DataFrame:
        """
        Evaluates the LLM linking theories on the following metrics:
            Correlation
            Accuracy
            Precision
            Recall
            F1

        For accuracy, precision, recall, and F1, scores are discretized
        by assuming that scores below the mean correspond to a label of
        "unacceptable."

        For precision, recall, and F1, sentences that are classified as
        unacceptable are considered to be "positives."
        """
        evals = {}
        for l_, e in self._cv_evals.items():
            evals[l_] = {m: (v.mean() if isinstance(v, np.ndarray) else v) for m, v in e.items()}

        results = {m: [v[m] for v in evals.values()] for m in self._scores}
        results["Function"] = self._cv_evals.keys()
        return pd.DataFrame(results)[["Function"] + list(self._scores.keys())]

    def results(self) -> pd.DataFrame:
        """
        creates a dataframe containing the function name, parameters, and eval metrics
        """
        params = self.params()
        all_fields = [
            "logprob_coef",
            "unigram_coef",
            "pos_unigram_coef",
            "length_coef",
            "norm_intercept",
            "intercept",
            "length_exp"
        ]
        dicts = []
        for param in params:
            data = {}
            for field in all_fields:
                data[field] = getattr(param, field, np.nan)
            dicts.append(data)
        df = pd.DataFrame(dicts)
        evals = self.evaluate()
        for col in evals.columns.tolist():
            df[col] = evals[col]
        cols = evals.columns.tolist() + all_fields
        df = df[cols]
        df = df.set_index("Function")
        return df

    def scatter(self, **sns_kwargs) -> sns.FacetGrid:
        """
        Creates scatter plots
        """
        self.df["Length Quartile"] = pd.qcut(self.df["Length"], q=4, precision=0)
        df = pd.melt(
            self.df,
            id_vars=["Sentence ID", "Acceptability", "Length Quartile"],
            value_vars=list(self.functions.keys()),
            var_name="Function", value_name="Predicted Acceptability")
        g = sns.lmplot(df, x="Acceptability", y="Predicted Acceptability",
                         col="Function", hue="Length Quartile", **sns_kwargs)
        g.set(xlim=(-2,2))
        for ax in g.axes.flat:
            ax.axline((0, 0), slope=1, color='k', ls='--')
            ax.grid(True, axis='both', ls=':')
        return g