"""
Calculates SLOR and generalized versions thereof.

Usage:
python linguistic_inquiry_laws.py --model <model_name> --prefix <prefix> --unigram_n <unigram_n> --use_pile <use_pile> --result_file_name <result_file_name> --save_file <save_file>

python linguistic_inquiry_laws.py --result_file_name per_token_probs.jsonl --save_file opt_n=100000.csv --unigram_n 100000 --model_family opt
python linguistic_inquiry_laws.py --result_file_name per_token_probs_bos-False.jsonl --save_file pythia-pile.csv --use_pile True --model_family pythia
"""
import click
import json
import math
import os
from pathlib import Path
from tqdm import tqdm

import pandas as pd

from constants import MODELS, PYTHIA_SUITE, OPT_SUITE
from linking_functions import LinkingFunction


def load_linguistic_inquiry_results(
        model_name: str,
        results_dir: str,
        result_file_name: str,
        unigram_n: int = 100000,
        use_pile: bool = False,
    ) -> pd.DataFrame:
    # Load LLM probability scores and lengths (in tokens)
    with open(results_dir / result_file_name) as f:
        lm_data = [json.loads(line) for line in f]

    good_token_p = [[math.log(p) for p in r["good_token_probs"]] for r in lm_data]
    bad_token_p = [[math.log(p) for p in r["bad_token_probs"]] for r in lm_data]

    good_p = [sum(probs) for probs in good_token_p]
    bad_p = [sum(probs) for probs in bad_token_p]
    good_l = [len(r["good_token_probs"]) for r in lm_data]
    bad_l = [len(r["bad_token_probs"]) for r in lm_data]

    # Load unigram distribution
    if use_pile:
        with open("../data/pile_unigram_logprobs.json") as f:
            u_dist = json.load(f)
    elif model_name in OPT_SUITE:
        filename = f"unigram_dist_opt-30b_n={unigram_n}.csv"
        u_dist = pd.read_csv(f"../unigram_logprobs/{filename}")
        u_dist = dict(zip(u_dist["Token"], u_dist["Unigram Logprob"]))

    good_token_u = [[u_dist[w] for w in d["good_tokens"].split()] for d in lm_data]
    bad_token_u = [[u_dist[w] for w in d["bad_tokens"].split()] for d in lm_data]
    good_u = [sum(probs) for probs in good_token_u]
    bad_u = [sum(probs) for probs in bad_token_u]

    # Load stimuli and acceptability ratings from Sprouse
    sprouse = pd.read_csv("../sprouse_data/sprouse_min_pairs.csv")

    good_idx = list(sprouse["Good ID"])
    bad_idx = list(sprouse["Bad ID"])
    good_s = list(sprouse["Good Sentence"])
    bad_s = list(sprouse["Bad Sentence"])
    good_a = list(sprouse["Good Sentence LS"])
    bad_a = list(sprouse["Bad Sentence LS"])

    # Compile the data
    n = len(good_p)
    df = pd.DataFrame({"Pair ID": list(range(n)) + list(range(n)),
                         "Grammaticality": ["Good"] * n + ["Bad"] * n,
                         "Sentence ID": good_idx + bad_idx,
                         "Sentence": good_s + bad_s,
                         "Length": good_l + bad_l,
                         "Acceptability": good_a + bad_a,
                         "Raw Logprob": good_p + bad_p,
                         "Raw Unigram": good_u + bad_u})
    return df


@click.command()
@click.argument(
    "model_family",
    help="Suite of models to evaluate",
    type=str,
    options=["opt", "pythia"],
    required=True,
)
@click.option(
    "--results_dir",
    help="Directory where LM logprobs are stored",
    type=str,
    default="../results/",
)
@click.option(
    "--unigram_n",
    help="Number of sentences sampled when estimating unigram logprobs",
    type=int,
    default=None,
)
@click.option(
    "--use_bos_token",
    help="For Pythia models, whether BOS token was used",
    type=bool,
    default=False,
)
@click.option(
    "--use_pile",
    help="Whether to use Pile unigram logprobs",
    type=bool,
    default=False,
)
@click.option(
    "--save_file",
    help="Name of the file to save linking function results to",
    type=str,
    required=True,
)
def main(
    model_family: str = None,
    results_dir: str = "../results/",
    unigram_n: int = None,
    use_bos_token: bool = False,
    use_pile: bool = False,
    save_file: str = None,
):
    if unigram_n is None and use_pile is False:
        unigram_n = 100000
    if model is not None:
        models = [model]
    else:
        if use_pile:
            models = PYTHIA_SUITE
        else:
            if model_family == "opt":
                models = OPT_SUITE
            elif model_family == "pythia":
                models = PYTHIA_SUITE
            else:
                models = MODELS.keys()
    model_results = []
    for model in tqdm(models):
        result_file_name = f"{model}.jsonl"
        if model in PYTHIA_SUITE:
            if not use_bos_token:
                result_file_name = f"{model}_bos-{use_bos_token}.jsonl"
        
        if unigram_n is None:
            results_exist = os.path.exists(os.path.join(results_dir, result_file_name))
        else:
            unigram_filename = f"{model}_n={unigram_n}.csv"
            results_exist = \
                os.path.exists(os.path.join(results_dir, result_file_name)) & \
                    os.path.exists(f"../unigram_logprobs/{unigram_filename}")
        if not results_exist:
            continue
        
        sprouse_raw = load_linguistic_inquiry_results(model, results_dir, result_file_name)
        function = LinkingFunction(sprouse_raw)
        function.add("Logprobs", None, 1, 0, 0, 0)
        function.add("SLOR", None, 1, -1, 0, 1)
        function.add("MORCELA, beta=1", None, 1, -1, None, 1)
        function.add("MORCELA, gamma=0", None, None, None, 0, 1)
        function.add("MORCELA", None, None, None, None, 1)

        result_df = function.results()
        result_df = result_df.round(3)
        model_col = [model] * len(result_df.index)
        result_df.insert(0, "Model", model_col)
        result_df = result_df.sort_values("corr", ascending=False)
        model_results.append(result_df)
        
    result_df = pd.concat(model_results)
    if not os.path.exists("../results"):
        os.makedirs("../results")
    result_df.to_csv(f"../results/{save_file}", index=True)


if __name__ == "__main__":
    main()
