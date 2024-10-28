"""
Calculates various linking functions using the Linguistic Inquiry data from Sprouse et. al (2017)

Outputs a CSV file with the linking function results for the specified model family to <save_dir>/<save_filename>
    as well as predictions per model in the <save_dir>/predictions directory
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
        logprobs_dir: str,
        result_file_name: str,
        unigram_file: str,
    ) -> pd.DataFrame:
    # Load LLM probability scores and lengths (in tokens)
    with open(logprobs_dir / result_file_name) as f:
        lm_data = [json.loads(line) for line in f]

    good_token_p = [[math.log(p) for p in r["good_token_probs"]] for r in lm_data]
    bad_token_p = [[math.log(p) for p in r["bad_token_probs"]] for r in lm_data]

    good_p = [sum(probs) for probs in good_token_p]
    bad_p = [sum(probs) for probs in bad_token_p]
    good_l = [len(r["good_token_probs"]) for r in lm_data]
    bad_l = [len(r["bad_token_probs"]) for r in lm_data]

    # Load unigram distribution
    if unigram_file.split(".")[-1] == "json":
        with open(unigram_file) as f:
            u_dist = json.load(f)
    else:
        u_dist = pd.read_csv(unigram_file)
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
)
@click.option(
    "--logprobs_dir",
    help="Directory where LM logprobs are stored",
    type=str,
    default="../logprobs/",
)
@click.option(
    "--unigram_file",
    help="Filepath of unigram logprobs",
    type=click.Path(exists=True),
    required=True,
)
@click.option(
    "--use_bos_token",
    help="For Pythia models, whether BOS token was used",
    is_flag=True,
)
@click.option(
    "--save_dir",
    help="Directory to save predictions and linking function results to",
    type=str,
    required=True,
)
@click.option(
    "--save_filename",
    help="Name of the file to save linking function results to",
    type=str,
    required=True,
)
def main(
    model_family,
    logprobs_dir: str = "../logprobs/",
    unigram_file: str = None,
    use_bos_token: bool = False,
    save_dir: str = "../results/",
    save_filename: str = None,
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    if model_family == "pythia":
            models = PYTHIA_SUITE
    else:
        models = OPT_SUITE

    model_results = []
    for model in tqdm(models):
        result_file_name = f"{model}.jsonl"
        if model in PYTHIA_SUITE:
            if not use_bos_token:
                result_file_name = f"{model}_bos-{use_bos_token}.jsonl"
        
        results_exist = \
            os.path.exists(os.path.join(logprobs_dir, result_file_name)) & \
                os.path.exists(unigram_file)
        if not results_exist:
            continue
        
        sprouse_raw = load_linguistic_inquiry_results(
            logprobs_dir, result_file_name, unigram_file)
        function = LinkingFunction(sprouse_raw)
        function.add("Logprobs", None, 1, 0, 0, 0)
        function.add("SLOR", None, 1, -1, 0, 1)
        function.add("MORCELA, beta=1", None, 1, -1, None, 1)
        function.add("MORCELA, gamma=0", None, None, None, 0, 1)
        function.add("MORCELA", None, None, None, None, 1)
        
        # save predictions per model
        if not os.path.exists(f"{save_dir}/predictions"):
            os.makedirs(f"{save_dir}/predictions")
        function.df.to_csv(f"{save_dir}/predictions/{model}.csv", index=False)

        result_df = function.results()
        result_df = result_df.round(3)
        model_col = [model] * len(result_df.index)
        result_df.insert(0, "Model", model_col)
        result_df = result_df.sort_values("corr", ascending=False)
        model_results.append(result_df)
        
    result_df = pd.concat(model_results)
    result_df.to_csv(f"{save_dir}/{save_filename}", index=True)


if __name__ == "__main__":
    main()
