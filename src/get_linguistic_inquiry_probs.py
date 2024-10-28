"""
code to generate per-token probabilities for sentences
in Linguistic Inquiry data from Sprouse & Almeida 2017
"""

import click
import math
import os
import csv

import pandas as pd
import json
import torch

from datasets import Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, \
    GPT2LMHeadModel, PreTrainedTokenizer, OPTForCausalLM, LlamaForCausalLM
from typing import Union, Tuple

from constants import MODELS, PYTHIA_SUITE, OPT_SUITE

if torch.cuda.is_available():
    _device = "cuda"
else:
    _device = "cpu"


def get_sentences() -> list[str]:
    """
    Loads the sentences from the Linguistic Inquiry dataset. This function
    is used in unigram.py.
    """
    with open("../data/linguistic_inquiry_data.csv", "r") as f:
        reader = csv.reader(f)
        _ = next(reader)
        bad, good = zip(*[(row[2], row[3]) for row in reader])
    return bad + good


def append_bos_token(tokenizer_output, bos_token_id):
    batch_input_ids = []
    batch_attention_masks = []
    for i in range(tokenizer_output["input_ids"].shape[0]):
        input_ids = tokenizer_output["input_ids"][i].tolist()
        attention_mask = tokenizer_output["attention_mask"][i].tolist()
        input_ids = [bos_token_id] + input_ids
        attention_mask = [1] + attention_mask
        batch_input_ids.append(torch.LongTensor(input_ids))
        batch_attention_masks.append(torch.LongTensor(attention_mask))
    tokenizer_output["input_ids"] = torch.unsqueeze(torch.cat(batch_input_ids, dim=0), 0)
    tokenizer_output["attention_mask"] = torch.unsqueeze(torch.cat(batch_attention_masks, dim=0), 0)
    return tokenizer_output


def get_token_probs(
    logits: torch.Tensor,  # [batch_size, seq_len, vocab_size]
    token_ids: torch.Tensor,  # [batch_size, seq_len]
) -> Tuple[torch.Tensor, list[int]]:
    # returns probabilities of token in a sequence along with their rank
    vocab_size = logits.shape[-1]
    m = torch.nn.Softmax(dim=-1)
    probs = m(logits)
    gather_token_ids = token_ids.unsqueeze(2).repeat(1, 1, vocab_size)
    token_probs = torch.gather(probs, -1, gather_token_ids)[:, :, 0].view(logits.shape[0], logits.shape[1])
    return token_probs


def evaluate(
    model: Union[AutoModelForCausalLM, GPT2LMHeadModel, OPTForCausalLM, LlamaForCausalLM],
    tokenizer: PreTrainedTokenizer,
    add_bos_token: bool = False
) -> list:
    """
    Evaluates a model on Linguistic Inquiry data.

    :param model: A Huggingface model to be evaluated
    :param tokenizer: The tokenizer for the model
    :param metric: Evaluating metric. By default, compare sequence logprobs.
    :param batch_size: The batch size for evaluation
    """
    df = pd.read_csv("../sprouse_data/sprouse_min_pairs.csv")
    df = df.rename(columns={"Good Sentence": "sentence_good", "Bad Sentence": "sentence_bad"})
    data = Dataset.from_pandas(df)
    n_total = len(data)

    model.eval()

    results = []
    batch_size = 1
    for batch in tqdm(data.iter(batch_size=batch_size),
                        total=math.ceil(n_total / batch_size)):
        good_input_tokens = tokenizer(
            batch["sentence_good"], return_tensors="pt", padding=False
        )
        if add_bos_token:
            good_input_tokens = append_bos_token(good_input_tokens, tokenizer.bos_token_id)
        good_input_tokens = good_input_tokens.to(_device)
        good_labels = good_input_tokens["input_ids"][:, 1:]
        good_logits = model(**good_input_tokens).logits[:, :-1]
        good_token_probs = get_token_probs(good_logits, good_labels)

        bad_input_tokens = tokenizer(
            batch["sentence_bad"], return_tensors="pt", padding=False, add_special_tokens=True
        )
        if add_bos_token:
            bad_input_tokens = append_bos_token(bad_input_tokens, tokenizer.bos_token_id)
        bad_input_tokens = bad_input_tokens.to(_device)
        bad_labels = bad_input_tokens["input_ids"][:, 1:]
        bad_logits = model(**bad_input_tokens).logits[:, :-1]
        bad_token_probs = get_token_probs(bad_logits, bad_labels)

        for i in range(batch_size):
            sent_result = {}
            sent_result["good_tokens"] = " ".join(tokenizer.decode(id for id in good_labels[i]))
            sent_result["bad_tokens"] = " ".join(tokenizer.decode(id for id in bad_labels[i]))
            sent_result["good_token_probs"] = good_token_probs[i].tolist()
            sent_result["bad_token_probs"] = bad_token_probs[i].tolist()
            results.append(sent_result)
    return results


@click.command()
@click.argument(
    "model_name",
    help="Model name",
    type=str,
)
@click.option(
    "--save_dir",
    help="Directory to save results",
    required=True,
)
@click.option(
    "--use_bos_token",
    help="Whether to add bos token for Pythia models",
    is_flag=True,
)
@click.option(
    "--distributed",
    help="Whether to use distributed inference",
    is_flag=True,
)
def main(
    model_name: str,
    save_dir: str = "../logprobs/",
    use_bos_token: bool = False,
    distributed: bool = False,
):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    assert model_name in MODELS.keys()
    tokenizer = AutoTokenizer.from_pretrained(MODELS[model_name])
    if distributed:
        if model_name in OPT_SUITE:
            model = OPTForCausalLM.from_pretrained(MODELS[model_name], device_map="auto")
        else:
            model = AutoModelForCausalLM.from_pretrained(MODELS[model_name], device_map="auto")
    else:
        if model_name in OPT_SUITE:
            model = OPTForCausalLM.from_pretrained(MODELS[model_name]).to(_device)
        else:
            model = AutoModelForCausalLM.from_pretrained(MODELS[model_name]).to(_device)

    add_bos_token = False
    if use_bos_token:
        add_bos_token = True

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    results = evaluate(
        model,
        tokenizer,
        add_bos_token=add_bos_token,
    )
    filepath = f"{save_dir}/{model_name}.jsonl"
    if model in PYTHIA_SUITE:
        filepath = f"{save_dir}/{model_name}_bos-{add_bos_token}.jsonl"    
    print(f"saving results to {filepath} ...")
    with open(filepath, 'w') as out:
        for dict in results:
            jout = json.dumps(dict, ensure_ascii=False) + '\n'
            out.write(jout)


if __name__ == "__main__":
    main()
