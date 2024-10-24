"""
Computes Rao–Blackwellized unigram probability scores from a language
model. Unigram probabilites are saved to: 
    unigram_logprobs/<model_name>_prompt=<prompt>_n=<n_samples>.csv

Usage: python estimate_unigrams.py <model_name> [-n <number of samples to generate>] [-b <batch size to generate>] [-p <prompt to generate text from>] [-d <use distributed inference>]
"""
import argparse
import itertools
from typing import List, Optional

import numpy as np
import os
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, \
    PreTrainedModel, PreTrainedTokenizer

from _utils import cache_df
from constants import MODELS
from get_linguistic_inquiry_probs import get_sentences

torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

_device = "cuda" if torch.cuda.is_available() else "cpu"


def _get_unigram_filename(clargs) -> str:
    name_ = (f"{clargs.model_name}_n={clargs.n_samples}.csv")
    name_ = name_.replace("/", "-")
    if not os.path.exists("../unigram_logprobs/"):
        os.makedirs("../unigram_logprobs/")
    return f"../unigram_logprobs/{name_}"


def _get_positional_unigram_probs(
        model: PreTrainedModel, tokenizer: PreTrainedTokenizer, length: int,
        batch_size: int = 50, n_samples: int = 50, prompt: str = "",
        vocab: Optional[List[int]] = None) -> np.ndarray:
    """
    :param model: A language model
    :param tokenizer: The tokenizer for model
    :param length: The length of responses for which the unigram
        distribution will be estimated
    :param batch_size: The batch size to use for sampling sentences
    :param n_samples: The number of sampled language model responses to
        be used in the estimation
    :param prompt: The prompt to which the language model is responding
    """
    global _device
    use_bos_token = tokenizer.add_bos_token
    tokenizer.add_bos_token = True

    x_len, y_len = None, None
    vocab = list(range(len(tokenizer))) if vocab is None else vocab
    counts = torch.zeros(length, len(vocab), requires_grad=False)
    for i in tqdm(range(0, n_samples, batch_size)):
        j = min(i + batch_size, n_samples)

        # Sample some sentences
        x = tokenizer([prompt] * (j - i), return_tensors="pt").to(_device)
        x_len = x["input_ids"].size(1) if x_len is None else x_len
        y_len = length + x_len if y_len is None else y_len
        y_hat = model.generate(
            **x, min_length=y_len, max_length=y_len, do_sample=True,
            return_dict_in_generate=True, output_logits=True)

        # Count frequencies
        with torch.no_grad():
            counts += F.softmax(torch.cat(
                [y.to("cpu").unsqueeze(1) for y in y_hat.logits], dim=1),
                dim=-1)[:, :, vocab].sum(dim=0)

    tokenizer.add_bos_token = use_bos_token
    return counts.numpy() / n_samples


@cache_df
def get_unigram_probs(
        positional_unigram_probs: np.ndarray, tokenizer: PreTrainedTokenizer,
        vocab: Optional[List[int]] = None) -> pd.DataFrame:
    """
    Given a prompt, this function estimates the unigram distribution for
    a randomly sampled response to that prompt of a certain length from
    a language model.

    :param positional_unigram_probs: Estimated probabilities from sampled 
        sentences
    :param tokenizer: The tokenizer for model
    :param length: The length of responses for which the unigram
        distribution will be estimated
    :param vocab: Tokens whose unigram log-probabilities will be
        estimated
        
    :return: The log-probabilities for each token in vocab
    """
    logprobs = np.log(positional_unigram_probs.mean(axis=0))
    tokens = [tokenizer.decode(id for id in vocab)]
    df = pd.DataFrame({
        "Index": vocab, "Token": tokens, "Unigram Logprob": logprobs})
    return df.sort_values(by="Unigram Logprob", ascending=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    parser.add_argument("-n", "--n_samples", type=int, default=100)
    parser.add_argument("-b", "--batch_size", type=int, default=100)
    parser.add_argument("-p", "--prompt", type=str, default="")
    parser.add_argument("-d", "--distributed", action="store_true")
    args = parser.parse_args()

    # Load model and tokenizer
    model_name = args.model_name
    tokenizer = AutoTokenizer.from_pretrained(
        MODELS[model_name], device_map="auto", add_bos_token=True)
    if args.distributed:
        model = AutoModelForCausalLM.from_pretrained(
            MODELS[model_name], device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODELS[model_name]).to(_device)

    # Get info about Sprouse sentences
    sentences = tokenizer(get_sentences())["input_ids"]
    length = max(len(s) for s in sentences)
    vocab = list(set(itertools.chain.from_iterable(sentences)))

    # Estimate unigram distribution
    raw_unigram_probs = _get_positional_unigram_probs(
        model, tokenizer, length, args.batch_size, 
        args.n_samples, args.prompt, vocab)

    unigram_logprobs_df = get_unigram_probs(
        raw_unigram_probs, tokenizer, vocab=vocab,
        cache_filename=_get_unigram_filename(args))
