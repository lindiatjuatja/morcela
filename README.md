# MORCELA: Magnitude-Optimized Regression for Controlling Effects on Linguistic Acceptability

This repository contains code to reproduce experiments from *What Goes Into an LM Acceptability Judgment? Rethinking the Impact of Frequency and Length* (Tjuatja et al., 2024).

## Setup
Install the following major dependencies (versions listed are those used in this work):
* accelerate (v0.33.0)
* click (v8.1.7)
* datasets (v2.18.0)
* pandas (v2.1.4)
* pytorch (v2.2.0)
* scikit-learn (v1.4.2)
* scipy (v1.13.1)
* transformers (v4.42.3)
* tqdm (v4.65.0)

## Calculating LM Log Probabilities
This work uses materials from the Linguistic Inquiry data collected by [Sprouse et al. (2013)](https://www.jonsprouse.com/papers/Sprouse%20et%20al.%202013.pdf) (see `data/linguistic_inquiry_data.csv`). To calculate log probabilties to be used as input to linking functions (e.g. SLOR, MORCELA), from the `src/` directory run:
```
python get_linguistic_inquiry_probs.py \
    <model name> \
    --save_dir <directory to save logprobs> \
    --use_bos_token <set flag if using BOS token (for Pythia models)> \
    --distributed <set flag if using distributed inference>
```
By default, the save_dir is `../logprobs/`

## Calculating Unigram Estimates
For models where we don't have access to the training corpus (OPT models), we estimate unigram frequency of tokens. To run this estimation, from the `src/` directory run:
```
python estimate_unigrams.py \
    <model_name> \
    -n <number of samples to generate> \
    -b <batch size to generate> \
    -p <prompt to generate text from> \
    -d <set flag if using distributed inference>
```
For the experiments with OPT models, we use unigram estimates from OPT-30B.

## Calculating Correlation of LM Acceptability Scores with Human Judgments
To calculate predictions and correlation of various linking functions (log probabilities, SLOR, MORCELA, etc.) with human judgments from the Linguistic Inquiry data, from the `src/` directory run:
```
python eval_linking_functions.py \
    <model_family ("opt" or "pythia")> \
    --logprobs_dir <directory where model log probs are saved> \
    --unigram_file <filepath to unigram logprobs>
    --use_bos_token <set flag if using BOS token (for Pythia models)> \
    --save_dir <directory to save predictions and eval results> \
    --save_filename <filename for eval results>
```
