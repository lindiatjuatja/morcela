MODELS = {
        "opt-125m": "facebook/opt-125m",
        "opt-350m": "facebook/opt-350m",
        "opt-1.3b": "facebook/opt-1.3b",
        "opt-2.7b": "facebook/opt-2.7b",
        "opt-6.7b": "facebook/opt-6.7b",
        "opt-13b": "facebook/opt-13b",
        "opt-30b": "facebook/opt-30b",
        "pythia-14m": "EleutherAI/pythia-14m",
        "pythia-70m": "EleutherAI/pythia-70m",
        "pythia-160m": "EleutherAI/pythia-160m",
        "pythia-410m": "EleutherAI/pythia-410m",
        "pythia-1b": "EleutherAI/pythia-1b",
        "pythia-1.4b": "EleutherAI/pythia-1.4b",
        "pythia-2.8b": "EleutherAI/pythia-2.8b",
        "pythia-6.9b": "EleutherAI/pythia-6.9b",
        "pythia-12b": "EleutherAI/pythia-12b",
    }

OPT_SUITE = [
        "opt-125m",
        "opt-350m",
        "opt-1.3b",
        "opt-2.7b",
        "opt-6.7b",
        "opt-13b",
        "opt-30b",
]

PYTHIA_SUITE = [
        "pythia-14m",
        "pythia-70m",
        "pythia-160m",
        "pythia-410m",
        "pythia-1b",
        "pythia-1.4b",
        "pythia-2.8b",
        "pythia-6.9b",
        "pythia-12b",
]

def get_model_size(shorthand_name: str) -> int:
    factors = {
        "m": 10**6,
        "b": 10**9,
    }
    name_split = shorthand_name.split("-")
    if name_split[-1][-1] not in factors.keys():
        size_str = name_split[-2]
    else:
        size_str = name_split[-1]
    return float(size_str[:-1]) * factors[size_str[-1]]
