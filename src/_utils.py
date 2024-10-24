"""
Helper functions
"""
import os.path
import pickle
import random
import time
from contextlib import contextmanager
from enum import Enum
from typing import Callable, Generator, Optional

import pandas as pd


def cache_pickle(func: Callable) -> Callable:
    """
    A decorator that caches the output of a function to a pickle file.
    It adds a keyword parameter called "cache_filename" that gives the
    name of the cached file.
    """

    def _func(*func_args, cache_filename: Optional[str] = None, **func_kwargs):
        if cache_filename is not None and os.path.isfile(cache_filename):
            message = "Loading cached function output from {}..." \
                      "".format(cache_filename)
            with timer(message):
                with open(cache_filename, "rb") as f:
                    return pickle.load(f)

        return_val = func(*func_args, **func_kwargs)

        if cache_filename is not None:
            message = "Saving function output to {}..." \
                      "".format(cache_filename)
            with timer(message):
                with open(cache_filename, "wb") as f:
                    pickle.dump(return_val, f)

        return return_val

    return _func


def cache_df(func: Callable[..., pd.DataFrame]) -> Callable[..., pd.DataFrame]:
    """
    A decorator that caches the output of a function to a csv file,
    asssuming that the output is a pandas DataFrame. It adds a keyword
    parameter called "cache_filename" that gives the name of the cached
    file.
    """

    def _func(*func_args, cache_filename: Optional[str] = None, **func_kwargs):
        if cache_filename is not None and os.path.isfile(cache_filename):
            msg = f"Loading the output of function {func.__name__} from " \
                  f"{cache_filename}..."
            with timer(msg):
                return pd.read_csv(cache_filename)

        return_val = func(*func_args, **func_kwargs)

        if cache_filename is not None:
            msg = f"Saving the output of function {func.__name__} to " \
                  f"{cache_filename}..."
            with timer(msg):
                return_val.to_csv(cache_filename)

        return return_val

    return _func


@contextmanager
def timer(message: str) -> Generator[Callable[[None], float], None, None]:
    """
    A timer that measures the time spent running code within a with-
    block.

    :param message: A message to be printed when the timer starts.
    :return: A generator that yields a function that returns the current
        time elapsed.
    """
    print_delay(message)

    # Start timer
    start_time = time.time()
    yield lambda: time.time() - start_time

    # Stop timer
    end_time = time.time()
    elapsed = end_time - start_time
    print_delay("Done. Time elapsed: {:.3f} seconds".format(elapsed))


@contextmanager
def random_seed(seed: Optional[int]) -> Generator[None, None, None]:
    """
    Temporarily sets a random seed within a with-block.

    :param seed: The seed to set
    """
    if seed is not None:
        state = random.getstate()
        random.seed(seed)
        yield
        random.setstate(state)
    else:
        yield


Casing = Enum("Casing", "none lower capitalize both")


def set_case(text: str, casing_style: Casing) -> str:
    """
    Converts a string to all lowercase, with the first letter optionally
    capitalized.

    :param text: The text to convert.
    :param casing_style: If "none," the text will be returned as-is. If
        "lower," the text will be made all lowercase. If "capitalize,"
        the first letter will be capitalized
    :return: The text, with modifications made according to casing_style
    """
    if casing_style == Casing.none:
        return text
    elif casing_style == Casing.lower:
        return text.lower()
    elif casing_style == Casing.capitalize:
        return text.capitalize()


def print_delay(*args, delay: float = .05, **kwargs):
    time.sleep(delay)
    print(*args, **kwargs)
    time.sleep(delay)
