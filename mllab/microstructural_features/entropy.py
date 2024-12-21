"""
Entropy calculation module (Shannon, Lempel-Ziv, Plug-In, Konto)
"""

import math
from typing import Union

import numpy as np
from numba import njit

def get_shannon_entropy(message: str) -> float:
    """
    Advances in Financial Machine Learning, page 263-264.

    Get Shannon entropy from message

    :param message: (str) Encoded message
    :return: (float) Shannon entropy
    """
    length = len(message)
    if length == 0:
        return 0.0

    frequency = {char: message.count(char) / length for char in set(message)}
    entropy = -sum(p * math.log2(p) for p in frequency.values())
    return entropy

def get_lempel_ziv_entropy(message: str) -> float:
    """
    Advances in Financial Machine Learning, Snippet 18.2, page 266.

    Get Lempel-Ziv entropy estimate

    :param message: (str) Encoded message
    :return: (float) Lempel-Ziv entropy
    """
    if not message:
        return 0.0

    substrings = set()
    i, k, l = 0, 1, len(message)
    while k <= l:
        while message[i:k] in substrings and k <= l:
            k += 1
        substrings.add(message[i:k])
        i = k
        k = i + 1

    return len(substrings) / len(message)

def _prob_mass_function(message: str, word_length: int) -> dict:
    """
    Advances in Financial Machine Learning, Snippet 18.1, page 266.

    Compute probability mass function for a one-dim discrete rv

    :param message: (str or array) Encoded message
    :param word_length: (int) Approximate word length
    :return: (dict) Dict of pmf for each word from message
    """
    if word_length <= 0 or word_length > len(message):
        raise ValueError("word_length must be greater than 0 and less than or equal to message length")

    words = [message[i:i + word_length] for i in range(len(message) - word_length + 1)]
    frequency = {word: words.count(word) / len(words) for word in set(words)}
    return frequency

def get_plug_in_entropy(message: str, word_length: int = None) -> float:
    """
    Advances in Financial Machine Learning, Snippet 18.1, page 265.

    Get Plug-in entropy estimator

    :param message: (str or array) Encoded message
    :param word_length: (int) Approximate word length
    :return: (float) Plug-in entropy
    """
    if not message:
        return 0.0

    if word_length is None:
        word_length = 1

    pmf = _prob_mass_function(message, word_length)
    entropy = -sum(p * math.log2(p) for p in pmf.values())
    return entropy

@njit()
def _match_length(message: str, start_index: int, window: int) -> Union[int, str]:    # pragma: no cover
    """
    Advances in Financial Machine Learning, Snippet 18.3, page 267.

    Function That Computes the Length of the Longest Match

    :param message: (str or array) Encoded message
    :param start_index: (int) Start index for search
    :param window: (int) Window length
    :return: (int, str) Match length and matched string
    """
    max_match = 0
    match_string = ""
    for i in range(max(0, start_index - window), start_index):
        length = 0
        while (start_index + length < len(message)) and (message[i + length] == message[start_index + length]):
            length += 1
            if i + length >= start_index:
                break

        if length > max_match:
            max_match = length
            match_string = message[start_index:start_index + length]

    return max_match, match_string

def get_konto_entropy(message: str, window: int = 0) -> float:
    """
    Advances in Financial Machine Learning, Snippet 18.4, page 268.

    Implementations of Algorithms Discussed in Gao et al.[2008]

    Get Kontoyiannis entropy

    :param message: (str or array) Encoded message
    :param window: (int) Expanding window length, can be negative
    :return: (float) Kontoyiannis entropy
    """
    if not message:
        return 0.0

    n = len(message)
    match_lengths = []
    for i in range(1, n):
        match_length, _ = _match_length(message, i, window if window > 0 else i)
        match_lengths.append(match_length)

    avg_match_length = np.mean(match_lengths)
    return math.log2(n) / avg_match_length if avg_match_length > 0 else 0.0
