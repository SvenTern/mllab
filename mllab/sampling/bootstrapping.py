"""
Logic regarding sequential bootstrapping from chapter 4.
"""
import pandas as pd
import numpy as np
from numba import jit, prange

def get_ind_matrix(samples_info_sets, price_bars):
    """
    Advances in Financial Machine Learning, Snippet 4.3, page 65.

    Build an Indicator Matrix

    Get indicator matrix. The book implementation uses bar_index as input, however there is no explanation
    how to form it. We decided that using triple_barrier_events and price bars by analogy with concurrency
    is the best option.

    :param samples_info_sets: (pd.Series): Triple barrier events (t1) from labeling.get_events
    :param price_bars: (pd.DataFrame): Price bars which were used to form triple barrier events
    :return: (np.array) Indicator binary matrix indicating what (price) bars influence the label for each observation
    """
    ind_matrix = np.zeros((len(samples_info_sets), len(price_bars)), dtype=int)
    for i, (start, end) in enumerate(samples_info_sets.iteritems()):
        ind_matrix[i, price_bars.index.get_loc(start):price_bars.index.get_loc(end) + 1] = 1
    return ind_matrix

def get_ind_mat_average_uniqueness(ind_mat):
    """
    Advances in Financial Machine Learning, Snippet 4.4, page 65.

    Compute Average Uniqueness

    Average uniqueness from indicator matrix.

    :param ind_mat: (np.array) Indicator binary matrix
    :return: (float) Average uniqueness
    """
    concurrency = ind_mat.sum(axis=0)
    uniqueness = ind_mat / concurrency[np.newaxis, :]
    avg_uniqueness = uniqueness.mean(axis=1)
    return avg_uniqueness.mean()

def get_ind_mat_label_uniqueness(ind_mat):
    """
    Advances in Financial Machine Learning, An adaptation of Snippet 4.4, page 65.

    Returns the indicator matrix element uniqueness.

    :param ind_mat: (np.array) Indicator binary matrix
    :return: (np.array) Element-wise uniqueness matrix
    """
    concurrency = ind_mat.sum(axis=0)
    uniqueness = ind_mat / concurrency[np.newaxis, :]
    return uniqueness

@jit(parallel=True, nopython=True)
def _bootstrap_loop_run(ind_mat, prev_concurrency):
    """
    Part of Sequential Bootstrapping for-loop. Using previously accumulated concurrency array, loops through all samples
    and generates averages uniqueness array of labels based on previously accumulated concurrency.

    :param ind_mat: (np.array) Indicator matrix from get_ind_matrix function
    :param prev_concurrency: (np.array) Accumulated concurrency from previous iterations of sequential bootstrapping
    :return: (np.array) Label average uniqueness based on prev_concurrency
    """
    avg_uniqueness = np.zeros(ind_mat.shape[0])
    for i in prange(ind_mat.shape[0]):
        concurrency = prev_concurrency + ind_mat[i]
        uniqueness = ind_mat[i] / concurrency
        avg_uniqueness[i] = uniqueness.mean()
    return avg_uniqueness

def seq_bootstrap(ind_mat, sample_length=None, warmup_samples=None, compare=False, verbose=False,
                  random_state=np.random.RandomState()):
    """
    Advances in Financial Machine Learning, Snippet 4.5, Snippet 4.6, page 65.

    Return Sample from Sequential Bootstrap

    Generate a sample via sequential bootstrap.
    Note: Moved from pd.DataFrame to np.matrix for performance increase.

    :param ind_mat: (np.array) Indicator matrix from triple barrier events
    :param sample_length: (int) Length of bootstrapped sample
    :param warmup_samples: (list) List of previously drawn samples
    :param compare: (bool) Flag to print standard bootstrap uniqueness vs sequential bootstrap uniqueness
    :param verbose: (bool) Flag to print updated probabilities on each step
    :param random_state: (np.random.RandomState) Random state
    :return: (list) Bootstrapped sample indices
    """
    n_samples = ind_mat.shape[0]
    sample_length = sample_length or n_samples
    sample = [] if warmup_samples is None else warmup_samples[:]

    prev_concurrency = np.zeros(ind_mat.shape[1])

    for _ in range(sample_length - len(sample)):
        avg_uniqueness = _bootstrap_loop_run(ind_mat, prev_concurrency)
        probs = avg_uniqueness / avg_uniqueness.sum()
        selected = random_state.choice(np.arange(n_samples), p=probs)
        sample.append(selected)
        prev_concurrency += ind_mat[selected]

        if verbose:
            print(f"Selected: {selected}, Probabilities: {probs}")

    if compare:
        standard_uniqueness = get_ind_mat_average_uniqueness(ind_mat)
        seq_uniqueness = get_ind_mat_average_uniqueness(ind_mat[sample])
        print(f"Standard Bootstrap Uniqueness: {standard_uniqueness}")
        print(f"Sequential Bootstrap Uniqueness: {seq_uniqueness}")

    return sample
