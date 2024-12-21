"""
Logic regarding concurrent labels from chapter 4.
"""
import pandas as pd
from mllab.util.multiprocess import mp_pandas_obj


def num_concurrent_events(close_series_index, label_endtime, molecule):
    """
    Advances in Financial Machine Learning, Snippet 4.1, page 60.

    Estimating the Uniqueness of a Label

    This function uses close series prices and label endtime (when the first barrier is touched) to compute the number
    of concurrent events per bar.

    :param close_series_index: (pd.Index) Close prices index.
    :param label_endtime: (pd.Series) Label endtime series (t1 for triple barrier events).
    :param molecule: (array-like) A subset of datetime index values for processing.
    :return: (pd.Series) Number of concurrent labels for each datetime index.
    """
    out = pd.Series(0, index=close_series_index)
    for t_in, t_out in label_endtime.loc[molecule].items():
        # Increment the counter for the duration of the event
        out.loc[t_in:t_out] += 1
    return out


def _get_average_uniqueness(label_endtime, num_conc_events, molecule):
    """
    Advances in Financial Machine Learning, Snippet 4.2, page 62.

    Estimating the Average Uniqueness of a Label

    This function calculates the average uniqueness of each label using the number of concurrent events.

    :param label_endtime: (pd.Series) Label endtime series (t1 for triple barrier events).
    :param num_conc_events: (pd.Series) Number of concurrent labels (output from num_concurrent_events function).
    :param molecule: (array-like) A subset of datetime index values for processing.
    :return: (pd.Series) Average uniqueness over event's lifespan.
    """
    out = pd.Series(index=molecule)
    for t_in, t_out in label_endtime.loc[molecule].items():
        # Compute uniqueness over the label's duration
        concurrency = num_conc_events.loc[t_in:t_out]
        avg_uniq = (1.0 / concurrency).mean()
        out.loc[t_in] = avg_uniq
    return out


def get_av_uniqueness_from_triple_barrier(triple_barrier_events, close_series, num_threads, verbose=True):
    """
    Orchestrator function to derive average sample uniqueness from a dataset labeled by the triple barrier method.

    :param triple_barrier_events: (pd.DataFrame) Events from labeling.get_events(), must contain 't1' column.
    :param close_series: (pd.Series) Close prices indexed by time.
    :param num_threads: (int) The number of threads concurrently used by the function.
    :param verbose: (bool) Flag to report progress on async jobs.
    :return: (pd.Series) Average uniqueness over event's lifespan for each index in triple_barrier_events.
    """
    label_endtime = triple_barrier_events['t1'].dropna()

    # Step 1: Compute the number of concurrent events
    num_conc_events = mp_pandas_obj(
        func=num_concurrent_events,
        pd_obj=('molecule', close_series.index),
        num_threads=num_threads,
        close_series_index=close_series.index,
        label_endtime=label_endtime,
    )

    # Step 2: Compute average uniqueness
    avg_uniqueness = mp_pandas_obj(
        func=_get_average_uniqueness,
        pd_obj=('molecule', label_endtime.index),
        num_threads=num_threads,
        label_endtime=label_endtime,
        num_conc_events=num_conc_events,
    )

    if verbose:
        print("Completed average uniqueness computation.")
    return avg_uniqueness
