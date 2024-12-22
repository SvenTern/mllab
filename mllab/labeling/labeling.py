"""
Logic regarding labeling from chapter 3. In particular the Triple Barrier Method and Meta-Labeling.
"""

import numpy as np
import pandas as pd

# Snippet 3.1, page 44, Daily Volatility Estimates
from mllab.util.multiprocess import mp_pandas_obj


# Snippet 3.2, page 45, Triple Barrier Labeling Method
def apply_pt_sl_on_t1(**kwargs):  # pragma: no cover
    """
    Advances in Financial Machine Learning, Snippet 3.2, page 45.

    Triple Barrier Labeling Method

    This function applies the triple-barrier labeling method. It works on a set of
    datetime index values (molecule). This allows the program to parallelize the processing.

    Mainly it returns a DataFrame of timestamps regarding the time when the first barriers were reached.

    :param close: (pd.Series) Close prices
    :param events: (pd.Series) Indices that signify "events" (see cusum_filter function
    for more details)
    :param pt_sl: (np.array) Element 0, indicates the profit taking level; Element 1 is stop loss level
    :param molecule: (an array) A set of datetime index values for processing
    :return: (pd.DataFrame) Timestamps of when first barrier was touched
    """
    opt = kwargs['kwargs']
    molecule = opt['molecule']
    close = opt['close']
    events = opt['events']
    pt_sl = opt['pt_sl']
    normalized_data = opt['normalized_data']

    results = pd.DataFrame(index=molecule)
    for idx, loc in enumerate(molecule, start=0):

        if loc not in events.index:
            continue
        df0 = close[loc:events.loc[loc, 't1']]  # Path prices
        trgt = events.loc[loc, 'trgt']

        # Profit taking and stop loss
        pt = trgt * pt_sl[0] if pt_sl[0] > 0 else np.nan
        sl = -trgt * pt_sl[1] if pt_sl[1] > 0 else np.nan

        # Barriers
        if normalized_data:
            # Barriers
            # нужен барьер trgt волатильность
            results.loc[loc, 'sl'] = df0[df0 <= sl].index.min()
            results.loc[loc, 'pt'] = df0[df0 >= pt].index.min()
        else:
            results.loc[loc, 'sl'] = df0[(df0 / close[loc] - 1) <= sl].index.min()
            results.loc[loc, 'pt'] = df0[(df0 / close[loc] - 1) >= pt].index.min()

        # нужно поставить минимальное время где происходит пересечение барьеров ...
        results.loc[loc, 't1'] = min(events.loc[loc, 't1'], results.loc[loc, 'sl'], results.loc[loc, 'pt'])

        # если минимальное время начало интервала то нужно перенести на начало следующего интервала
        if results.loc[loc, 't1'] == loc and idx + 1 < len(molecule):
            results.loc[loc, 't2'] = molecule[idx + 1]
        else:
            results.loc[loc, 't2'] = results.loc[loc, 't1']

    return results


# Snippet 3.4 page 49, Adding a Vertical Barrier
def add_vertical_barrier(t_events, close, num_days=0, num_hours=0, num_minutes=0, num_seconds=0):
    """
    Advances in Financial Machine Learning, Snippet 3.4 page 49.

    Adding a Vertical Barrier

    For each index in t_events, it finds the timestamp of the next price bar at or immediately after
    a number of days num_days. This vertical barrier can be passed as an optional argument t1 in get_events.

    This function creates a series that has all the timestamps of when the vertical barrier would be reached.

    :param t_events: (pd.Series) Series of events (symmetric CUSUM filter)
    :param close: (pd.Series) Close prices
    :param num_days: (int) Number of days to add for vertical barrier
    :param num_hours: (int) Number of hours to add for vertical barrier
    :param num_minutes: (int) Number of minutes to add for vertical barrier
    :param num_seconds: (int) Number of seconds to add for vertical barrier
    :return: (pd.Series) Timestamps of vertical barriers
    """
    timedelta = pd.Timedelta(days=num_days, hours=num_hours, minutes=num_minutes, seconds=num_seconds)
    t1 = t_events + timedelta
    t1 = t1[t1 <= close.index[-1]]  # Ensure barriers are within data range
    return t1


# Snippet 3.3 -> 3.6 page 50, Getting the Time of the First Touch, with Meta Labels
def get_events(close, t_events, pt_sl, target, min_ret=None, num_threads=1, vertical_barrier_times=None,
               side_prediction=None, normalized_data:bool=False, verbose=True ):
    """
    Advances in Financial Machine Learning, Snippet 3.6 page 50.

    Getting the Time of the First Touch, with Meta Labels

    This function is orchestrator to meta-label the data, in conjunction with the Triple Barrier Method.

    :param close: (pd.Series) Close prices
    :param t_events: (pd.Series) of t_events. These are timestamps that will seed every triple barrier.
        These are the timestamps selected by the sampling procedures discussed in Chapter 2, Section 2.5.
        Eg: CUSUM Filter
    :param pt_sl: (2 element array) Element 0, indicates the profit taking level; Element 1 is stop loss level.
        A non-negative float that sets the width of the two barriers. A 0 value means that the respective
        horizontal barrier (profit taking and/or stop loss) will be disabled.
    :param target: (pd.Series) of values that are used (in conjunction with pt_sl) to determine the width
        of the barrier. In this program this is daily volatility series.
    :param min_ret: (float) The minimum target return required for running a triple barrier search.
                    If None, it will be set to the median of the target series.
    :param num_threads: (int) The number of threads concurrently used by the function.
    :param vertical_barrier_times: (pd.Series) A pandas series with the timestamps of the vertical barriers.
        We pass a False when we want to disable vertical barriers.
    :param side_prediction: (pd.Series) Side of the bet (long/short) as decided by the primary model
    :param verbose: (bool) Flag to report progress on asynch jobs
    :return: (pd.DataFrame) Events
            -events.index is event's starttime
            -events['t1'] is event's endtime
            -events['trgt'] is event's target
            -events['side'] (optional) implies the algo's position side
            -events['pt'] is profit taking multiple
            -events['sl']  is stop loss multiple
    """
    # проверка на соответствие длины t_events и количества барьеров vertical_barrier_times
    if len(t_events) > len(vertical_barrier_times):
        t_events = t_events[:len(vertical_barrier_times)+1]

    # Auto-set min_ret if not provided
    if not min_ret is None:
        min_ret = target.median()
        # Filter events based on min_ret
        target = target[target > min_ret]

    t_events = t_events.intersection(target.index)

    # Set vertical barriers
    if isinstance(vertical_barrier_times, (pd.Series, pd.DataFrame, pd.DatetimeIndex)):
        t1 = pd.Series(vertical_barrier_times, index=t_events)
    else:
        t1 = pd.Series(pd.NaT, index=t_events)

    # Create events DataFrame
    events = pd.DataFrame({'t1': t1, 'trgt': target.loc[t_events]}, index=t_events)
    if side_prediction is not None:
        events['side'] = side_prediction.loc[events.index]

    # Apply Triple Barrier Labeling
    # close, events, pt_sl, molecule
    df0 = mp_pandas_obj(
        func=apply_pt_sl_on_t1,
        pd_obj=('molecule', events.index),
        num_threads=num_threads,
        close=close,
        events=events,
        pt_sl=pt_sl,
        normalized_data=normalized_data
    )
    events = events.assign(**df0)
    return events


# Snippet 3.9, pg 55, Question 3.3
def barrier_touched(out_df, events):
    """
    Advances in Financial Machine Learning, Snippet 3.9, page 55, Question 3.3.

    Adjust the getBins function (Snippet 3.7) to return a 0 whenever the vertical barrier is the one touched first.

    Top horizontal barrier: 1
    Bottom horizontal barrier: -1
    Vertical barrier: 0

    :param out_df: (pd.DataFrame) Returns and target
    :param events: (pd.DataFrame) The original events data frame. Contains the pt sl multiples needed here.
    :return: (pd.DataFrame) Returns, target, and labels
    """
    for loc in out_df.index:
        sl = out_df.loc[loc, 'sl']
        pt = out_df.loc[loc, 'pt']
        t1 = out_df.loc[loc, 't1']

        if pd.isna(sl) and pd.isna(pt):
            out_df.loc[loc, 'label'] = 0
        elif not pd.isna(sl) and (pd.isna(pt) or sl <= pt):
            out_df.loc[loc, 'label'] = -1
        elif not pd.isna(pt) and (pd.isna(sl) or pt <= sl):
            out_df.loc[loc, 'label'] = 1
        else:
            out_df.loc[loc, 'label'] = 0
    return out_df


# Snippet 3.4 -> 3.7, page 51, Labeling for Side & Size with Meta Labels
def get_bins(triple_barrier_events, close, normalized_data: bool = False):
    """
    Advances in Financial Machine Learning, Snippet 3.7, page 51.

    Labeling for Side & Size with Meta Labels

    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:

    Now the possible values for labels in out['bin'] are {0,1}, as opposed to whether to take the bet or pass,
    a purely binary prediction. When the predicted label the previous feasible values {−1,0,1}.
    The ML algorithm will be trained to decide is 1, we can use the probability of this secondary prediction
    to derive the size of the bet, where the side (sign) of the position has been set by the primary model.

    :param triple_barrier_events: (pd.DataFrame)
                -events.index is event's starttime
                -events['t1'] is event's endtime
                -events['trgt'] is event's target
                -events['side'] (optional) implies the algo's position side
                Case 1: ('side' not in events): bin in (-1,1) <-label by price action
                Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    :param close: (pd.Series) Close prices
    :return: (pd.DataFrame) Meta-labeled events
    """
    out = triple_barrier_events[['t1', 'trgt']].copy()
    for loc, event in triple_barrier_events.iterrows():
        df0 = close[loc:event['t2']]  # Path prices
        if normalized_data:
            df0 = (df0) * event['side'] if 'side' in event else df0
        else:
            df0 = (df0 / close[loc] - 1) * event['side'] if 'side' in event else df0 / close[loc] - 1

        # Assign labels
        out.loc[loc, 'bin'] = 0
        if df0.max() > event['trgt']:
            out.loc[loc, 'bin'] = 1
        if df0.min() < -event['trgt']:
            out.loc[loc, 'bin'] = -1

        if out.loc[loc, 'bin'] > 0:
            out.loc[loc, 'return'] = close[loc:event['t2']].max()
        elif out.loc[loc, 'bin'] < 0:
            out.loc[loc, 'return'] = close[loc:event['t2']].min()
        else:
            out.loc[loc, 'return'] = 0

    return out


# Snippet 3.8 page 54
def drop_labels(events, min_pct=.05):
    """
    Advances in Financial Machine Learning, Snippet 3.8 page 54.

    This function recursively eliminates rare observations.

    :param events: (dp.DataFrame) Events.
    :param min_pct: (float) A fraction used to decide if the observation occurs less than that fraction.
    :return: (pd.DataFrame) Events.
    """
    while True:
        df0 = events['bin'].value_counts(normalize=True)
        if df0.min() > min_pct or df0.shape[0] < 3:
            break
        events = events[events['bin'] != df0.idxmin()]
    return events
