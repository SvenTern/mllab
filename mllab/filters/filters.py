"""
Filters are used to filter events based on some kind of trigger. For example a structural break filter can be
used to filter events where a structural break occurs. This event is then used to measure the return from the event
to some event horizon, say a day.
"""

import numpy as np
import pandas as pd

# Snippet 2.4, page 39, The Symmetric CUSUM Filter.
def cusum_filter(raw_time_series, threshold, time_stamps=True, normalized_data: bool = False):
    """
    Advances in Financial Machine Learning, Snippet 2.4, page 39.

    The Symmetric Dynamic/Fixed CUSUM Filter.

    The CUSUM filter is a quality-control method, designed to detect a shift in the mean value of a measured quantity
    away from a target value. The filter is set up to identify a sequence of upside or downside divergences from any
    reset level zero. We sample a bar t if and only if S_t >= threshold, at which point S_t is reset to 0.

    One practical aspect that makes CUSUM filters appealing is that multiple events are not triggered by raw_time_series
    hovering around a threshold level, which is a flaw suffered by popular market signals such as Bollinger Bands.
    It will require a full run of length threshold for raw_time_series to trigger an event.

    Once we have obtained this subset of event-driven bars, we will let the ML algorithm determine whether the occurrence
    of such events constitutes actionable intelligence. Below is an implementation of the Symmetric CUSUM filter.

    Note: As per the book this filter is applied to closing prices but we extended it to also work on other
    time series such as volatility.

    :param raw_time_series: (pd.Series) Close prices (or other time series, e.g. volatility).
    :param threshold: (float or pd.Series) When the abs(change) is larger than the threshold, the function captures
                      it as an event, can be dynamic if threshold is pd.Series
    :param time_stamps: (bool) Default is to return a DateTimeIndex, change to false to have it return a list.
    :return: (datetime index vector) Vector of datetimes when the events occurred. This is used later to sample.
    """
    t_events = []
    s_pos, s_neg = 0, 0

    if isinstance(threshold, (float, int)):
        threshold = pd.Series(threshold, index=raw_time_series.index)

    if normalized_data:
        diff = raw_time_series.dropna()
    else:
        diff = raw_time_series.pct_change().dropna()

    for i in diff.index:
        s_pos = max(0, s_pos + diff.loc[i])
        s_neg = min(0, s_neg + diff.loc[i])

        if s_pos > threshold.loc[i]:
            s_pos = 0
            t_events.append(i)
        elif s_neg < -threshold.loc[i]:
            s_neg = 0
            t_events.append(i)

    if time_stamps:
        return pd.DatetimeIndex(t_events)
    return t_events

def z_score_filter(raw_time_series, mean_window, std_window, z_score=3, time_stamps=True):
    """
    Filter which implements z_score filter
    (https://stackoverflow.com/questions/22583391/peak-signal-detection-in-realtime-timeseries-data)

    :param raw_time_series: (pd.Series) Close prices (or other time series, e.g. volatility).
    :param mean_window: (int): Rolling mean window
    :param std_window: (int): Rolling std window
    :param z_score: (float): Number of standard deviations to trigger the event
    :param time_stamps: (bool) Default is to return a DateTimeIndex, change to false to have it return a list.
    :return: (datetime index vector) Vector of datetimes when the events occurred. This is used later to sample.
    """
    rolling_mean = raw_time_series.rolling(window=mean_window).mean()
    rolling_std = raw_time_series.rolling(window=std_window).std()

    z_scores = (raw_time_series - rolling_mean) / rolling_std
    t_events = z_scores[abs(z_scores) > z_score].index

    if time_stamps:
        return pd.DatetimeIndex(t_events)
    return list(t_events)
