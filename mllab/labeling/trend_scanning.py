"""
Implementation of Trend-Scanning labels described in `Advances in Financial Machine Learning: Lecture 3/10
<https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2708678>`_
"""

import pandas as pd
import numpy as np

from mllab.structural_breaks.sadf import get_betas

def trend_scanning_labels(price_series: pd.Series, t_events: list = None, observation_window: int = 20,
                          look_forward: bool = True, min_sample_length: int = 5, step: int = 1, normalized_data:bool = False) -> pd.DataFrame:
    """
    Trend scanning labels implementation.

    :param price_series: (pd.Series) Close prices used to label the data set
    :param t_events: (list) Filtered events, array of pd.Timestamps
    :param observation_window: (int) Maximum look forward window used to get the trend value
    :param look_forward: (bool) True if using a forward-looking window, False if using a backward-looking one
    :param min_sample_length: (int) Minimum sample length used to fit regression
    :param step: (int) Optimal t-value index is searched every 'step' indices
    :return: (pd.DataFrame) Consists of t1, t-value, ret, bin (label information). t1 - label endtime, tvalue,
        ret - price change %, bin - label value based on price change sign
    """
    # Initialize results DataFrame
    results = pd.DataFrame(index=t_events, columns=['t1', 'tvalue', 'ret', 'bin'])

    for t in t_events:
        if look_forward:
            window_prices = price_series[t:t + pd.Timedelta(minutes=observation_window)]
        else:
            window_prices = price_series[t - pd.Timedelta(minutes=observation_window):t]

        if len(window_prices) < min_sample_length:
            continue

        max_tvalue, max_tvalue_index = -np.inf, None

        for i in range(min_sample_length, len(window_prices), step):
            sub_window_prices = window_prices.iloc[:i]
            if normalized_data:
                y = sub_window_prices.dropna().values
            else:
                y = sub_window_prices.pct_change().dropna().values
            X = np.arange(len(y)).reshape(-1, 1)

            if len(y) < min_sample_length:
                continue

            # Regression fit and t-value computation
            betas = get_betas(X, y, add_intercept=True)
            tvalue = betas['t_values'][1]  # t-value of slope

            if tvalue > max_tvalue:
                max_tvalue = tvalue
                max_tvalue_index = sub_window_prices.index[-1]

        if max_tvalue_index is not None:
            t1 = max_tvalue_index
            if normalized_data:
                # нужно суммировать все изменения за период от t - t1
                ret = sum(price_series[t:t1])
            else:
                ret = price_series[t1] / price_series[t] - 1

            bin_label = np.sign(max_tvalue)

            results.loc[t] = [t1, max_tvalue, ret, bin_label]

    results.dropna(inplace=True)
    return results
