"""
Chapter 3.2 Fixed-Time Horizon Method, in Advances in Financial Machine Learning, by M. L. de Prado.

Work "Classification-based Financial Markets Prediction using Deep Neural Networks" by Dixon et al. (2016) describes how
labeling data this way can be used in training deep neural networks to predict price movements.
"""



import warnings
import pandas as pd

def fixed_time_horizon(prices, threshold=0, resample_by=None, lag=True, standardized=False, window=None):
    """
    Fixed-Time Horizon Labeling Method.

    Originally described in the book Advances in Financial Machine Learning, Chapter 3.2, p.43-44.

    Returns 1 if return is greater than the threshold, -1 if less, and 0 if in between. If no threshold is
    provided then it will simply take the sign of the return.

    :param prices: (pd.Series or pd.DataFrame) Time-indexed stock prices used to calculate returns.
    :param threshold: (float or pd.Series) When the absolute value of return exceeds the threshold, the observation is
                    labeled with 1 or -1, depending on the sign of the return. If return is less, it's labeled as 0.
                    Can be dynamic if threshold is inputted as a pd.Series, and threshold.index must match prices.index.
                    If resampling is used, the index of threshold must match the index of prices after resampling.
                    If threshold is negative, then the directionality of the labels will be reversed. If no threshold
                    is provided, it is assumed to be 0 and the sign of the return is returned.
    :param resample_by: (str) If not None, the resampling period for price data prior to calculating returns. 'B' = per
                        business day, 'W' = week, 'M' = month, etc. Will take the last observation for each period.
                        For full details see `here.
                        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects>`_
    :param lag: (bool) If True, returns will be lagged to make them forward-looking.
    :param standardized: (bool) Whether returns are scaled by mean and standard deviation.
    :param window: (int) If standardized is True, the rolling window period for calculating the mean and standard
                    deviation of returns.
    :return: (pd.Series or pd.DataFrame) -1, 0, or 1 denoting whether the return for each observation is
                    less/between/greater than the threshold at each corresponding time index. First or last row will be
                    NaN, depending on lag.
    """
    if not isinstance(prices, (pd.Series, pd.DataFrame)):
        raise TypeError("Prices must be a pandas Series or DataFrame.")

    # Resample prices if resample_by is provided
    if resample_by:
        prices = prices.resample(resample_by).last()

    # Calculate returns
    returns = prices.pct_change()
    if lag:
        returns = returns.shift(-1)  # Forward-looking returns

    if standardized:
        if window is None:
            raise ValueError("Window must be provided for standardization.")
        rolling_mean = returns.rolling(window=window, min_periods=1).mean()
        rolling_std = returns.rolling(window=window, min_periods=1).std()
        returns = (returns - rolling_mean) / rolling_std

    # Apply threshold
    if isinstance(threshold, (int, float)):
        labels = returns.applymap(lambda x: 1 if x > threshold else (-1 if x < -threshold else 0)) if isinstance(prices, pd.DataFrame) else \
                 returns.apply(lambda x: 1 if x > threshold else (-1 if x < -threshold else 0))
    elif isinstance(threshold, pd.Series):
        if not threshold.index.equals(returns.index):
            raise ValueError("Threshold index must match the returns index.")
        labels = returns.apply(lambda x, th: 1 if x > th else (-1 if x < -th else 0), args=(threshold,))

    return labels

def classify_price_movements(prices, thresholds, resample_by=None, lag=True):
    """
    Classification-based Financial Markets Prediction.

    This function demonstrates how to use fixed-time horizon labeling to classify price movements for training
    deep neural networks or other predictive models.

    Work "Classification-based Financial Markets Prediction using Deep Neural Networks" by Dixon et al. (2016) describes how
    labeling data this way can be used in training deep neural networks to predict price movements.

    :param prices: (pd.Series or pd.DataFrame) Time-indexed stock prices used to calculate returns.
    :param thresholds: (float or pd.Series) Threshold values for labeling price movements.
    :param resample_by: (str) Optional, period for resampling prices.
    :param lag: (bool) Optional, if True, returns will be lagged.
    :return: (pd.DataFrame) A DataFrame with classified labels.
    """
    labels = fixed_time_horizon(prices, threshold=thresholds, resample_by=resample_by, lag=lag)
    return labels
