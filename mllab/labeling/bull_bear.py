"""
Detection of bull and bear markets.
"""
import numpy as np
import pandas as pd

def pagan_sossounov(prices, window=8, censor=6, cycle=16, phase=4, threshold=0.2):
    """
    Pagan and Sossounov's labeling method for detecting bull and bear markets.

    :param prices: (pd.Series) Close prices of the market index.
    :param window: (int) Rolling window length to determine local extrema.
    :param censor: (int) Number of periods to eliminate for start and end.
    :param cycle: (int) Minimum length for a complete cycle.
    :param phase: (int) Minimum length for a phase.
    :param threshold: (float) Minimum threshold for phase change (proportional change).
    :return: (pd.Series) Labels: 1 for Bull, -1 for Bear.
    """
    # Calculate rolling maxima and minima
    peaks = prices.rolling(window, center=True).max()
    troughs = prices.rolling(window, center=True).min()

    # Remove extrema within censoring period
    peaks[:censor] = np.nan
    peaks[-censor:] = np.nan
    troughs[:censor] = np.nan
    troughs[-censor:] = np.nan

    # Identify turning points
    turning_points = pd.Series(index=prices.index, dtype='float64')
    for i in range(1, len(prices) - 1):
        if not np.isnan(peaks[i]) and prices[i] == peaks[i]:
            turning_points[i] = 1  # Peak
        elif not np.isnan(troughs[i]) and prices[i] == troughs[i]:
            turning_points[i] = -1  # Trough

    # Ensure alternation of peaks and troughs
    turning_points = _alternation(turning_points)

    # Assign labels based on phases
    labels = pd.Series(index=prices.index, dtype='float64')
    for i in range(1, len(turning_points)):
        if turning_points[i] == 1 and turning_points[i - 1] == -1:
            # Bull phase
            if (prices[i] / prices[i - 1]) - 1 >= threshold:
                labels[i] = 1
        elif turning_points[i] == -1 and turning_points[i - 1] == 1:
            # Bear phase
            if (prices[i] / prices[i - 1]) - 1 <= -threshold:
                labels[i] = -1

    return labels.fillna(method='ffill').fillna(method='bfill')

def _alternation(turning_points):
    """
    Helper function to ensure alternation of peaks and troughs.

    :param turning_points: (pd.Series) Detected turning points (1 for peak, -1 for trough).
    :return: (pd.Series) Alternating turning points.
    """
    alternated = turning_points.copy()
    for i in range(1, len(turning_points)):
        if turning_points[i] == turning_points[i - 1]:
            alternated[i] = np.nan
    return alternated.dropna()

def lunde_timmermann(prices, bull_threshold=0.15, bear_threshold=0.15):
    """
    Lunde and Timmermann's labeling method for detecting bull and bear markets.

    :param prices: (pd.Series) Close prices of the market index.
    :param bull_threshold: (float) Threshold for identifying a bull market (proportional change).
    :param bear_threshold: (float) Threshold for identifying a bear market (proportional change).
    :return: (pd.Series) Labels: 1 for Bull, -1 for Bear.
    """
    labels = pd.Series(index=prices.index, dtype='float64')

    for i in range(1, len(prices)):
        change = (prices[i] / prices[i - 1]) - 1
        if change >= bull_threshold:
            labels[i] = 1  # Bull market
        elif change <= -bear_threshold:
            labels[i] = -1  # Bear market

    return labels.fillna(0)
