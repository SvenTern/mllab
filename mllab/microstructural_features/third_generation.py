"""
Third generation models implementation (VPIN)

This module provides the implementation of the Volume-Synchronized Probability of Informed Trading (VPIN) model.
The VPIN metric is derived from bar data to estimate the probability of informed trading.

References:
Advances in Financial Machine Learning, p. 292-293.
"""

import pandas as pd

def get_vpin(volume: pd.Series, buy_volume: pd.Series, window: int = 1) -> pd.Series:
    """
    Get Volume-Synchronized Probability of Informed Trading (VPIN) from bars.

    This function calculates the VPIN metric, which measures the probability of informed trading
based on volume imbalances between buy and sell trades within a given estimation window.

    :param volume: (pd.Series) Bar volume.
    :param buy_volume: (pd.Series) Bar volume classified as buy (either tick rule, BVC or aggressor side methods applied).
    :param window: (int) Estimation window.
    :return: (pd.Series) VPIN series.
    """
    # Calculate sell volume
    sell_volume = volume - buy_volume

    # Calculate volume imbalance (absolute difference between buy and sell volumes)
    imbalance = abs(buy_volume - sell_volume)

    # Calculate normalized imbalance
    normalized_imbalance = imbalance / volume

    # Compute VPIN as rolling mean of normalized imbalance
    vpin = normalized_imbalance.rolling(window=window, min_periods=1).mean()

    return vpin
