"""
Various miscellaneous microstructural features (VWAP, average tick size)
"""

import numpy as np

def vwap(dollar_volume: list, volume: list) -> float:
    """
    Get Volume Weighted Average Price (VWAP).

    :param dollar_volume: (list) Dollar volumes
    :param volume: (list) Trades sizes
    :return: (float) VWAP value
    """
    if len(dollar_volume) != len(volume):
        raise ValueError("dollar_volume and volume must have the same length")

    total_dollar_volume = sum(dollar_volume)
    total_volume = sum(volume)

    if total_volume == 0:
        raise ValueError("Total volume cannot be zero")

    return total_dollar_volume / total_volume

def get_avg_tick_size(tick_size_arr: list) -> float:
    """
    Get average tick size in a bar.

    :param tick_size_arr: (list) Trade sizes
    :return: (float) Average trade size
    """
    if not tick_size_arr:
        raise ValueError("tick_size_arr cannot be empty")

    return np.mean(tick_size_arr)
