"""
Second generation models features: Kyle lambda, Amihud Lambda, Hasbrouck lambda (bar and trade based)
"""

from typing import List
import numpy as np
import pandas as pd

# pylint: disable=invalid-name
def get_bar_based_kyle_lambda(close: pd.Series, volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Advances in Financial Machine Learning, p. 286-288.

    Get Kyle lambda from bars data.

    :param close: (pd.Series) Close prices
    :param volume: (pd.Series) Bar volume
    :param window: (int) Rolling window used for estimation
    :return: (pd.Series) Kyle lambdas
    """
    price_diff = close.diff().dropna()
    volume = volume.loc[price_diff.index]  # Align volumes with price differences

    # Calculate lambda using rolling regression
    def calculate_lambda(prices, vols):
        regression = np.polyfit(vols, prices, 1)  # Linear regression (slope is lambda)
        return regression[0]

    return price_diff.rolling(window).apply(
        lambda x: calculate_lambda(x, volume.loc[x.index]),
        raw=False
    )

def get_bar_based_amihud_lambda(close: pd.Series, dollar_volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Advances in Financial Machine Learning, p.288-289.

    Get Amihud lambda from bars data.

    :param close: (pd.Series) Close prices
    :param dollar_volume: (pd.Series) Dollar volumes
    :param window: (int) rolling window used for estimation
    :return: (pd.Series) of Amihud lambda
    """
    log_ret = np.log(close).diff().abs().dropna()
    dollar_volume = dollar_volume.loc[log_ret.index]  # Align with returns

    return log_ret.rolling(window).apply(
        lambda x: (x / dollar_volume.loc[x.index]).mean(),
        raw=False
    )

def get_bar_based_hasbrouck_lambda(close: pd.Series, dollar_volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Advances in Financial Machine Learning, p.289-290.

    Get Hasbrouck lambda from bars data.

    :param close: (pd.Series) Close prices
    :param dollar_volume: (pd.Series) Dollar volumes
    :param window: (int) Rolling window used for estimation
    :return: (pd.Series) Hasbrouck lambda
    """
    log_ret = np.log(close).diff().dropna()
    dollar_volume = dollar_volume.loc[log_ret.index]  # Align with returns

    def calculate_lambda(logrets, dvols):
        regression = np.polyfit(dvols, logrets, 1)  # Linear regression
        return regression[0]

    return log_ret.rolling(window).apply(
        lambda x: calculate_lambda(x, dollar_volume.loc[x.index]),
        raw=False
    )

def get_trades_based_kyle_lambda(price_diff: list, volume: list, aggressor_flags: list) -> List[float]:
    """
    Advances in Financial Machine Learning, p.286-288.

    Get Kyle lambda from trades data.

    :param price_diff: (list) Price diffs
    :param volume: (list) Trades sizes
    :param aggressor_flags: (list) Trade directions [-1, 1]  (tick rule or aggressor side can be used to define)
    :return: (list) Kyle lambda for a bar and t-value
    """
    price_impact = np.multiply(price_diff, aggressor_flags)  # Directional price movement
    regression = np.polyfit(volume, price_impact, 1)  # Linear regression
    return [regression[0], regression[1]]  # Lambda and t-value

def get_trades_based_amihud_lambda(log_ret: list, dollar_volume: list) -> List[float]:
    """
    Advances in Financial Machine Learning, p.288-289.

    Get Amihud lambda from trades data.

    :param log_ret: (list) Log returns
    :param dollar_volume: (list) Dollar volumes (price * size)
    :return: (float) Amihud lambda for a bar
    """
    amihud = np.mean(np.abs(log_ret) / dollar_volume)
    return [amihud]

def get_trades_based_hasbrouck_lambda(log_ret: list, dollar_volume: list, aggressor_flags: list) -> List[float]:
    """
    Advances in Financial Machine Learning, p.289-290.

    Get Hasbrouck lambda from trades data.

    :param log_ret: (list) Log returns
    :param dollar_volume: (list) Dollar volumes (price * size)
    :param aggressor_flags: (list) Trade directions [-1, 1]  (tick rule or aggressor side can be used to define)
    :return: (list) Hasbrouck lambda for a bar and t value
    """
    price_impact = np.multiply(log_ret, aggressor_flags)  # Directional log returns
    regression = np.polyfit(dollar_volume, price_impact, 1)  # Linear regression
    return [regression[0], regression[1]]  # Lambda and t-value
