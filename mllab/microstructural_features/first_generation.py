"""
First generation features (Roll Measure/Impact, Corwin-Schultz spread estimator)
"""

import numpy as np
import pandas as pd


def get_roll_measure(close_prices: pd.Series, window: int = 20) -> pd.Series:
    """
    Advances in Financial Machine Learning, page 282.

    Get Roll Measure

    Roll Measure gives the estimate of effective bid-ask spread
    without using quote-data.

    :param close_prices: (pd.Series) Close prices
    :param window: (int) Estimation window
    :return: (pd.Series) Roll measure
    """
    price_diff = close_prices.diff()
    roll_cov = price_diff.rolling(window).cov(price_diff.shift(1))
    roll_measure = 2 * np.sqrt(-roll_cov)
    return roll_measure.dropna()


def get_roll_impact(close_prices: pd.Series, dollar_volume: pd.Series, window: int = 20) -> pd.Series:
    """
    Get Roll Impact.

    Derivate from Roll Measure which takes into account dollar volume traded.

    :param close_prices: (pd.Series) Close prices
    :param dollar_volume: (pd.Series) Dollar volume series
    :param window: (int) Estimation window
    :return: (pd.Series) Roll impact
    """
    roll_measure = get_roll_measure(close_prices, window)
    roll_impact = roll_measure / dollar_volume.rolling(window).mean()
    return roll_impact.dropna()


def _get_beta(high: pd.Series, low: pd.Series, window: int) -> pd.Series:
    """
    Advances in Financial Machine Learning, Snippet 19.1, page 285.

    Get beta estimate from Corwin-Schultz algorithm

    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :param window: (int) Estimation window
    :return: (pd.Series) Beta estimates
    """
    hl_ratio = (high / low).rolling(window=2).mean()
    beta = np.log(hl_ratio) ** 2
    return beta.dropna()


def _get_gamma(high: pd.Series, low: pd.Series) -> pd.Series:
    """
    Advances in Financial Machine Learning, Snippet 19.1, page 285.

    Get gamma estimate from Corwin-Schultz algorithm.

    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :return: (pd.Series) Gamma estimates
    """
    gamma = np.log(high / low) ** 2
    return gamma.dropna()


def _get_alpha(beta: pd.Series, gamma: pd.Series) -> pd.Series:
    """
    Advances in Financial Machine Learning, Snippet 19.1, page 285.

    Get alpha from Corwin-Schultz algorithm.

    :param beta: (pd.Series) Beta estimates
    :param gamma: (pd.Series) Gamma estimates
    :return: (pd.Series) Alphas
    """
    alpha = np.sqrt(2 * beta) - np.sqrt(2 * gamma)
    return alpha.dropna()


def get_corwin_schultz_estimator(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """
    Advances in Financial Machine Learning, Snippet 19.1, page 285.

    Get Corwin-Schultz spread estimator using high-low prices

    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :param window: (int) Estimation window
    :return: (pd.Series) Corwin-Schultz spread estimators
    """
    beta = _get_beta(high, low, window)
    gamma = _get_gamma(high, low)
    alpha = _get_alpha(beta, gamma)
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread.dropna()


def get_bekker_parkinson_vol(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """
    Advances in Financial Machine Learning, Snippet 19.2, page 286.

    Get Bekker-Parkinson volatility from gamma and beta in Corwin-Schultz algorithm.

    :param high: (pd.Series) High prices
    :param low: (pd.Series) Low prices
    :param window: (int) Estimation window
    :return: (pd.Series) Bekker-Parkinson volatility estimates
    """
    gamma = _get_gamma(high, low)
    beta = _get_beta(high, low, window)
    bp_vol = np.sqrt(gamma - beta)
    return bp_vol.dropna()
