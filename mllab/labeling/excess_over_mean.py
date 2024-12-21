"""
Return in excess of mean method.

Chapter 5, Machine Learning for Factor Investing, by Coqueret and Guida, (2020).
"""
import numpy as np
import pandas as pd

class ExcessReturnCalculator:
    """
    A class to calculate returns in excess of the mean for a portfolio of stocks.

    This method is based on Chapter 5.5.1 of "Machine Learning for Factor Investing"
    by Coqueret, G. and Guida, T. (2020).
    """
    def __init__(self, prices, resample_by=None, lag=True):
        """
        Initialize the ExcessReturnCalculator with stock price data and optional parameters.

        :param prices: (pd.DataFrame) Close prices of all tickers in the market that are used to establish the mean.
        :param resample_by: (str) If not None, the resampling period for price data prior to calculating returns.
                            For example, 'B' for business day, 'W' for week, etc.
        :param lag: (bool) If True, returns will be lagged to make them forward-looking.
        """
        self.prices = prices
        self.resample_by = resample_by
        self.lag = lag

    def calculate_excess_returns(self, binary=False):
        """
        Calculate returns in excess of the mean for the portfolio.

        :param binary: (bool) If False, the numerical value of excess returns over mean will be given. If True,
                       only the sign of the excess return over mean will be given (-1, 1, or 0 if equal to the mean).
        :return: (pd.DataFrame) DataFrame of numerical returns or binary signs of returns in excess of the market mean.
        """
        prices = self.prices

        if self.resample_by:
            prices = prices.resample(self.resample_by).last()

        returns = prices.pct_change()

        if self.lag:
            returns = returns.shift(-1)

        mean_returns = returns.mean(axis=1)
        excess_returns = returns.sub(mean_returns, axis=0)

        if binary:
            excess_returns = excess_returns.applymap(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))

        return excess_returns

# Example function retained for standalone use
def excess_over_mean(prices, binary=False, resample_by=None, lag=True):
    """
    Return in excess of mean labeling method. Sourced from Chapter 5.5.1 of Machine Learning for Factor Investing,
    by Coqueret, G. and Guida, T. (2020).

    Returns a DataFrame containing returns of stocks over the mean of all stocks in the portfolio. Returns a DataFrame
    of signs of the returns if binary is True. In this case, an observation may be labeled as 0 if it itself is the
    mean.

    :param prices: (pd.DataFrame) Close prices of all tickers in the market that are used to establish the mean. NaN
                    values are ok. Returns on each ticker are then compared to the mean for the given timestamp.
    :param binary: (bool) If False, the numerical value of excess returns over mean will be given. If True, then only
                    the sign of the excess return over mean will be given (-1 or 1). A label of 0 will be given if
                    the observation itself equals the mean.
    :param resample_by: (str) If not None, the resampling period for price data prior to calculating returns. 'B' = per
                        business day, 'W' = week, 'M' = month, etc. Will take the last observation for each period.
                        For full details see `here.
                        <https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#dateoffset-objects>`_
    :param lag: (bool) If True, returns will be lagged to make them forward-looking.
    :return: (pd.DataFrame) Numerical returns in excess of the market mean return, or sign of return depending on
                whether binary is False or True respectively.
    """
    calculator = ExcessReturnCalculator(prices, resample_by, lag)
    return calculator.calculate_excess_returns(binary=binary)
