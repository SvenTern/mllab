# pylint: disable=missing-module-docstring
import numpy as np
import pandas as pd


class TailSetLabels:
    """
    Tail set labels are a classification labeling technique introduced in the following paper: Nonlinear support vector
    machines can systematically identify stocks with high and low future returns. Algorithmic Finance, 2(1), pp.45-58.

    A tail set is defined to be a group of stocks whose volatility-adjusted return is in the highest or lowest
    quantile, for example the highest or lowest 5%.

    A classification model is then fit using these labels to determine which stocks to buy and sell in a long / short
    portfolio.
    """

    def __init__(self, prices, n_bins, vol_adj=None, window=None):
        """
        :param prices: (pd.DataFrame) Asset prices.
        :param n_bins: (int) Number of bins to determine the quantiles for defining the tail sets. The top and
                        bottom quantiles are considered to be the positive and negative tail sets, respectively.
        :param vol_adj: (str) Whether to take volatility adjusted returns. Allowable inputs are ``None``,
                        ``mean_abs_dev``, and ``stdev``.
        :param window: (int) Window period used in the calculation of the volatility adjusted returns, if vol_adj is not
                        None. Has no impact if vol_adj is None.
        """

        self.prices = prices
        self.n_bins = n_bins
        self.vol_adj = vol_adj
        self.window = window

    def get_tail_sets(self):
        """
        Computes the tail sets (positive and negative) and then returns a tuple with 3 elements, positive set, negative
        set, full matrix set.

        The positive and negative sets are each a series of lists with the names of the securities that fall within each
        set at a specific timestamp.

        For the full matrix a value of 1 indicates the volatility adjusted returns were in the top quantile, a value of
        -1 for the bottom quantile.
        :return: (tuple) positive set, negative set, full matrix set.
        """

        vol_adj_rets = self._vol_adjusted_rets()
        full_matrix = vol_adj_rets.apply(self._extract_tail_sets, axis=1)

        positive_set = full_matrix.apply(self._positive_tail_set)
        negative_set = full_matrix.apply(self._negative_tail_set)

        return positive_set, negative_set, full_matrix

    def _vol_adjusted_rets(self):
        """
        Computes the volatility adjusted returns. This is simply the log returns divided by a volatility estimate. We
        have provided 2 techniques for volatility estimation: an exponential moving average and the traditional standard
        deviation.
        """
        log_rets = np.log(self.prices / self.prices.shift(1))

        if self.vol_adj == "mean_abs_dev":
            vol = log_rets.rolling(self.window).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=False)
        elif self.vol_adj == "stdev":
            vol = log_rets.rolling(self.window).std()
        else:
            vol = 1

        return log_rets / vol

    def _extract_tail_sets(self, row):
        """
        Method used in a .apply() setting to transform each row in a DataFrame to the positive and negative tail sets.

        This method splits the data into quantiles determined by the user, with n_bins.

        :param row: (pd.Series) Vol adjusted returns for a given date.
        :return: (pd.Series) Tail set with positive and negative labels.
        """
        pos_threshold = row.quantile(1 - 1 / self.n_bins)
        neg_threshold = row.quantile(1 / self.n_bins)

        return row.apply(lambda x: 1 if x >= pos_threshold else (-1 if x <= neg_threshold else 0))

    @staticmethod
    def _positive_tail_set(row):
        """
        Takes as input a row from the vol_adj_ret DataFrame and then returns a list of names of the securities in the
        positive tail set, for this specific row date.

        This method is used in an apply() setting.

        :param row: (pd.Series) Labeled row of several stocks where each is already labeled with +1 (positive tail set),
                    -1 (negative tail set), or 0.
        :return: (list) Securities in the positive tail set.
        """
        return row[row == 1].index.tolist()

    @staticmethod
    def _negative_tail_set(row):
        """
        Takes as input a row from the vol_adj_ret DataFrame and then returns a list of names of the securities in the
        negative tail set, for this specific row date.

        This method is used in an apply() setting.

        :param row: (pd.Series) Labeled row of several stocks where each is already labeled with +1 (positive tail set),
                    -1 (negative tail set), or 0.
        :return: (list) Securities in the negative tail set.
        """
        return row[row == -1].index.tolist()
