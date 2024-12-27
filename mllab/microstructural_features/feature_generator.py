"""
Inter-bar feature generator which uses trades data and bars index to calculate inter-bar features
"""

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm.notebook import tqdm

from mllab.microstructural_features.entropy import get_shannon_entropy, get_plug_in_entropy, get_lempel_ziv_entropy, \
    get_konto_entropy
from mllab.microstructural_features.encoding import encode_array
from mllab.microstructural_features.second_generation import get_trades_based_kyle_lambda, \
    get_trades_based_amihud_lambda, get_trades_based_hasbrouck_lambda
from mllab.microstructural_features.misc import get_avg_tick_size, vwap
from mllab.microstructural_features.encoding import encode_tick_rule_array
from mllab.util.misc import crop_data_frame_in_batches


class MicrostructuralFeaturesGenerator:
    """
    Class which is used to generate inter-bar features when bars are already compressed.

    :param trades_input: (str or pd.DataFrame) Path to the csv file or Pandas DataFrame containing raw tick data
                                               in the format [date_time, price, volume]
    :param tick_num_series: (pd.Series) Series of tick number where bar was formed.
    :param batch_size: (int) Number of rows to read in from the csv, per batch.
    :param volume_encoding: (dict) Dictionary of encoding scheme for trades size used to calculate entropy on encoded messages
    :param pct_encoding: (dict) Dictionary of encoding scheme for log returns used to calculate entropy on encoded messages
    """

    def __init__(self, trades_input: (str, pd.DataFrame), tick_num_series: pd.Series, batch_size: int = 2e7,
                 volume_encoding: dict = None, pct_encoding: dict = None):
        """
        Constructor

        :param trades_input: (str or pd.DataFrame) Path to the csv file or Pandas DataFrame containing raw tick data
                                                   in the format [date_time, price, volume]
        :param tick_num_series: (pd.Series) Series of tick number where bar was formed.
        :param batch_size: (int) Number of rows to read in from the csv, per batch.
        :param volume_encoding: (dict) Dictionary of encoding scheme for trades size used to calculate entropy on encoded messages
        :param pct_encoding: (dict) Dictionary of encoding scheme for log returns used to calculate entropy on encoded messages
        """
        self.trades_input = trades_input
        self.tick_num_series = tick_num_series
        self.batch_size = batch_size
        self.volume_encoding = volume_encoding
        self.pct_encoding = pct_encoding

    def get_features(self, verbose=True, to_csv=False, output_path=None):
        """
        Reads a csv file of ticks or pd.DataFrame in batches and then constructs corresponding microstructural intra-bar features:
        average tick size, tick rule sum, VWAP, Kyle lambda, Amihud lambda, Hasbrouck lambda, tick/volume/pct Shannon, Lempel-Ziv,
        Plug-in entropies if corresponding mapping dictionaries are provided (self.volume_encoding, self.pct_encoding).
        The csv file must have only 3 columns: date_time, price, & volume.

        :param verbose: (bool) Flag whether to print message on each processed batch or not
        :param to_csv: (bool) Flag for writing the results of bars generation to local csv file, or to in-memory DataFrame
        :param output_path: (str) Path to results file, if to_csv = True
        :return: (DataFrame or None) Microstructural features for bar index
        """
        pass

    def _reset_cache(self):
        """
        Reset price_diff, trade_size, tick_rule, log_ret arrays to empty when bar is formed and features are
        calculated

        :return: None
        """
        pass

    def _extract_bars(self, data):
        """
        For loop which calculates features for formed bars using trades data.

        :param data: (tuple) Contains 3 columns - date_time, price, and volume.
        :return: None
        """
        pass

    def _get_bar_features(self, date_time: pd.Timestamp, list_bars: list) -> list:
        """
        Calculate inter-bar features: lambdas, entropies, avg_tick_size, vwap.

        :param date_time: (pd.Timestamp) When bar was formed.
        :param list_bars: (list) Previously formed bars.
        :return: (list) Inter-bar features.
        """
        pass

    @staticmethod
    def _apply_tick_rule(price: float) -> int:
        """
        Advances in Financial Machine Learning, page 29.

        Applies the tick rule.

        :param price: (float) Price at time t.
        :return: (int) The signed tick.
        """
        pass

    @staticmethod
    def _get_price_diff(price: float) -> float:
        """
        Get price difference between ticks.

        :param price: (float) Price at time t.
        :return: (float) Price difference.
        """
        pass

    @staticmethod
    def _get_log_ret(price: float) -> float:
        """
        Get log return between ticks.

        :param price: (float) Price at time t.
        :return: (float) Log return.
        """
        pass

    @staticmethod
    def _assert_csv(test_batch: pd.DataFrame):
        """
        Tests that the csv file read has the format: date_time, price, and volume.
        If not then the user needs to create such a file. This format is in place to remove any unwanted overhead.

        :param test_batch: (pd.DataFrame) The first row of the dataset.
        :return: None
        """
        pass

# Feature Engineering

# нужно найти все корреляции

#авто регрессия
#история
#объемы торгов
#взвешенные объемы торгов
#бенчмарки SP_500_TICKE и по отрасллям
#отношения с линиями боллинджера
#рси
#иакроиндексами

def calculate_indicators(data):
    """
    # Feature Engineering

    # нужно найти все корреляции

    #авто регрессия
    #история
    #объемы торгов
    #взвешенные объемы торгов
    #бенчмарки SP_500_TICKE и по отрасллям
    #отношения с линиями боллинджера
    #рси
    #иакроиндексами

    """


    def process_ticker(group, tic):
        group = group.copy()
        x = pd.DataFrame(index=group.index)
        data_row = group['close'].shift(1)
        data_row_volume = group['volume'].shift(1)
        data_row_vwap = group['vwap'].shift(1)
        data_row_transactions = group['transactions'].shift(1)

        # Log-returns
        group["log_ret"] = np.log(data_row).diff()

        # Log-returns volumes
        group["log_ret_volumes"] = np.log(data_row_volume).diff()

        # Volatility
        #x["volatility_50"] = group["log_ret"].rolling(window=50, min_periods=50, center=False).std()
        #x["volatility_31"] = group["log_ret"].rolling(window=31, min_periods=31, center=False).std()
        #x["volatility_15"] = group["log_ret"].rolling(window=15, min_periods=15, center=False).std()

        # Autocorrelation
        #window_autocorr = 10
        #autocorrs = group["log_ret"].rolling(window=window_autocorr, min_periods=window_autocorr).apply(
        #    lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 1 else np.nan, raw=True
        #)
        #for lag in range(1, 6):
        #    x[f"autocorr_{lag}"] = autocorrs.shift(lag - 1)

        # Log-return momentum
        for lag in range(1, 6):
            x[f"log_t{lag}"] = group["log_ret"].shift(lag)

        # Moving Averages
        x["ma_10"] = (data_row.rolling(window=10).mean() - data_row) / data_row
        x["ma_50"] = (data_row.rolling(window=50).mean() - data_row) / data_row
        x["ma_200"] = (data_row.rolling(window=200).mean() - data_row) / data_row

        # Bollinger Bands (20-day moving average, 2 standard deviations)
        rolling_window = 20
        x["bollinger_ma"] = data_row.rolling(window=rolling_window).mean()
        std_dev = data_row.rolling(window=rolling_window).std()
        x["bollinger_upper"] = (x["bollinger_ma"] + 2 * std_dev - data_row) / data_row
        x["bollinger_lower"] = (x["bollinger_ma"] - 2 * std_dev - data_row) / data_row

        # Exponential Moving Averages
        #x["ema_12"] = (data_row.ewm(span=12, adjust=False).mean() - data_row) / data_row
        #x["ema_26"] = (data_row.ewm(span=26, adjust=False).mean() - data_row) / data_row

        # MACD (Moving Average Convergence Divergence)
        x["macd"] = x["ema_12"] - x["ema_26"]
        x["macd_signal"] = x["macd"].ewm(span=9, adjust=False).mean()

        # RSI (Relative Strength Index)
        delta = data_row.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        x["rsi"] = 100 - (100 / (1 + rs))

        # Volume Indicators
        #x["volume_ma_10"] = data_row_volume.rolling(window=10).mean()
        #x["volume_ma_50"] = data_row_volume.rolling(window=50).mean()
        #x["volume_ma_200"] = data_row_volume.rolling(window=200).mean()
        #x["volume_volatility"] = data_row_volume.rolling(window=20).std() / data_row_volume.mean()

        # VWAP (Volume Weighted Average Price)
        x["vwap_diff"] = (data_row_vwap - data_row) / data_row

        # Transaction-based Indicators
        #x["transactions_ma_10"] = data_row_transactions.rolling(window=10).mean()
        #x["transactions_ma_50"] = data_row_transactions.rolling(window=50).mean()
        #x["transactions_volatility"] = data_row_transactions.rolling(window=20).std() / data_row_transactions.mean()

        # Add ticker information to the results
        x["tic"] = tic
        return x

    # Parallel processing of tickers with progress bar
    tickers = data['tic'].unique()
    results = []
    with tqdm(total=len(tickers), desc="Processing tickers") as pbar:
        for group, tic in Parallel(n_jobs=-1)(
            delayed(lambda g, t: (g, t))(group, tic) for tic, group in data.groupby('tic')
        ):
            results.append(process_ticker(group, tic))
            pbar.update(1)

    # Combine all results into a single DataFrame
    final_result = pd.concat(results)
    final_result = final_result.reindex(data.index)

    return final_result.fillna(0)

