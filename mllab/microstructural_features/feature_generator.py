"""
Inter-bar feature generator which uses trades data and bars index to calculate inter-bar features
"""

import pandas as pd
import numpy as np
from joblib import Parallel, delayed
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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


def calculate_indicators(data,
                         sp500_data=None,
                         sector_data=None,
                         macro_data=None,
                         correlation_windows=[20, 60, 120]):
    """
    Feature Engineering

    Нужно найти все корреляции:
      - Авто регрессия
      - История
      - Объемы торгов, взвешенные объемы
      - Бенчмарки (SP500, отраслевые)
      - Отношения с линиями Боллинджера
      - RSI
      - Макроиндексы

    Parameters
    ----------
    data : pd.DataFrame
        Данные цен, объёмов и т. д. (обязательно содержит колонки
        ['tic', 'timestamp', 'close', 'volume', 'vwap', 'transactions']).
    sp500_data : pd.DataFrame or None
        Данные для индекса S&P 500 (или любого другого бенчмарка).
        Должен содержать колонку 'close' и те же даты, что и в data.
    sector_data : pd.DataFrame or None
        Аналогично для отраслевого индекса/ETF.
    macro_data : pd.DataFrame or None
        Макро-показатели (например, CPI, VIX, Interest Rates и пр.).
    correlation_windows : list
        Размеры скользящих окон, по которым считать корреляции.

    Returns
    -------
    pd.DataFrame
        Финальный DataFrame с вычисленными фичами.
        Изначально индексом будет (tic, timestamp),
        но при reset_index мы перенесём их в колонки.
    """

    def process_ticker(group, tic):
        """
        Вычисляет фичи для одного тикера (группа строк).
        """
        group = group.copy()
        x = pd.DataFrame(index=group.index)

        # --- Сдвигаем видимый интервал на 1 шаг ---
        data_row = group['close'].shift(1)
        data_row_volume = group['volume'].shift(1)
        data_row_vwap = group['vwap'].shift(1)
        data_row_transactions = group['transactions'].shift(1)

        # --- Log Returns ---
        group["log_ret"] = np.log(data_row).diff()

        # --- Log Returns of Volumes ---
        group["log_ret_volumes"] = np.log(data_row_volume).diff()

        # --- Волатильность на разных окнах ---
        x["volatility_5"] = group["log_ret"].rolling(window=50, min_periods=50).std()
        x["volatility_10"] = group["log_ret"].rolling(window=31, min_periods=31).std()
        x["volatility_15"] = group["log_ret"].rolling(window=15, min_periods=15).std()

        # --- Автокорреляция log_ret ---
        window_autocorr = 10
        autocorrs = (
            group["log_ret"]
            .rolling(window=window_autocorr, min_periods=window_autocorr)
            .apply(
                lambda series: np.corrcoef(series[:-1], series[1:])[0, 1]
                if len(series) > 1 else np.nan,
                raw=True
            )
        )
        for lag in range(1, 6):
            x[f"autocorr_{lag}"] = autocorrs.shift(lag - 1)

        # --- Моментум (log_return) ---
        for lag in range(1, 6):
            x[f"log_t{lag}"] = group["log_ret"].shift(lag)

        # --- Скользящие средние (MA) ---
        x["ma_10"] = (data_row.rolling(window=10).mean() - data_row) / data_row
        x["ma_50"] = (data_row.rolling(window=50).mean() - data_row) / data_row
        x["ma_200"] = (data_row.rolling(window=200).mean() - data_row) / data_row

        # --- Полосы Боллинджера (Bollinger Bands) ---
        rolling_window = 20
        x["bollinger_ma"] = data_row.rolling(window=rolling_window).mean()
        std_dev = data_row.rolling(window=rolling_window).std()
        x["bollinger_upper"] = (x["bollinger_ma"] + 2 * std_dev - data_row) / data_row
        x["bollinger_lower"] = (x["bollinger_ma"] - 2 * std_dev - data_row) / data_row

        # --- Экспоненциальные скользящие средние (EMA) ---
        x["ema_12"] = (data_row.ewm(span=12, adjust=False).mean() - data_row) / data_row
        x["ema_26"] = (data_row.ewm(span=26, adjust=False).mean() - data_row) / data_row

        # --- MACD ---
        x["macd"] = x["ema_12"] - x["ema_26"]
        x["macd_signal"] = x["macd"].ewm(span=9, adjust=False).mean()
        for lag in range(1, 6):
            x[f"macd_log_t{lag}"] = x["macd"].shift(lag)

        # --- RSI (Relative Strength Index) ---
        delta = data_row.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        x["rsi"] = 100 - (100 / (1 + rs))
        for lag in range(1, 6):
            x[f"rsi_log_t{lag}"] = x["rsi"].shift(lag)

        # --- VWAP (Volume Weighted Average Price) ---
        x["vwap_diff"] = (data_row_vwap - data_row) / data_row

        # --- Корреляции с бенчмарками / макро ---
        if sp500_data is not None and 'close' in sp500_data.columns:
            sp500_close = sp500_data.loc[group.index, 'close'].shift(1)
            sp500_logret = np.log(sp500_close).diff()
            for window in correlation_windows:
                x[f"corr_sp500_{window}"] = (
                    group["log_ret"]
                    .rolling(window=window, min_periods=1)
                    .corr(sp500_logret)
                )

        if sector_data is not None and 'close' in sector_data.columns:
            sector_close = sector_data.loc[group.index, 'close'].shift(1)
            sector_logret = np.log(sector_close).diff()
            for window in correlation_windows:
                x[f"corr_sector_{window}"] = (
                    group["log_ret"]
                    .rolling(window=window, min_periods=1)
                    .corr(sector_logret)
                )

        if macro_data is not None:
            for col in macro_data.columns:
                if col not in ['tic', 'close', 'volume']:
                    macro_series = macro_data.loc[group.index, col].shift(1).fillna(method="ffill")
                    macro_series_ret = macro_series.pct_change()
                    for window in correlation_windows:
                        x[f"corr_{col}_{window}"] = (
                            group["log_ret"]
                            .rolling(window=window, min_periods=1)
                            .corr(macro_series_ret)
                        )

        # ВАЖНО: не добавляем x["tic"] = tic, т.к. tic у нас уже в индексе.
        return x

    # --- Устанавливаем (tic, timestamp) как MultiIndex ---
    data = data.set_index(['tic', 'timestamp']).sort_index()

    # Получаем список тикеров из уровня индекса 'tic'
    tickers = data.index.get_level_values('tic').unique()

    results = []

    # Группируем по уровню 'tic'
    groups = data.groupby(level='tic', group_keys=False)

    # Параллельная обработка
    with tqdm(total=len(tickers), desc="Processing tickers") as pbar:
        parallel_results = Parallel(n_jobs=-1)(
            delayed(process_ticker)(grp, name)
            for name, grp in groups
        )

        for res in parallel_results:
            results.append(res)
            pbar.update(1)

    # Склеиваем все результаты
    final_result = pd.concat(results)

    # Перестраиваем индекс (tic, timestamp), если нужно
    final_result = final_result.reindex(data.index)

    # Заполняем NaN
    final_result.fillna(0, inplace=True)

    # Теперь сбрасываем MultiIndex, перенесём 'tic' и 'timestamp' в колонки
    final_result.reset_index(inplace=True)

    return final_result

def get_correlation(labels, indicators, column_main='bin', threshold=0.03, show_heatmap = True):
    """
    Входящий dataframe `labels` имеет index=timestamp
    и колонки ['tic', 'bin'], где 'bin' — значение цены (или таргет).

    Входящий dataframe `indicators` имеет index=timestamp
    и колонки ['tic'] + набор признаков (индикаторов).

    Выводит матрицу корреляции (heatmap) и
    возвращает кортеж (correlation_matrix, high_corr_cols), где:
      - correlation_matrix — pd.DataFrame с корреляциями
      - high_corr_cols — список колонок, у которых |corr| > threshold
        (по умолчанию 3%) к столбцу column_main
    """

    # Преобразуем index в столбцы, чтобы можно было делать merge
    labels_merged = labels.reset_index()  # получаем столбец 'timestamp'
    indicators_merged = indicators.reset_index()

    # Мерджим по ["timestamp", "tic"]
    merged_data = pd.merge(
        labels_merged[['timestamp', 'tic', column_main]],
        indicators_merged,
        on=["timestamp", "tic"],
        how="inner"
    )

    # Считаем матрицу корреляции только по числовым столбцам
    numeric_cols = merged_data.select_dtypes(include=["number"]).columns
    correlation_matrix = merged_data[numeric_cols].corr()

    if show_heatmap:
        # Строим heatmap
        plt.figure(figsize=(20, 20))
        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt='.2f',
            cmap='coolwarm',
            cbar=True
        )
        plt.title("Correlation Matrix")
        plt.show()

    # Выбираем признаки, у которых модуль корреляции с column_main больше threshold
    if column_main in correlation_matrix.columns:
        main_corr = correlation_matrix[column_main]
        high_corr = main_corr[abs(main_corr) >= threshold].sort_values(ascending=False)

        # Убираем саму target-колонку (column_main), чтобы она не попадала в список
        high_corr_cols = [col for col in high_corr.index if col != column_main]

        print(f"\nФичи с корреляцией к '{column_main}' больше {threshold*100:.2f}% (по модулю):")
        print(high_corr_cols)

    else:
        print(f"Колонка '{column_main}' не найдена в матрице корреляции.")
        high_corr_cols = []

    # Возвращаем матрицу корреляции и список колонок с высокой корреляцией
    return correlation_matrix, high_corr_cols
