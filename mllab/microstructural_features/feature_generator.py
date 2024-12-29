import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed

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
        Данные цен, объёмов и т. д. (должен содержать колонки ['tic', 'close', 'volume', 'vwap', 'transactions'])
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
    """

    def process_ticker(group, tic):
        """
        Вычисляет фичи для отдельного тикера.
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
        #    (пример: вы можете менять окно и лаги под свои нужды)
        window_autocorr = 10
        autocorrs = (
            group["log_ret"]
            .rolling(window=window_autocorr, min_periods=window_autocorr)
            .apply(lambda series: np.corrcoef(series[:-1], series[1:])[0, 1]
                   if len(series) > 1 else np.nan, raw=True)
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

        # --- Объёмные фичи ---
        # x["volume_ma_10"] = data_row_volume.rolling(window=10).mean()
        # x["volume_ma_50"] = data_row_volume.rolling(window=50).mean()

        # --- VWAP (Volume Weighted Average Price) ---
        x["vwap_diff"] = (data_row_vwap - data_row) / data_row

        # --- Корреляции с бенчмарком (SP500), отраслью (sector), макро (macro) ---
        # Предположим, что во входном data уже лежат соответствующие close-цены или индексы.
        # Если нет — вы можете приджойнить sp500_data/sector_data/macro_data по индексу.

        if sp500_data is not None and 'close' in sp500_data.columns:
            # Получаем сопоставимый ряд для S&P500
            sp500_close = sp500_data.loc[group.index, 'close'].shift(1)
            sp500_logret = np.log(sp500_close).diff()
            for window in correlation_windows:
                x[f"corr_sp500_{window}"] = (
                    group["log_ret"]
                    .rolling(window=window, min_periods=1)
                    .corr(sp500_logret)
                )

        if sector_data is not None and 'close' in sector_data.columns:
            # Аналогично для sector
            sector_close = sector_data.loc[group.index, 'close'].shift(1)
            sector_logret = np.log(sector_close).diff()
            for window in correlation_windows:
                x[f"corr_sector_{window}"] = (
                    group["log_ret"]
                    .rolling(window=window, min_periods=1)
                    .corr(sector_logret)
                )

        if macro_data is not None:
            # Предположим, что macro_data может содержать несколько колонок: например, CPI, IR (interest rate), VIX...
            # Для каждой такой колонки можно считать корреляцию.
            for col in macro_data.columns:
                if col not in ['tic', 'close', 'volume']:  # пример фильтрации
                    macro_series = macro_data.loc[group.index, col].shift(1).fillna(method="ffill")
                    # Нормализуем или берём темп роста
                    macro_series_ret = macro_series.pct_change()  # Или log_diff
                    for window in correlation_windows:
                        x[f"corr_{col}_{window}"] = (
                            group["log_ret"]
                            .rolling(window=window, min_periods=1)
                            .corr(macro_series_ret)
                        )

        # --- Добавляем тикер в результат, чтобы после concat не терять информацию ---
        x["tic"] = tic

        return x

    data = data.set_index(['tic', 'date']).sort_index()

    # --- Параллельная обработка тикеров ---
    tickers = data['tic'].unique()
    results = []

    with tqdm(total=len(tickers), desc="Processing tickers") as pbar:
        # Разбиваем data на группы по столбцу 'tic'
        groups = data.groupby('tic', group_keys=False)

        # Запускаем параллельно для каждого тикера
        parallel_results = Parallel(n_jobs=-1)(
            delayed(lambda grp, name: (process_ticker(grp, name)))(grp, name)
            for name, grp in groups
        )

        for res in parallel_results:
            results.append(res)
            pbar.update(1)

    # --- Склеиваем все результаты ---
    final_result = pd.concat(results)
    # --- Выравниваем индексы, чтобы соответствовать исходным ---
    final_result = final_result.reindex(data.index)

    # --- Заполняем NaN ---
    final_result = final_result.fillna(0)

    return final_result
