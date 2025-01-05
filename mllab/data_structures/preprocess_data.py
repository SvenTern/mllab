"""Reference: https://github.com/AI4Finance-LLC/FinRL"""

from __future__ import annotations

import datetime
from datetime import date
from datetime import timedelta
from sqlite3 import Timestamp
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import TypeVar
from typing import Union

import exchange_calendars as tc
import numpy as np
import pandas as pd
import pytz
import yfinance as yf
from polygon import RESTClient
import mplfinance as mpf
from stockstats import StockDataFrame as Sdf

from google.colab import drive
import os

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors


class FinancePreprocessor:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API
    """

    def __init__(self, source : str = 'Yahoo', ticker_list : list[str] = None, time_interval : str = "1d", file_path:str = None, extended_interval: bool = False, proxy: str | dict = None):
        self.ticker_list = ticker_list

        self.time_interval = self.convert_interval(time_interval)
        self.proxy = proxy
        folder_path = '/content/drive/My Drive/DataTrading'
        os.makedirs(folder_path, exist_ok=True)
        self.file_path = os.path.join(folder_path, file_path)
        self.clean = False
        self.source = source
        self.extended_interval = extended_interval
        self.fill_dates()
        self.start = self.TRAIN_START_DATE
        self.end = self.TEST_END_DATE

        if self.source == "polygon":
            file_path = '/content/drive/My Drive/DataTrading/polygon_api_keys.txt'
            with open(file_path, 'r') as file:
                self.POLYGON_API_KEY =  file.read()

    """
    Param
    ----------
        start_date : str
            start date of the data
        end_date : str
            end date of the data
        ticker_list : list
            a list of stock tickers
    Example
    -------
    input:
    ticker_list = config_tickers.DOW_30_TICKER
    start_date = '2009-01-01'
    end_date = '2021-10-31'
    time_interval == "1D"

    output:
        date	    tic	    open	    high	    low	        close	    volume
    0	2009-01-02	AAPL	3.067143	3.251429	3.041429	2.767330	746015200.0
    1	2009-01-02	AMGN	58.590000	59.080002	57.750000	44.523766	6547900.0
    2	2009-01-02	AXP	    18.570000	19.520000	18.400000	15.477426	10955700.0
    3	2009-01-02	BA	    42.799999	45.560001	42.779999	33.941093	7010200.0
    ...
    """

    def read_csv(self, file_name: str = "", udate_dates = False):
        data_return = pd.read_csv(self.file_path + file_name)

        index_name = None
        # Determine index column
        if 'timestamp' in data_return.columns:
            index_name = 'timestamp'
        elif 'date' in data_return.columns:
            index_name = 'date'
        elif 'Unnamed: 0' in data_return.columns:
            # Rename column to 'timestamp'
            data_return.rename(columns={'Unnamed: 0': 'timestamp'}, inplace=True)
            index_name = 'timestamp'

        if not index_name is None:
            # Set the determined index and convert to datetime
            data_return.set_index(index_name, inplace=True)
            data_return.index = pd.to_datetime(data_return.index)

        # нужно определить начальную дату в датаесете и конечную дату
        if udate_dates:
            # Определяем минимальную и максимальную дату в индексе
            start = data_return.index.min().strftime('%Y-%m-%d')
            end = data_return.index.max().strftime('%Y-%m-%d')
            self.fill_dates(start, end)


        return data_return


    def convert_interval(self, time_interval: str) -> str:
        # Convert FinRL 'standardised' time periods to Yahoo format: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        if time_interval in [
            "1Min",
            "2Min",
            "5Min",
            "15Min",
            "30Min",
            "60Min",
            "90Min",
        ]:
            time_interval = time_interval.replace("Min", "m")
        elif time_interval in ["1H", "1D", "5D", "1h", "1d", "5d"]:
            time_interval = time_interval.lower()
        elif time_interval == "1W":
            time_interval = "1wk"
        elif time_interval in ["1M", "3M"]:
            time_interval = time_interval.replace("M", "mo")
        else:
            raise ValueError("wrong time_interval")

        return time_interval

    def download_data(
        self, download_from_disk : bool = False, clean_data : bool = False
    ) -> pd.DataFrame:
        if clean_data:
            self.clean = True
        if download_from_disk:
          return self.read_csv('.csv', udate_dates = True)

        # Download and save the data in a pandas DataFrame
        start_date = pd.Timestamp(self.start)
        end_date = pd.Timestamp(self.end)
        delta = timedelta(days=1)
        data_df = pd.DataFrame()
        total_tickers = len(self.ticker_list)
        for idx, tic in enumerate(self.ticker_list, start=1):
            current_tic_start_date = start_date
            while (
                current_tic_start_date <= end_date
            ):  # downloading daily to workaround yfinance only allowing  max 7 calendar (not trading) days of 1 min data per single download
                if self.source == "Yahoo":
                    temp_df = yf.download(
                    tic,
                    start=current_tic_start_date,
                    end=current_tic_start_date + delta,
                    interval=self.time_interval,
                    proxy=self.proxy,
                )
                else:
                    # загрузка polygon
                    client = RESTClient(self.POLYGON_API_KEY)
                    aggs = []
                    for a in client.list_aggs(ticker=tic, multiplier=1, timespan="minute", from_=current_tic_start_date, to=current_tic_start_date + delta):
                        aggs.append(a)
                    temp_df = pd.DataFrame(aggs)
                    if temp_df.empty:
                        current_tic_start_date += delta
                        continue
                    temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'], unit='ms').dt.tz_localize('UTC').dt.tz_convert('America/New_York')
                    # Фильтрация по времени
                    if self.extended_interval:
                        temp_df = temp_df[(temp_df['timestamp'].dt.time >= pd.Timestamp("04:00:00").time()) &
                                          (temp_df['timestamp'].dt.time <= pd.Timestamp("20:00:00").time())]
                    else:
                        temp_df = temp_df[(temp_df['timestamp'].dt.time >= pd.Timestamp("09:30:00").time()) &
                                          (temp_df['timestamp'].dt.time <= pd.Timestamp("16:00:00").time())]

                    print("Загружено % ", 100 * (current_tic_start_date -  start_date) / (end_date - start_date) / total_tickers + 100 * (idx - 1) / total_tickers )
                    #break


                temp_df = temp_df.reset_index()
                if self.source == "Yahoo":
                    temp_df.columns = [
                      "timestamp",
                      "adjclose",
                      "close",
                      "high",
                      "low",
                      "open",
                      "volume"]
                temp_df["tic"] = tic

                data_df = pd.concat([data_df, temp_df])
                current_tic_start_date += delta
        if 'adjclose' in data_df.columns:
            data_df["close"] = data_df["adjclose"]
            data_df = data_df.reset_index(drop=True).drop(columns=["adjclose"])
        data_df.to_csv(self.file_path + '.csv', index=False)
        return data_df

    def convert_local_time(self, local_time, time_zone):
      # Check if the local_time is str
      if type(local_time) == str:
        local_time = pd.Timestamp(local_time)
      # Check if the timestamp is already tz-aware
      if local_time.tz is not None:
        # If tz-aware, convert to desired timezone using tz_convert
        return local_time.tz_convert(time_zone)
      else:
        # If tz-naive, localize to desired timezone using tz_localize
        return local_time.tz_localize(time_zone)

    def evaluate_optimal_threshold(self, data, max_threshold=None, num_thresholds=50, target_bar_ratio=0.4):
        """
        Оценивает оптимальный порог для функции create_dollar_bars для нескольких тикеров.

        :param data: DataFrame с минутными данными (обязательные столбцы: 'tic', 'close', 'volume').
        :param max_threshold: Максимальный порог для анализа (если None, берётся 90-й процентиль долларового объема для каждого тикера).
        :param num_thresholds: Количество порогов для анализа.
        :param target_bar_ratio: Целевое соотношение между количеством баров и количеством минутных данных.
        :return: Словарь с оптимальными порогами для каждого тикера и графиками зависимости количества баров от порога.
        """
        unique_tickers = data['tic'].unique()
        optimal_thresholds = {}

        for ticker in unique_tickers:
            ticker_data = data[data['tic'] == ticker]

            ticker_data['DollarVolume'] = ticker_data['close'] * ticker_data['volume']

            if max_threshold is None:
                max_threshold = np.percentile(ticker_data['DollarVolume'].cumsum(), 90)

            thresholds = np.logspace(np.log10(1), np.log10(max_threshold), num=num_thresholds)
            bars_count = []

            for threshold in thresholds:
                dollar_bars = self.create_dollar_bars(ticker_data, download_from_disk = False, optimal_thresholds = threshold, evaluate = True)
                bars_count.append(len(dollar_bars))

            total_minutes = len(ticker_data)
            target_bars = total_minutes * target_bar_ratio
            optimal_threshold = None
            for i, count in enumerate(bars_count):
                if count <= target_bars:
                    optimal_threshold = thresholds[i]
                    break

            # График зависимости количества баров от порога
            plt.figure(figsize=(10, 6))
            plt.plot(thresholds, bars_count, marker='o', label=f"Number of Dollar Bars ({ticker})")
            if optimal_threshold:
                plt.axvline(optimal_threshold, color='red', linestyle='--', label=f"Optimal Threshold: {optimal_threshold:.2f}")
            plt.xscale('log')
            plt.xlabel("Threshold (Dollar Volume)")
            plt.ylabel("Number of Bars")
            plt.title(f"Threshold vs. Number of Dollar Bars ({ticker})")
            plt.legend()
            plt.grid()
            plt.show()


            print(f"Ticker: {ticker}, Optimal Threshold: {optimal_threshold:.2f}")
            optimal_thresholds[ticker] = optimal_threshold


        return optimal_thresholds

    def create_dollar_bars(self, data: pd.DataFrame = None, download_from_disk: bool = False, evaluate: bool = False, optimal_thresholds: Union[Dict[str, float], float] = None) -> pd.DataFrame:
        """
        Создаёт долларовые бары из минутных данных для нескольких тикеров.

        :param data: DataFrame с минутными данными (обязательные столбцы: 'tic', 'close', 'volume').
                    'tic' - тикер актива, 'close' - цена закрытия, 'volume' - объём торгов.
        :param optimal_thresholds: Словарь с оптимальными порогами для каждого тикера (tic -> threshold)
                                  или единое число для всех тикеров.
        :return: DataFrame с долларовыми барами, включая столбцы ['timestamp', 'open', 'high', 'low', 'close', 'tic'].
        """
        if data is None or download_from_disk:
            return self.read_csv('_final.csv')

        # Проверка на наличие обязательных столбцов
        required_columns = {'tic', 'close', 'volume'}
        if not required_columns.issubset(data.columns):
            raise ValueError(f"DataFrame должен содержать столбцы: {required_columns}")

        # Получение уникальных тикеров
        unique_tickers = data['tic'].unique()
        all_dollar_bars = []

        if optimal_thresholds is None:
            optimal_thresholds = self.evaluate_optimal_threshold(data)

        # Обработка каждого тикера отдельно
        for ticker in unique_tickers:
            ticker_data = data[data['tic'] == ticker]

            # Определение порога
            if isinstance(optimal_thresholds, dict):
                if ticker not in optimal_thresholds:
                    raise ValueError(f"Порог для тикера {ticker} отсутствует в словаре optimal_thresholds.")
                threshold = optimal_thresholds[ticker]
            else:
                threshold = optimal_thresholds

            dollar_bars = []
            cum_dollar_volume = 0
            cum_transactions_volume = 0
            cum_vwap_volume = 0
            cum_volume = 0

            bar = {'open': None, 'high': -np.inf, 'low': np.inf, 'close': None, 'vwap': None, 'transactions': None, 'timestamp': None, 'tic': ticker}

            # Итерация по строкам данных
            for index, row in ticker_data.iterrows():
                dollar_volume = row['close'] * row['volume']
                cum_dollar_volume += dollar_volume
                cum_vwap_volume = cum_vwap_volume + row['vwap'] * row['volume']
                cum_volume = cum_volume + row['volume']

                if bar['open'] is None:
                    bar['open'] = row['close']
                    bar['timestamp'] = row['timestamp'] #index  # Сохраняем реальную метку времени

                bar['high'] = max(bar['high'], row['close'])
                bar['low'] = min(bar['low'], row['close'])

                if 'vwap' in data.columns:
                    bar['vwap'] = ( cum_vwap_volume ) / ( cum_volume )

                bar['close'] = row['close']

                if 'transactions' in data.columns:
                    cum_transactions_volume = cum_transactions_volume + row['transactions']
                    bar['transactions'] = cum_transactions_volume

                if cum_dollar_volume >= threshold:
                    dollar_bars.append(bar)
                    cum_dollar_volume = 0
                    cum_transactions_volume = 0
                    cum_vwap_volume = 0
                    cum_volume = 0
                    bar = {'open': None, 'high': -np.inf, 'low': np.inf, 'close': None, 'vwap': None, 'transactions': None, 'timestamp': None, 'tic': ticker}


            # Добавление последнего бара, если данные остались
            if bar['open'] is not None:
                dollar_bars.append(bar)

            # Конвертация списка баров в DataFrame
            dollar_bars_df = pd.DataFrame(dollar_bars)

            all_dollar_bars.append(dollar_bars_df)

        # Объединение всех тикеров в один DataFrame
        all_dollar_bars_df = pd.concat(all_dollar_bars, ignore_index=True)

        # возврат значения оценки без сохранения на диске
        if evaluate:
            return all_dollar_bars_df

        all_dollar_bars_df.set_index('timestamp', inplace=True)
        # Преобразование Index в DatetimeIndex
        all_dollar_bars_df.index = pd.to_datetime(all_dollar_bars_df.index)

        # сохраняем нормализованный датасет на диске
        all_dollar_bars_df.to_csv(self.file_path + '_final.csv', index=True)

        return all_dollar_bars_df

    # required_columns - список обязательных колонок
    # timestamp, tic - основные колонки
    # required_columns - список обязательных колонок
    # timestamp, tic - основные колонки
    def clean_data(self, df: pd.DataFrame, required_columns: list, clean: bool = None) -> pd.DataFrame:
        if clean is not None:
            self.clean = clean
        if self.clean:
            return self.read_csv('_clean.csv')

        # Проверяем обязательные колонки
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Если 'timestamp' это индекс, делаем ее колонкой
        if df.index.name == 'timestamp':
            df = df.reset_index()

        # Проверяем наличие ключевых колонок 'timestamp' и 'tic'
        if 'timestamp' not in df.columns or 'tic' not in df.columns:
            raise KeyError("DataFrame must contain 'timestamp' and 'tic' columns.")

        # Оставляем только необходимые колонки
        df = df[required_columns + ['tic', 'timestamp']]

        tic_list = df['tic'].unique()
        NY = "America/New_York"
        tz = pytz.timezone(NY)

        # Получаем список торговых дней
        trading_days = self.get_trading_days()

        # Генерируем полный индекс времени
        # Генерируем полный индекс времени для каждого дня
        if self.time_interval == "1d":
            times = pd.Index(trading_days)
        elif self.time_interval == "1m":
            times = []
            for day in trading_days:

                if self.extended_interval:
                    day_times = pd.date_range(
                        start=pd.Timestamp(f"{day} 04:00:00", tz=tz),
                        end=pd.Timestamp(f"{day} 20:00:00", tz=tz),
                        freq='T'
                    )
                else:
                    day_times = pd.date_range(
                        start=pd.Timestamp(f"{day} 09:30:00", tz=tz),
                        end=pd.Timestamp(f"{day} 16:00:00", tz=tz),
                        freq='T'
                    )
                times.extend(day_times)
            times = pd.Index(times)  # Преобразуем в Index для дальнейшего использования
        else:
            raise ValueError("Unsupported time interval for data cleaning.")

        # Подготовка нового DataFrame с полным временным индексом
        data_frames = []

        df_grouped = df.groupby('tic')

        for tic, tic_df in df_grouped:
            # Удаляем дубликаты по индексу 'timestamp' для предотвращения ошибок
            tic_df = tic_df[~tic_df['timestamp'].duplicated(keep='first')]

            # Инициализируем DataFrame для текущего тикера
            tmp_df = pd.DataFrame(index=times, columns=required_columns)

            # Объединяем временные ряды
            tic_df = tic_df.set_index("timestamp")
            tmp_df.update(tic_df)

            # Заполняем NaN-значения
            tmp_df = self.fill_missing_data(tmp_df, required_columns, tic)

            # Добавляем тикер в итоговый DataFrame
            tmp_df["tic"] = tic
            data_frames.append(tmp_df)

        # Сбрасываем индекс и сортируем
        new_df = pd.concat(data_frames).reset_index().rename(columns={"index": "timestamp"})
        new_df = new_df.sort_values(['timestamp', 'tic'], ignore_index=True)

        # Сохраняем очищенные данные
        new_df.to_csv(self.file_path + '_clean.csv', index=False)
        self.clean = True
        return new_df

    def fill_missing_data(self, df: pd.DataFrame, required_columns: list, tic: str) -> pd.DataFrame:
        # Если первая строка содержит NaN, заполняем первым доступным значением для всех обязательных колонок
        df.ffill(inplace=True)
        df.bfill(inplace=True)

        # Проверка остатков NaN
        if df[required_columns].isna().any().any():
            raise ValueError(f"NaN values remain in data for ticker {tic} in columns {required_columns}")

        return df


    def add_technical_indicator(
        self, data: pd.DataFrame, tech_indicator_list: list[str]
    ):
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(by=["tic", "timestamp"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["tic"] = unique_ticker[i]
                    temp_indicator["timestamp"] = df[df.tic == unique_ticker[i]][
                        "timestamp"
                    ].to_list()
                    indicator_df = pd.concat(
                        [indicator_df, temp_indicator], ignore_index=True
                    )
                except Exception as e:
                    print(e)
            df = df.merge(
                indicator_df[["tic", "timestamp", indicator]],
                on=["tic", "timestamp"],
                how="left",
            )
        df = df.sort_values(by=["timestamp", "tic"])
        return df

    def add_vix(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        add vix from yahoo finance
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        vix_df = self.download_data(["VIXY"], self.start, self.end, self.time_interval)
        cleaned_vix = self.clean_data(vix_df)
        print("cleaned_vix\n", cleaned_vix)
        vix = cleaned_vix[["timestamp", "close"]]
        print('cleaned_vix[["timestamp", "close"]\n', vix)
        vix = vix.rename(columns={"close": "VIXY"})
        print('vix.rename(columns={"close": "VIXY"}\n', vix)

        df = data.copy()
        print("df\n", df)
        df = df.merge(vix, on="timestamp")
        df = df.sort_values(["timestamp", "tic"]).reset_index(drop=True)
        return df

    def calculate_turbulence(
        self, data: pd.DataFrame, time_period: int = 252
    ) -> pd.DataFrame:
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(index="timestamp", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.timestamp.unique()
        # start after a fixed timestamp period
        start = time_period
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - time_period])
            ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[
                hist_price.isna().sum().min() :
            ].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
                filtered_hist_price, axis=0
            )
            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)

        turbulence_index = pd.DataFrame(
            {"timestamp": df_price_pivot.index, "turbulence": turbulence_index}
        )
        return turbulence_index

    def add_turbulence(
        self, data: pd.DataFrame, time_period: int = 252
    ) -> pd.DataFrame:
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df, time_period=time_period)
        df = df.merge(turbulence_index, on="timestamp")
        df = df.sort_values(["timestamp", "tic"]).reset_index(drop=True)
        return df

    def df_to_array(
        self, df: pd.DataFrame, tech_indicator_list: list[str], if_vix: bool
    ) -> list[np.ndarray]:
        df = df.copy()
        unique_ticker = df.tic.unique()
        if_first_time = True
        for tic in unique_ticker:
            if if_first_time:
                price_array = df[df.tic == tic][["close"]].values
                tech_array = df[df.tic == tic][tech_indicator_list].values
                if if_vix:
                    turbulence_array = df[df.tic == tic]["VIXY"].values
                else:
                    turbulence_array = df[df.tic == tic]["turbulence"].values
                if_first_time = False
            else:
                price_array = np.hstack(
                    [price_array, df[df.tic == tic][["close"]].values]
                )
                tech_array = np.hstack(
                    [tech_array, df[df.tic == tic][tech_indicator_list].values]
                )
        #        print("Successfully transformed into array")
        return price_array, tech_array, turbulence_array

    def fill_dates(self, start = None, end = None):
        if start is None:
            self.TRAIN_START_DATE = '2024-01-01'
        else:
            self.TRAIN_START_DATE = start

        current_date = datetime.datetime.now().strftime('%Y-%m-%d')

        self.start = self.TRAIN_START_DATE

        if end is None:
            self.end = current_date
        else:
            self.end = end

        # Get trading days from the start of the year to the current date
        trading_days = self.get_trading_days()

        # Ensure TEST_END_DATE is the last trading day up to the current date
        self.TEST_END_DATE = trading_days[-1]

        # Determine TEST_START_DATE (14 trading days back) and TRAIN_END_DATE (previous trading day)
        test_end_index = trading_days.index(self.TEST_END_DATE)
        self.TEST_START_DATE = trading_days[max(0, test_end_index - 14)]
        self.TRAIN_END_DATE = trading_days[max(0, test_end_index - 15)]

    def get_trading_days(self) -> list[str]:
        nyse = tc.get_calendar("NYSE")
        df = nyse.sessions_in_range(pd.Timestamp(self.start), pd.Timestamp(self.end))
        trading_days = []
        for day in df:
            trading_days.append(str(day)[:10])

        return trading_days

    # ****** NB: YAHOO FINANCE DATA MAY BE IN REAL-TIME OR DELAYED BY 15 MINUTES OR MORE, DEPENDING ON THE EXCHANGE ******
    def fetch_latest_data(
        self,
        tech_indicator_list: list[str],
        limit: int = 100,
    ) -> pd.DataFrame:
        time_interval = self.convert_interval(time_interval)

        end_datetime = datetime.datetime.now()
        start_datetime = end_datetime - datetime.timedelta(
            minutes=limit + 1
        )  # get the last rows up to limit

        data_df = pd.DataFrame()
        for tic in self.ticker_list:
            barset = yf.download(
                tic, start_datetime, end_datetime, interval=time_interval
            )  # use start and end datetime to simulate the limit parameter
            barset["tic"] = tic
            data_df = pd.concat([data_df, barset])

        data_df = data_df.reset_index().drop(
            columns=["Adj Close"]
        )  # Alpaca data does not have 'Adj Close'

        data_df.columns = [  # convert to Alpaca column names lowercase
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "tic",
        ]

        start_time = data_df.timestamp.min()
        end_time = data_df.timestamp.max()
        times = []
        current_time = start_time
        end = end_time + pd.Timedelta(minutes=1)
        while current_time != end:
            times.append(current_time)
            current_time += pd.Timedelta(minutes=1)

        df = data_df.copy()
        new_df = pd.DataFrame()
        for tic in ticker_list:
            tmp_df = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"], index=times
            )
            tic_df = df[df.tic == tic]
            for i in range(tic_df.shape[0]):
                tmp_df.loc[tic_df.iloc[i]["timestamp"]] = tic_df.iloc[i][
                    ["open", "high", "low", "close", "volume"]
                ]

                if str(tmp_df.iloc[0]["close"]) == "nan":
                    for i in range(tmp_df.shape[0]):
                        if str(tmp_df.iloc[i]["close"]) != "nan":
                            first_valid_close = tmp_df.iloc[i]["close"]
                            tmp_df.iloc[0] = [
                                first_valid_close,
                                first_valid_close,
                                first_valid_close,
                                first_valid_close,
                                0.0,
                            ]
                            break
                if str(tmp_df.iloc[0]["close"]) == "nan":
                    print(
                        "Missing data for ticker: ",
                        tic,
                        " . The prices are all NaN. Fill with 0.",
                    )
                    tmp_df.iloc[0] = [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]

            for i in range(tmp_df.shape[0]):
                if str(tmp_df.iloc[i]["close"]) == "nan":
                    previous_close = tmp_df.iloc[i - 1]["close"]
                    if str(previous_close) == "nan":
                        previous_close = 0.0
                    tmp_df.iloc[i] = [
                        previous_close,
                        previous_close,
                        previous_close,
                        previous_close,
                        0.0,
                    ]
            tmp_df = tmp_df.astype(float)
            tmp_df["tic"] = tic
            new_df = pd.concat([new_df, tmp_df])

        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "timestamp"})

        df = self.add_technical_indicator(new_df, tech_indicator_list)
        df["VIXY"] = 0

        price_array, tech_array, turbulence_array = self.df_to_array(
            df, tech_indicator_list, if_vix=True
        )
        latest_price = price_array[-1]
        latest_tech = tech_array[-1]
        start_datetime = end_datetime - datetime.timedelta(minutes=1)
        turb_df = yf.download("VIXY", start_datetime, limit=1)
        latest_turb = turb_df["Close"].values
        return latest_price, latest_tech, latest_turb

    def candle_plot(self, df: pd.DataFrame, tic: str = 'AAPL', style: str = 'yahoo', interval: str = '1T', type_chart : str = 'candle'):
        """
        Построение свечного графика с возможностью задания интервала.

        :param df: DataFrame с данными
        :param tic: Тикер (символ акции)
        :param style: Стиль графика mplfinance
        :param interval: Интервал для ресемплинга ('1T' - 1 минута, '5T' - 5 минут и т.д.)
        """
        # Проверка на наличие необходимых колонок
        required_columns = {"open", "high", "low", "close", "tic"}
        if not required_columns.issubset(df.columns):
            raise ValueError(f"DataFrame должен содержать следующие столбцы: {required_columns}")

        if not "volume" in df.columns:
            dont_use_volume = True
        else:
            dont_use_volume = False

        # Подготовка данных
        index_name = "timestamp"
        df_copy = df.copy()
        if not df_copy.index.name == index_name:
            if index_name not in df_copy.columns:
                index_name = "date"
            df_copy = df.copy()
            df_copy[index_name] = pd.to_datetime(df_copy[index_name])
            df_copy = df_copy.set_index(index_name)
        if dont_use_volume:
            df_copy = df_copy.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close"})
        else:
            df_copy = df_copy.rename(columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"})

        # Фильтрация данных по тикеру
        filtered_data = df_copy[df_copy.tic == tic]

        # Проверка наличия данных после фильтрации
        if filtered_data.empty:
            raise ValueError(f"Нет данных для тикера {tic}")

        resampled_data = filtered_data
        if not interval == "dollar bars":
            # Ресемплинг данных для указанного интервала
            if dont_use_volume:
                resampled_data = filtered_data.resample(interval).agg({
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last"
                }).dropna()
            else:
                resampled_data = filtered_data.resample(interval).agg({
                    "Open": "first",
                    "High": "max",
                    "Low": "min",
                    "Close": "last",
                    "Volume": "sum"
                }).dropna()



        # Диагностическая информация
        #print(f"Данные для {tic} (первые 5 строк):\n", resampled_data.head())

        # Построение свечного графика
        mpf.plot(
            resampled_data,
            type=type_chart,  # Тип графика: свечной
            volume= not dont_use_volume,    # Добавить объём
            title=f"Candlestick {tic} ({interval} interval)",  # Заголовок графика
            style=style,  # Стиль графика
            ylabel='Price',  # Подпись оси Y
            ylabel_lower='Volume'  # Подпись оси объёма
        )

    def normalize_by_ticker(self, df: pd.DataFrame = None, download_from_disk: bool = False, method: str = 'log'):
        """Нормализует данные для каждого тикера отдельно."""

        if download_from_disk or df is None:
            return self.read_csv('_normalize.csv')

        normalized_data = []
        for ticker, group in df.groupby('tic'):
            numeric_cols = group.select_dtypes(include=np.number).columns
            if method == 'log':
                group[numeric_cols] = np.log(group[numeric_cols]).diff().fillna(0)
            else:
                group[numeric_cols] = group[numeric_cols].pct_change().fillna(0)
            normalized_data.append(group)

        data_normalized = pd.concat(normalized_data).reset_index(drop=True)

        data_normalized.set_index('timestamp', inplace=True)
        # Преобразование Index в DatetimeIndex
        data_normalized.index = pd.to_datetime(data_normalized.index)

        # сохраняем нормализованный датасет на диске
        data_normalized.to_csv(self.file_path + '_normalize.csv', index=True)

        return data_normalized


def add_takeprofit_stoploss_volume(predicted_data, coeff_tp=1, coeff_sl=1):
    """
    Оптимизирует массивы в колонке 'prediction' для работы на TPU.
    Обновляет 4-й элемент (return) и добавляет vol, tp, sl в конец массива.

    Parameters:
    predicted_data (pd.DataFrame): DataFrame с колонкой 'prediction', содержащей массивы.
    coeff_tp (float): Коэффициент для расчета take-profit.
    coeff_sl (float): Коэффициент для расчета stop-loss.

    Returns:
    pd.DataFrame: Обновленный DataFrame с модифицированными массивами в 'prediction'.
    """
    # Преобразуем prediction в NumPy массив для эффективной обработки
    predictions = np.array(predicted_data['prediction'].tolist(), dtype=object)

    # Извлекаем основные данные
    first_three = np.vstack(predictions[:, :3])  # Первые три элемента
    fourth_element = np.array([x[3] if len(x) > 3 else None for x in predictions])

    # Определяем max_bin и корректируем return
    max_bin = np.argmax(first_three, axis=1) + 1
    return_value = np.copy(fourth_element)

    # Условная корректировка return_value
    return_value = np.where(
        (return_value < 0) & (max_bin == 3),
        np.abs(return_value),
        return_value
    )
    return_value = np.where(
        (return_value > 0) & (max_bin == 1),
        -np.abs(return_value),
        return_value
    )

    # Инициализация vol, tp, sl
    vol = np.zeros(len(predictions))
    tp = np.zeros(len(predictions))
    sl = np.zeros(len(predictions))

    # Обработка bin-1
    mask_bin_minus1 = max_bin == 1
    p_minus1 = first_three[mask_bin_minus1, 0]
    b_minus1 = p_minus1 / (1 - p_minus1)
    vol[mask_bin_minus1] = np.maximum(0, p_minus1 - (1 - p_minus1) / b_minus1)
    tp[mask_bin_minus1] = coeff_tp * return_value[mask_bin_minus1]
    sl[mask_bin_minus1] = coeff_sl * return_value[mask_bin_minus1] / b_minus1

    # Обработка bin+1
    mask_bin_plus1 = max_bin == 3
    p_plus1 = first_three[mask_bin_plus1, 2]
    b_plus1 = p_plus1 / (1 - p_plus1)
    vol[mask_bin_plus1] = np.maximum(0, p_plus1 - (1 - p_plus1) / b_plus1)
    tp[mask_bin_plus1] = coeff_tp * return_value[mask_bin_plus1]
    sl[mask_bin_plus1] = coeff_sl * return_value[mask_bin_plus1] / b_plus1

    # Обновляем массив prediction: заменяем 4-й элемент и добавляем vol, tp, sl
    updated_predictions = []
    for i in range(len(predictions)):
        pred = list(predictions[i])
        if len(pred) > 3:
            pred[3] = return_value[i]
        pred.extend([vol[i], tp[i], sl[i]])
        updated_predictions.append(pred)

    # Обновляем DataFrame
    predicted_data['prediction'] = updated_predictions

    return predicted_data



