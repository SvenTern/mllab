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

    def __init__(self, source : str = 'Yahoo', start_date : str = None, end_date : str = None, ticker_list : list[str] = None, time_interval : str = "1d", file_path:str = None, proxy: str | dict = None):
        self.ticker_list = ticker_list
        self.start = start_date
        self.end = end_date
        self.time_interval = self.convert_interval(time_interval)
        self.proxy = proxy
        folder_path = '/content/drive/My Drive/DataTrading'
        os.makedirs(folder_path, exist_ok=True)
        self.file_path = os.path.join(folder_path, file_path)
        self.clean = False
        self.source = source

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
          return pd.read_csv(self.file_path + '.csv')

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
                    temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'], unit='ms')
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
                print('начинаем делать оценку с threshold',threshold)
                dollar_bars = self.create_dollar_bars(ticker_data, threshold, evaluate = True)
                print('данные после оценки', dollar_bars)
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


            print(optimal_threshold)
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
            return pd.read_csv(self.file_path + '_final.csv')

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

            print('threshold for evaluate',threshold)

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
                    print('набранное количество баров при оценке ', len(dollar_bars))

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




    def clean_data(self, df: pd.DataFrame, clean : bool = None) -> pd.DataFrame:
        if clean is not None:
          self.clean = clean
        if self.clean:
            return pd.read_csv(self.file_path + '_clean.csv')
        tic_list = np.unique(df.tic.values)
        NY = "America/New_York"

        trading_days = self.get_trading_days()
        # produce full timestamp index
        if self.time_interval == "1d":
            times = trading_days
        elif self.time_interval == "1m":
            times = []
            for day in trading_days:
                current_time = self.convert_local_time(pd.Timestamp(day + " 09:30:00"), NY)
                for i in range(390):  # 390 minutes in trading day
                    times.append(current_time)
                    current_time += pd.Timedelta(minutes=1)
        else:
            raise ValueError(
                "Data clean at given time interval is not supported for YahooFinance data."
            )
        # create a new dataframe with full timestamp series
        print('Cleaning data, recheck for nan, recheck for empty start/end date replacing for previous/next data')
        new_df = pd.DataFrame()
        for tic in tic_list:
            tmp_df = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"], index=times
            )
            tic_df = df[
                df.tic == tic
            ]  # extract just the rows from downloaded data relating to this tic
            for i in range(tic_df.shape[0]):  # fill empty DataFrame using original data
                #tmp_df.loc[self.convert_local_time(tic_df.iloc[i]["timestamp"], NY)] = tic_df.iloc[i][["open", "high", "low", "close", "volume"]]
                tmp_df.loc[times[i]] = tic_df.iloc[i][["open", "high", "low", "close", "volume"]]
            # print("(9) tmp_df\n", tmp_df.to_string()) # print ALL dataframe to check for missing rows from download

            # if close on start date is NaN, fill data with first valid close
            # and set volume to 0.
            if str(tmp_df.iloc[0]["close"]) == "nan":
                print("NaN data on start date, fill using first valid data.")
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

            # if the close price of the first row is still NaN (All the prices are NaN in this case)
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

            # fill NaN data with previous close and set volume to 0.
            for i in range(tmp_df.shape[0]):
                if str(tmp_df.iloc[i]["close"]) == "nan":
                    previous_close = tmp_df.iloc[i - 1]["close"]
                    if str(previous_close) == "nan":
                        raise ValueError
                    tmp_df.iloc[i] = [
                        previous_close,
                        previous_close,
                        previous_close,
                        previous_close,
                        0.0,
                    ]
                    # print(tmp_df.iloc[i], " Filled NaN data with previous close and set volume to 0. ticker: ", tic)

            # merge single ticker data to new DataFrame
            tmp_df = tmp_df.astype(float)
            tmp_df["tic"] = tic
            new_df = pd.concat([new_df, tmp_df])

            print(("Data clean for ") + tic + (" is finished."))

        # reset index and rename columns
        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "timestamp"})

        new_df.sort_values(['timestamp','tic'],ignore_index=True)
        #        print("Data clean all finished!")
        new_df.to_csv(self.file_path+ '_clean.csv', index=False)
        self.clean = True
        return new_df

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

    def normalize_by_ticker(self, df: pd.DataFrame = None, download_from_disk: bool = False):
        """Нормализует данные для каждого тикера отдельно."""

        if download_from_disk or df is None:
            return pd.read_csv(self.file_path + '_normalize.csv')

        normalized_data = []
        for ticker, group in df.groupby('tic'):
            numeric_cols = group.select_dtypes(include=np.number).columns
            group[numeric_cols] = group[numeric_cols].pct_change().fillna(0)
            normalized_data.append(group)

        data_normalized = pd.concat(normalized_data).reset_index(drop=True)

        data_normalized.set_index('timestamp', inplace=True)
        # Преобразование Index в DatetimeIndex
        data_normalized.index = pd.to_datetime(data_normalized.index)

        # сохраняем нормализованный датасет на диске
        data_normalized.to_csv(self.file_path + '_normalize.csv', index=True)

        return data_normalized
