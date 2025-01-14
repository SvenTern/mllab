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


import os
from tqdm.auto import tqdm

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import json

from pandas.errors import EmptyDataError
from mllab.labeling.labeling import short_long_box

class FinancePreprocessor:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API
    """

    def __init__(self, ticker_list, ticker_indicator_list, source : str = 'Yahoo', time_interval : str = "1d", file_path:str = None, extended_interval: bool = False, proxy: str | dict = None):
        self.ticker_list = ticker_list

        self.time_interval = self.convert_interval(time_interval)
        self.proxy = proxy
        folder_path = '/content/drive/My Drive/DataTrading'
        os.makedirs(folder_path, exist_ok=True)
        self.file_path = os.path.join(folder_path, file_path)
        os.makedirs(self.file_path, exist_ok=True)
        self.source = source
        self.extended_interval = extended_interval
        self.fill_dates()
        self.start = self.TRAIN_START_DATE
        self.end = self.TEST_END_DATE

        if self.source == "polygon":
            file_path = '/content/drive/My Drive/DataTrading/polygon_api_keys.txt'
            with open(file_path, 'r') as file:
                self.POLYGON_API_KEY =  file.read()
        self.ticker_list = ticker_list
        self.ticker_indicator_list = ticker_indicator_list

        self.raw_data = 'raw_data'
        self.cleaned_data = 'cleaned_data'
        self.labels = 'labels'
        self.indiicators = 'indiicators'

        os.makedirs(os.path.join(self.file_path, self.raw_data), exist_ok=True)
        os.makedirs(os.path.join(self.file_path, self.cleaned_data), exist_ok=True)
        os.makedirs(os.path.join(self.file_path, self.labels), exist_ok=True)
        os.makedirs(os.path.join(self.file_path, self.indiicators), exist_ok=True)




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

    def convert_to_ny_time(self, date_str):
        """
        Конвертирует строку даты в формате '%Y-%m-%d' в локализованное время Нью-Йорка.

        :param date_str: Строка даты в формате '%Y-%m-%d'
        :return: Локализованное время Нью-Йорка в формате datetime
        """
        try:
            # Преобразуем строку в объект datetime
            date_obj = datetime.datetime.strptime(date_str, '%Y-%m-%d')

            # Устанавливаем таймзону Нью-Йорка
            ny_timezone = pytz.timezone('America/New_York')

            # Локализуем дату
            localized_date = ny_timezone.localize(date_obj)

            return localized_date
        except ValueError as e:
            raise ValueError(f"Неверный формат даты: {e}")

    def read_csv(self, file_name: str = "", udate_dates = False):
        data_return = pd.read_csv(os.path.join(self.file_path, file_name))

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

        # Восстанавливаем np.array из JSON строк
        data_return = data_return.applymap(
            lambda x: np.array(json.loads(x)) if isinstance(x, str) and x.startswith('[') else x)

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

    def get_ticker_data_by_ticker(self, ticker: str, clean: bool = True) -> pd.DataFrame:
        """
        Возвращает DataFrame по конкретному тикеру с диска:
          - Если clean=True, пытается загрузить очищенный файл (..._cleaned.csv).
            Если он отсутствует, автоматически вызывает clean_data(tickers=[ticker], clean=True)
            и затем читает результат.
          - Если clean=False, читает «сырые» (неочищенные) данные (...csv).

        Параметры
        ----------
        ticker : str
            Название тикера (например, "AAPL").
        clean : bool
            Если True, пытаемся работать с очищенными данными.
            Если их нет, вызываем clean_data(...) для этого тикера и читаем результат.
            Если False, читаем сырые данные.

        Возвращает
        ----------
        pd.DataFrame
            Данные тикера (очищенные или сырые, в зависимости от аргумента clean).
        """
        # Формируем имя файла (даты из self.start и self.end)
        start_str = pd.Timestamp(self.start).strftime("%Y%m%d")
        end_str = pd.Timestamp(self.end).strftime("%Y%m%d")

        # Основной (сырой) файл
        raw_file_name = f"{ticker}_{start_str}_{end_str}.csv"
        raw_path = os.path.join(self.file_path, self.raw_data, raw_file_name)

        # «Очищенный» файл
        cleaned_file_name = f"{ticker}_{start_str}_{end_str}_cleaned.csv"
        cleaned_path = os.path.join(self.file_path, self.cleaned_data, cleaned_file_name)

        # Если хотим работать с очищенными данными
        if clean:
            print(f"[get_ticker_data] Пытаемся загрузить очищенные данные: {cleaned_file_name}")
            if os.path.isfile(cleaned_path):
                print(f"[get_ticker_data] Найден файл {cleaned_file_name}, читаем...")
                return pd.read_csv(cleaned_path, parse_dates=["timestamp"])
            else:
                print(
                    f"[get_ticker_data] Файл {cleaned_file_name} не найден. Запускаем clean_data для тикера {ticker}...")
                # Вызываем clean_data, чтобы сгенерировать «очищенный» файл
                self.clean_data(tickers=[ticker], clean=False)

                # После успешной очистки файл должен появиться. Читаем.
                if os.path.isfile(cleaned_path):
                    print(f"[get_ticker_data] Очистка завершена. Читаем {cleaned_file_name}...")
                    return pd.read_csv(cleaned_path, parse_dates=["timestamp"])
                else:
                    raise FileNotFoundError(
                        f"[get_ticker_data] Ошибка! Не удалось найти файл после очистки: {cleaned_file_name}"
                    )

        # Иначе, если clean=False, читаем «сырые» данные
        else:
            print(f"[get_ticker_data] Запрошены сырые данные (clean=False). Файл: {raw_file_name}")
            if not os.path.isfile(raw_path):
                raise FileNotFoundError(f"[get_ticker_data] Сырой файл не найден: {raw_file_name}")
            return pd.read_csv(raw_path, parse_dates=["timestamp"])

    def get_ticker_data(self, tickers=None, clean:bool = True) -> pd.DataFrame:
        """
        Считывает данные по одному или нескольким тикерам с диска и
        возвращает единый DataFrame со всеми строчками.

        Параметры:
        -----------
        tickers : list[str] или None
            Список тикеров для загрузки. Если None, берём объединённый список:
            self.ticker_list + self.ticker_indicator_list.

        Возвращает:
        -----------
        pd.DataFrame
            Объединённый DataFrame по всем тикерам. Если ни один файл не найден,
            вернётся пустой DataFrame.
        """
        # Если список тикеров не задан, берём объединённый (например, чтобы загрузить все)
        if tickers is None:
            tickers = self.ticker_list + self.ticker_indicator_list

        all_data = []

        start_str = pd.Timestamp(self.start).strftime("%Y%m%d")
        end_str = pd.Timestamp(self.end).strftime("%Y%m%d")

        # Перебираем каждый тикер
        for ticker in tickers:
            # считываем данные отдельноого тикера, при необходимости очищаем данные
            df = self.get_ticker_data_by_ticker(self, ticker, clean)
            # Добавляем в общий список
            all_data.append(df)

        # Объединяем все прочитанные данные в один DataFrame
        if not all_data:
            print("Не удалось найти ни одного файла. Возвращаем пустой DataFrame.")
            return pd.DataFrame()

        final_df = pd.concat(all_data, ignore_index=True)

        # На всякий случай удаляем дубликаты по timestamp + tic (если есть)
        final_df.drop_duplicates(subset=["timestamp", "tic"], keep="last", inplace=True)
        final_df.sort_values(by=["timestamp", "tic"], inplace=True)

        return final_df

    def download_data(self, download_from_disk: bool = False) -> None:
        """
        Загрузка данных по тикерам из self.ticker_list и self.ticker_indicator_list:
        - Ничего не возвращает (return None).
        - Сохраняет данные в отдельные CSV-файлы по каждому тикеру.
        - Выводит сообщения через print(...).
        - Отображает прогресс с помощью tqdm.
        - Приводит время к локальному времени Нью-Йорка (America/New_York).
        - При self.extended_interval=True захватывает период с 4:00 до 20:00, иначе 9:30–16:00.

        Параметры:
        download_from_disk=True:
          - Игнорирует любые локальные файлы, принудительно перезагружает данные из Интернета.
        download_from_disk=False:
          - Если локальный файл по тикеру есть, докачивает недостающие периоды.
        """

        # Предположим, self.start и self.end — строковые представления или объекты datetime
        # в UTC. Если другой часовой пояс, замените 'UTC' на нужный.
        start_date = pd.Timestamp(self.start).tz_localize('UTC')
        end_date = pd.Timestamp(self.end).tz_localize('UTC')
        delta = timedelta(days=1)

        # Объединяем тикеры из основного списка и списка «индикаторов»
        combined_tickers = np.concatenate((self.ticker_list, self.ticker_indicator_list))

        print("Начинаем загрузку данных...")
        # Проходимся по списку тикеров с помощью tqdm
        for tic in tqdm(combined_tickers, desc="Downloading data", total=len(combined_tickers)):
            # Формируем имя файла для сохранения
            file_name = f"{tic}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv"
            full_path = os.path.join(self.file_path, self.raw_data, file_name)

            # Принудительная перезагрузка из Интернета
            if download_from_disk:
                print(
                    f"[{tic}] Принудительно перезагружаем весь период {start_date.date()} - {end_date.date()} из Интернета.")
                new_df = self._download_for_ticker(tic, start_date, end_date, delta)
                new_df.to_csv(full_path, index=False)
                continue

            # Если докачка (download_from_disk=False)
            if os.path.isfile(full_path):
                print(f"[{tic}] Найден файл: {file_name}. Проверяем недостающие периоды...")
                try:
                    existing_df = pd.read_csv(full_path, parse_dates=["timestamp"])
                except EmptyDataError:
                    print(f"[{tic}] Файл {file_name} пуст или поврежден. Перезагружаем данные заново.")
                    new_df = self._download_for_ticker(tic, start_date, end_date, delta)
                    new_df.to_csv(full_path, index=False)
                    continue  # Переходим к следующему тикеру

                # Приводим к единому формату, если есть столбец 'adjclose'
                if "adjclose" in existing_df.columns:
                    existing_df["close"] = existing_df["adjclose"]
                    existing_df.drop(columns=["adjclose"], inplace=True)

                # Ensure 'timestamp' column is in datetime format
                if not pd.api.types.is_datetime64_any_dtype(existing_df["timestamp"]):
                    # Use utc=True to normalize tz-aware datetime objects
                    existing_df["timestamp"] = pd.to_datetime(existing_df["timestamp"], utc=True)

                # Check if the timestamp is tz-aware
                if existing_df["timestamp"].dt.tz is None:
                    # If the timestamp is naive, localize it
                    existing_min_date = existing_df["timestamp"].min().tz_localize('UTC')
                    existing_max_date = existing_df["timestamp"].max().tz_localize('UTC')
                else:
                    # If the timestamp is tz-aware, use tz_convert
                    existing_min_date = existing_df["timestamp"].min().tz_convert('UTC')
                    existing_max_date = existing_df["timestamp"].max().tz_convert('UTC')

                # Проверяем, покрывает ли уже файл нужный диапазон
                if existing_min_date.date() <= start_date.date() and existing_max_date.date() >= end_date.date():
                    print(f"[{tic}] Файл уже содержит данные за весь период.")
                    continue

                # Докачиваем недостающие периоды
                left_gap_df = pd.DataFrame()
                right_gap_df = pd.DataFrame()

                if existing_min_date.date() > start_date.date() and (existing_min_date - delta).date() > start_date.date():
                    left_gap_start = start_date
                    left_gap_end = existing_min_date - delta
                    print(f"[{tic}] Докачиваем слева: {left_gap_start.date()} -> {left_gap_end.date()}")
                    left_gap_df = self._download_for_ticker(tic, left_gap_start, left_gap_end, delta)

                if existing_max_date.date() < end_date.date() and end_date.date() > (existing_max_date + delta).date():
                    right_gap_start = existing_max_date + delta
                    right_gap_end = end_date
                    print(f"[{tic}] Докачиваем справа: {right_gap_start.date()} -> {right_gap_end.date()}")
                    right_gap_df = self._download_for_ticker(tic, right_gap_start, right_gap_end, delta)

                combined_df = pd.concat([existing_df, left_gap_df, right_gap_df], ignore_index=True)
                combined_df.drop_duplicates(subset=["timestamp", "tic"], keep="last", inplace=True)
                combined_df.sort_values(by=["timestamp"], inplace=True)
                combined_df.to_csv(full_path, index=False)
                print(f"[{tic}] Данные дополнены и сохранены в {file_name}.")

            else:
                # Файл не найден — скачиваем полностью
                print(f"[{tic}] Файл не найден. Скачиваем весь период {start_date.date()} -> {end_date.date()}.")
                new_df = self._download_for_ticker(tic, start_date, end_date, delta)
                new_df.to_csv(full_path, index=False)
                print(f"[{tic}] Данные сохранены в {file_name}.")

        print("Загрузка всех данных завершена.")
        return None

    def _download_for_ticker(self,
                             ticker: str,
                             start_date: pd.Timestamp,
                             end_date: pd.Timestamp,
                             delta: timedelta) -> pd.DataFrame:
        """
        Вспомогательная функция для скачивания данных за период [start_date, end_date]
        с шагом в 1 день. Возвращает DataFrame, где 'timestamp' переведён в локальное
        время Нью-Йорка (America/New_York), отфильтрованный по нужному временному диапазону.
        """
        current_date = start_date
        data_df = pd.DataFrame()

        while current_date <= end_date:
            if self.source == "Yahoo":
                # Для получения расширенных часов у Yahoo используем prepost=True
                temp_df = yf.download(
                    ticker,
                    start=current_date,
                    end=current_date + delta,
                    interval=self.time_interval,
                    proxy=self.proxy,
                    prepost=True  # Расширенные часы
                )

                # Сбрасываем индекс, чтобы получить столбец с датой/временем
                temp_df = temp_df.reset_index()

                # yfinance обычно возвращает локально-наивные Timestamp'ы (без таймзоны).
                # Считаем их "UTC" и конвертируем в "America/New_York".
                temp_df["timestamp"] = (
                    temp_df["Date"]
                    .dt.tz_localize("UTC")  # считаем исходный datetime за UTC
                    .dt.tz_convert("America/New_York")  # переводим в Нью-Йорк
                )

                # Приводим колонки к единому виду
                temp_df.drop(columns=["Date"], inplace=True)
                temp_df.columns = [
                    "open",
                    "high",
                    "low",
                    "close",
                    "adjclose",
                    "volume",
                    "timestamp"
                ]
                # Часто для backtest используют close = adjclose
                temp_df["close"] = temp_df["adjclose"]
                temp_df.drop(columns=["adjclose"], inplace=True)

            else:
                # Загрузка с polygon
                client = RESTClient(self.POLYGON_API_KEY)
                aggs = []
                from_str = current_date.strftime("%Y-%m-%d")
                to_str = (current_date + delta).strftime("%Y-%m-%d")
                for a in client.list_aggs(
                        ticker=ticker,
                        multiplier=1,
                        timespan="minute",
                        from_=from_str,
                        to=to_str
                ):
                    aggs.append(a)
                temp_df = pd.DataFrame(aggs)
                if temp_df.empty:
                    current_date += delta
                    continue
                #print('temp_df', temp_df)
                # Переводим timestamp из миллисекунд Unix (UTC) в datetime с таймзоной NYC
                # 1. Преобразуем столбец 'timestamp' в datetime, указывая, что значение – в миллисекундах и в UTC
                temp_df['timestamp'] = pd.to_datetime(temp_df['timestamp'], unit='ms', utc=True)

                # 2. Конвертируем время в часовой пояс Нью-Йорка
                temp_df['timestamp'] = temp_df['timestamp'].dt.tz_convert('America/New_York')


            # Если столбец с временными метками не называется 'timestamp' по умолчанию,
            # обязательно убедитесь, что он именно 'timestamp' (выше в Yahoo-кейсе мы его явно задаём).

            # Фильтрация по времени торгов
            if self.extended_interval:
                # Расширенные торги: 4:00 - 20:00
                start_time = pd.Timestamp("04:00:00").time()
                end_time = pd.Timestamp("20:00:00").time()
            else:
                # Обычная сессия: 9:30 - 16:00
                start_time = pd.Timestamp("09:30:00").time()
                end_time = pd.Timestamp("16:00:00").time()

            # Применяем фильтр по локальному времени Нью-Йорка
            temp_df = temp_df[
                (temp_df["timestamp"].dt.time >= start_time) &
                (temp_df["timestamp"].dt.time <= end_time)
                ]

            # Если данные с Polygon, там уже есть open, high, low, close, volume.
            # Для Yahoo мы адаптировали выше, чтобы всё называлось аналогично.
            # Убедимся, что есть столбцы open/close/high/low/volume в temp_df.

            # Добавим столбец с тикером
            temp_df["tic"] = ticker

            data_df = pd.concat([data_df, temp_df], ignore_index=True)

            current_date += delta

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

    def create_dollar_bars(self, file_name:str, data: pd.DataFrame = None, download_from_disk: bool = False, evaluate: bool = False, optimal_thresholds: Union[Dict[str, float], float] = None) -> pd.DataFrame:
        """
        Создаёт долларовые бары из минутных данных для нескольких тикеров.

        :param data: DataFrame с минутными данными (обязательные столбцы: 'tic', 'close', 'volume').
                    'tic' - тикер актива, 'close' - цена закрытия, 'volume' - объём торгов.
        :param optimal_thresholds: Словарь с оптимальными порогами для каждого тикера (tic -> threshold)
                                  или единое число для всех тикеров.
        :return: DataFrame с долларовыми барами, включая столбцы ['timestamp', 'open', 'high', 'low', 'close', 'tic'].
        """
        if data is None or download_from_disk:
            return self.read_csv(os.path.join(self.file_path, file_name))

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
        all_dollar_bars_df.to_csv(os.path.join(self.file_path, file_name), index=True)

        return all_dollar_bars_df

    def clean_data(
            self,
            tickers=None,
            required_columns=None,
            clean: bool = False
    ) -> dict:
        """
        Очищает (или считывает уже очищенные) данные сразу для нескольких тикеров.
        Использует tqdm для индикации прогресса и print(...) для вывода сообщений.
        ...
        """
        if required_columns is None:
            required_columns = ["open", "high", "low", "close", "volume", "vwap", "transactions"]

        if tickers is None:
            tickers = np.concatenate((self.ticker_list, self.ticker_indicator_list))

        start_date = pd.Timestamp(self.start).tz_localize('UTC')
        end_date = pd.Timestamp(self.end).tz_localize('UTC')
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")

        results = {}

        for ticker in tqdm(tickers, desc="Cleaning data", total=len(tickers)):
            original_file_name = f"{ticker}_{start_str}_{end_str}.csv"
            original_path = os.path.join(self.file_path, self.raw_data, original_file_name)

            cleaned_file_name = f"{ticker}_{start_str}_{end_str}_cleaned.csv"
            cleaned_path = os.path.join(self.file_path, self.cleaned_data, cleaned_file_name)

            print(f"\n[Ticker: {ticker}] Обработка файла: {original_file_name}")

            if not os.path.isfile(original_path):
                msg = f"[{ticker}] Файл не найден: {original_file_name}"
                print(msg)
                raise FileNotFoundError(msg)

            if not clean:
                print(f"[{ticker}] clean=False, ищем очищенный файл: {cleaned_file_name}")
                if os.path.isfile(cleaned_path):
                    print(f"[{ticker}] Найден {cleaned_file_name}, читаем...")
                    # df_cleaned = pd.read_csv(cleaned_path, parse_dates=["timestamp"])
                    # results[ticker] = df_cleaned
                    continue

            print(f"[{ticker}] Загружаем сырой файл и выполняем очистку...")
            df_raw = pd.read_csv(original_path, parse_dates=["timestamp"])

            # Удаление устаревших очищенных файлов для данного тикера с другими датами
            cleaned_dir = os.path.join(self.file_path, self.cleaned_data)
            pattern = f"{ticker}_*_cleaned.csv"
            for filename in os.listdir(cleaned_dir):
                if filename.startswith(f"{ticker}_") and filename.endswith(
                        "_cleaned.csv") and filename != cleaned_file_name:
                    file_to_remove = os.path.join(cleaned_dir, filename)
                    try:
                        os.remove(file_to_remove)
                        print(f"[{ticker}] Удалён устаревший файл: {filename}")
                    except Exception as e:
                        print(f"[{ticker}] Ошибка при удалении файла {filename}: {e}")

            # Проверяем обязательные колонки
            missing_columns = set(required_columns) - set(df_raw.columns)
            if missing_columns:
                msg = f"[{ticker}] Missing required columns: {missing_columns}"
                print(msg)
                raise ValueError(msg)

            #  корректировка ошибки загрузки данных ...
            if 'datetime_ny' in df_raw.columns:
                df_raw['timestamp'] = pd.to_datetime(df_raw['datetime_ny'])

            if df_raw.index.name == 'timestamp':
                df_raw = df_raw.reset_index()

            if 'timestamp' not in df_raw.columns or 'tic' not in df_raw.columns:
                msg = f"[{ticker}] DataFrame must contain 'timestamp' and 'tic' columns."
                print(msg)
                raise KeyError(msg)

            df_raw = df_raw[required_columns + ['tic', 'timestamp']]
            df_raw = df_raw.loc[df_raw['tic'] == ticker].copy()

            df_cleaned = self._build_cleaned_df(df_raw, ticker, required_columns)

            df_cleaned.to_csv(cleaned_path, index=False)
            print(f"[{ticker}] Очищенные данные сохранены в {cleaned_file_name}")

            # results[ticker] = df_cleaned

        print("\nОчистка данных завершена для всех тикеров.")
        return results

    def _build_cleaned_df(self, df: pd.DataFrame, ticker: str, required_columns: list) -> pd.DataFrame:
        """
        Формирует «чистый» DataFrame для заданного тикера:
        - Генерирует сплошной временной индекс,
        - Заполняет пропуски (forward-fill, backward-fill),
        - Проверяет на NaN,
        - Возвращает готовый DataFrame.
        """
        # Удаляем дубликаты по timestamp
        df.drop_duplicates(subset='timestamp', keep='first', inplace=True)
        df.set_index("timestamp", inplace=True)

        # Генерируем полный индекс времени
        times = self._generate_time_index()

        # Создаём DataFrame со сплошным временным индексом
        tmp_df = pd.DataFrame(index=times, columns=required_columns)

        # Переносим значения
        tmp_df.update(df)

        # Заполняем пропуски (ffill/bfill)
        tmp_df.ffill(inplace=True)
        tmp_df.bfill(inplace=True)

        # Проверяем, остались ли NaN
        if tmp_df[required_columns].isna().any().any():
            msg = f"[{ticker}] NaN values remain in cleaned data."
            print(msg)
            raise ValueError(msg)

        # Пропишем тикер
        tmp_df["tic"] = ticker

        # Сбросим индекс, получим колонку "timestamp"
        final_df = tmp_df.reset_index().rename(columns={"index": "timestamp"})
        final_df.sort_values("timestamp", inplace=True)
        return final_df

    def _generate_time_index(self) -> pd.Index:
        """
        Генерирует сплошной временной индекс с учётом self.time_interval и self.extended_interval.
        """
        trading_days = self.get_trading_days()
        tz = pytz.timezone("America/New_York")

        if self.time_interval == "1d":
            # По торговым дням (Date)
            return pd.Index(trading_days)  # array of date objects
        elif self.time_interval == "1m":
            all_minutes = []
            for day in trading_days:
                if self.extended_interval:
                    # Расширенные торги: 04:00 - 20:00
                    day_times = pd.date_range(
                        start=pd.Timestamp(f"{day} 04:00:00", tz=tz),
                        end=pd.Timestamp(f"{day} 20:00:00", tz=tz),
                        freq='T'
                    )
                else:
                    # Обычная сессия: 09:30 - 16:00
                    day_times = pd.date_range(
                        start=pd.Timestamp(f"{day} 09:30:00", tz=tz),
                        end=pd.Timestamp(f"{day} 16:00:00", tz=tz),
                        freq='T'
                    )
                all_minutes.extend(day_times)
            return pd.Index(all_minutes)
        else:
            raise ValueError("Unsupported time interval (only '1d' or '1m').")

    def top_100_tickers_by_bin_share(self):
        """
        Для списка тикеров:
          - Загружает соответствующие CSV-файлы из указанной директории.
          - Подсчитывает значения в колонке `bin`.
          - Сортирует тикеры по наибольшей доле значений -1 и 1.

        Параметры:
        ticker_list: список тикеров для обработки.
        file_path: базовая директория, где хранятся файлы.
        labels: подпапка в базовой директории, содержащая CSV-файлы.
        start_str, end_str: строки, определяющие временной интервал в именах файлов.

        Возвращает:
        Список из 100 кортежей вида (тикер, доля), отсортированных по убыванию доли значений -1 и 1.
        """
        ticker_list = np.concatenate((self.ticker_list, self.ticker_indicator_list))

        required_columns = ['bin']

        start_date = pd.Timestamp(self.start).tz_localize('UTC')
        end_date = pd.Timestamp(self.end).tz_localize('UTC')
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")

        bin_value_counts = {}

        # Используем tqdm для отображения прогресса
        for ticker in tqdm(ticker_list, desc="Processing tickers"):
            # Формируем имя файла по шаблону
            labeled_file_name = f"{ticker}_{start_str}_{end_str}_labeled.csv"
            # Полный путь к файлу
            labeled_path = os.path.join(self.file_path, self.labels, labeled_file_name)

            # Проверяем существование файла
            if os.path.exists(labeled_path):
                df = pd.read_csv(labeled_path)
                if 'bin' in df.columns:
                    # Подсчитываем значения -1, 0 и 1 в колонке 'bin'
                    counts = df['bin'].value_counts().to_dict()
                    # Обеспечиваем наличие ключей -1, 0, 1
                    counts_complete = {k: counts.get(k, 0) for k in [-1, 0, 1]}
                    bin_value_counts[ticker] = counts_complete
                else:
                    print(f"В файле для {ticker} отсутствует колонка 'bin'.")
            else:
                print(f"Файл для {ticker} не найден по пути: {labeled_path}")

        # Список для хранения тикеров с вычисленной долей
        ticker_shares = []

        # Вычисляем долю значений -1 и 1 для каждого тикера
        for ticker, counts in bin_value_counts.items():
            total = sum(counts.values())
            share = (counts.get(-1, 0) + counts.get(1, 0)) / total if total > 0 else 0
            ticker_shares.append((ticker, share))

        # Сортировка тикеров по доле в порядке убывания
        ticker_shares_sorted = sorted(ticker_shares, key=lambda x: x[1], reverse=True)

        # Возвращаем первые 100 элементов
        return ticker_shares_sorted[:100], ticker_shares_sorted

    def label_data(
            self,
            tickers=None,
            label: bool = False
    ) -> dict:
        """
        Применяет функцию short_long_box к очищённым данным для нескольких тикеров.
        Использует tqdm для индикации прогресса и print(...) для вывода сообщений.
        Возвращает словарь с DataFrame для каждого тикера, содержащим разметку.
        """
        if tickers is None:
            tickers = np.concatenate((self.ticker_list, self.ticker_indicator_list))

        required_columns = ['close']

        start_date = pd.Timestamp(self.start).tz_localize('UTC')
        end_date = pd.Timestamp(self.end).tz_localize('UTC')
        start_str = start_date.strftime("%Y%m%d")
        end_str = end_date.strftime("%Y%m%d")

        for ticker in tqdm(tickers, desc="Labeling data", total=len(tickers)):
            cleaned_file_name = f"{ticker}_{start_str}_{end_str}_cleaned.csv"
            cleaned_path = os.path.join(self.file_path, self.cleaned_data, cleaned_file_name)

            labeled_file_name = f"{ticker}_{start_str}_{end_str}_labeled.csv"
            labeled_path = os.path.join(self.file_path, self.labels, labeled_file_name)

            if not os.path.isfile(labeled_path):
                msg = f"[{ticker}] Размеченный файл не найден, формирование нового размеченного файла: {labeled_file_name}"
                print(msg)
            elif not label:
                # не переделываем разметку, если она уже есть
                msg = f"[{ticker}] Размеченный файл найден: {labeled_file_name}"
                print(msg)
                continue  # или выбросить ошибку, если это критично

            # Проверяем наличие очищённого файла
            if not os.path.isfile(cleaned_path):
                msg = f"[{ticker}] Очищённый файл не найден: {cleaned_file_name}"
                print(msg)
                continue  # или выбросить ошибку, если это критично

            # Читаем очищённые данные
            df_cleaned = pd.read_csv(cleaned_path, parse_dates=["timestamp"])

            # Удаление устаревших очищенных файлов для данного тикера с другими датами
            labels_dir = os.path.join(self.file_path, self.labels)
            pattern = f"{ticker}_*_labeled.csv"
            for filename in os.listdir(labels_dir):
                if filename.startswith(f"{ticker}_") and filename.endswith(
                        "_labeled.csv") and filename != cleaned_file_name:
                    file_to_remove = os.path.join(labels_dir, filename)
                    try:
                        os.remove(file_to_remove)
                        print(f"[{ticker}] Удалён устаревший файл: {filename}")
                    except Exception as e:
                        print(f"[{ticker}] Ошибка при удалении файла {filename}: {e}")

            # Проверка наличия необходимых столбцов в очищённых данных
            missing_columns = set(required_columns + ['tic', 'timestamp']) - set(df_cleaned.columns)
            if missing_columns:
                msg = f"[{ticker}] Missing required columns in cleaned data: {missing_columns}"
                print(msg)
                continue

            # Применяем функцию short_long_box для разметки
            try:
                df_labeled = short_long_box(df_cleaned)
            except Exception as e:
                print(f"[{ticker}] Ошибка при разметке: {e}")
                continue

            # Сохраняем размеченные данные
            try:
                df_labeled.to_csv(labeled_path, index=False)
                print(f"[{ticker}] Размеченные данные сохранены в {labeled_file_name}")
            except Exception as e:
                print(f"[{ticker}] Ошибка при сохранении файла {labeled_file_name}: {e}")
                continue

            #results[ticker] = df_labeled

        print("\nРазметка данных завершена для всех тикеров.")
        return True

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

        yesterday_date = (datetime.datetime.now() - datetime.timedelta(days=1)).strftime('%Y-%m-%d')

        self.start = self.TRAIN_START_DATE

        end = '2025-01-10'

        if end is None:
            self.end = yesterday_date
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


def add_takeprofit_stoploss_volume(
        predicted_data: pd.DataFrame,
        coeff_tp: float = 1.0,
        coeff_sl: float = 1.0,
        eps: float = 1e-6
) -> pd.DataFrame:
    """
    Оптимизирует массивы в колонке 'prediction' DataFrame для работы на TPU.
    При этом финальный массив НЕ будет содержать исходный 4-й элемент (return).

    Структура входных данных (каждая строка в колонке 'prediction'):
        [p(-1), p(0), p(+1), return, ...]

    Итоговая структура после обработки:
        [p(-1), p(0), p(+1), vol, tp, sl]

    Parameters
    ----------
    predicted_data : pd.DataFrame
        DataFrame с колонкой 'prediction', содержащей массивы вида
        [p(-1), p(0), p(+1), return, ...].
    coeff_tp : float, optional
        Коэффициент для расчёта take-profit, по умолчанию 1.
    coeff_sl : float, optional
        Коэффициент для расчёта stop-loss, по умолчанию 1.
    eps : float, optional
        Небольшой зазор для избежания деления на 0 (clip вероятностей).

    Returns
    -------
    pd.DataFrame
        Обновлённый DataFrame, где в каждом элементе 'prediction':
          * НЕ сохраняется исходный 4-й элемент (return)
          * Добавляются значения vol, tp, sl
    """

    if 'prediction' not in predicted_data.columns:
        raise KeyError("В DataFrame отсутствует колонка 'prediction'.")

    # Преобразуем 'prediction' в объект np.ndarray (список списков -> 2D-массив dtype=object)
    predictions = np.array(predicted_data['prediction'].tolist(), dtype=object)
    if len(predictions) == 0:
        return predicted_data

    # Извлекаем первые три элемента (вероятности) и четвёртый элемент (return)
    first_three = []
    fourth_element = []
    for row in predictions:
        # Если меньше трёх элементов, дополним нулями
        if len(row) >= 3:
            first_three.append(row[:3])
        else:
            first_three.append([0.0, 0.0, 0.0])

        # Берём return, если он есть; иначе 0
        if len(row) >= 4 and row[3] is not None:
            fourth_element.append(row[3])
        else:
            fourth_element.append(0.0)

    first_three = np.array(first_three, dtype=float)
    return_value = np.array(fourth_element, dtype=float)

    # Определяем bin = argmax по первым трём элементам => значения 1, 2 или 3
    max_bin = np.argmax(first_three, axis=1) + 1

    # Корректируем return_value по знаку:
    #   bin=3 (long): если return < 0 => берем модуль
    #   bin=1 (short): если return > 0 => берем отрицательный модуль
    cond_long_negative = (max_bin == 3) & (return_value < 0)
    return_value[cond_long_negative] = np.abs(return_value[cond_long_negative])

    cond_short_positive = (max_bin == 1) & (return_value > 0)
    return_value[cond_short_positive] = -np.abs(return_value[cond_short_positive])

    # Инициализация vol, tp, sl
    vol = np.zeros_like(return_value, dtype=float)
    tp = np.zeros_like(return_value, dtype=float)
    sl = np.zeros_like(return_value, dtype=float)

    # --- bin = 1 (short) ---
    mask_bin_minus1 = (max_bin == 1)
    p_minus1 = first_three[mask_bin_minus1, 0]
    # Зажимаем вероятности, чтобы не было деления на ноль
    p_minus1 = np.clip(p_minus1, eps, 1 - eps)
    b_minus1 = p_minus1 / (1.0 - p_minus1)

    # vol = p(-1) - (1 - p(-1)) / b(-1), но не меньше 0
    raw_vol_minus1 = p_minus1 - (1 - p_minus1) / b_minus1
    vol[mask_bin_minus1] = np.maximum(0.0, raw_vol_minus1)

    return_minus1 = return_value[mask_bin_minus1]
    tp[mask_bin_minus1] = coeff_tp * return_minus1
    sl[mask_bin_minus1] = - coeff_sl * return_minus1 / b_minus1

    # --- bin = 3 (long) ---
    mask_bin_plus1 = (max_bin == 3)
    p_plus1 = first_three[mask_bin_plus1, 2]
    p_plus1 = np.clip(p_plus1, eps, 1 - eps)
    b_plus1 = p_plus1 / (1.0 - p_plus1)

    raw_vol_plus1 = p_plus1 - (1 - p_plus1) / b_plus1
    vol[mask_bin_plus1] = np.maximum(0.0, raw_vol_plus1)

    return_plus1 = return_value[mask_bin_plus1]
    tp[mask_bin_plus1] = coeff_tp * return_plus1
    sl[mask_bin_plus1] = - coeff_sl * return_plus1 / b_plus1

    # Формируем финальную структуру: [p(-1), p(0), p(+1), vol, tp, sl]
    updated_predictions = []
    for i, row in enumerate(predictions):
        # Берём первые три вероятности
        row_3 = list(row[:3])
        # Добавляем vol, tp, sl
        new_row = row_3 + [vol[i], tp[i], sl[i]]
        updated_predictions.append(new_row)

    # Записываем обновлённую колонку обратно в DataFrame
    predicted_data['prediction'] = updated_predictions

    return predicted_data




