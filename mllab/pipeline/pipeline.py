
import os
import time
import threading
import logging
import requests
import pandas as pd
import joblib
import pickle

from pathlib import Path
from tqdm import tqdm  # не забудьте установить: pip install tqdm

# from your_rl_module import DRLAgent, env_train  # пример, если нужно
from mllab.pipeline.drive import GoogleDriveHandler



def log_execution_time(func):
    """Декоратор для логирования времени выполнения методов."""
    def wrapper(self, *args, **kwargs):
        start_time = time.time()
        result = func(self, *args, **kwargs)
        end_time = time.time()
        elapsed_time = end_time - start_time
        self.logger.info(f"Execution time for {func.__name__}: {elapsed_time:.4f} seconds")
        if func.__name__ not in self.execution_times:
            self.execution_times[func.__name__] = []
        self.execution_times[func.__name__].append(elapsed_time)
        return result
    return wrapper


class TradingPipeline:
    def __init__(
            self,
            stock_list=None,
            polygon_api_key=None,
            indicators=None,
            ibkr_module=None,
            start="2021-01-01",
            end="2021-12-31",
            local_data_root="data"
    ):
        """
        Инициализация TradingPipeline:
          - stock_list: список тикеров
          - polygon_api_key: ключ для Polygon API
          - indicators: список функций, которые рассчитывают индикаторы
          - ibkr_module: модуль для работы с IBKR
          - start / end: даты (используются в названиях файлов)
          - local_data_root: корневая локальная папка для кэша (data/)
        """

        # Список тикеров по умолчанию
        if stock_list is None:
            stock_list = ['NVR', 'MPWR']

        self.stock_list = stock_list
        self.polygon_api_key = polygon_api_key
        self.indicators = indicators if indicators else []
        self.ibkr_module = ibkr_module

        # Даты для формирования названий файлов
        self.start = pd.Timestamp(start)
        self.end = pd.Timestamp(end)
        self.start_str = self.start.strftime("%Y%m%d")
        self.end_str = self.end.strftime("%Y%m%d")

        # Локальная папка (кэш)
        self.local_data_root = local_data_root
        os.makedirs(self.local_data_root, exist_ok=True)

        # GoogleDriveHandler для скачивания
        self.drive_handler = GoogleDriveHandler()

        # Логгер
        logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
        self.logger = logging.getLogger(self.__class__.__name__)

        # Корневая папка на Google Диске, где лежат модели: root/DataTrading/SP500_1m
        self.gdrive_root = "DataTrading/SP500_1m"

        # Папки для bagging
        self.bagging = 'bagging'
        self.bagging_model = 'bagging_model'
        self.bagging_scaler = 'bagging_scaler'
        self.bagging_indicator = 'bagging_indicator'

        # Папки для regression
        self.regression = 'regression'
        self.regression_model = 'regression_model'
        self.regression_scaler = 'regression_scaler'
        self.regression_indicator = 'regression_indicator'

        # Переменные пайплайна
        self.raw_data = {}
        self.data_lock = threading.Lock()
        self.execution_times = {}

        # Здесь сохраняем загруженные объекты:
        #   self.model_data["bagging"][ticker] = {
        #       "model": <модель>,
        #       "scaler": <скейлер>,
        #       "indicator": <индикатор>
        #   }
        #   self.model_data["regression"][ticker] = {...}
        self.model_data = {
            "bagging": {},
            "regression": {}
        }

        # Загружаем все файлы (6 на тикер) с единым прогресс-баром
        self._load_all_models_for_all_tickers()

    # ------------------------------------------------------------------------
    # Метод load для локальных файлов (по расширению)
    # ------------------------------------------------------------------------
    def load(self, file_path):
        """
        Загрузка данных/моделей по расширению:
          - .pkl -> joblib.load
          - .csv -> pd.read_csv (с индексом по 'timestamp', если есть)
          - .zip -> заглушка (NotImplementedError) для RL
          - прочие (.lst или др.) -> pickle.load
        """
        if file_path is None:
            raise ValueError("file_path cannot be None.")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file '{file_path}' does not exist.")

        _, ext = os.path.splitext(file_path)
        ext = ext.lower()

        if ext == ".pkl":
            # sklearn / joblib-модель
            return joblib.load(file_path)

        elif ext == ".csv":
            # DataFrame
            result = pd.read_csv(file_path, parse_dates=["timestamp"])
            if result.index.name != 'timestamp':
                if 'timestamp' in result.columns:
                    result = result.set_index('timestamp')
                else:
                    raise ValueError("The 'timestamp' column is not present in the DataFrame.")
            return result

        elif ext == ".zip":
            # Пример для RL-моделей
            # Нужно адаптировать под реальную логику
            raise NotImplementedError("Загрузка RL-моделей (.zip) не реализована!")

        else:
            # Считаем, что .lst или другое — это Pickle
            with open(file_path, 'rb') as f:
                return pickle.load(f)

    # ------------------------------------------------------------------------
    # Методы для скачивания файлов с Google Диска и кэширования
    # ------------------------------------------------------------------------
    def _download_and_cache_file(self, drive_path: str, local_filename: str) -> str:
        """
        Проверяет, есть ли файл local_filename в локальном кэше (data/),
        если нет — скачивает с Google Диска (используя GoogleDriveHandler).
        Возвращает полный путь к локальному файлу.
        """
        local_path = os.path.join(self.local_data_root, local_filename)
        if os.path.exists(local_path):
            self.logger.info(f"Файл уже локально: {local_path}")
            return local_path

        self.logger.info(f"Скачиваем '{local_filename}' из '{drive_path}' -> '{local_path}'...")
        file_id = self.drive_handler.find_file_id_by_path(drive_path)
        if not file_id:
            raise FileNotFoundError(f"Файл не найден на Google Диске: {drive_path}")

        # Скачиваем
        self.drive_handler.download_file(
            file_id=file_id,
            file_name=local_filename,
            local_folder=self.local_data_root,
            drive_file_size=0,
            progress_bar=None  # chunk-прогресс не используем, у нас общий tqdm
        )
        self.logger.info(f"Загружен файл: {local_path}")
        return local_path

    def _download_files_for_ticker(self, ticker, pbar):
        """
        Загрузить 6 файлов (bagging + regression) для одного тикера,
        после каждого загруженного файла pbar.update(1).

        Возвращает словарь с объектами:
           {
             "bagging_model": <объект модели>,
             "bagging_scaler": <скейлер>,
             "bagging_indicator": <что-то>,
             "regression_model": <объект модели>,
             "regression_scaler": <скейлер>,
             "regression_indicator": <что-то>,
           }
        """
        results = {}

        # ---------- Bagging Model ----------
        bm_filename = f"{ticker}_{self.start_str}_{self.end_str}_bagging_model.pkl"
        bm_drive_path = f"{self.gdrive_root}/{self.bagging}/{self.bagging_model}/{bm_filename}"
        bm_local = self._download_and_cache_file(bm_drive_path, bm_filename)
        results["bagging_model"] = self.load(bm_local)
        pbar.update(1)

        # ---------- Bagging Scaler ----------
        bs_filename = f"{ticker}_{self.start_str}_{self.end_str}_bagging_scaler.pkl"
        bs_drive_path = f"{self.gdrive_root}/{self.bagging}/{self.bagging_scaler}/{bs_filename}"
        bs_local = self._download_and_cache_file(bs_drive_path, bs_filename)
        results["bagging_scaler"] = self.load(bs_local)
        pbar.update(1)

        # ---------- Bagging Indicator ----------
        bi_filename = f"{ticker}_{self.start_str}_{self.end_str}_bagging_indicator.lst"
        bi_drive_path = f"{self.gdrive_root}/{self.bagging}/{self.bagging_indicator}/{bi_filename}"
        bi_local = self._download_and_cache_file(bi_drive_path, bi_filename)
        results["bagging_indicator"] = self.load(bi_local)
        pbar.update(1)

        # ---------- Regression Model ----------
        rm_filename = f"{ticker}_{self.start_str}_{self.end_str}_regression_model.pkl"
        rm_drive_path = f"{self.gdrive_root}/{self.regression}/{self.regression_model}/{rm_filename}"
        rm_local = self._download_and_cache_file(rm_drive_path, rm_filename)
        results["regression_model"] = self.load(rm_local)
        pbar.update(1)

        # ---------- Regression Scaler ----------
        rs_filename = f"{ticker}_{self.start_str}_{self.end_str}_regression_scaler.pkl"
        rs_drive_path = f"{self.gdrive_root}/{self.regression}/{self.regression_scaler}/{rs_filename}"
        rs_local = self._download_and_cache_file(rs_drive_path, rs_filename)
        results["regression_scaler"] = self.load(rs_local)
        pbar.update(1)

        # ---------- Regression Indicator ----------
        ri_filename = f"{ticker}_{self.start_str}_{self.end_str}_regression_indicator.lst"
        ri_drive_path = f"{self.gdrive_root}/{self.regression}/{self.regression_indicator}/{ri_filename}"
        ri_local = self._download_and_cache_file(ri_drive_path, ri_filename)
        results["regression_indicator"] = self.load(ri_local)
        pbar.update(1)

        return results

    def _load_all_models_for_all_tickers(self):
        """
        Единый прогресс-бар на (6 файлов * число тикеров).
        Для каждого тикера загружаем (bagging + regression).
        """
        from tqdm import tqdm

        total_files = len(self.stock_list) * 6
        with tqdm(total=total_files, desc="Loading all models/scalers/indicators", unit="file") as pbar:
            for ticker in self.stock_list:
                self.logger.info(f"--- ЗАГРУЗКА ФАЙЛОВ ДЛЯ {ticker} ---")
                try:
                    files_dict = self._download_files_for_ticker(ticker, pbar)
                except Exception as e:
                    self.logger.error(f"[{ticker}] Ошибка при загрузке: {e}")
                    continue

                # Раскладываем по словарю model_data
                self.model_data["bagging"][ticker] = {
                    "model": files_dict["bagging_model"],
                    "scaler": files_dict["bagging_scaler"],
                    "indicator": files_dict["bagging_indicator"]
                }
                self.model_data["regression"][ticker] = {
                    "model": files_dict["regression_model"],
                    "scaler": files_dict["regression_scaler"],
                    "indicator": files_dict["regression_indicator"]
                }

    @log_execution_time
    def fetch_data(self):
        """Fetches trading data from the Polygon API for the given stock list.

        Retrieves the previous day's aggregate trading data for each stock in the stock list.
        """
        data = {}
        for stock in self.stock_list:
            response = requests.get(
                f"https://api.polygon.io/v2/aggs/ticker/{stock}/prev",
                params={"apiKey": self.polygon_api_key}
            )
            if response.status_code == 200:
                data[stock] = response.json()
            else:
                print(f"Error fetching data for {stock}: {response.text}")
        with self.data_lock:
            self.raw_data = data

    @log_execution_time
    def preprocess_data(self, raw_data):
        """Preprocesses raw data fetched from Polygon API.

        Extracts relevant fields like open, close, high, low, volume, and timestamp for each stock.

        Args:
            raw_data (dict): Raw data fetched from Polygon API.

        Returns:
            dict: Processed data with essential fields.
        """
        processed_data = {}
        for stock, data in raw_data.items():
            try:
                if "results" in data and data["results"]:
                    stock_data = data["results"][0]
                    processed_data[stock] = {
                        "open": stock_data.get("o", 0),
                        "close": stock_data.get("c", 0),
                        "high": stock_data.get("h", 0),
                        "low": stock_data.get("l", 0),
                        "volume": stock_data.get("v", 0),
                        "timestamp": stock_data.get("t", 0)
                    }
                else:
                    print(f"No results for {stock}, skipping.")
            except Exception as e:
                print(f"Error processing data for {stock}: {e}")
        return processed_data

    @log_execution_time
    def calculate_indicators(self, data):
        """Calculates indicators based on the preprocessed data.

        Args:
            data (dict): Preprocessed trading data for each stock.

        Returns:
            dict: Calculated indicators for each stock.
        """
        calculated_indicators = {}
        for stock, stock_data in data.items():
            stock_indicators = {}
            for indicator in self.indicators:
                stock_indicators[indicator.__name__] = indicator(stock_data)
            calculated_indicators[stock] = stock_indicators
        return calculated_indicators

    @log_execution_time
    def predict_trend(self, indicators):
        """Predicts the trend direction using the trend model.

        Args:
            indicators (dict): Calculated indicators for each stock.

        Returns:
            dict: Predicted trend direction (e.g., up or down) for each stock.
        """
        predictions = {}
        for stock, indicator_values in indicators.items():
            df = pd.DataFrame([indicator_values])
            predictions[stock] = self.trend_model.predict(df)[0]
        return predictions

    @log_execution_time
    def predict_growth(self, indicators):
        """Predicts the growth level using the growth model.

        Args:
            indicators (dict): Calculated indicators for each stock.

        Returns:
            dict: Predicted growth level for each stock.
        """
        predictions = {}
        for stock, indicator_values in indicators.items():
            df = pd.DataFrame([indicator_values])
            predictions[stock] = self.growth_model.predict(df)[0]
        return predictions

    @log_execution_time
    def calculate_weights_and_limits(self, trend_predictions, growth_predictions):
        """Calculates weights, stop loss, and take profit for each stock.

        Args:
            trend_predictions (dict): Predicted trend directions for each stock.
            growth_predictions (dict): Predicted growth levels for each stock.

        Returns:
            list: Actions with calculated weights, stop loss, and take profit for each stock.
        """
        actions = []
        for stock in trend_predictions.keys():
            trend = trend_predictions[stock]
            growth = growth_predictions[stock]
            weight = growth if trend == 1 else 0
            stop_loss = growth * 0.95  # Example stop loss at 95% of growth
            take_profit = growth * 1.1  # Example take profit at 110% of growth
            actions.append({
                "stock": stock,
                "weight": weight,
                "stop_loss": stop_loss,
                "take_profit": take_profit
            })
        return actions

    def execute_actions(self, actions):
        """Executes the actions through the IBKR module.

        Args:
            actions (list): List of actions with trading parameters for each stock.
        """
        for action in actions:
            self.ibkr_module.place_order(action)

    def data_fetching_loop(self, interval):
        """Continuously fetch data in a separate thread.

        Args:
            interval (int): Time interval in seconds between fetches.
        """
        while True:
            self.fetch_data()
            time.sleep(interval)

    def run_online(self, interval=60):
        """Runs the pipeline in online mode, fetching and processing data in parallel.

        Args:
            interval (int): Time interval in seconds between data fetches.
        """
        threading.Thread(target=self.data_fetching_loop, args=(interval,), daemon=True).start()
        while True:
            with self.data_lock:
                raw_data = self.raw_data.copy()
            processed_data = self.preprocess_data(raw_data)
            indicators = self.calculate_indicators(processed_data)
            trend_predictions = self.predict_trend(indicators)
            growth_predictions = self.predict_growth(indicators)
            actions = self.calculate_weights_and_limits(trend_predictions, growth_predictions)
            self.execute_actions(actions)

    def run_backtest(self, historical_data):
        """Runs the pipeline in backtesting mode with historical data.

        Args:
            historical_data (dict): Historical trading data for backtesting.
        """
        for timestamp, raw_data in historical_data.items():
            processed_data = self.preprocess_data(raw_data)
            indicators = self.calculate_indicators(processed_data)
            trend_predictions = self.predict_trend(indicators)
            growth_predictions = self.predict_growth(indicators)
            actions = self.calculate_weights_and_limits(trend_predictions, growth_predictions)
            print(f"{timestamp}: Actions: {actions}")

# Example usage:
# trading_pipeline = TradingPipeline(stock_list, polygon_api_key, indicators, 'MyDrive/Models/trend_model.pkl', 'MyDrive/Models/growth_model.pkl', ibkr_module)
# trading_pipeline.connect_google_drive()
# trading_pipeline.run_online()
