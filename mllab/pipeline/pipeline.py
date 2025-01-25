import time
import requests
import pandas as pd
import os
import threading
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
import joblib
from mllab.drive import GoogleDriveHandler


class TradingPipeline:
    def __init__(self, stock_list, polygon_api_key, indicators, trend_model_path, growth_model_path, ibkr_module):
        """Initializes the TradingPipeline with required parameters.

        Args:
            stock_list (list): List of stock tickers to trade.
            polygon_api_key (str): API key for accessing Polygon data.
            indicators (list): List of indicator functions to compute.
            trend_model_path (str): Path to the trend prediction model on Google Drive.
            growth_model_path (str): Path to the growth prediction model on Google Drive.
            ibkr_module (module): Module for interacting with IBKR API for placing orders.
        """
        self.stock_list = stock_list
        self.polygon_api_key = polygon_api_key
        self.indicators = indicators
        self.trend_model_path = trend_model_path
        self.growth_model_path = growth_model_path
        self.ibkr_module = ibkr_module
        self.drive_handler = GoogleDriveHandler()
        self.trend_model = self.load_model(trend_model_path, "trend_model.pkl")
        self.growth_model = self.load_model(growth_model_path, "growth_model.pkl")
        self.raw_data = {}
        self.data_lock = threading.Lock()
        self.execution_times = {}  # Dictionary to store execution times

    def log_execution_time(func):
        """Decorator to log the execution time of a method and save it to execution_times."""

        def wrapper(self, *args, **kwargs):
            start_time = time.time()
            result = func(self, *args, **kwargs)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Execution time for {func.__name__}: {elapsed_time:.4f} seconds")
            if func.__name__ not in self.execution_times:
                self.execution_times[func.__name__] = []
            self.execution_times[func.__name__].append(elapsed_time)
            return result

        return wrapper

    def load_model(self, drive_path, local_filename):
        """Loads a model from Google Drive using GoogleDriveHandler.

        Args:
            drive_path (str): Path to the model file on Google Drive.
            local_filename (str): Name of the cached file locally.

        Returns:
            object: The loaded model.
        """
        local_cache_path = os.path.join("/content/cache", local_filename)
        os.makedirs("/content/cache", exist_ok=True)

        if not os.path.exists(local_cache_path):
            self.logger.info(f"Downloading {local_filename} from Google Drive...")
            file_id = self.drive_handler.file_exists(os.path.basename(drive_path))['id']
            self.drive_handler.download_file(file_id, local_cache_path)
        else:
            self.logger.info(f"Using cached model {local_filename}.")

        return joblib.load(local_cache_path)

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
