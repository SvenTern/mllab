import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, LSTM
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.utils import resample
from tensorflow.keras.utils import Sequence
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import joblib
from mllab.cross_validation import score_confusion_matrix
import os
import random
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.utils.class_weight import compute_class_weight
from mllab.cross_validation import score_confusion_matrix
from gym.utils import seeding
import gym
from gym import spaces
from stable_baselines3.common.vec_env import DummyVecEnv


class ensemble_models:
    def __init__(self, indicators, labels):
        list_main_indicators = ['log_t1', 'log_t2', 'log_t3', 'log_t4', 'log_t5', 'ma_10', 'ma_50', 'ma_200',
                                'bollinger_upper', 'bollinger_lower', 'rsi', 'vwap_diff', 'macd_signal']
        label = 'bin'
        self.X = indicators[list_main_indicators]
        self.y = labels[label]
        self.scaler = StandardScaler()
        self.X = pd.DataFrame(self.scaler.fit_transform(self.X), columns=self.X.columns).values
        self.all_classes = np.unique(self.y)
        self.class_weights = compute_class_weight('balanced', classes=np.unique(self.y), y=self.y)
        self.class_weights_dict = {cls: weight for cls, weight in zip(np.unique(self.y), self.class_weights)}
        self.class_weights_dict = {cls: max(1.0, weight) for cls, weight in self.class_weights_dict.items()}
        self.strategy = self.setup_tpu()

    # Set up TPU strategy if available
    def setup_tpu(self):
        try:
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # Detect TPU
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.TPUStrategy(tpu)
            print("Running on TPU")
        except ValueError:
            strategy = tf.distribute.MirroredStrategy()  # Fallback to multi-GPU or CPU
            print("Running on CPU/GPU")
        return strategy

    def load_ensemble_models(self, directory="ensemble_models"):
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory {directory} does not exist.")

        ensemble_models = []
        for filename in sorted(os.listdir(directory)):
            if filename.endswith(".h5"):
                model_path = os.path.join(directory, filename)
                try:
                    model = tf.keras.models.load_model(model_path)
                    ensemble_models.append(model)
                    print(f"Loaded model from {model_path}")
                except Exception as e:
                    print(f"Error loading model from {model_path}: {e}")
                    continue
        return ensemble_models

    def create_dense_model(self, input_dim):
        with self.strategy.scope():
            model = Sequential([
                Dense(128, activation='relu', input_shape=(input_dim,),
                      kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                BatchNormalization(),
                Dropout(0.5),
                Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                BatchNormalization(),
                Dropout(0.5),
                Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
                BatchNormalization(),
                Dropout(0.5),
                Dense(1, activation='tanh')
            ])
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model

    def create_lstm_model(self, input_dim):
        with self.strategy.scope():
            model = Sequential([
                LSTM(64, input_shape=(input_dim, 1), return_sequences=True, activation='tanh',
                     recurrent_activation='sigmoid'),
                Dropout(0.5),
                LSTM(64, activation='tanh', recurrent_activation='sigmoid'),
                Dropout(0.5),
                Dense(1, activation='tanh')
            ])
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model

    def train_ensemble(self, X_train, y_train, n_estimators=5, batch_size=32, epochs=20, use_saved_weights=False):
        ensemble_models = []
        for i in tqdm(range(n_estimators), desc="Training Ensemble Models", unit="model"):
            X_resampled, y_resampled = resample(np.array(X_train), np.array(y_train))
            sample_weights = np.array([self.class_weights_dict[label] for label in y_resampled])

            model_type = random.choice(['dense', 'lstm'])
            if model_type == 'dense':
                model = self.create_dense_model(X_train.shape[1])
            elif model_type == 'lstm':
                X_resampled = X_resampled.reshape(X_resampled.shape[0], X_resampled.shape[1], 1)
                model = self.create_lstm_model(X_resampled.shape[1])
            else:
                raise ValueError("Unsupported model_type. Use 'dense' or 'lstm'.")

            if use_saved_weights:
                try:
                    model.load_weights(f"ensemble_weights/model_{model_type}_weights.h5")
                    print(f"Loaded weights for model {model_type}")
                except FileNotFoundError:
                    print(f"Weights not found for model {model_type}, training from scratch.")

            print(f"Training model {i + 1}/{n_estimators} as {model_type}")

            early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
            model.fit(X_resampled, y_resampled, sample_weight=sample_weights, batch_size=batch_size, epochs=epochs,
                      verbose=1,
                      callbacks=[early_stopping])
            ensemble_models.append(model)

            if use_saved_weights:
                try:
                    os.makedirs("ensemble_weights", exist_ok=True)
                    model.save_weights(f"ensemble_weights/model_{model_type}_weights.h5")
                    print(f"Saved weights for model {model_type}")
                except Exception as e:
                    print(f"Error saving weights for model {model_type}: {e}")

        return ensemble_models

    def ensemble_predict(self, ensemble_models, X_test, digitize=True):
        X_test = np.array(X_test)
        predictions = np.zeros((X_test.shape[0], len(ensemble_models)))
        for i, model in enumerate(ensemble_models):
            if isinstance(model.layers[0], LSTM):
                X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                predictions[:, i] = model.predict(X_test_reshaped).flatten()
            else:
                predictions[:, i] = model.predict(X_test).flatten()

        averaged_predictions = np.mean(predictions, axis=1)
        mapped_predictions = np.digitize(averaged_predictions, bins=[-0.33, 0.33]) - 1
        return mapped_predictions if digitize else (mapped_predictions, averaged_predictions)

    def train_and_evaluate_ensemble(self, test_size=0.2, batch_size=32, n_estimators=5, epochs=20, use_saved_weights=False):
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42,
                                                            stratify=self.y)
        ensemble_models = self.train_ensemble(X_train, y_train, n_estimators=n_estimators, batch_size=batch_size, epochs=epochs,
                                              use_saved_weights=use_saved_weights)

        os.makedirs("ensemble_models", exist_ok=True)
        for idx, model in enumerate(ensemble_models):
            model.save(f"ensemble_models/model_{idx}.h5")

        y_pred, y_pred_auc = self.ensemble_predict(ensemble_models, X_test, digitize=False)

        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        print("Confusion Matrix:")
        score_confusion_matrix(y_test, y_pred)


def train_regression(labels, indicators, list_main_indicators, label, dropout_rate=0.3, base_folder='models_and_scalers', test_size=0.2, random_state = 42 ):
    """
    Function to train regression models sequentially for unique tickers in the dataset.

    Args:
        labels (pd.DataFrame): DataFrame containing target labels.
        indicators (pd.DataFrame): DataFrame containing features.
        list_main_indicators (list): List of feature column names.
        label (str): Name of the target variable.
        dropout_rate (float): Dropout rate for the model.
        base_folder (str): Folder to save models and scalers.
        test_size (float): Proportion of test data.
        random_state (int): Random state for reproducibility.
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print("TPU найден. Адрес:", tpu.master())
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
    except ValueError:
        print("TPU не найден. Используем стратегию по умолчанию (CPU/GPU).")
        strategy = tf.distribute.get_strategy()

    print("Используемая стратегия:", strategy)

    def create_complex_model(num_features, dropout_rate):
        model = keras.Sequential([
            keras.layers.Dense(256, activation='relu', input_shape=(num_features,)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(dropout_rate),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(1)
        ])
        return model

    unique_tickers = indicators['tic'].unique()
    os.makedirs(base_folder, exist_ok=True)

    previous_ticker_model_path = None

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    for idx, ticker in enumerate(unique_tickers):
        print(f"\n=== Обработка тикера: {ticker} ===")

        ticker_data = indicators[indicators['tic'] == ticker]
        ticker_labels = labels[labels['tic'] == ticker]

        X = ticker_data[list_main_indicators]
        y = ticker_labels[label]

        if len(X) == 0 or len(y) == 0:
            print(f"    Пропускаем (нет данных) для тикера {ticker}.")
            continue

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        scaler_path = f"{base_folder}/regression_scaler_{ticker}.joblib"
        joblib.dump(scaler, scaler_path)
        print(f"    Scaler сохранён в файл: {scaler_path}")

        num_features = X_scaled.shape[1]

        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=random_state
        )

        if len(X_train) < 10:
            print("    Недостаточно данных для обучения. Пропускаем.")
            continue

        with strategy.scope():
            if idx > 0 and previous_ticker_model_path is not None:
                print(f"    Загрузка полной модели из: {previous_ticker_model_path}")
                model = joblib.load(previous_ticker_model_path)
            else:
                model = create_complex_model(num_features, dropout_rate)
                model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        val_split = 0.2
        val_size = int(len(X_train) * val_split)

        X_val = X_train[:val_size]
        y_val = y_train[:val_size]
        X_train_part = X_train[val_size:]
        y_train_part = y_train[val_size:]

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train_part, y_train_part)).shuffle(buffer_size=len(X_train_part)).batch(32)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)
        test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(32)

        print("    Обучаем новую модель...")
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=50,
            verbose=1,
            callbacks=[early_stopping]
        )

        test_loss, test_mae = model.evaluate(test_dataset, verbose=0)
        print(f"    Test MSE = {test_loss:.4f}, Test MAE = {test_mae:.4f}")

        model_path = f"{base_folder}/regression_model_{ticker}.joblib"
        joblib.dump(model, model_path)
        print(f"    Полная модель сохранена в файл: {model_path}")

        previous_ticker_model_path = model_path

        if len(X_test) > 3:
            sample_for_prediction = X_test[:3]
            sample_dataset = tf.data.Dataset.from_tensor_slices(sample_for_prediction).batch(1)
            predictions = model.predict(sample_dataset)
            print("    Прогноз (первые 3 записи из теста):", predictions.reshape(-1))


def train_bagging(labels, indicators, list_main_indicators, label, base_folder='model bagging', test_size=0.2, random_state=42, n_estimators=20):

    # Создание базовой директории, если она не существует
    os.makedirs(base_folder, exist_ok=True)

    # Допустим, у нас есть список тикеров
    unique_tickers = indicators['tic'].unique()

    for ticker in unique_tickers:
        print(f"\nProcessing ticker: {ticker}")

        # Фильтрация данных по тикеру
        ticker_data = indicators[indicators['tic'] == ticker]
        ticker_labels = labels[labels['tic'] == ticker]

        # Разделение на признаки и метки
        X = ticker_data[list_main_indicators]
        y = ticker_labels[label]

        # Опционально: проверить баланс классов
        # print("Class distribution:\n", y.value_counts())

        # Получаем все классы и вычисляем веса
        all_classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=all_classes, y=y)
        class_weights_dict = {cls: weight for cls, weight in zip(all_classes, class_weights)}
        # Пример ручного исправления веса для класса 2 (если нужно)
        class_weights_dict[2] = 1

        print("Class weights:", class_weights_dict)

        # Разделение данных на train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )

        # МАСШТАБИРОВАНИЕ после разделения на train/test
        scaler = StandardScaler()
        # fit только на обучающей выборке
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns
        )
        # transform на тестовой выборке
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns
        )

        # Создание ансамбля Bagging + RandomForest
        bagging_classifier = BaggingClassifier(
            estimator=RandomForestClassifier(
                random_state=random_state,
                class_weight=class_weights_dict
            ),
            n_estimators=n_estimators,
            bootstrap=True,
            random_state=random_state
        )

        # Обучение модели
        bagging_classifier.fit(X_train_scaled, y_train)

        # Сохранение обученного скейлера и модели
        model_filename = os.path.join(base_folder, f"classifier_model_{ticker}.pkl")
        scaler_filename = os.path.join(base_folder, f"classifier_scaler_{ticker}.pkl")
        joblib.dump(bagging_classifier, model_filename, compress=3)
        joblib.dump(scaler, scaler_filename)
        print(f"Сохранены файлы: {model_filename}, {scaler_filename}")

        # Проверка модели на тестовой выборке
        y_pred = bagging_classifier.predict(X_test_scaled)

        # Оценка качества модели
        print(f"\nEvaluation for ticker {ticker}:")
        score_confusion_matrix(y_test, y_pred)



class StockPortfolioEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 df,
                 stock_dim,
                 hmax,
                 initial_amount,
                 transaction_cost_amount,
                 reward_scaling,
                 tech_indicator_list,
                 features_list,
                 turbulence_threshold=None,
                 lookback=5):
        """
        Initialize the environment with the given parameters.

        Parameters:
        - df: DataFrame containing market data.
        - stock_dim: Number of stocks in the portfolio.
        - hmax: Maximum shares per transaction.
        - initial_amount: Starting portfolio cash value.
        - transaction_cost_amount: Cost per share transaction.
        - reward_scaling: Scaling factor for rewards.
        - state_space: Dimensions of the state space.
        - action_space: Dimensions of the action space.
        - tech_indicator_list: List of technical indicators to include in the state.
        - turbulence_threshold: Optional threshold for turbulence, unused in this version.
        - lookback: Number of historical time steps for constructing the state.
        """
        self.min = 0  # Current time index
        self.lookback = lookback  # Number of previous steps for state construction
        self.df = df  # Market data
        self.stock_dim = stock_dim  # Number of stocks
        self.hmax = hmax  # Max shares per transaction
        self.initial_amount = initial_amount  # Starting portfolio cash value
        self.transaction_cost_amount = transaction_cost_amount  # Cost per share transaction
        self.reward_scaling = reward_scaling  # Scaling factor for reward
        self.state_space = (calculate_total_length_of_features(df, features_list) + len(
            tech_indicator_list)) * lookback  # Dimensions of state space
        self.action_space = spaces.Box(low=-1, high=1, shape=(stock_dim, 3))  # Long/Short/StopLoss/TakeProfit
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(stock_dim, self.state_space)
        )
        self.tech_indicator_list = tech_indicator_list  # List of technical indicators
        self.features_list = features_list  # List of features

        # Precompute timestamps and data mapping
        self.timestamps = df['timestamp'].sort_values().unique()
        self.data_map = {ts: df[df['timestamp'] == ts] for ts in self.timestamps}
        self.state = self._construct_state()
        self.terminal = False
        self.portfolio_value = self.initial_amount  # Portfolio value
        self.cash = self.initial_amount  # Cash is now
        self.share_holdings = np.zeros(self.stock_dim)  # Share holdings

        # Memory for tracking and logging
        self.asset_memory = [
            {
                'cash': self.initial_amount,
                'portfolio_value': self.initial_amount,
                'holdings': np.zeros(self.stock_dim).tolist()  # Convert array to list for compatibility
            }
        ]
        self.portfolio_return_memory = [0]
        self.actions_memory = [[[0]] * self.stock_dim]
        self.date_memory = [self.timestamps[0]]

    def calculate_total_length_of_features(df, features_list):
        """
        Calculate the total length of all data elements in the specified columns of the DataFrame.

        Parameters:
        - df: The input DataFrame containing the data.
        - features_list: List of column names to calculate lengths for.

        Returns:
        - total_length: The total length of all data elements in the specified columns.
        """
        total_length = 0

        for feature in features_list:
            if feature in df.columns:
                total_length += df[feature].apply(
                    lambda x: len(x) if isinstance(x, (list, np.ndarray, str)) else 1).sum()

        return total_length


    def get_transaction_cost(self, amount, current_price):
        """
        Calculate the transaction cost for a deal.

        Parameters:
        - amount: The number of shares to buy or sell.
        - current_price: The current price of the stock.

        Returns:
        - transaction_cost: The calculated transaction cost.
        """
        value_deal = abs(amount) * current_price
        base_cost = value_deal * self.transaction_cost_amount['base']
        minimal_cost = self.transaction_cost_amount['minimal']
        maximal_cost = self.transaction_cost_amount['maximal'] * value_deal

        # Apply cost boundaries
        transaction_cost = max(minimal_cost, min(base_cost, maximal_cost))

        return transaction_cost

    def _construct_state(self):
        """
        Construct the current state with historical data.
        """
        start_index = max(0, self.min - self.lookback + 1)
        historical_data = self.df.iloc[start_index:self.min + 1]

        # Collect features
        state_data = []
        for feature in self.features_list:
            feature_values = historical_data[feature].values[-self.lookback:] if len(
                historical_data) >= self.lookback else np.zeros(self.lookback)
            state_data.append(feature_values)

        # Collect technical indicators
        for tech in self.tech_indicator_list:
            tech_values = historical_data[tech].values[-self.lookback:] if len(
                historical_data) >= self.lookback else np.zeros(self.lookback)
            state_data.append(tech_values)

        return np.concatenate(state_data)

    def _sell_stock(self, stock_index, amount, current_price):
        """
        Sell stock and handle cash and holdings, accounting for long and short positions.

        Parameters:
        - stock_index: Index of the stock to sell.
        - amount: Amount of stock to sell (positive for selling long, negative for short).
        - current_price: Current price of the stock.
        """
        if amount == 0:
            return

        # Calculate transaction cost
        transaction_cost = self.get_transaction_cost(amount, current_price)

        # Update portfolio and cash
        sell_value = amount * current_price
        self.cash += sell_value - transaction_cost
        self.portfolio_value -= transaction_cost
        self.share_holdings[stock_index] -= amount

    def _sell_all_stocks(self):
        """
        Sell all long and short positions.
        """
        for i, holding in enumerate(self.share_holdings):
            current_price = self.data_map[self.timestamps[self.min]]['close'].values[i]
            self._sell_stock(i, holding, current_price)

    def step(self, actions):
        """
        Execute one step in the environment.

        Parameters:
        - actions: Array of actions (weights, stop_loss, take_profit) for each stock.

        Returns:
        - state: Updated state after the step.
        - reward: Reward for the step.
        - terminal: Boolean indicating if the episode is finished.
        - info: Additional information (currently empty).
        """
        self.terminal = self.min >= len(self.timestamps) - 1
        last_minute_of_day = self.min < len(self.timestamps) - 1 and self.timestamps[self.min].date() != \
                             self.timestamps[self.min + 1].date()

        if self.terminal:
            self._sell_all_stocks()
            df = pd.DataFrame(self.portfolio_return_memory, columns=['return'])
            plt.plot(df['return'].cumsum())
            plt.savefig('cumulative_reward.png')
            plt.close()

            return self.state, self.reward, self.terminal, {}

        # Normalize weights for non-terminal, non-last-minute steps
        new_weights = np.zeros_like(actions[:, 0]) if last_minute_of_day else self.softmax_normalization(actions[:, 0])
        stop_loss = actions[:, 1]
        take_profit = actions[:, 2]
        weight_diff = new_weights - np.array(self.actions_memory[-1][0])

        for i, diff in enumerate(weight_diff):
            current_price = self.data_map[self.timestamps[self.min]]['close'].values[i]
            self._sell_stock(i, int(diff * self.portfolio_value / current_price), current_price)

        self.min += 1
        self.state = self._construct_state()

        portfolio_return, updated_weights = self.calculate_portfolio_return(stop_loss, take_profit)
        self.actions_memory.append(
            np.vstack((updated_weights, stop_loss, take_profit)))  # Update weights in action memory
        self.portfolio_return_memory.append(portfolio_return)
        self.asset_memory.append(
            {'cash': self.cash, 'portfolio_value': self.portfolio_value, 'holdings': self.share_holdings.copy()})

        self.reward = self.portfolio_value * self.reward_scaling
        return self.state, self.reward, self.terminal, {}

    def calculate_portfolio_return(self, stop_loss, take_profit):
        """
        Calculate returns for the portfolio, including stop-loss and take-profit handling.
        """
        updated_weights = np.zeros_like(self.share_holdings)
        returns = []

        for i, holding in enumerate(self.share_holdings):
            low = self.data_map[self.timestamps[self.min]]['low'].values[i]
            high = self.data_map[self.timestamps[self.min]]['high'].values[i]
            close_price = self.data_map[self.timestamps[self.min]]['close'].values[i]
            open_price = self.data_map[self.timestamps[self.min]]['open'].values[i]
            last_close = self.data_map[self.timestamps[self.min - 1]]['close'].values[i]

            stop_loss_price = last_close * (1 - stop_loss[i])
            take_profit_price = last_close * (1 + take_profit[i])

            # Handle stop-loss and take-profit for long and short positions
            if low <= stop_loss_price and holding > 0:  # Long stop-loss
                current_return = (stop_loss_price - last_close) * holding
                transaction_cost = self.get_transaction_cost(holding, stop_loss_price)
                current_return -= transaction_cost
                self.cash += stop_loss_price * holding - transaction_cost
                self.share_holdings[i] = 0
            elif high >= stop_loss_price and holding < 0:  # Short stop-loss
                current_return = (stop_loss_price - last_close) * holding
                transaction_cost = self.get_transaction_cost(holding, stop_loss_price)
                current_return -= transaction_cost
                self.cash += stop_loss_price * holding - transaction_cost
                self.share_holdings[i] = 0
            elif high >= take_profit_price and holding > 0:  # Long take-profit
                current_return = (take_profit_price - last_close) * holding
                transaction_cost = self.get_transaction_cost(holding, take_profit_price)
                current_return -= transaction_cost
                self.cash += take_profit_price * holding - - transaction_cost
                self.share_holdings[i] = 0
            elif low <= take_profit_price and holding < 0:  # Short take-profit
                current_return = (take_profit_price - last_close) * holding
                transaction_cost = self.get_transaction_cost(holding, take_profit_price)
                current_return -= transaction_cost
                self.cash += take_profit_price * holding - transaction_cost
                self.share_holdings[i] = 0
            else:  # Regular price change
                current_return = (close_price - last_close) * holding

            # Append return
            returns.append(current_return)

        # Calculate portfolio return
        portfolio_return = sum(returns) / self.portfolio_value
        self.portfolio_value += sum(returns)

        # Update portfolio weights based on new holdings
        for i, holding in enumerate(self.share_holdings):
            updated_weights[i] = (holding * self.data_map[self.timestamps[self.min]]['close'].values[
                i]) / self.portfolio_value if self.portfolio_value > 0 else 0

        return portfolio_return, updated_weights

    def softmax_normalization(self, actions):
        """
        Normalize actions to valid weights where the sum of absolute weights equals 1,
        and each weight does not exceed self.risk_volume.

        Parameters:
        - actions: Array of raw action weights.

        Returns:
        - normalized_weights: Array of weights balanced and capped by self.risk_volume.
        """
        abs_sum = np.sum(np.abs(actions))

        # Handle the edge case where all actions are zero
        if abs_sum == 0:
            return np.zeros_like(actions)

        # Normalize actions to make the sum of absolute values equal 1
        normalized_actions = actions / abs_sum

        # Cap each normalized action to self.risk_volume
        capped_actions = np.clip(normalized_actions, -self.risk_volume, self.risk_volume)

        # Re-normalize to ensure the sum of absolute weights equals 1 after capping
        abs_capped_sum = np.sum(np.abs(capped_actions))
        if abs_capped_sum == 0:
            return np.zeros_like(actions)  # Handle the case where all weights are capped to zero
        final_weights = capped_actions / abs_capped_sum

        return final_weights

    def reset(self):
        """
        Reset the environment to its initial state.
        """
        self.min = 0
        self.state = self._construct_state()
        self.portfolio_value = self.initial_amount  # Reset portfolio
        self.cash = self.initial_amount  # Reset cash
        self.share_holdings = np.zeros(self.stock_dim)  # Reset share holdings
        self.asset_memory = [
            {
                'cash': self.initial_amount,
                'portfolio_value': self.initial_amount,
                'holdings': np.zeros(self.stock_dim).tolist()  # Convert array to list for compatibility
            }
        ]
        self.portfolio_return_memory = [0]
        self.actions_memory = [[[0]] * self.stock_dim]
        self.date_memory = [self.timestamps[0]]
        return self.state

    def save_asset_memory(self):
        """
        Save portfolio values over time.
        """
        date_list = self.date_memory
        portfolio_return = self.portfolio_return_memory
        df_account_value = pd.DataFrame({'date': date_list, 'minutely_return': portfolio_return})
        return df_account_value

    def save_action_memory(self):
        """
        Save actions taken over time.
        """
        date_list = self.date_memory
        df_date = pd.DataFrame(date_list)
        df_date.columns = ['timestamp']
        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data_map[self.timestamps[self.min]]['tic'].values
        df_actions.index = df_date.timestamp
        return df_actions

    def render(self, mode='human'):
        """Render the current environment state."""
        return self.state

    def _seed(self, seed=None):
        """Set random seed for reproducibility."""
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_sb_env(self):
        """Get the stable-baselines environment."""
        e = DummyVecEnv([lambda: self])
        obs = e.reset()
        return e, obs