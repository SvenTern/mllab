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
from tqdm.auto import tqdm
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

from concurrent.futures import ThreadPoolExecutor
from mllab.microstructural_features.feature_generator import get_correlation
from scipy.stats import pearsonr, spearmanr
import ast


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


def train_regression(labels, indicators, list_main_indicators, label, previous_ticker_model_path = None, dropout_rate=0.3, test_size=0.2, random_state = 42 ):
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

    previous_ticker_model_path = None

    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    )
    total_score = []

    for idx, ticker in enumerate(unique_tickers):
        print(f"\n=== Обработка тикера: {ticker} ===")

        ticker_data = indicators[indicators['tic'] == ticker]
        ticker_labels = labels[labels['tic'] == ticker]

        X = ticker_data[list_main_indicators]
        y = np.array(ticker_labels[label])

        if len(X) == 0 or len(y) == 0:
            print(f"    Пропускаем (нет данных) для тикера {ticker}.")
            continue

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        scaler = StandardScaler()
        X_train_scaled = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns
        )
        # transform на тестовой выборке
        X_test_scaled = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns
        )

        num_features = X_train_scaled.shape[1]

        if len(X_train_scaled) < 10:
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
        val_size = int(len(X_train_scaled) * val_split)

        X_val = X_train_scaled[:val_size]
        y_val = y_train[:val_size]
        X_train_part = X_train_scaled[val_size:]
        y_train_part = y_train[val_size:]

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train_part, y_train_part)).shuffle(buffer_size=len(X_train_part)).batch(32)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(32)


        test_dataset = tf.data.Dataset.from_tensor_slices((X_test_scaled, y_test)).batch(32)

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
        total_score.append((f'Test MSE {ticker}', test_loss))
        total_score.append((f'Test MAE {ticker}', test_mae))

        #sample_for_prediction = X_test
        #sample_dataset = tf.data.Dataset.from_tensor_slices(sample_for_prediction).batch(1)
        predictions = model.predict(X_test_scaled).flatten()
        #print('predictions',predictions, len(predictions))
        #print('y_test',y_test, len(y_test))

        # 1. Коэффициент Пирсона (Pearson)
        pearson_corr, pearson_pval = pearsonr(y_test, predictions)
        print(f"Pearson correlation: {pearson_corr:.4f}")
        #print(f"Pearson p-value: {pearson_pval:.6f}")

        # 2. Коэффициент Спирмена (Spearman)
        #spearman_corr, spearman_pval = spearmanr(y_test, predictions)
        #print(f"Spearman correlation: {spearman_corr:.4f}")
        #print(f"Spearman p-value: {spearman_pval:.6f}")

        total_score.append((f'Pearson correlation {ticker}', pearson_corr))

    return model, total_score, scaler


def train_bagging(labels, indicators, list_main_indicators, label, test_size=0.2, random_state=42, n_estimators=20):

     # Допустим, у нас есть список тикеров
    unique_tickers = indicators['tic'].unique()

    total_score = []
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

        # Проверка модели на тестовой выборке
        y_pred = bagging_classifier.predict(X_test_scaled)

        # Оценка качества модели
        print(f"\nEvaluation bagging for ticker {ticker}:")
        score = score_confusion_matrix(y_test, y_pred)
        total_score.append((f'{ticker} accuaracy', score))

    return bagging_classifier, total_score, scaler

def update_indicators(labels, indicators, type='bagging', short_period:int = 1):
    # Extract list of tickers
    list_tickers = indicators['tic'].unique()

    # Load models
    basefolder = '/content/drive/My Drive/DataTrading/'
    folder_bagging = 'model bagging/'
    folder_regression = 'model regression/'
    models = {}

    for tic in list_tickers:
        try:
            if type == 'bagging':
                models[f'classifier_model_{tic}'] = joblib.load(basefolder + folder_bagging + f'classifier_model_{tic}_{short_period}.pkl')
                models[f'classifier_scaler_{tic}'] = joblib.load(basefolder + folder_bagging + f'classifier_scaler_{tic}_{short_period}.pkl')
                models[f'classifier_indicators_{tic}'] = joblib.load(
                    basefolder + folder_bagging + f'classifier_indicators_{tic}_{short_period}.lst')
            elif type == 'regression':
                models[f'regression_model_{tic}'] = joblib.load(basefolder + folder_regression + f'regression_model_{tic}_{short_period}.joblib')
                models[f'regression_scaler_{tic}'] = joblib.load(basefolder + folder_regression + f'regression_scaler_{tic}_{short_period}.joblib')
                models[f'classifier_indicators_{tic}'] = joblib.load(
                    basefolder + folder_regression + f'classifier_indicators_{tic}_{short_period}.lst')
        except Exception as e:
            print(f"Error loading model or scaler for ticker {tic}: {e}")
            continue


    # Prepare data for prediction
    data_for_prediction = indicators

    def process_ticker(tic):
        try:
            # Filter data for the current ticker
            filtered_data = data_for_prediction[data_for_prediction['tic'] == tic]
            #print('filtered_data', filtered_data)

            if type == 'bagging':
                # Scale data for prediction
                scale_data_classifire = models[f'classifier_scaler_{tic}'].transform(
                    filtered_data[models[f'classifier_indicators_{tic}']])
                #print('scale_data_classifire', scale_data_classifire)

                # Generate predictions
                predicted_classifire = models[f'classifier_model_{tic}'].predict_proba(scale_data_classifire)
                #print('predicted_classifire', predicted_classifire)

                # Combine predictions into DataFrame
                return pd.DataFrame({
                    'timestamp': filtered_data.index,
                    'tic': tic,
                    'bin-1': [predicted_classifire[i, 0] for i in range(len(filtered_data.index))],
                    'bin-0': [predicted_classifire[i, 1] for i in range(len(filtered_data.index))],
                    'bin+1': [predicted_classifire[i, 2] for i in range(len(filtered_data.index))]
                })
            elif type == 'regression':
                # Scale data for prediction
                scale_data_regression = models[f'regression_scaler_{tic}'].transform(
                    filtered_data[models[f'classifier_indicators_{tic}']])

                # Generate predictions
                predicted_regression = models[f'regression_model_{tic}'].predict(scale_data_regression).flatten()

                # Combine predictions into DataFrame
                return pd.DataFrame({
                    'timestamp': filtered_data.index,
                    'tic': tic,
                    'regression': [predicted_regression[i] for i in range(len(filtered_data.index))]
                })
        except Exception as e:
            print(f"Error processing ticker {tic}: {e}")
            return pd.DataFrame()

    # Parallel processing of tickers
    try:
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(process_ticker, list_tickers))

        #print('results', results)
        # Concatenate results
        predicted_data = pd.concat(results, ignore_index=True)

        if type == 'bagging':
            # Удалить колонки bin+1, bin-0, bin-1, если они есть в indicators
            columns_to_remove = ['bin+1', 'bin-0', 'bin-1']
            indicators = indicators.drop(
                columns=[col for col in columns_to_remove if col in indicators.columns]
            )
        elif type == 'regression':
            # Удалить колонку 'regression', если она есть
            if 'regression' in indicators.columns:
                indicators = indicators.drop('regression', axis=1)

        # Merge predicted data back into indicators DataFrame
        indicators = indicators.reset_index()
        indicators = indicators.merge(predicted_data, on=['timestamp', 'tic'], how='left')
        #print('predicted_data, shape',predicted_data, predicted_data.shape)
        indicators = indicators.set_index('timestamp')
        #print('indicators, shape',indicators, indicators.shape)

        # нужно перезаписать индикаторы на диск

        return indicators

    except Exception as e:
        print(f"Error during parallel processing: {e}")
        return indicators


class StockPortfolioEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self,
                 df,
                 stock_dim,
                 hmax,
                 risk_volume,
                 initial_amount,
                 transaction_cost_amount,
                 tech_indicator_list,
                 features_list,
                 FEATURE_LENGTHS,
                 turbulence_threshold=None,
                 reward_scaling=100,
                 lookback=5,
                 initial=True,
                 previous_state=[],
                 minimal_cash = 0.1,
                 use_sltp = False,
                 use_logging = 1,
                 sl_scale = 1.001,
                 tp_scale = 1
                 ):
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
        self.FEATURE_LENGTHS = FEATURE_LENGTHS
        self.minimal_cash = minimal_cash * initial_amount
        self.use_sltp = use_sltp
        # play mode для тестирования label, predictions
        self.play_mode = False

        # контроль максимального провала
        self.drowdown = 0

        self.ticker_list = df['tic'].unique().tolist()

        self.annual_risk_free_rate = 0.0385
        self.trading_days_per_year = 252
        self.minutes_per_day = 390
        self.minutes_per_year = self.trading_days_per_year * self.minutes_per_day
        self.risk_free_rate_per_min = (1 + self.annual_risk_free_rate) ** (1 / (self.minutes_per_year)) - 1

        if self.use_sltp:
            self.sl_scale = sl_scale
            self.tp_scale = tp_scale
        else:
            self.sl_scale = sl_scale
            self.tp_scale = tp_scale

        self.min = 0  # Current time index
        self.episode =0 # подсчет количества полных циклов через reset
        self.lookback = lookback  # Number of previous steps for state construction
        self.df = df  # Market data
        self.stock_dim = stock_dim  # Number of stocks
        self.hmax = hmax  # Max shares per transaction
        self.initial_amount = initial_amount  # Starting portfolio cash value
        self.transaction_cost_amount = transaction_cost_amount  # Cost per share transaction
        self.reward_scaling = reward_scaling  # Scaling factor for reward
        self.risk_volume = risk_volume  # Risk volume for portfolio
        # state_dim = (self.calculate_total_length_of_features(df, features_list) + len(tech_indicator_list)) * lookback
        self.initial = initial
        self.previous_state = previous_state

        self.tech_indicator_list = tech_indicator_list  # List of technical indicators
        self.features_list = features_list  # List of features

        # Precompute timestamps and data mapping
        self.dates = np.sort(df['date'].unique())

        # Преобразуем DataFrame в формат numpy для более быстрой обработки на TPU
        self.df_numpy = df.to_numpy()
        self.columns = df.columns.tolist()  # Сохраняем порядок колонок для доступа к данным

        # Группируем данные по дате
        self.grouped_data = df.groupby('date')

        # self.data_map = {ts: df[df['date'] == ts] for ts in self.dates}
        self.terminal = False
        self.portfolio_value = self.initial_amount  # Portfolio value
        self.cash = self.initial_amount  # Cash is now
        self.share_holdings = np.zeros(self.stock_dim)  # Share holdings
        # initalize state
        self.state = self._initiate_state()
        self.state_space = len(self.state)  # Dimensions of state space
        self.action_space = spaces.Box(low=-1, high=1, shape=(stock_dim, 3)) if self.use_sltp else spaces.Box(low=-1, high=1, shape=(stock_dim, ))  # Long/Short/StopLoss/TakeProfit
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.state_space,)
        )
        # Memory for tracking and logging
        self.asset_memory = [{'cash': self.cash, 'portfolio_value': self.portfolio_value, 'holdings': self.share_holdings.copy()}]
        self.portfolio_return_memory = [0]
        self.actions_memory = [[[0]] * self.stock_dim]
        self.date_memory = [self.dates[0]]
        self.data = self.get_data_by_date()

        self.use_logging = use_logging
        self.logging_data = []

    def logging(self, text, value = None):

        if self.use_logging == 0:
            return
        #время
        if value is not None and value > 0:
            self.logging_data.append(f'{self.min}: {text} : {value:,.0f} : cash : {self.cash:,.0f} : value : {self.portfolio_value:,.0f}')
        else:
            self.logging_data.append(
                f'{self.min}: {text} : cash :{self.cash:,.0f} : value :{self.portfolio_value:,.0f}')

    def convert_absolute_to_relative_returns(self,df, portfolio_value_column='portfolio_value', return_column='return'):
        """
        Преобразует абсолютные доходности в относительные.

        :param df: pandas.DataFrame с колонкой portfolio_value и абсолютными доходностями.
        :param portfolio_value_column: Имя столбца с абсолютными значениями портфеля.
        :param return_column: Имя столбца с абсолютными доходностями.
        :return: pandas.DataFrame с добавленным столбцом relative_return.
        """
        if portfolio_value_column not in df.columns:
            raise ValueError(f"Столбец '{portfolio_value_column}' не найден в DataFrame.")
        if return_column not in df.columns:
            raise ValueError(f"Столбец '{return_column}' не найден в DataFrame.")

        # Вычисление относительных доходностей
        df['relative_return'] = df[return_column] / df[portfolio_value_column].shift(1)

        # Удаление строк с NaN, возникающих из-за сдвига
        df = df.dropna(subset=['relative_return'])

        return df

    def calculate_annual_sharpe_ratio(self, df, return_column='return'):
        """
        Вычисляет годовой коэффициент Шарпа на основе доходностей, автоматически определяя
        количество периодов в торговом дне и продолжительность периода в календарных днях.

        :param df: pandas.DataFrame с индексом datetime и столбцом с доходностями.
        :param return_column: Имя столбца в df, содержащего доходности. По умолчанию 'return'.
        :param risk_free_rate: Годовая безрисковая ставка (например, 0.02 для 2%). По умолчанию 0.0.
        :return: Годовой коэффициент Шарпа.
        """

        # Проверка наличия столбца
        if return_column not in df.columns:
            raise ValueError(f"Столбец '{return_column}' не найден в DataFrame.")

        df['date'] = pd.to_datetime(df['date'], utc=True)

        ## Установка колонки 'date' в качестве индекса
        df.set_index('date', inplace=True)
        df.index.names = ['timestamp']

        # Убедимся, что индекс типа datetime
        #if not pd.api.types.is_datetime64_any_dtype(df.index):
        #    raise TypeError("Индекс DataFrame должен быть типа datetime.")

        # Удаление пропущенных значений
        returns = df[return_column].dropna()

        # Проверка достаточности данных
        if returns.empty:
            raise ValueError("Нет доступных данных для вычисления коэффициента Шарпа.")

        # Извлечение даты и времени из индекса
        df_clean = returns.to_frame()
        df_clean['date'] = df_clean.index.date
        df_clean['time'] = df_clean.index.time


        # Определение количества периодов в день
        periods_per_day = df_clean.groupby('date').size().mode()
        if periods_per_day.empty:
            raise ValueError("Не удалось определить количество периодов в день из данных.")
        periods_per_day = periods_per_day.iloc[0]

        # Определение количества календарных дней
        start_date = df_clean['date'].min()
        end_date = df_clean['date'].max()
        total_days = (end_date - start_date).days + 1  # +1 для включения конечного дня

        # Определение количества торговых дней (уникальные даты с данными)
        trading_days = df_clean['date'].nunique()

        # Определение продолжительности периода в годах
        period_years = total_days / 365.25  # Используем среднее количество дней в году с учетом високосных

        #print(
        #    f"Определено {periods_per_day} периодов в торговом дне, {trading_days} торговых дней и период длится {period_years:.2f} года(лет).")

        # Расчет средней доходности и стандартного отклонения
        mean_return = returns.mean()
        std_return = returns.std()

        # Масштабирование до годовых значений
        annual_mean_return = mean_return * periods_per_day * trading_days / period_years
        annual_std_return = std_return * np.sqrt(periods_per_day * trading_days) / np.sqrt(period_years)

        # Корректировка безрисковой ставки
        annual_excess_return = annual_mean_return - self.annual_risk_free_rate

        # Вычисление коэффициента Шарпа
        sharpe_ratio = annual_excess_return / annual_std_return

        return sharpe_ratio

    def sharpe_ratio_minutely(
            self

    ) -> float:
        """
        Рассчитывает годовой Sharpe Ratio на основе МИНУТНЫХ доходностей портфеля.

        Параметры
        ---------
        self.portfolio_return_memory : np.ndarray
            Последовательность минутных доходностей (например, r_t = (P(t) - P(t-1)) / P(t-1)).
        self.annual_risk_free_rate : float
            Годовая безрисковая ставка в долях (по умолчанию 2% = 0.02).

        Возвращает
        ---------
        float
            Годовой Sharpe Ratio. Если стандартное отклонение 0 (нет волатильности),
            функция вернёт 0.0.
        """

        # Преобразуем входные данные в numpy-массив
        returns = np.asarray(self.portfolio_return_memory, dtype=float)
        if len(returns) < 2:
            raise ValueError("Массив минутных доходностей должен содержать как минимум 2 значения.")

        # Средняя доходность за 1 минуту
        mean_return_per_min = np.mean(returns)

        # Стандартное отклонение минутных доходностей
        std_return_per_min = np.std(returns, ddof=1)

        # Защита от деления на ноль
        if std_return_per_min == 0:
            return 0.0

        # Sharpe Ratio за минуту
        sharpe_per_min = (mean_return_per_min - self.risk_free_rate_per_min) / std_return_per_min

        # Годовой Sharpe Ratio
        sharpe_annual = sharpe_per_min * np.sqrt(self.minutes_per_year)

        return sharpe_annual

    def get_data_by_date(self, min = None):
        """
        Получить данные за конкретную дату.
        """
        if min is None:
            date = self.dates[self.min]
        else:
            date = self.dates[min]
        if date in self.grouped_data.groups:
            return self.grouped_data.get_group(date)
        else:
            return None

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
        base_cost = abs(amount) * self.transaction_cost_amount['base']
        minimal_cost = self.transaction_cost_amount['minimal']
        maximal_cost = self.transaction_cost_amount['maximal'] * value_deal

        # Apply cost boundaries
        transaction_cost = min(maximal_cost, max(base_cost, minimal_cost))

        return transaction_cost

    def parse_to_1d_array(self, value):
        """
        Преобразует строку/число/список в numpy-массив float.
        Если на входе строка вида '[0.049, 0.8265, 0.1245]',
        то с помощью ast.literal_eval превращаем её в список float.
        Если одиночное число - делаем массив из одного элемента.
        """
        if isinstance(value, str):
            # Парсим строку как Python-объект (список и т.д.)
            value = ast.literal_eval(value)  # теперь value может стать list/float
        if isinstance(value, list) or isinstance(value, np.ndarray):
            return np.array(value, dtype=np.float32)
        else:
            # Если это одиночное число (float/int), сделаем массив из одного элемента
            return np.array([value], dtype=np.float32)

    def _update_state(self, weights):
        """
        Сбор всех чисел из столбца feature (по текущему tic_data) в один большой массив
        длины (lookback * desired_len), с «дополнением» или «обрезанием» при необходимости.
        """
        if self.play_mode:
            return self.state

        # Инициализируем state_data как NumPy-массив
        state_data = np.empty(0, dtype=np.float32)

        # 1) Добавим портфельную стоимость
        state_data = np.concatenate([
            state_data,
            np.array([self.portfolio_value], dtype=np.float32)
        ])

        # 2) Добавим веса
        weights = weights.astype(np.float32)
        state_data = np.concatenate([state_data, weights])

        # 3) Добавим кэш
        state_data = np.concatenate([
            state_data,
            np.array([self.cash], dtype=np.float32)
        ])

        # 4) Срез исторических данных
        start_index = max(0, self.min - self.lookback + 1)
        historical_data = self.df.iloc[start_index:self.min + 1]

        # 5) Цикл по тикерам
        for tic in self.ticker_list:
            tic_data = historical_data[historical_data['tic'] == tic]

            # --- (A) Обработка рыночных признаков ---
            for feature in self.features_list:
                # desired_len: сколько элементов мы *ожидаем* для данного feature
                desired_len = self.FEATURE_LENGTHS.get(feature, 1)

                if feature in tic_data.columns:
                    # Собираем все строки в один «сплошной» массив
                    parsed_arrays = [
                        self.parse_to_1d_array(val)
                        for val in tic_data[feature].values
                    ]
                    if len(parsed_arrays) > 0:
                        flattened = np.concatenate(parsed_arrays, axis=0)
                        n = len(flattened)
                        # Полная целевая длина
                        full_len = self.lookback * desired_len

                        if n >= full_len:
                            # Если данных больше или равно, берём *последние* full_len
                            feature_values = flattened[-full_len:]
                        else:
                            # Дозаполняем нулями в начале
                            feature_values = np.zeros(full_len, dtype=np.float32)
                            feature_values[-n:] = flattened
                    else:
                        # Нет строк (parsed_arrays пуст)
                        full_len = self.lookback * desired_len
                        feature_values = np.zeros(full_len, dtype=np.float32)
                else:
                    # Нет такого столбца
                    full_len = self.lookback * desired_len
                    feature_values = np.zeros(full_len, dtype=np.float32)

                # Конкатенируем в основной state_data
                state_data = np.concatenate([state_data, feature_values])

            # --- (B) Аналогично - для технических индикаторов ---
            for tech in self.tech_indicator_list:
                if tech in tic_data.columns:
                    parsed_arrays = [
                        self.parse_to_1d_array(val)
                        for val in tic_data[tech].values
                    ]
                    if len(parsed_arrays) > 0:
                        flattened = np.concatenate(parsed_arrays, axis=0)
                        n = len(flattened)

                        if n >= self.lookback:
                            tech_values = flattened[-self.lookback:]
                        else:
                            tech_values = np.zeros(self.lookback, dtype=np.float32)
                            tech_values[-n:] = flattened
                    else:
                        tech_values = np.zeros(self.lookback, dtype=np.float32)
                else:
                    tech_values = np.zeros(self.lookback, dtype=np.float32)

                state_data = np.concatenate([state_data, tech_values])

        return state_data

    def _adjust_sell_amount(self, amount, current_price):
        """
        Корректируем 'amount' так, чтобы после сделки self.cash не опустился ниже self.minimal_cash,
        учитывая, что transaction_cost мал, и мы считаем его один раз.
        """
        if amount == 0:
            return 0

        # 1. Посчитаем комиссию один раз
        transaction_cost = self.get_transaction_cost(amount, current_price)

        # 2. Посчитаем, каким станет cash после сделки с исходным amount
        potential_new_cash = self.cash + amount * current_price - transaction_cost

        # 3. Если уже удовлетворяем условие (не уходим ниже minimal_cash) — корректировка не нужна
        if potential_new_cash >= self.minimal_cash:
            return amount

        # 4. "значение кэша ушло в минус уже до этого, поэтому мы ничего не делаем,
        #    текущая сделка только увеличивает количество кэша"
        if amount * current_price - transaction_cost > 0:
            return amount

        # 5. Ниже идёт логика для покрытия короткой позиции (amount < 0).
        max_cover = int((self.cash - self.minimal_cash - transaction_cost) / current_price)
        x = -amount

        if max_cover < 0:
            # max_cover < 0 означает, что даже покупка 0 акций (вообще не покрывать шорт) уже
            # не позволяет остаться выше minimal_cash. Возвращаем 0.
            return 0

        # Если x (сколько хотели купить) больше, чем max_cover, то уменьшаем до max_cover
        if x > max_cover:
            return -max_cover
        else:
            return amount

    def _sell_stock(self, stock_index, amount, current_price):
        if amount == 0:
            return


        # Узнаём, сколько в итоге можем/хотим продать (или покрыть) без нарушения minimal_cash
        adjusted_amount = self._adjust_sell_amount(amount, current_price)

        if adjusted_amount == 0:
            # Если стало 0 — значит, сделку не совершаем.
            return

        # Пересчитываем комиссию и итоговое поступление/списание
        transaction_cost = self.get_transaction_cost(adjusted_amount, current_price)
        sell_value = adjusted_amount * current_price

        self.cash += sell_value - transaction_cost
        #self.logging('cash from # Sell stock', sell_value - transaction_cost)

        self.portfolio_value -= transaction_cost
        self.share_holdings[stock_index] -= adjusted_amount
        #self.logging(f'holding {stock_index} amount {adjusted_amount:,.0f} price: {current_price:,.3f} cost {transaction_cost:,.0f} from # Sell stock', sell_value)


    def _sell_all_stocks(self):
        """
        Sell all long and short positions.
        """
        for i, holding in enumerate(self.share_holdings):
            current_price = self.data['close'].values[i]
            self._sell_stock(i, holding, current_price)

    def get_grow_value(self):

        # Предполагается, что self.asset_memory — это список объектов с атрибутом portfolio_value
        # и self.dates — это список объектов datetime

        # Извлекаем значения portfolio_value и даты
        portfolio_values = [item['portfolio_value'] for item in self.asset_memory]
        dates = self.dates

        # Создаем DataFrame
        df = pd.DataFrame({
            'date': dates,
            'portfolio_value': portfolio_values
        })

        # Убедимся, что данные отсортированы по дате
        df = df.sort_values('date').reset_index(drop=True)

        # Рассчитываем прирост стоимости портфеля (разницу с предыдущим значением)
        df['return'] = df['portfolio_value'].diff()

        # Убираем первую строку, где прирост будет NaN
        df = df.dropna()

        return df

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
        self.terminal = self.min >= len(self.dates) - 1

        if self.terminal:
            self._sell_all_stocks()

            df = self.get_grow_value()
            plt.plot(df['return'].cumsum())
            plt.savefig('cumulative_reward.png')
            plt.close()
            #if self.episode % self.print_verbosity == 0:
            #print(f"day: {self.date_memory[-1]}, episode: {self.episode}")
            begin_total_asset = self.asset_memory[0]['portfolio_value']
            end_total_asset = self.asset_memory[-1]['portfolio_value']
            total_reward = 100 * (end_total_asset - begin_total_asset) / begin_total_asset
            total_drowdown = 0 if self.drowdown > 0 else 100 * (self.drowdown) / begin_total_asset
            #print(f"begin_total_asset: {begin_total_asset:0.2f}")
            #print(f"end_total_asset: {end_total_asset:0.2f}")
            #print(f"total_reward: {total_reward:0.2f}")
            #print(f'maximal drowdown : {total_drowdown:0.2f}')
            #print(f"total_cost: {self.cost:0.2f}")
            #print(f"total_trades: {self.trades}")
            total_sharp_ratio =  self.calculate_annual_sharpe_ratio(self.convert_absolute_to_relative_returns(df))
            #print(f"Annual Sharpe: {total_sharp_ratio:0.3f}")
            #print("=================================")

            return self.state, self.reward, self.terminal, {'total_sharp_ratio': total_sharp_ratio, 'total_reward':total_reward, 'total_drowdown': total_drowdown}

        last_minute_of_day = (
                self.dates[self.min].floor('D') != self.dates[self.min + 1].floor('D')
        )

        if last_minute_of_day:
            self._sell_all_stocks()

        else:


            # Normalize weights for non-terminal, non-last-minute steps
            if self.use_sltp:
                new_weights = np.zeros_like(actions[:, 0]) if last_minute_of_day else self.softmax_normalization(actions[:, 0])
                weight_diff = new_weights - np.array(self.actions_memory[-1][0])
            else:
                new_weights = np.zeros_like(actions) if last_minute_of_day else self.softmax_normalization(actions)
                weight_diff = new_weights - np.array(self.actions_memory[-1][0])


            for i, diff in enumerate(weight_diff):
                # входим в сделку только если есть какие то значения sl tp
                current_price = self.data['close'].values[i]
                self._sell_stock(i, -int(diff * self.portfolio_value / current_price), current_price)

        stop_loss, take_profit = self.get_sltp(actions)

        #print('stop_loss', stop_loss)
        #print('take_profit', take_profit)
        #print('new_weights', new_weights)
        #print('weight_diff', weight_diff)

        #print('share_holdings', self.share_holdings)
        #print('portfolio_value', self.portfolio_value)

        self.min += 1
        self.data = self.get_data_by_date()

        portfolio_return, updated_weights = self.calculate_portfolio_return(stop_loss, take_profit)
        #print('portfolio_return, updated_weights' ,portfolio_return, updated_weights)

        self.actions_memory.append(
            np.vstack((updated_weights, stop_loss, take_profit)))  # Update weights in action memory
        self.portfolio_return_memory.append(portfolio_return)
        self.asset_memory.append(
            {'cash': self.cash, 'portfolio_value': self.portfolio_value, 'holdings': self.share_holdings.copy()})

        self.drowdown = min(self.drowdown, self.portfolio_value - self.initial_amount)

        self.date_memory.append(self.dates[self.min])

        self.state = self._update_state(updated_weights)

        self.reward = self.sharpe_ratio_minutely() * self.reward_scaling
        return self.state, self.reward, self.terminal, {}

    def get_sltp_volatility(self, volatility, holdings):

        if volatility == 0:
            volatility = 0.001

        if holdings >= 0:
            return -abs(volatility) * self.sl_scale, abs(volatility) * self.tp_scale
        else:
            return abs(volatility) * self.sl_scale, - abs(volatility) * self.tp_scale

    def get_sltp(self, actions):
        if self.use_sltp:

            stop_loss = actions[:, 1]
            take_profit = actions[:, 2]

            for idx, tic in enumerate(self.ticker_list):
                sl_value = stop_loss[idx]
                tp_value = take_profit[idx]

                # Фильтрация данных для текущего тикера
                tic_data = self.data[self.data['tic'] == tic]

                if tic_data.empty:
                    # Обработка случая, когда данные для тикера отсутствуют
                    #parsed_prediction = [0] * 6  # Предполагаем минимум 6 элементов
                    volatility = 0.001  # Можно задать нулевую волатильность или другое значение по умолчанию
                else:
                    # Парсинг предсказанных значений и извлечение волатильности
                    #parsed_prediction = self.parse_to_1d_array(tic_data['prediction'].values[0])
                    volatility = tic_data['volatility'].values[0]

                # Проверка и корректировка значений stop_loss и take_profit на основе волатильности
                if self.share_holdings[idx] > 0:
                    if (1 + tp_value) <= (1 + sl_value):  # нужно чтобы take_profit был выше stop_loss
                        sl_value, tp_value = self.get_sltp_volatility(volatility, self.share_holdings[idx])
                elif self.share_holdings[idx] < 0:
                    if (1 + tp_value) >= (1 + sl_value):  # нужно чтобы take_profit был ниже stop_loss
                        sl_value, tp_value = self.get_sltp_volatility(volatility, self.share_holdings[idx])

                # Заполнение массивов рассчитанными значениями для текущего индекса
                stop_loss[idx] = sl_value
                take_profit[idx] = tp_value

            return stop_loss* self.sl_scale, take_profit * self.tp_scale

        else:
            # Инициализация numpy массивов для stop_loss и take_profit
            n = len(self.ticker_list)
            stop_loss = np.empty(n)
            take_profit = np.empty(n)

            for idx, tic in enumerate(self.ticker_list):
                # Фильтрация данных для текущего тикера
                tic_data = self.data[self.data['tic'] == tic]

                if tic_data.empty:
                    # Обработка случая, когда данные для тикера отсутствуют
                    parsed_prediction = [0] * 6  # Предполагаем минимум 6 элементов
                    volatility = 0.001  # Можно задать нулевую волатильность или другое значение по умолчанию
                else:
                    # Парсинг предсказанных значений и извлечение волатильности
                    parsed_prediction = self.parse_to_1d_array(tic_data['prediction'].values[0])
                    volatility = tic_data['volatility'].values[0]

                # Извлечение исходных значений stop_loss и take_profit для текущего тикера
                sl_value = parsed_prediction[5] if len(parsed_prediction) > 5 else 0
                tp_value = parsed_prediction[6] if len(parsed_prediction) > 6 else 0

                # Проверка и корректировка значений stop_loss и take_profit на основе волатильности
                if self.share_holdings[idx] > 0:
                    if (1 + tp_value) <= (1 + sl_value): #нужно чтобы take_profit был выше stop_loss
                        sl_value, tp_value = self.get_sltp_volatility(volatility, self.share_holdings[idx])
                elif self.share_holdings[idx] < 0:
                    if (1 + tp_value) >= (1 + sl_value):  #нужно чтобы take_profit был ниже stop_loss
                        sl_value, tp_value = self.get_sltp_volatility(volatility, self.share_holdings[idx])

                # Заполнение массивов рассчитанными значениями для текущего индекса
                stop_loss[idx] = sl_value
                take_profit[idx] = tp_value

            return stop_loss, take_profit

    def calculate_portfolio_return(self, stop_loss, take_profit):
        """
        Calculate returns for the portfolio, including stop-loss and take-profit handling.
        """
        updated_weights = np.zeros_like(self.share_holdings)
        returns = []

        for i, holding in enumerate(self.share_holdings):
            low = self.data['low'].values[i]
            high = self.data['high'].values[i]
            close_price = self.data['close'].values[i]
            open_price = self.data['open'].values[i]
            previous_data = self.get_data_by_date(self.min - 1)
            if previous_data is None:
                last_close = open_price
            else:
                last_close = previous_data['close'].values[i]

            # вот здесь сложный вопрос, так как важно понимать какого знака должен быть SL, TP
            # если stop_loss[i] < 0 то считаем, что это short сделка
            # stop_loss_price - в это случае будет больше, т.е. мы страхуемся от роста ...
            stop_loss_price = last_close * (1 + stop_loss[i])
            take_profit_price = last_close * (1 + take_profit[i])

            self.logging(
                f'price {i} last_close: {last_close}, open :{open_price}, low ;{low}, high :{high}, close :{close_price}, stop_loss_price :{stop_loss_price:,.3f}, take_profit_price :{take_profit_price:,.3f}',
                0)

            # Handle stop-loss and take-profit for long and short positions
            if low <= stop_loss_price and holding > 0:  # Long stop-loss

                if stop_loss_price >= take_profit_price:
                    raise ValueError(f"# Long stop-loss stop_loss_price {stop_loss_price:,.3f} не должно быть больше или равно take_profit_price {take_profit_price:,.3f}")

                current_return = (stop_loss_price - last_close) * holding
                transaction_cost = self.get_transaction_cost(holding, stop_loss_price)
                current_return -= transaction_cost
                self.cash += stop_loss_price * holding - transaction_cost
                #self.logging('cash from # Long stop-loss',  stop_loss_price * holding - transaction_cost)
                self.logging(
                    f'price {i} last_close: {last_close}, open :{open_price}, low ;{low}, high :{high}, close :{close_price}, stop_loss_price :{stop_loss_price:,.3f}, take_profit_price :{take_profit_price:,.3f}',
                    0)
                self.logging(f'Потери Long stop-loss {current_return:,.2f} holding {i} amount {holding} price {stop_loss_price:,.3f} cost {transaction_cost:,.2f} from # Long stop-loss', - stop_loss_price * holding)

                self.share_holdings[i] = 0
            elif high >= stop_loss_price and holding < 0:  # Short stop-loss

                if stop_loss_price <= take_profit_price:  #нужно чтобы take_profit был ниже stop_loss
                    raise ValueError(f"# Short stop-loss stop_loss_price {stop_loss_price:,.3f} не должно быть меньше или равно take_profit_price {take_profit_price:,.3f}")

                current_return = (stop_loss_price - last_close) * holding
                transaction_cost = self.get_transaction_cost(holding, stop_loss_price)
                current_return -= transaction_cost
                self.cash += stop_loss_price * holding - transaction_cost
                #self.logging('cash from # Short stop-loss', stop_loss_price * holding - transaction_cost)
                #self.logging(f'Потери Short stop-loss {current_return:,.2f} holding {i} amount {holding} price {stop_loss_price:,.3f} cost {transaction_cost:,.2f} from # Short stop-loss', - stop_loss_price * holding)
                self.share_holdings[i] = 0
            elif high >= take_profit_price and holding > 0:  # Long take-profit

                if stop_loss_price >= take_profit_price:
                    raise ValueError(f"# Long take-profit stop_loss_price {stop_loss_price:,.3f} не должно быть больше или равно take_profit_price {take_profit_price:,.3f}")

                current_return = (take_profit_price - last_close) * holding
                transaction_cost = self.get_transaction_cost(holding, take_profit_price)
                current_return -= transaction_cost
                self.cash += take_profit_price * holding - transaction_cost
                #self.logging('cash from # Long take-profit', take_profit_price * holding - transaction_cost)
                #self.logging(f'Заработок Long take-profit {current_return:,.2f} holding {i} amount {holding} price {take_profit_price:,.3f} cost {transaction_cost:,.2f} from # Long take-profit', - take_profit_price * holding)
                self.share_holdings[i] = 0
            elif low <= take_profit_price and holding < 0:  # Short take-profit

                if stop_loss_price <= take_profit_price:
                    raise ValueError(f"# Short take-profit stop_loss_price {stop_loss_price:,.3f} не должно быть меньше или равно take_profit_price {take_profit_price:,.3f}")

                current_return = (take_profit_price - last_close) * holding
                transaction_cost = self.get_transaction_cost(holding, take_profit_price)
                current_return -= transaction_cost
                self.cash += take_profit_price * holding - transaction_cost
                #self.logging('cash from # Short take-profit', take_profit_price * holding - transaction_cost)
                #self.logging(f'Заработок Short take-profit {current_return:,.2f} holding {i} amount {holding} price {take_profit_price:,.3f} cost {transaction_cost:,.2f} from # Short take-profit', - take_profit_price * holding)
                self.share_holdings[i] = 0
            else:  # Regular price change
                current_return = (close_price - last_close) * holding

            # Append return
            returns.append(current_return)

        # дополнительный профит от кэша
        current_return = self.cash * self.risk_free_rate_per_min
        self.cash += current_return
        #self.logging(f'cash from # risk free {current_return:.2f}', 0)
        returns.append(current_return)

        # Calculate portfolio return
        portfolio_return = sum(returns) / self.portfolio_value
        self.portfolio_value += sum(returns)

        # Update portfolio weights based on new holdings
        for i, holding in enumerate(self.share_holdings):
            updated_weights[i] = (holding * self.data['close'].values[
                i]) / self.portfolio_value if self.portfolio_value > 0 else 0

        return portfolio_return, updated_weights

    # структура state
    # текущий размер портфеля
    # объемы портфеля по акциям и в кэше
    # все future list вместе с loopback
    def _initiate_state(self):
        if self.initial:
            # For Initial State
            # for multiple stock
            state_data = [
                np.array([self.initial_amount]),  # Преобразовать в массив
                np.zeros(self.stock_dim),  # Массив нулей для акций
                np.array([self.initial_amount]),  # Преобразовать в массив
            ]

            # Вычисляем стартовый индекс для lookback
            start_index = 0
            historical_data = self.df.iloc[start_index:self.min + 1]

            # Обрабатываем данные по каждому тикеру
            for tic in self.ticker_list:
                tic_data = historical_data[historical_data['tic'] == tic]

                # Добавляем рыночные признаки
                for feature in self.features_list:
                    desired_len = self.FEATURE_LENGTHS.get(feature, 1)
                    if feature in tic_data.columns:
                        feature_values = np.zeros(self.lookback*desired_len)
                    else:
                        feature_values = np.zeros(self.lookback*desired_len)
                    state_data.append(feature_values)

                # Добавляем технические индикаторы
                for tech in self.tech_indicator_list:
                    if tech in tic_data.columns:
                        tech_values = np.zeros(self.lookback)
                    else:
                        tech_values = np.zeros(self.lookback)
                    state_data.append(tech_values)
            #print('state_data', state_data)
            # Объединяем все данные в единый вектор
            state = np.concatenate(state_data)

        else:
            # Using Previous State
            # for multiple stock
            state_data = (
                    [self.previous_state[0]]
                    + self.previous_state[
                      (self.stock_dim + 1): (self.stock_dim * 2 + 2)
                      ]
                    + sum(
                [
                    self.df[feature].values if np.issubdtype(self.df[feature].dtype, np.number) else self.df[
                        feature].tolist()
                    for feature in self.features_list
                ],
            )
                    + sum(
                [
                    self.data[tech].values.tolist()
                    for tech in self.tech_indicator_list
                ],
            )
            )

            # Вычисляем стартовый индекс для lookback
            start_index = 0
            historical_data = self.df.iloc[start_index:1]

            # Обрабатываем данные по каждому тикеру
            for tic in self.ticker_list:
                tic_data = historical_data[historical_data['tic'] == tic]

                # Добавляем рыночные признаки
                for feature in self.features_list:
                    if feature in tic_data.columns:
                        feature_values = tic_data[feature].values[-self.lookback:] if len(
                            tic_data) >= self.lookback else np.zeros(self.lookback)
                    else:
                        feature_values = np.zeros(self.lookback)
                    state_data.append(feature_values)

                # Добавляем технические индикаторы
                for tech in self.tech_indicator_list:
                    if tech in tic_data.columns:
                        tech_values = tic_data[tech].values[-self.lookback:] if len(
                            tic_data) >= self.lookback else np.zeros(self.lookback)
                    else:
                        tech_values = np.zeros(self.lookback)
                    state_data.append(tech_values)

            # Объединяем все данные в единый вектор
            return np.concatenate(state_data)

        return state

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
        if abs_sum == 0:
            return np.zeros_like(actions)

        # First, scale so sum of abs is 1 (or self.risk_volume).
        normalized = actions / abs_sum

        # Then clamp each weight in magnitude by self.risk_volume.
        clamped = np.clip(normalized, -self.risk_volume, self.risk_volume)

        # Note: once you clamp, the sum of absolute values might be below 1.
        # If you must preserve sum=1 while also clamping, you'd need more complex logic.
        return clamped

    def reset(self):
        """
        Reset the environment to its initial state.
        """
        self.min = 0
        self.data = self.get_data_by_date()
        self.portfolio_value = self.initial_amount  # Reset portfolio
        self.cash = self.initial_amount  # Reset cash
        self.share_holdings = np.zeros(self.stock_dim)  # Reset share holdings
        self.state = self._initiate_state()
        self.asset_memory = [{'cash': self.cash, 'portfolio_value': self.portfolio_value, 'holdings': self.share_holdings.copy()}]
        self.portfolio_return_memory = [0]
        self.actions_memory = [[[0]] * self.stock_dim]
        self.date_memory = [self.dates[0]]
        self.episode += 1
        self.logging_data = []
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
        df_date.columns = ['date']
        action_list = self.actions_memory
        df_actions = pd.DataFrame(action_list)
        df_actions.columns = self.data_map[self.dates[self.min]]['tic'].values
        df_actions.index = df_date.date
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

    # проскальзывание
    def simulate_stop_loss_slippage_dynamic(entry_price, stop_loss_price, low_price, max_slippage_percent=5):
        """
        Моделирование проскальзывания при срабатывании стоп-лосса с динамическим диапазоном проскальзывания.

        :param entry_price: float, цена открытия позиции.
        :param stop_loss_price: float, заданная цена стоп-лосса.
        :param low_price: float, минимальная цена свечи (low).
        :param max_slippage_percent: float, максимальный процент дополнительного проскальзывания.
        :return: float, итоговая цена закрытия позиции с учетом проскальзывания.
        """
        if low_price > stop_loss_price:
            # Если low выше стоп-лосса, проскальзывание отсутствует
            return stop_loss_price

        # Разница между стоп-лоссом и уровнем low
        distance_to_low = stop_loss_price - low_price

        # Динамический диапазон проскальзывания (в зависимости от расстояния до low)
        dynamic_slippage_percent = max_slippage_percent * (distance_to_low / stop_loss_price)
        dynamic_slippage_range = (-dynamic_slippage_percent, dynamic_slippage_percent)

        # Генерация случайного дополнительного проскальзывания
        additional_slippage_percent = random.uniform(dynamic_slippage_range[0], dynamic_slippage_range[1])
        additional_slippage = low_price * additional_slippage_percent / 100

        # Итоговая цена с учетом low и дополнительного проскальзывания
        final_price = low_price + additional_slippage

        return final_price

    import numpy as np
    import pandas as pd

    def __get_predictions__(self, type: str = 'prediction'):
        if type == 'prediction':
            # Формируем массив прогнозов из столбца 'prediction'
            predictions_array = np.stack(self.df['prediction'].values)

            # Используем первые три столбца для сравнения
            comparison_array = predictions_array[:, :3]
            max_indices = np.argmax(comparison_array, axis=1)

            # Извлекаем необходимые столбцы из массива
            tp = predictions_array[:, 4]
            sl = predictions_array[:, 5]

            # Вычисляем значения столбца 'bin' на основе max_indices
            col3 = predictions_array[:, 3]
            bin_minus1 = -col3
            bin_0 = np.zeros_like(col3)
            bin_plus1 = col3
            chosen_values = np.choose(max_indices, [bin_minus1, bin_0, bin_plus1])

            # Записываем результаты в DataFrame
            self.df['sl'] = sl
            self.df['tp'] = tp
            self.df['bin'] = chosen_values

        # Общие операции для обоих типов
        self.df.sort_values(['date', 'tic'], inplace=True)
        grouped = self.df.groupby('date')

        # Инициализация прогресс-бара для этапа группировки
        grouped_data = {}
        # Используем tqdm для отображения прогресса по группам
        for date, group in tqdm(grouped, desc="Подготовка массива actions: ", total=len(grouped), leave=False, position=2, dynamic_ncols=True):
            matrix = group[['bin', 'sl', 'tp']].to_numpy()
            grouped_data[date] = matrix

        return grouped_data

    def __run__(self, type: str = 'prediction'):

        # включение play_mode для более быстрого проведения игры
        self.play_mode = True

        grouped_data = self.__get_predictions__(type=type)
        dates = sorted(grouped_data.keys())
        results = None

        # Используем tqdm для создания итератора с прогресс-баром
        pbar = tqdm(dates, desc="Игра по датам :", leave=False, position=2, dynamic_ncols=True)
        try:
            for current_date in pbar:
                # ваши действия
                actions = grouped_data[current_date]
                results = self.step(actions)

                if self.terminal:
                    # прерываем цикл
                    break
        finally:
            # Закрываем прогресс-бар, если нужно
            pbar.n = pbar.total
            pbar.refresh()
            pbar.close()

        return results




def __check__(self):
        # Проверка наличия атрибута predictions; при необходимости вызов метода для получения предсказаний
        if not hasattr(self, 'predictions') or self.predictions is None:
            self.predictions = self.__get_predictions__()

        # Предварительная сортировка DataFrame по тикеру и дате для оптимизации последующих операций
        df_sorted = self.df.sort_values(by=['tic', 'date']).reset_index(drop=True)
        # Группировка по тикерам для вычислений
        grouped = df_sorted.groupby('tic')

        # Векторизированный расчёт процентного изменения цены закрытия:
        # Вычисление изменения от предыдущего периода и до следующего периода для каждой строки
        df_sorted['change_prev_%'] = grouped['close'].pct_change() * 100
        df_sorted['change_next_%'] = grouped['close'].pct_change(periods=-1) * 100

        # Определение направления изменений цен для предыдущего и следующего периодов
        df_sorted['actual_direction_prev'] = np.where(
            df_sorted['change_prev_%'] > 0, 'growth',
            np.where(df_sorted['change_prev_%'] < 0, 'fall', 'neutral')
        )
        df_sorted['actual_direction_next'] = np.where(
            df_sorted['change_next_%'] > 0, 'growth',
            np.where(df_sorted['change_next_%'] < 0, 'fall', 'neutral')
        )

        # Получаем список уникальных тикеров (уже отсортированный)
        tickers = df_sorted['tic'].unique()

        # Подготовка структуры для хранения количества верных/неверных предсказаний по каждому тикеру
        results_by_ticker = {
            tic: {
                'correct_prev': 0,
                'incorrect_prev': 0,
                'correct_next': 0,
                'incorrect_next': 0
            } for tic in tickers
        }

        # Создаём словарь предсказаний для быстрого доступа: ключ - (date, tic), значение - predicted_direction
        prediction_dict = {}
        # Предполагаем, что self.predictions – словарь: ключами являются даты, значениями – массивы предсказаний
        for date, prediction_array in self.predictions.items():
            # Для каждого тикера извлекаем предсказанное направление
            for idx, tic in enumerate(tickers):
                prediction_value = prediction_array[idx, 0]
                # Если предсказание не нейтральное, сохраняем направление
                if prediction_value != 0:
                    direction = 'growth' if prediction_value > 0 else 'fall'
                    prediction_dict[(date, tic)] = direction

        # Оценка предсказаний: сравнение предсказанного направления с фактическим для каждого случая
        for (date, tic), predicted_direction in prediction_dict.items():
            # Находим строку DataFrame по тикеру и дате
            mask = (df_sorted['tic'] == tic) & (df_sorted['date'] == date)
            row = df_sorted.loc[mask]
            if row.empty:
                continue

            # Извлекаем фактические направления для предыдущего и следующего периодов
            actual_prev = row['actual_direction_prev'].values[0]
            actual_next = row['actual_direction_next'].values[0]

            # Обновляем счётчики верных/неверных предсказаний для каждого тикера
            if actual_prev == predicted_direction:
                results_by_ticker[tic]['correct_prev'] += 1
            else:
                results_by_ticker[tic]['incorrect_prev'] += 1

            if actual_next == predicted_direction:
                results_by_ticker[tic]['correct_next'] += 1
            else:
                results_by_ticker[tic]['incorrect_next'] += 1

        # Вывод текстовых результатов по каждому тикеру
        for tic, res in results_by_ticker.items():
            print(f"\nТикер: {tic}")
            print(f"По change_prev - Верных: {res['correct_prev']}, Неверных: {res['incorrect_prev']}")
            print(f"По change_next - Верных: {res['correct_next']}, Неверных: {res['incorrect_next']}")

        # Подготовка данных для визуализации по change_prev
        correct_prev_counts = [results_by_ticker[tic]['correct_prev'] for tic in tickers]
        incorrect_prev_counts = [results_by_ticker[tic]['incorrect_prev'] for tic in tickers]

        # Визуализация результатов по change_prev
        x = np.arange(len(tickers))
        width = 0.35
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width / 2, correct_prev_counts, width, label='Верные (prev)')
        rects2 = ax.bar(x + width / 2, incorrect_prev_counts, width, label='Неверные (prev)')

        ax.set_ylabel('Количество предсказаний')
        ax.set_title('Оценка предсказаний по change_prev')
        ax.set_xticks(x)
        ax.set_xticklabels(tickers, rotation=45)
        ax.legend()

        def autolabel(rects):
            """Добавление числовых подписей над столбцами для наглядности."""
            for rect in rects:
                height = rect.get_height()
                ax.annotate(f'{height}', xy=(rect.get_x() + rect.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom')

        autolabel(rects1)
        autolabel(rects2)
        fig.tight_layout()
        plt.show()

        # Подготовка данных для визуализации по change_next
        correct_next_counts = [results_by_ticker[tic]['correct_next'] for tic in tickers]
        incorrect_next_counts = [results_by_ticker[tic]['incorrect_next'] for tic in tickers]

        # Визуализация результатов по change_next
        fig, ax = plt.subplots(figsize=(10, 6))
        rects1 = ax.bar(x - width / 2, correct_next_counts, width, label='Верные (next)')
        rects2 = ax.bar(x + width / 2, incorrect_next_counts, width, label='Неверные (next)')

        ax.set_ylabel('Количество предсказаний')
        ax.set_title('Оценка предсказаний по change_next')
        ax.set_xticks(x)
        ax.set_xticklabels(tickers, rotation=45)
        ax.legend()

        autolabel(rects1)
        autolabel(rects2)
        fig.tight_layout()
        plt.show()

        return results_by_ticker









