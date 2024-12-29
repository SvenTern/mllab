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


class ensemble_models():
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
