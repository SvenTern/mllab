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
        self.class_weights_dict = {cls: weight for cls, weight in zip(np.unique(y), class_weights)}
        self.class_weights_dict = {cls: max(1.0, weight) for cls, weight in class_weights_dict.items()}

    # Custom Data Generator
    class DataGenerator(Sequence):
        def __init__(self, X, y, batch_size, augment=False):
            self.X = np.array(X)  # Ensure numpy array
            self.y = np.array(y)  # Ensure numpy array
            self.batch_size = batch_size
            self.augment = augment

        def __len__(self):
            return int(np.ceil(len(self.X) / self.batch_size))

        def __getitem__(self, index):
            start_idx = index * self.batch_size
            end_idx = (index + 1) * self.batch_size
            batch_X = self.X[start_idx:end_idx]
            batch_y = self.y[start_idx:end_idx]

            if self.augment:
                batch_X = self._augment_data(batch_X)

            return batch_X, batch_y

        def _augment_data(self, batch_X):
            # Example augmentation: adding Gaussian noise
            noise = np.random.normal(0, 0.01, batch_X.shape)
            return batch_X + noise

    # Function to load all models in the ensemble
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

    # Improved Dense Model Creation
    def create_dense_model(self, input_dim):
        model = Sequential([
            Dense(128, activation='relu', input_shape=(input_dim,), kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01)),
            BatchNormalization(),
            Dropout(0.5),
            Dense(1, activation='tanh')  # Regression or binary-like output
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model

    # LSTM Model Creation
    def create_lstm_model(self, input_dim):
        model = Sequential([
            LSTM(64, input_shape=(input_dim, 1), return_sequences=True, activation='tanh',
                 recurrent_activation='sigmoid'),
            Dropout(0.5),
            LSTM(64, activation='tanh', recurrent_activation='sigmoid'),
            Dropout(0.5),
            Dense(1, activation='tanh')  # Regression or binary-like output
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
        return model

    # Train Ensemble of Models
    def train_ensemble(self, X_train, y_train, n_estimators=5, batch_size=32, epochs=20, use_saved_weights=False):
        ensemble_models = []
        for i in tqdm(range(n_estimators), desc="Training Ensemble Models", unit="model"):
            # Bootstrap resampling
            X_resampled, y_resampled = resample(np.array(X_train), np.array(y_train))

            # Calculate sample weights
            sample_weights = np.array([class_weights_dict[label] for label in y_resampled])

            # Randomly select model type
            model_type = random.choice(['dense', 'lstm'])
            if model_type == 'dense':
                model = create_dense_model(X_train.shape[1])
            elif model_type == 'lstm':
                X_resampled = X_resampled.reshape(X_resampled.shape[0], X_resampled.shape[1], 1)  # Reshape for LSTM
                model = create_lstm_model(X_resampled.shape[1])
            else:
                raise ValueError("Unsupported model_type. Use 'dense' or 'lstm'.")

            if use_saved_weights:
                try:
                    model.load_weights(f"ensemble_weights/model_{model_type}_weights.h5")
                    print(f"Loaded weights for model {model_type}")
                except FileNotFoundError:
                    print(f"Weights not found for model {model_type}, training from scratch.")

            print(f"Training model {i + 1}/{n_estimators} as {model_type}")

            # Train the model
            early_stopping = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
            model.fit(X_resampled, y_resampled, sample_weight=sample_weights, batch_size=batch_size, epochs=epochs,
                      verbose=1,
                      callbacks=[early_stopping])
            ensemble_models.append(model)

            if use_saved_weights:
                try:
                    os.makedirs("ensemble_weights", exist_ok=True)
                    model.save_weights(f"ensemble_weights/model_{model_type}_weights.h5")
                    print(f"Saveed weights for model {model_type}")
                except FileNotFoundError:
                    print(f"Didnt save for model {model_type}, training from scratch.")

        return ensemble_models

    # Ensemble Prediction
    def ensemble_predict(self, ensemble_models, X_test, digitize=True):
        X_test = np.array(X_test)  # Ensure input is numpy array
        predictions = np.zeros((X_test.shape[0], len(ensemble_models)))
        for i, model in enumerate(ensemble_models):
            if isinstance(model.layers[0], LSTM):
                X_test_reshaped = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)  # Reshape for LSTM
                predictions[:, i] = model.predict(X_test_reshaped).flatten()
            else:
                predictions[:, i] = model.predict(X_test).flatten()

        # Average predictions
        averaged_predictions = np.mean(predictions, axis=1)

        # Map predictions to classes [-1, 0, 1]
        mapped_predictions = np.digitize(averaged_predictions, bins=[-0.33, 0.33]) - 1
        if digitize:
            return mapped_predictions
        else:
            return mapped_predictions, averaged_predictions

    # Train and Evaluate Model on Train/Test Split
    def train_and_evaluate_ensemble(self, test_size=0.2, batch_size=32, n_estimators=5, epochs=20,
                                    use_saved_weights=False):
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42,
                                                            stratify=self.y)

        # Train the ensemble
        ensemble_models = self.train_ensemble(X_train, y_train, n_estimators=n_estimators, batch_size=batch_size,
                                              epochs=epochs, use_saved_weights=use_saved_weights)

        # Save the ensemble models
        os.makedirs("ensemble_models", exist_ok=True)
        for idx, model in enumerate(ensemble_models):
            model.save(f"ensemble_models/model_{idx}.h5")

        # Evaluate the ensemble
        y_pred, y_pred_auc = self.ensemble_predict(ensemble_models, X_test, digitize=False)

        # Print evaluation metrics
        print("Classification Report:")
        print(classification_report(y_test, y_pred))

        # Print confusion matrix
        print("Confusion Matrix:")
        score_confusion_matrix(y_test, y_pred, y_pred_auc)

        return ensemble_models
