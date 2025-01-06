import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
import datetime
from os.path import exists
import joblib as joblib
import json

import warnings
warnings.filterwarnings('ignore')

from finrl import config
from finrl import config_tickers
from finrl.meta.data_processors.processor_yahoofinance import YahooFinanceProcessor
from finrl.meta.preprocessor.yahoodownloader import YahooDownloader
from finrl.meta.preprocessor.preprocessors import FeatureEngineer, data_split
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from finrl.agents.stablebaselines3.models import DRLAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline, convert_daily_return_to_pyfolio_ts
from finrl.main import check_and_make_directories
from pprint import pprint
from stable_baselines3.common.logger import configure
import sys

# Import MlFinLab tools
from mllab.util.volatility import get_daily_vol
from mllab.filters.filters import cusum_filter
from mllab.labeling import labeling
from mllab.data_structures.preprocess_data import FinancePreprocessor, add_takeprofit_stoploss_volume
from mllab.labeling.trend_scanning import trend_scanning_labels
from sklearn.ensemble import RandomForestClassifier
from mllab.ensemble.sb_bagging import SequentiallyBootstrappedBaggingClassifier
from mllab.cross_validation import score_confusion_matrix
from mllab.microstructural_features.feature_generator import calculate_indicators, get_correlation
from mllab.ensemble.model_train import ensemble_models, train_regression, train_bagging, StockPortfolioEnv


# sys.path.append("../FinRL")

import pickle

import itertools

from finrl.config import (
    DATA_SAVE_DIR,
    TRAINED_MODEL_DIR,
    TENSORBOARD_LOG_DIR,
    RESULTS_DIR,
    INDICATORS,
    TRAIN_START_DATE,
    TRAIN_END_DATE,
    TEST_START_DATE,
    TEST_END_DATE,
    TRADE_START_DATE,
    TRADE_END_DATE,
)

from finrl.config_tickers import SP_500_TICKER, DOW_30_TICKER

from google.colab import drive
import os

# Подключение Google Диска
drive.mount('/content/drive')

# Directory setup
check_and_make_directories([DATA_SAVE_DIR, TRAINED_MODEL_DIR, TENSORBOARD_LOG_DIR, RESULTS_DIR])


# File paths and tickers
file_path = 'DOW30_1m'
list_tickers = ['AAPL', 'TSLA', 'NVDA']

coeff_tp = 1
coeff_sl = 1

download = True

# Initialize processor
processor = FinancePreprocessor('polygon', list_tickers, "1Min", file_path)

stock_dimension = 3
risk_volume = 0.2

FEATURE_LENGTHS = {
    'prediction': 6,   # массив из 6 элементов
    'covariance': 3   # массив из 3 элементов
    # и т.д. для остальных feature...
}

# 'base':0.0035 стоимость за акцию, 'minimal':0.35 минимальная комиссия в сделке, 'maximal':0.01 максимальный процент 1% комиссии, 'short_loan': 0.17 ставка займа в short 0.17% в день,
# 'short_rebate': 0.83 выплата rebate по коротким позициям 0.83% в день, 'short_rebate_limit':100000 лимит начиная с которого выплачивается rebate
transaction_cost_amount = {
    'base': 0.0035,
    'minimal': 0.35,
    'maximal': 0.01,
    'short_loan': 0.17,
    'short_rebate': 0.83,
    'short_rebate_limit': 100000
}
slippage = 0.02


env_kwargs = {
    "stock_dim": stock_dimension,
    "hmax": 100,
    'risk_volume': risk_volume,
    "initial_amount": 1000000,
    "transaction_cost_amount": transaction_cost_amount,
    "tech_indicator_list": [],
    'features_list': features_list,
    'FEATURE_LENGTHS' : FEATURE_LENGTHS
}

# Load or generate final prepared dataset
if True:
    train = processor.read_csv('train_final.csv')
    train.reset_index(inplace=True)

# Extract unique timestamps and tickers
timestamps = train['date'].unique()
tickers = train['tic'].unique()

# Generate repeated action data for each ticker at each timestamp
actions_data = []
for timestamp in timestamps:
    for tic in tickers:
        actions_data.append({
            'date': timestamp,
            'tic': tic,
            'weight': np.random.uniform(-1, 1),  # Random portfolio weight (allows shorts)
            'stop_loss': np.random.uniform(0.01, 0.1),  # Random stop-loss percentage
            'take_profit': np.random.uniform(0.1, 0.2)  # Random take-profit percentage
        })

# Create a DataFrame from the list of dictionaries
df_actions = pd.DataFrame(actions_data)


e_train = StockPortfolioEnv(df = train, **env_kwargs)

for i in range(10):

    # Filter actions for the i timestamp
    second_timestamp = df_actions['date'].sort_values().unique()[i]  # Get the second timestamp
    second_actions = df_actions[df_actions['date'] == second_timestamp]

    # Prepare the actions as a NumPy array for the environment
    actions = second_actions[['weight', 'stop_loss', 'take_profit']].to_numpy()
    print(actions)

    # Execute the second step
    state, reward, terminal, info = e_train.step(actions)

    # Print the results
    print("State after second step:")
    #print(state)
    #print("\nReward after second step:", reward)
    print("\ncash status:", e_train.cash)
    print("\nvalue status:", e_train.portfolio_value)
    print("\nHoldings info:", e_train.share_holdings)
    print("\nreturne rate:", e_train.portfolio_return_memory[-1])








