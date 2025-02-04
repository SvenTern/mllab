# pylint: disable=missing-module-docstring
import numpy as np
import scipy.stats as ss
from scipy import linalg
from mllab.ensemble.model_train import StockPortfolioEnv
from mllab.data_structures.preprocess_data import add_takeprofit_stoploss_volume
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # иногда нужен явный импорт
from tqdm.auto import tqdm
from os.path import exists
from os import makedirs
import os as os


class CampbellBacktesting:
    """
    This class implements the Haircut Sharpe Ratios and Profit Hurdles algorithms described in the following paper:
    `Campbell R. Harvey and Yan Liu, Backtesting, (Fall 2015). Journal of Portfolio Management,
    2015 <https://papers.ssrn.com/abstract_id=2345489>`_; The code is based on the code provided by the authors of the paper.

    The Haircut Sharpe Ratios algorithm lets the user adjust the observed Sharpe Ratios to take multiple testing into account
    and calculate the corresponding haircuts. The haircut is the percentage difference between the original Sharpe ratio
    and the new Sharpe ratio.

    The Profit Hurdle algorithm lets the user calculate the required mean return for a strategy at a given level of
    significance, taking multiple testing into account.
    """

    def __init__(self, simulations=2000):
        """
        Set the desired number of simulations to make in Haircut Sharpe Ratios or Profit Hurdle algorithms.

        :param simulations: (int) Number of simulations
        """

        self.simulations = simulations

    @staticmethod
    def _sample_random_multest(rho, n_trails, prob_zero_mean, lambd, n_simulations, annual_vol=0.15, n_obs=240):
        """
        Generates empirical p-value distributions.

        The algorithm is described in the paper and is based on the model estimated by `Harvey, C.R., Y. Liu,
        and H. Zhu., … and the Cross-section of Expected Returns. Review of Financial Studies, forthcoming 2015`,
        referred to as the HLZ model.

        It provides a set of simulated t-statistics based on the parameters recieved from the _parameter_calculation
        method.

        Researchers propose a structural model to capture trading strategies’ underlying distribution.
        With probability p0 (prob_zero_mean), a strategy has a mean return of zero and therefore comes
        from the null distribution. With probability 1 – p0, a strategy has a nonzero mean and therefore
        comes from the alternative distribution - exponential.

        :param rho: (float) Average correlation among returns
        :param n_trails: (int) Total number of trials inside a simulation
        :param prob_zero_mean: (float) Probability for a random factor to have a zero mean
        :param lambd: (float) Average of monthly mean returns for true strategies
        :param n_simulations: (int) Number of rows (simulations)
        :param annual_vol: (float) HLZ assume that the innovations in returns follow a normal distribution with a mean
                                   of zero and a standard deviation of ma = 15%
        :param n_obs: (int) Number of observations of used for volatility estimation from HLZ
        :return: (np.ndarray) Array with distributions calculated
        """

        pass

    @staticmethod
    def _parameter_calculation(rho):
        """
        Estimates the parameters used to generate the distributions in _sample_random_multest - the HLZ model.

        Based on the work of HLZ, the pairwise correlation of returns is used to estimate the probability (prob_zero_mean),
        total number of trials (n_simulations) and (lambd) - parameter of the exponential distribution. Levels and
        parameters taken from the HLZ research.

        :param rho: (float) Average correlation coefficient between strategy returns
        :return: (np.array) Array of parameters
        """

        pass

    @staticmethod
    def _annualized_sharpe_ratio(sharpe_ratio, sampling_frequency='A', rho=0, annualized=False,
                                 autocorr_adjusted=False):
        """
        Calculate the equivalent annualized Sharpe ratio after taking the autocorrelation of returns into account.

        Adjustments are based on the work of `Lo, A., The Statistics of Sharpe Ratios. Financial Analysts Journal,
        58 (2002), pp. 36-52` and are described there in more detail.

        :param sharpe_ratio: (float) Sharpe ratio of the strategy
        :param sampling_frequency: (str) Sampling frequency of returns
                                   ['D','W','M','Q','A'] = [Daily, Weekly, Monthly, Quarterly, Annual]
        :param rho: (float) Autocorrelation coefficient of returns at specified frequency
        :param annualized: (bool) Flag if annualized, 'ind_an' = 1, otherwise = 0
        :param autocorr_adjusted: (bool) Flag if Sharpe ratio was adjusted for returns autocorrelation
        :return: (float) Adjusted annualized Sharpe ratio
        """

        pass

    @staticmethod
    def _monthly_observations(num_obs, sampling_frequency):
        """
        Calculates the number of monthly observations based on sampling frequency and number of observations.

        :param num_obs: (int) Number of observations used for modelling
        :param sampling_frequency: (str) Sampling frequency of returns
                                   ['D','W','M','Q','A'] = [Daily, Weekly, Monthly, Quarterly, Annual]
        :return: (np.float64) Number of monthly observations
        """

        pass

    @staticmethod
    def _holm_method_sharpe(all_p_values, num_mult_test, p_val):
        """
        Runs one cycle of the Holm method for the Haircut Shape ratio algorithm.

        :param all_p_values: (np.array) Sorted p-values to adjust
        :param num_mult_test: (int) Number of multiple tests allowed
        :param p_val: (float) Significance level p-value
        :return: (np.float64) P-value adjusted at a significant level
        """

        pass

    @staticmethod
    def _bhy_method_sharpe(all_p_values, num_mult_test, p_val):
        """
        Runs one cycle of the BHY method for the Haircut Shape ratio algorithm.

        :param all_p_values: (np.array) Sorted p-values to adjust
        :param num_mult_test: (int) Number of multiple tests allowed
        :param p_val: (float) Significance level p-value
        :param c_constant: (float) Constant used in BHY method
        :return: (np.float64) P-value adjusted at a significant level
        """

        pass

    @staticmethod
    def _sharpe_ratio_haircut(p_val, monthly_obs, sr_annual):
        """
        Calculates the adjusted Sharpe ratio and the haircut based on the final p-value of the method.

        :param p_val: (float) Adjusted p-value of the method
        :param monthly_obs: (int) Number of monthly observations
        :param sr_annual: (float) Annualized Sharpe ratio to compare to
        :return: (np.array) Elements (Adjusted annual Sharpe ratio, Haircut percentage)
        """

        pass

    @staticmethod
    def _holm_method_returns(p_values_simulation, num_mult_test, alpha_sig):
        """
        Runs one cycle of the Holm method for the Profit Hurdle algorithm.

        :param p_values_simulation: (np.array) Sorted p-values to adjust
        :param num_mult_test: (int) Number of multiple tests allowed
        :param alpha_sig: (float) Significance level (e.g., 5%)
        :return: (np.float64) P-value adjusted at a significant level
        """

        pass

    @staticmethod
    def _bhy_method_returns(p_values_simulation, num_mult_test, alpha_sig):
        """
        Runs one cycle of the BHY method for the Profit Hurdle algorithm.

        :param p_values_simulation: (np.array) Sorted p-values to adjust
        :param num_mult_test: (int) Number of multiple tests allowed
        :param alpha_sig: (float) Significance level (e.g., 5%)
        :return: (np.float64) P-value adjusted at a significant level
        """

        pass

    def haircut_sharpe_ratios(self, sampling_frequency, num_obs, sharpe_ratio, annualized,
                              autocorr_adjusted, rho_a, num_mult_test, rho):
        # pylint: disable=too-many-locals
        """
        Calculates the adjusted Sharpe ratio due to testing multiplicity.

        This algorithm lets the user calculate Sharpe ratio adjustments and the corresponding haircuts based on
        the key parameters of returns from the strategy. The adjustment methods are Bonferroni, Holm,
        BHY (Benjamini, Hochberg and Yekutieli) and the Average of them. The algorithm calculates adjusted p-value,
        adjusted Sharpe ratio and the haircut.

        The haircut is the percentage difference between the original Sharpe ratio and the new Sharpe ratio.

        :param sampling_frequency: (str) Sampling frequency ['D','W','M','Q','A'] of returns
        :param num_obs: (int) Number of returns in the frequency specified in the previous step
        :param sharpe_ratio: (float) Sharpe ratio of the strategy. Either annualized or in the frequency specified in the previous step
        :param annualized: (bool) Flag if Sharpe ratio is annualized
        :param autocorr_adjusted: (bool) Flag if Sharpe ratio was adjusted for returns autocorrelation
        :param rho_a: (float) Autocorrelation coefficient of returns at the specified frequency (if the Sharpe ratio
                              wasn't corrected)
        :param num_mult_test: (int) Number of other strategies tested (multiple tests)
        :param rho: (float) Average correlation among returns of strategies tested
        :return: (np.ndarray) Array with adjuted p-value, adjusted Sharpe ratio, and haircut as rows
                              for Bonferroni, Holm, BHY and average adjustment as columns
        """

        pass

    def profit_hurdle(self, num_mult_test, num_obs, alpha_sig, vol_anu, rho):
        # pylint: disable=too-many-locals
        """
        Calculates the required mean monthly return for a strategy at a given level of significance.

        This algorithm uses four adjustment methods - Bonferroni, Holm, BHY (Benjamini, Hochberg and Yekutieli)
        and the Average of them. The result is the Minimum Average Monthly Return for the strategy to be significant
        at a given significance level, taking into account multiple testing.

        This function doesn't allow for any autocorrelation in the strategy returns.

        :param num_mult_test: (int) Number of tests in multiple testing allowed (number of other strategies tested)
        :param num_obs: (int) Number of monthly observations for a strategy
        :param alpha_sig: (float) Significance level (e.g., 5%)
        :param vol_anu: (float) Annual volatility of returns(e.g., 0.05 or 5%)
        :param rho: (float) Average correlation among returns of strategies tested
        :return: (np.ndarray) Minimum Average Monthly Returns for
                              [Independent tests, Bonferroni, Holm, BHY and Average for Multiple tests]
        """

        pass

def test_label_game(data, labels):
    # подготовим data для проверки разметки на торговле

    # Исходная обработка
    data_final = data[['open', 'low', 'high', 'close', 'tic']].copy()
    data_final['date'] = data.index
    data_final.sort_values(by=['date', 'tic'], inplace=True)
    labels_final = labels.copy()
    labels_final.sort_values(by=['timestamp', 'tic'], inplace=True)
    data_final['volatility'] = 1.0

    # Сброс индексов для синхронизации строк
    data_final.reset_index(drop=True, inplace=True)
    labels_final.reset_index(drop=True, inplace=True)

    # Присвоение новых столбцов
    data_final[['bin', 'sl', 'tp']] = labels_final[['bin', 'vr_low', 'vr_high']]

    stock_dimension = len(data.tic.unique())
    risk_volume = 0.2

    FEATURE_LENGTHS = {
        'prediction': 6,  # массив из 6 элементов
        'covariance': 3  # массив из 3 элементов
        # и т.д. для остальных feature...
    }
    # Constants and Transaction Cost Definitions
    lookback = 5
    features_list = ['prediction', 'covariance', 'volatility', 'last_return']
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
        'FEATURE_LENGTHS': FEATURE_LENGTHS,
        'use_logging': 0,
        'use_sltp': True
    }

    e_train_gym = StockPortfolioEnv(df=data_final, **env_kwargs)

    _, _, _, results = e_train_gym.__run__(type='label')

    return results

def test_prediction_game(data, indicators, coeff_tp = 1.0, coeff_sl = 1.0):
    # подготовим data для проверки игры на predictions
    step = {}
    step['coeff_tp'] = coeff_tp
    step['coeff_sl'] = coeff_sl

    # Исходная обработка
    data_final = data[['open', 'low', 'high', 'close', 'tic']].copy()
    data_final['date'] = data.index
    data_final.sort_values(by=['date', 'tic'], inplace=True)
    indicators_final = indicators.copy()
    if 'date' not in indicators_final.columns:
        indicators_final['date'] = indicators_final.index
    indicators_final.sort_values(by=['date', 'tic'], inplace=True)

    # Сброс индексов для синхронизации строк
    data_final.reset_index(drop=True, inplace=True)
    indicators_final.reset_index(drop=True, inplace=True)

    # нужно собрать из датасета indicators только колонки timestamp, tic
    # потом сделать колонку prediction - которая есть массив из данных колонок bin-1, bin-0, bin+1, regression
    data.sort_values(by=['timestamp', 'tic'], inplace=True)
    indicators.sort_values(by=['timestamp', 'tic'], inplace=True)

    # Создаём новую колонку, в которую упакуем нужные значения в виде списка
    data_final['prediction'] = indicators[['bin-1', 'bin-0', 'bin+1', 'regression']].values.tolist()

    data_final = add_takeprofit_stoploss_volume(data_final, coeff_tp=coeff_tp, coeff_sl=coeff_sl)

    # 2. Считаем логарифмическую доходность для каждого тикера
    #    groupby('tic') нужен, чтобы сдвигать close только внутри одного тикера.
    data_final['log_return'] = (
        data_final
        .groupby('tic')['close']
        .transform(lambda x: np.log(x / x.shift(1)))
    )

    # 3. Вычисляем стандартное отклонение логарифмической доходности на 5-шаговом окне
    #    (тоже отдельно по каждому тикеру)
    data_final['volatility'] = (
        data_final
        .groupby('tic')['log_return']
        .transform(lambda x: x.rolling(window=5).std())
    )

    # 4. При необходимости убираем NaN, которые появляются на первых строках при скользящем okне
    data_final['volatility'].fillna(0, inplace=True)

    # Посмотрим на результат
    # print(data_prediction.head())

    stock_dimension = len(data.tic.unique())
    risk_volume = 0.2

    FEATURE_LENGTHS = {
        'prediction': 6,  # массив из 6 элементов
        'covariance': 3  # массив из 3 элементов
        # и т.д. для остальных feature...
    }
    # Constants and Transaction Cost Definitions
    lookback = 5
    features_list = ['prediction', 'covariance', 'volatility', 'last_return']
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
        'FEATURE_LENGTHS': FEATURE_LENGTHS,
        'use_logging': 0,
        'use_sltp': True
    }

    e_train_gym = StockPortfolioEnv(df=data_final, **env_kwargs)

    _, _, _, results = e_train_gym.__run__(type='prediction')

    for key in results.keys():
        step[key] = results[key]

    return step

def show_heatmap(df_results):
    # Предположим, df_results содержит:
    # ['coeff_tp', 'coeff_sl', 'total_sharp_ratio', 'total_reward', 'total_drowdown']

    # Для построения тепловой карты нам нужна сводная таблица (pivot)
    df_pivot_sharp = df_results.pivot(
        index='coeff_tp',  # строки
        columns='coeff_sl',  # столбцы
        values='total_sharp_ratio'  # значения в ячейках
    )

    df_pivot_reward = df_results.pivot(
        index='coeff_tp',
        columns='coeff_sl',
        values='total_reward'
    )

    df_pivot_dd = df_results.pivot(
        index='coeff_tp',
        columns='coeff_sl',
        values='total_drowdown'
    )

    plt.figure(figsize=(18, 5))  # Ширина 18 дюймов, высота 5

    # Тепловая карта для total_sharp_ratio
    plt.subplot(1, 3, 1)
    sns.heatmap(df_pivot_sharp, annot=True, fmt=".2f", cmap='viridis')
    plt.title("Heatmap: total_sharp_ratio")
    plt.xlabel("coeff_sl")
    plt.ylabel("coeff_tp")

    # Тепловая карта для total_reward
    plt.subplot(1, 3, 2)
    sns.heatmap(df_pivot_reward, annot=True, fmt=".2f", cmap='viridis')
    plt.title("Heatmap: total_reward")
    plt.xlabel("coeff_sl")
    plt.ylabel("coeff_tp")

    # Тепловая карта для total_drowdown
    plt.subplot(1, 3, 3)
    sns.heatmap(df_pivot_dd, annot=True, fmt=".2f", cmap='viridis')
    plt.title("Heatmap: total_drowdown")
    plt.xlabel("coeff_sl")
    plt.ylabel("coeff_tp")

    plt.tight_layout()  # Чуть сожмём, чтобы подписи не налезали
    plt.show()

def d3_map(df_results, value):
    # Сводная таблица (pivot) по total_reward
    df_pivot_reward = df_results.pivot(
        index='coeff_tp',
        columns='coeff_sl',
        values=value
    )

    # X и Y — это значения coeff_sl и coeff_tp, превращаем их в сетку
    X = df_pivot_reward.columns.values  # coeff_sl
    Y = df_pivot_reward.index.values  # coeff_tp
    X, Y = np.meshgrid(X, Y)

    # Z — сама метрика (размер должен совпадать с X, Y)
    Z = df_pivot_reward.values

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax.set_title(f'3D Surface: {value}')
    ax.set_xlabel('coeff_sl')
    ax.set_ylabel('coeff_tp')
    ax.set_zlabel(f'{value}')

    fig.colorbar(surf, shrink=0.5, aspect=5)  # полоса цвета
    plt.show()


def test_sltp_run(data, indicators, processor):
    """
    Запускает перебор coeff_tp и coeff_sl с учетом того,
    какие значения уже были просчитаны и записаны в CSV.
    Используем round(..., 5), чтобы избежать проблем
    с точностью при сравнении float.
    """

    # Убедимся, что директория существует (если нет — создаём)
    makedirs(processor.file_path, exist_ok=True)

    # Файл, куда сохраняются результаты
    csv_file_path = os.path.join(processor.file_path, 'results_sl_tp_test.csv')
    # print("CSV file path:", csv_file_path)
    # print("Exists?", os.path.exists(csv_file_path))

    # Создаём диапазоны значений
    tp_values = np.arange(2.0, 4.5, 0.5)
    sl_values = np.arange(3.0, 5.5, 0.5)

    # Считываем уже посчитанные результаты, если файл существует
    if exists(csv_file_path):
        existing_df = pd.read_csv(csv_file_path)

        existing_df['coeff_tp'] = existing_df['coeff_tp'].astype(float)
        existing_df['coeff_sl'] = existing_df['coeff_sl'].astype(float)

        # Здесь считаем, что в CSV уже хранятся округлённые значения coeff_tp и coeff_sl
        # (см. ниже блок кода, где мы делаем round())
        processed_pairs = set(zip(existing_df['coeff_tp'], existing_df['coeff_sl']))
        print('processed_pairs', processed_pairs)

        # Превращаем в список словарей, чтобы удобно пополнять
        list_results = existing_df.to_dict('records')
    else:
        existing_df = pd.DataFrame()
        processed_pairs = set()
        list_results = []

    # Перебираем все пары (tp_val, sl_val)
    for tp_val in tqdm(tp_values, desc="Перебор coeff_tp", leave=True, position=0, dynamic_ncols=True):
        for sl_val in tqdm(sl_values, desc="Перебор coeff_sl", leave=True, position=1, dynamic_ncols=True):

            # Округляем значения, чтобы избежать неточностей (например, 0.30000000000000004)
            tp_rounded = round(tp_val, 5)
            sl_rounded = round(sl_val, 5)

            # print('(tp_rounded, sl_rounded)', (tp_rounded, sl_rounded))

            # Проверяем, не было ли уже вычислений для этой пары
            if (tp_rounded, sl_rounded) in processed_pairs:
                continue  # Пропускаем, так как уже считали

            # Если пары ещё нет, запускаем тест
            result_dict = test_prediction_game(data, indicators, coeff_tp=tp_val, coeff_sl=sl_val)

            # Добавляем в словарь «признаки» текущих tp/sl (в уже округлённом виде)
            result_dict['coeff_tp'] = tp_rounded
            result_dict['coeff_sl'] = sl_rounded

            # Дописываем в общий список
            list_results.append(result_dict)

            # Обновляем processed_pairs
            processed_pairs.add((tp_rounded, sl_rounded))

            # На каждом шаге сохраняем результаты в CSV
            # (или можно реже, если запись в CSV слишком долгая)
            temp_df = pd.DataFrame([result_dict])
            # mode='a' -> добавлять в конец файла
            # header=not os.path.exists(...) -> шапка CSV только если файл ещё не создан
            temp_df.to_csv(csv_file_path, mode='a', header=not os.path.exists(csv_file_path), index=False)

    # После окончания всех итераций можно перезаписать CSV (итоговое «чистое» сохранение):
    df_results = pd.DataFrame(list_results)
    df_results.to_csv(csv_file_path, index=False)

    # Вызовы функций визуализации (пример):
    show_heatmap(df_results)
    d3_map(df_results, 'total_sharp_ratio')
    d3_map(df_results, 'total_reward')
    d3_map(df_results, 'total_drowdown')