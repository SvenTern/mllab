# pylint: disable=missing-module-docstring
import numpy as np
import scipy.stats as ss
from scipy import linalg
from mllab.ensemble.model_train import StockPortfolioEnv
from mllab.data_structures.preprocess_data import add_takeprofit_stoploss_volume


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

        e_train_gym.__run__(type='label')

def test_prediction_game(data, indicators, coeff_tp = 1, coeff_sl = 1):
    # подготовим data для проверки игры на predictions

    # Исходная обработка
    data_final = data[['open', 'low', 'high', 'close', 'tic']].copy()
    data_final['date'] = data.index
    data_final.sort_values(by=['date', 'tic'], inplace=True)
    indicators_final = indicators.copy()
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
        .apply(lambda x: np.log(x / x.shift(1)))
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

    e_train_gym.__run__(type='predictions')