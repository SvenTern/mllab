"""
Explosiveness tests: SADF
"""

from typing import Union, Tuple
import pandas as pd
import numpy as np
from statsmodels.api import OLS, add_constant
from mllab.util.multiprocess import mp_pandas_obj

def _get_sadf_at_t(X: pd.DataFrame, y: pd.DataFrame, min_length: int, model: str, phi: float) -> float:
    """
    Advances in Financial Machine Learning, Snippet 17.2, page 258.

    SADF's Inner Loop (get SADF value at t)

    :param X: (pd.DataFrame) Lagged values, constants, trend coefficients
    :param y: (pd.DataFrame) Y values (either y or y.diff())
    :param min_length: (int) Minimum number of samples needed for estimation
    :param model: (str) Either 'linear', 'quadratic', 'sm_poly_1', 'sm_poly_2', 'sm_exp', 'sm_power'
    :param phi: (float) Coefficient to penalize large sample lengths when computing SMT, in [0, 1]
    :return: (float) SADF statistics for y.index[-1]
    """
    adf_stats = []
    for start in range(0, len(y) - min_length + 1):
        X_window = X.iloc[start:]
        y_window = y.iloc[start:]

        model_fit = OLS(y_window, X_window).fit()
        t_stat = model_fit.tvalues[0]
        # Penalize ADF statistic if phi > 0
        if phi > 0:
            t_stat /= (len(y_window) ** phi)
        adf_stats.append(t_stat)

    return max(adf_stats)

def _get_y_x(series: pd.Series, model: str, lags: Union[int, list], add_const: bool) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Advances in Financial Machine Learning, Snippet 17.2, page 258-259.

    Preparing The Datasets

    :param series: (pd.Series) Series to prepare for test statistics generation (for example log prices)
    :param model: (str) Either 'linear', 'quadratic', 'sm_poly_1', 'sm_poly_2', 'sm_exp', 'sm_power'
    :param lags: (int or list) Either number of lags to use or array of specified lags
    :param add_const: (bool) Flag to add constant
    :return: (pd.DataFrame, pd.DataFrame) Prepared y and X for SADF generation
    """
    y = series.diff().dropna()
    X = _lag_df(series, lags).iloc[1:]

    if model == 'linear':
        pass  # No additional modifications
    elif model == 'quadratic':
        X['trend'] = np.arange(len(X)) ** 2
    
    if add_const:
        X = add_constant(X)

    return y, X

def _lag_df(df: pd.DataFrame, lags: Union[int, list]) -> pd.DataFrame:
    """
    Advances in Financial Machine Learning, Snipet 17.3, page 259.

    Apply Lags to DataFrame

    :param df: (int or list) Either number of lags to use or array of specified lags
    :param lags: (int or list) Lag(s) to use
    :return: (pd.DataFrame) Dataframe with lags
    """
    if isinstance(lags, int):
        lags = list(range(1, lags + 1))
    
    lagged_df = pd.concat([df.shift(lag) for lag in lags], axis=1)
    lagged_df.columns = [f'lag_{lag}' for lag in lags]

    return lagged_df.dropna()

def get_betas(X: pd.DataFrame, y: pd.DataFrame) -> Tuple[np.array, np.array]:
    """
    Advances in Financial Machine Learning, Snippet 17.4, page 259.

    Fitting The ADF Specification (get beta estimate and estimate variance)

    :param X: (pd.DataFrame) Features(factors)
    :param y: (pd.DataFrame) Outcomes
    :return: (np.array, np.array) Betas and variances of estimates
    """
    model_fit = OLS(y, X).fit()
    betas = model_fit.params
    variances = model_fit.bse

    return betas.values, variances.values

def _sadf_outer_loop(X: pd.DataFrame, y: pd.DataFrame, min_length: int, model: str, phi: float,
                     molecule: list) -> pd.Series:
    """
    Advances in Financial Machine Learning, Snippet 17.3, page 259.

    This function gets SADF for t times from molecule

    :param X: (pd.DataFrame) Features(factors)
    :param y: (pd.DataFrame) Outcomes
    :param min_length: (int) Minimum number of observations
    :param model: (str) Either 'linear', 'quadratic', 'sm_poly_1', 'sm_poly_2', 'sm_exp', 'sm_power'
    :param phi: (float) Coefficient to penalize large sample lengths when computing SMT, in [0, 1]
    :param molecule: (list) Indices to get SADF
    :return: (pd.Series) SADF statistics
    """
    sadf_series = pd.Series(index=molecule)
    for idx in molecule:
        sadf_series.loc[idx] = _get_sadf_at_t(X[:idx], y[:idx], min_length, model, phi)
    
    return sadf_series

def get_sadf(series: pd.Series, model: str, lags: Union[int, list], min_length: int, add_const: bool = False,
             phi: float = 0, num_threads: int = 8, verbose: bool = True) -> pd.Series:
    """
    Advances in Financial Machine Learning, p. 258-259.

    Multithread implementation of SADF

    SADF fits the ADF regression at each end point t with backwards expanding start points. For the estimation
    of SADF(t), the right side of the window is fixed at t. SADF recursively expands the beginning of the sample
    up to t - min_length, and returns the sup of this set.

    When doing with sub- or super-martingale test, the variance of beta of a weak long-run bubble may be smaller than
    one of a strong short-run bubble, hence biasing the method towards long-run bubbles. To correct for this bias,
    ADF statistic in samples with large lengths can be penalized with the coefficient phi in [0, 1] such that:

    ADF_penalized = ADF / (sample_length ^ phi)

    :param series: (pd.Series) Series for which SADF statistics are generated
    :param model: (str) Either 'linear', 'quadratic', 'sm_poly_1', 'sm_poly_2', 'sm_exp', 'sm_power'
    :param lags: (int or list) Either number of lags to use or array of specified lags
    :param min_length: (int) Minimum number of observations needed for estimation
    :param add_const: (bool) Flag to add constant
    :param phi: (float) Coefficient to penalize large sample lengths when computing SMT, in [0, 1]
    :param num_threads: (int) Number of cores to use
    :param verbose: (bool) Flag to report progress on asynch jobs
    :return: (pd.Series) SADF statistics
    """
    y, X = _get_y_x(series, model, lags, add_const)
    molecules = np.array_split(range(min_length, len(series) + 1), num_threads)

    sadf_stats = mp_pandas_obj(
        func=_sadf_outer_loop,
        pd_obj=('molecule', molecules),
        num_threads=num_threads,
        X=X,
        y=y,
        min_length=min_length,
        model=model,
        phi=phi
    )

    return sadf_stats
