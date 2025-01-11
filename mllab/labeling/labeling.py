"""
Logic regarding labeling from chapter 3. In particular the Triple Barrier Method and Meta-Labeling.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from joblib import Parallel, delayed
from numba import njit

# Snippet 3.1, page 44, Daily Volatility Estimates
from mllab.util.multiprocess import mp_pandas_obj


# Snippet 3.2, page 45, Triple Barrier Labeling Method
def apply_pt_sl_on_t1(**kwargs):  # pragma: no cover
    """
    Advances in Financial Machine Learning, Snippet 3.2, page 45.

    Triple Barrier Labeling Method

    This function applies the triple-barrier labeling method. It works on a set of
    datetime index values (molecule). This allows the program to parallelize the processing.

    Mainly it returns a DataFrame of timestamps regarding the time when the first barriers were reached.

    :param close: (pd.Series) Close prices
    :param events: (pd.Series) Indices that signify "events" (see cusum_filter function
    for more details)
    :param pt_sl: (np.array) Element 0, indicates the profit taking level; Element 1 is stop loss level
    :param molecule: (an array) A set of datetime index values for processing
    :return: (pd.DataFrame) Timestamps of when first barrier was touched
    """
    opt = kwargs['kwargs']
    molecule = opt['molecule']
    close = opt['close']
    events = opt['events']
    pt_sl = opt['pt_sl']
    normalized_data = opt['normalized_data']

    results = pd.DataFrame(index=molecule)
    for idx, loc in enumerate(molecule, start=0):

        if loc not in events.index:
            continue
        df0 = close[loc:events.loc[loc, 't1']]  # Path prices
        trgt = events.loc[loc, 'trgt']

        # Profit taking and stop loss
        pt = trgt * pt_sl[0] if pt_sl[0] > 0 else np.nan
        sl = -trgt * pt_sl[1] if pt_sl[1] > 0 else np.nan

        # Barriers
        if normalized_data:
            # Barriers
            # нужен барьер trgt волатильность
            results.loc[loc, 'sl'] = df0[df0 <= sl].index.min()
            results.loc[loc, 'pt'] = df0[df0 >= pt].index.min()
        else:
            results.loc[loc, 'sl'] = df0[(df0 / close[loc] - 1) <= sl].index.min()
            results.loc[loc, 'pt'] = df0[(df0 / close[loc] - 1) >= pt].index.min()

        # нужно поставить минимальное время, где происходит пересечение барьеров ...
        results.loc[loc, 't1'] = min(events.loc[loc, 't1'], results.loc[loc, 'sl'], results.loc[loc, 'pt'])

        # если минимальное время начало интервала, то нужно перенести на начало следующего интервала
        if results.loc[loc, 't1'] == loc and idx + 1 < len(molecule):
            results.loc[loc, 't2'] = molecule[idx + 1]
        else:
            results.loc[loc, 't2'] = results.loc[loc, 't1']

    return results


def add_vertical_barrier(t_events, close, num_days=0, num_hours=0, num_minutes=0, num_seconds=0):
    """
    Advances in Financial Machine Learning, Snippet 3.4 page 49.

    Adding a Vertical Barrier

    For each index in t_events, it finds the timestamp of the next price bar at or immediately after
    a number of days num_days. This vertical barrier can be passed as an optional argument t1 in get_events.

    This function creates a series that has all the timestamps of when the vertical barrier would be reached.

    :param t_events: (pd.Series) Series of events (symmetric CUSUM filter)
    :param close: (pd.Series) Close prices
    :param num_days: (int) Number of days to add for vertical barrier
    :param num_hours: (int) Number of hours to add for vertical barrier
    :param num_minutes: (int) Number of minutes to add for vertical barrier
    :param num_seconds: (int) Number of seconds to add for vertical barrier
    :return: (pd.DatetimeIndex) Timestamps of vertical barriers
    """
    timedelta = pd.Timedelta(days=num_days, hours=num_hours, minutes=num_minutes, seconds=num_seconds)
    barrier_times = t_events + timedelta
    barrier_times = barrier_times[barrier_times <= close.index[-1]]  # Ensure barriers are within data range

    # Align to the closest future index in close
    aligned_barriers = []
    for barrier_time in barrier_times:
        future_indices = close.index[close.index >= barrier_time]
        if not future_indices.empty:
            aligned_barriers.append(future_indices[0])
        else:
            aligned_barriers.append(close.index[-1])  # Fallback to the last available close index

    return pd.DatetimeIndex(aligned_barriers)


# Snippet 3.3 -> 3.6 page 50, Getting the Time of the First Touch, with Meta Labels
def get_events(close, t_events, pt_sl, target, min_ret=None, num_threads=1, vertical_barrier_times=None,
               side_prediction=None, normalized_data: bool = False, verbose=True):
    """
    Advances in Financial Machine Learning, Snippet 3.6 page 50.

    Getting the Time of the First Touch, with Meta Labels

    This function is orchestrator to meta-label the data, in conjunction with the Triple Barrier Method.

    :param close: (pd.Series) Close prices
    :param t_events: (pd.Series) of t_events. These are timestamps that will seed every triple barrier.
        These are the timestamps selected by the sampling procedures discussed in Chapter 2, Section 2.5.
        Eg: CUSUM Filter
    :param pt_sl: (2 element array) Element 0, indicates the profit taking level; Element 1 is stop loss level.
        A non-negative float that sets the width of the two barriers. A 0 value means that the respective
        horizontal barrier (profit taking and/or stop loss) will be disabled.
    :param target: (pd.Series) of values that are used (in conjunction with pt_sl) to determine the width
        of the barrier. In this program this is daily volatility series.
    :param min_ret: (float) The minimum target return required for running a triple barrier search.
                    If None, it will be set to the median of the target series.
    :param num_threads: (int) The number of threads concurrently used by the function.
    :param vertical_barrier_times: (pd.Series) A pandas series with the timestamps of the vertical barriers.
        We pass a False when we want to disable vertical barriers.
    :param side_prediction: (pd.Series) Side of the bet (long/short) as decided by the primary model
    :param verbose: (bool) Flag to report progress on asynch jobs
    :return: (pd.DataFrame) Events
            -events.index is event's starttime
            -events['t1'] is event's endtime
            -events['trgt'] is event's target
            -events['side'] (optional) implies the algo's position side
            -events['pt'] is profit taking multiple
            -events['sl']  is stop loss multiple
    """
    # проверка на соответствие длины t_events и количества барьеров vertical_barrier_times
    if len(t_events) > len(vertical_barrier_times):
        t_events = t_events[:len(vertical_barrier_times) + 1]

    # Auto-set min_ret if not provided
    if not min_ret is None:
        min_ret = target.median()
        # Filter events based on min_ret
        target = target[target > min_ret]

    t_events = t_events.intersection(target.index)

    # Set vertical barriers
    if isinstance(vertical_barrier_times, (pd.Series, pd.DataFrame, pd.DatetimeIndex)):
        t1 = pd.Series(vertical_barrier_times, index=t_events)
    else:
        t1 = pd.Series(pd.NaT, index=t_events)

    # Create events DataFrame
    events = pd.DataFrame({'t1': t1, 'trgt': target.loc[t_events]}, index=t_events)
    if side_prediction is not None:
        events['side'] = side_prediction.loc[events.index]

    # Apply Triple Barrier Labeling
    # close, events, pt_sl, molecule
    df0 = mp_pandas_obj(
        func=apply_pt_sl_on_t1,
        pd_obj=('molecule', events.index),
        num_threads=num_threads,
        close=close,
        events=events,
        pt_sl=pt_sl,
        normalized_data=normalized_data
    )
    events = events.assign(**df0)
    return events


# Snippet 3.9, pg 55, Question 3.3
def barrier_touched(out_df, events):
    """
    Advances in Financial Machine Learning, Snippet 3.9, page 55, Question 3.3.

    Adjust the getBins function (Snippet 3.7) to return a 0 whenever the vertical barrier is the one touched first.

    Top horizontal barrier: 1
    Bottom horizontal barrier: -1
    Vertical barrier: 0

    :param out_df: (pd.DataFrame) Returns and target
    :param events: (pd.DataFrame) The original events data frame. Contains the pt sl multiples needed here.
    :return: (pd.DataFrame) Returns, target, and labels
    """
    for loc in out_df.index:
        sl = out_df.loc[loc, 'sl']
        pt = out_df.loc[loc, 'pt']
        t1 = out_df.loc[loc, 't1']

        if pd.isna(sl) and pd.isna(pt):
            out_df.loc[loc, 'label'] = 0
        elif not pd.isna(sl) and (pd.isna(pt) or sl <= pt):
            out_df.loc[loc, 'label'] = -1
        elif not pd.isna(pt) and (pd.isna(sl) or pt <= sl):
            out_df.loc[loc, 'label'] = 1
        else:
            out_df.loc[loc, 'label'] = 0
    return out_df


# Snippet 3.4 -> 3.7, page 51, Labeling for Side & Size with Meta Labels
def get_bins(triple_barrier_events, close, normalized_data: bool = False):
    """
    Advances in Financial Machine Learning, Snippet 3.7, page 51.

    Labeling for Side & Size with Meta Labels

    Compute event's outcome (including side information, if provided).
    events is a DataFrame where:

    Now the possible values for labels in out['bin'] are {0,1}, as opposed to whether to take the bet or pass,
    a purely binary prediction. When the predicted label the previous feasible values {−1,0,1}.
    The ML algorithm will be trained to decide is 1, we can use the probability of this secondary prediction
    to derive the size of the bet, where the side (sign) of the position has been set by the primary model.

    :param triple_barrier_events: (pd.DataFrame)
                -events.index is event's starttime
                -events['t1'] is event's endtime
                -events['trgt'] is event's target
                -events['side'] (optional) implies the algo's position side
                Case 1: ('side' not in events): bin in (-1,1) <-label by price action
                Case 2: ('side' in events): bin in (0,1) <-label by pnl (meta-labeling)
    :param close: (pd.Series) Close prices
    :return: (pd.DataFrame) Meta-labeled events
    """
    out = triple_barrier_events[['t1', 'trgt']].copy()
    for loc, event in triple_barrier_events.iterrows():
        df0 = close[loc:event['t2']]  # Path prices
        if normalized_data:
            df0 = (df0) * event['side'] if 'side' in event else df0
        else:
            df0 = (df0 / close[loc] - 1) * event['side'] if 'side' in event else df0 / close[loc] - 1

        # Assign labels
        out.loc[loc, 'bin'] = 0
        if df0.max() > event['trgt']:
            out.loc[loc, 'bin'] = 1
        if df0.min() < -event['trgt']:
            out.loc[loc, 'bin'] = -1

        if out.loc[loc, 'bin'] > 0:
            out.loc[loc, 'return'] = close[loc:event['t2']].max()
        elif out.loc[loc, 'bin'] < 0:
            out.loc[loc, 'return'] = close[loc:event['t2']].min()
        else:
            out.loc[loc, 'return'] = 0

    return out


@njit
def calculate_segments(group_close, group_low, group_high, group_last_minute, group_end_day_index, short_period,
                       long_period, group_threshold):
    n = len(group_close)
    bins = [0] * n
    vr_lows = [0.0] * n
    vr_highs = [0.0] * n
    returns = [0.0] * n
    period_lengths = [0] * n

    current_bin = 0
    cumulative_return = 0.0
    start_index = short_period
    pred_index = 0

    for i in range(short_period, n):
        # вот здесь нужно понимать, если pred_index в прошлом торговом дне, а i в текущем то нужно считать, что нет роста или падений, чтобы не было такой разметки
        short_return = (group_close[i] - group_close[pred_index]) / group_close[pred_index]
        if group_end_day_index[pred_index] < group_end_day_index[i]:
            short_return = 0
        new_bin = 1 if short_return > group_threshold else -1 if short_return < -group_threshold else 0

        # нужно добавить конец дня, т.е. когда i это конец торгового дня, тоже нужно завершить тренд ...
        if new_bin != current_bin or (i - start_index + 1) > long_period or group_last_minute[i]:
            if current_bin != 0:
                period_length = i - start_index
                for j in range(start_index - short_period, i - short_period):
                    bins[j] = current_bin
                    last_vr_lows = min(group_low[pred_index:j + short_period]) / group_close[pred_index] - 1
                    last_vr_highs = max(group_high[pred_index:j+ short_period]) / group_close[pred_index] - 1
                    last_returns = (group_close[j+ short_period] - group_close[pred_index]) / group_close[pred_index]
                    vr_lows[j] = last_vr_lows
                    vr_highs[j] = last_vr_highs
                    returns[j] = last_returns
                    period_lengths[j] = period_length - (j - start_index) - 1
                    # последний период i -  short_period -1

                # дозаполняем разметку до конца short_period
                for j in range(i - short_period, i - 1):
                    bins[j] = current_bin
                    vr_lows[j] = last_vr_lows
                    vr_highs[j] = last_vr_highs
                    returns[j] = last_returns
                    period_lengths[j] = period_length - (j - start_index) - 1

            start_index = i
            pred_index = start_index - short_period
            # нужно пересчитать current_bin
            short_return = (group_close[i] - group_close[pred_index]) / group_close[pred_index]
            if group_end_day_index[pred_index] < group_end_day_index[i]:
                short_return = 0
            current_bin = 1 if short_return > group_threshold else -1 if short_return < -group_threshold else 0

    if current_bin != 0:
        period_length = n - start_index
        for j in range(start_index - short_period, n):
            bins[j] = current_bin
            vr_lows[j] = min(group_low[pred_index:j+ short_period]) / group_close[pred_index] - 1
            vr_highs[j] = max(group_high[pred_index:j+ short_period]) / group_close[pred_index] - 1
            returns[j] = (group_close[j+ short_period] - group_close[pred_index]) / group_close[pred_index]
            period_lengths[j] = period_length - (j - start_index) -1

    return bins, vr_lows, vr_highs, returns, period_lengths


def short_long_box(data: pd.DataFrame, short_period: int = 1, long_period: int = 5, threshold: float = 0.005):
    """
    Identifies price trends and outliers in the provided OHLC data, optionally grouped by 'tic'.
    Includes a progress indicator and optimized computation.

    Parameters:
        data (pd.DataFrame): Input DataFrame with columns ['timestamp', 'tic', 'open', 'high', 'low', 'close'] (optional 'tic').
        short_period (int): Minimum period to evaluate a trend (short-term).
        long_period (int): Maximum period to accumulate trend data (long-term).
        threshold (float or dict): Threshold for detecting trend direction change.
                                If 'tic' is present, it must be a dictionary with keys as 'tic' values.

    Returns:
        pd.DataFrame: A DataFrame with trend and outlier calculations for each timestamp.
    """
    # Если уже индексировано по timestamp, то data.index.name == 'timestamp'.
    # Если нет — пробуем установить из столбца.
    if data.index.name != 'timestamp':
        if 'timestamp' in data.columns:
            data = data.set_index('timestamp').copy()
        else:
            raise ValueError("DataFrame не индексирован по 'timestamp' и не содержит столбец 'timestamp'.")

    # Предполагается, что DataFrame 'data' уже содержит столбец или индекс 'timestamp'.
    # Убедимся, что индекс DataFrame — это datetime индекс.
    #if data.index.name != 'timestamp' or not pd.api.types.is_datetime64_any_dtype(data.index):
    #    if 'timestamp' in data.columns:
    #        data['timestamp'] = pd.to_datetime(data['timestamp'])
    #        data.set_index('timestamp', inplace=True)
    #    else:
    #        raise ValueError("DataFrame должен содержать индекс или столбец 'timestamp' с типом datetime.")

    # Создаём столбец 'date' для группировки по дате (без времени)
    data['date'] = pd.to_datetime(data.index, utc=True).date

    # Проверяем наличие столбца 'row_num' в last_entries
    if 'row_num' not in data.columns:
        # 1. Создаем вспомогательный столбец с порядковым номером для каждой строки
        data['row_num'] = range(len(data))

    # Группируем по дате и находим последнюю запись для каждого дня
    last_entries = data.groupby('date').tail(1)

    # Инициализируем новый столбец 'is_last_minute' значениями False
    data['is_last_minute'] = False

    # Устанавливаем True для строк, которые соответствуют последней минуте каждого дня
    # Используем индекс из last_entries для обновления основной таблицы
    data.loc[last_entries.index, 'is_last_minute'] = True

    ## 4. Создаем словарь: ключ – дата, значение – порядковый номер последней записи этого дня
    #mapping = {row.date: row.row_num for _, row in last_entries.iterrows()}

    # 4. Создаем словарь: ключ – дата, значение – порядковый номер последней записи этого дня,
    # используя set_index для формирования словаря напрямую
    mapping = last_entries.set_index('date')['row_num'].to_dict()

    # 4. Добавляем колонку с индексом конца торгового дня для каждой строки
    data['end_day_index'] = data['date'].map(mapping)

    # (Опционально) Удаляем вспомогательный столбец 'date', если он больше не нужен
    data.drop(columns='date', inplace=True)

    has_tic = 'tic' in data.columns
    result_list = []

    # Sort data for faster grouping
    if has_tic:
        data.sort_values(['tic', 'timestamp'], inplace=True)

    # Group data by 'tic' if it exists
    groups = [('default', data)] if not has_tic else data.groupby('tic')

    # Calculate threshold per tic if necessary
    if isinstance(threshold, dict):
        calculated_threshold = threshold
    elif has_tic:
        calculated_threshold = {}
        for tic in data['tic'].unique():
            group = data[data['tic'] == tic]
            mean_deal = 0.02 * 400000
            commission = 0.0035
            minimal = 0.35
            cost_transaction = commission * int(mean_deal / group['close'].mean())
            if cost_transaction < minimal:
                cost_transaction = minimal
            tic_threshold = 2 * cost_transaction / group['close'].mean()
            calculated_threshold[tic] = tic_threshold
    else:
        calculated_threshold = {"default": threshold}

    # Parallel processing for groups
    def process_group(tic, group):
        if len(group) < short_period:
            return pd.DataFrame()

        group_result = pd.DataFrame(index=group.index)
        group_result['bin'] = 0
        group_result['vr_low'] = 0.0
        group_result['vr_high'] = 0.0
        group_result['return'] = 0.0
        group_result['period_length'] = 0
        if has_tic:
            group_result['tic'] = tic

        group_threshold = threshold if not isinstance(calculated_threshold, dict) else calculated_threshold.get(tic,
                                                                                                                threshold)

        bins, vr_lows, vr_highs, returns, period_lengths = calculate_segments(
            group['close'].values, group['low'].values, group['high'].values, group['is_last_minute'].values,
            group['end_day_index'].values, short_period, long_period, group_threshold
        )

        group_result['bin'] = bins
        group_result['vr_low'] = vr_lows
        group_result['vr_high'] = vr_highs
        group_result['return'] = returns
        group_result['period_length'] = period_lengths

        return group_result

    result_list = Parallel(n_jobs=-1)(
        delayed(process_group)(tic, group) for tic, group in tqdm(groups, desc="Processing Groups", unit="group")
    )

    # т.е алгоритм такой, в раздельности по Тикерам, идем вдоль массива и если находим в колонке period_length сочетание short_period, 0 то переносим в следующие short_period -1 строки содержание последней колонки

    # Combine results
    final_result = pd.concat(result_list, ignore_index=True)
    final_result.index = data.index  # Preserve the original index from the input DataFrame
    return final_result


def check_trend_labels_with_period_length(data: pd.DataFrame, labels: pd.DataFrame, short_period: int = 2):
    """
    Проверяет корректность разметки изменения тренда с учетом интервалов времени period_length,
    начиная цикл со второй строки и устанавливая previous_end_time на предыдущую строку.

    Параметры:
      data   - DataFrame с индексом 'timestamp' и колонками ['tic', 'open', 'low', 'high', 'close']
      labels - DataFrame с индексом 'timestamp' и колонками ['tic', 'period_length', 'bin'],
               где 'period_length' — количество интервалов (строк) текущего тренда,
               'bin' — направление: 1 для роста, -1 для падения, 0 для нейтрального.

    Возвращает:
      Список кортежей (tic, period_end_timestamp, сообщение), где найдены несоответствия.
    """
    discrepancies = []
    labels_grouped = labels.groupby('tic')

    for tic, tic_labels in labels_grouped:
        tic_labels = tic_labels.sort_index().reset_index()
        tic_data = data[data['tic'] == tic].sort_index().reset_index()

        if tic_data.empty or len(tic_labels) < 2:
            continue

        # Начинаем со второй строки
        start_index = 0
        previous_close = None
        # указатель точку завершения текущего периода
        #previous_end_time = start_index - short_period

        # Цикл начинается со второй строки
        for idx in range(start_index, len(tic_labels) - short_period):
            label_row = tic_labels.loc[idx]
            bin_value = label_row['bin']
            period_length = int(label_row['period_length'])
            current_label_time = label_row['timestamp']

            # Определение начала периода
            # start_time = current_label_time if current_label_time > previous_end_time else previous_end_time

            # Обработка bin == 0
            if bin_value == 0:
                # Сдвиг previous_end_time на текущую строку
                #previous_end_time = idx +  1
                previous_close = tic_data.loc[idx+1]['close']
                continue

            # Обработка bin == 1 или bin == -1
            period_rows = tic_data.loc[idx + short_period:].head(period_length - short_period + 1)
            if period_rows.empty:
                continue

            #period_end_time = period_rows.index[-1]
            period_end_close = period_rows.iloc[-1]['close']
            period_end_time_time = tic_data.loc[idx]['timestamp']

            if previous_close is not None:
                if bin_value == 1 and period_end_close < previous_close:
                    discrepancies.append(
                        (tic, period_end_time_time, idx,
                         f"Несоответствие: тикер {tic}, ожидаемый рост, но "
                         f"цена упала с {previous_close} до {period_end_close}")
                    )
                elif bin_value == -1 and period_end_close > previous_close:
                    discrepancies.append(
                        (tic, period_end_time_time, idx,
                         f"Несоответствие: тикер {tic}, ожидается падение, но "
                         f"цена выросла с {previous_close} до {period_end_close}")
                    )
            # нужно передвигать период, только если сменился тренд
            # может уже начаться новый тренд .. поэтому нужно переключатся раньше ...
            if period_length == short_period:
                previous_close = tic_data.loc[idx + period_length - short_period + 1]['close']
                #previous_end_time = period_end_time

    return discrepancies


# Snippet 3.8 page 54
def drop_labels(events, min_pct=.05):
    """
    Advances in Financial Machine Learning, Snippet 3.8 page 54.

    This function recursively eliminates rare observations.

    :param events: (dp.DataFrame) Events.
    :param min_pct: (float) A fraction used to decide if the observation occurs less than that fraction.
    :return: (pd.DataFrame) Events.
    """
    while True:
        df0 = events['bin'].value_counts(normalize=True)
        if df0.min() > min_pct or df0.shape[0] < 3:
            break
        events = events[events['bin'] != df0.idxmin()]
    return events
