import pandas as pd


def short_long_box(data: pd.DataFrame, short_period: int = 3, long_period: int = 5, threshold: float = 0.005):
    """
    Identifies price trends and outliers in the provided OHLC data, optionally grouped by 'tic'.

    Parameters:
        data (pd.DataFrame): Input DataFrame with columns ['timestamp', 'tic', 'open', 'high', 'low', 'close'] (optional 'tic').
        short_period (int): Minimum period to evaluate a trend (short-term).
        long_period (int): Maximum period to accumulate trend data (long-term).
        threshold (float or dict): Threshold for detecting trend direction change.
                                If 'tic' is present, it must be a dictionary with keys as 'tic' values.

    Returns:
        pd.DataFrame: A DataFrame with trend and outlier calculations for each timestamp.
    """
    result_list = []

    groups = [('', data)] #if not has_tic else data.groupby('tic')

    for tic, group in groups:
        group_result = pd.DataFrame(index=group.index)
        group_result['bin'] = 0
        #group_result['vr_low'] = 0.0
        #group_result['vr_high'] = 0.0
        group_result['return'] = 0.0
        group_result['period_length'] = 0
        #if has_tic:
        #    group_result['tic'] = tic

        group_threshold = threshold if not isinstance(threshold, dict) else threshold.get(tic, threshold)
        current_bin = None
        cumulative_return = 0.0
        start_index = 0
        start_pred = 0


        i = 0
        while i < len(group):
            if i < short_period - 1:
                i += 1
                continue

            short_return = (group['close'].iloc[i] - group['close'].iloc[start_pred])
            new_bin = 1 if short_return > group_threshold else -1 if short_return < -group_threshold else 0

            if new_bin != current_bin or (i - start_index + 1) > long_period:
                if current_bin is not None:
                    #vr_low = group['low'].iloc[start_index:i].min() / group['close'].iloc[start_index] - 1
                    #vr_high = group['high'].iloc[start_index:i].max() / group['close'].iloc[start_index] - 1
                    period_length = i - start_index
                    group_result.iloc[start_index:i, group_result.columns.get_loc('bin')] = current_bin
                    #group_result.iloc[start_index:i, group_result.columns.get_loc('vr_low')] = vr_low
                    #group_result.iloc[start_index:i, group_result.columns.get_loc('vr_high')] = vr_high
                    group_result.iloc[start_index:i, group_result.columns.get_loc('return')] = cumulative_return
                    group_result.iloc[start_index:i, group_result.columns.get_loc('period_length')] = period_length

                if current_bin is not None:
                    start_index = i # должно указывать на начало предыдущего периода !!!
                    start_pred = start_index - short_period + 1
                    cumulative_return = 0.0
                current_bin = new_bin
            else:
                cumulative_return = short_return
                i += 1

        if current_bin is not None:
            #vr_low = group['low'].iloc[start_index:].min() / group['close'].iloc[start_index] - 1
            #vr_high = group['high'].iloc[start_index:].max() / group['close'].iloc[start_index] - 1
            period_length = len(group) - start_index
            group_result.iloc[start_index:, group_result.columns.get_loc('bin')] = current_bin
            #group_result.iloc[start_index:, group_result.columns.get_loc('vr_low')] = vr_low
            #group_result.iloc[start_index:, group_result.columns.get_loc('vr_high')] = vr_high
            group_result.iloc[start_index:, group_result.columns.get_loc('return')] = cumulative_return
            group_result.iloc[start_index:, group_result.columns.get_loc('period_length')] = period_length

        result_list.append(group_result.reset_index())

    final_result = pd.concat(result_list, ignore_index=True)
    return final_result


data = pd.DataFrame({'close': [1, 3, 4, 1, 8, 9, 10, 11, 12]})
print(short_long_box(data, short_period=3, long_period=5, threshold=1))
