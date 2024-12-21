import numpy as np


def encode_tick_rule_array(tick_rule_array: list) -> str:
    """
    Encode array of tick signs (-1, 1, 0)

    :param tick_rule_array: (list) Tick rules
    :return: (str) Encoded message
    """
    ascii_table = _get_ascii_table()
    tick_to_ascii_map = {-1: ascii_table[0], 0: ascii_table[1], 1: ascii_table[2]}
    return ''.join(tick_to_ascii_map.get(tick, '?') for tick in tick_rule_array)


def _get_ascii_table() -> list:
    """
    Get all ASCII symbols

    :return: (list) ASCII symbols
    """
    return [chr(i) for i in range(32, 127)]  # Printable ASCII characters


def quantile_mapping(array: list, num_letters: int = 26) -> dict:
    """
    Generate dictionary of quantile-letters based on values from array and dictionary length (num_letters).

    :param array: (list) Values to split on quantiles
    :param num_letters: (int) Number of letters (quantiles) to encode
    :return: (dict) Dict of quantile-symbol
    """
    if not array:
        raise ValueError("Input array is empty")
    ascii_table = _get_ascii_table()
    letters = ascii_table[:num_letters]
    quantiles = np.linspace(0, 1, num_letters + 1)
    thresholds = np.quantile(array, quantiles)
    return {value: letters[idx] for idx, value in enumerate(thresholds[:-1])}


def sigma_mapping(array: list, step: float = 0.01) -> dict:
    """
    Generate dictionary of sigma encoded letters based on values from array and discretization step.

    :param array: (list) Values to split on quantiles
    :param step: (float) Discretization step (sigma)
    :return: (dict) Dict of value-symbol
    """
    if not array:
        raise ValueError("Input array is empty")
    ascii_table = _get_ascii_table()
    letters = ascii_table
    mean = np.mean(array)
    std_dev = np.std(array)
    thresholds = np.arange(mean - 3 * std_dev, mean + 3 * std_dev, step * std_dev)
    return {threshold: letters[idx % len(letters)] for idx, threshold in enumerate(thresholds)}


def _find_nearest(array: list, value: float) -> float:
    """
    Find the nearest element from array to value.

    :param array: (list) Values
    :param value: (float) Value for which the nearest element needs to be found
    :return: (float) The nearest to the value element in array
    """
    array = np.array(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def _get_letter_from_encoding(value: float, encoding_dict: dict) -> str:
    """
    Get letter for float/int value from encoding dict.

    :param value: (float/int) Value to use
    :param encoding_dict: (dict) Used dictionary
    :return: (str) Letter from encoding dict
    """
    keys = np.array(list(encoding_dict.keys()))
    nearest_key = _find_nearest(keys, value)
    return encoding_dict[nearest_key]


def encode_array(array: list, encoding_dict: dict) -> str:
    """
    Encode array with strings using encoding dict, in case of multiple occurrences of the minimum values,
    the indices corresponding to the first occurrence are returned

    :param array: (list) Values to encode
    :param encoding_dict: (dict) Dict of quantile-symbol
    :return: (str) Encoded message
    """
    return ''.join(_get_letter_from_encoding(value, encoding_dict) for value in array)
