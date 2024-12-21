"""
Logic regarding return and time decay attribution for sample weights from chapter 4.
And stacked sample weights logic: return and time based sample weights for a multi-asset dataset.
"""

import numpy as np
import pandas as pd

from mllab.sampling.concurrent import (num_concurrent_events, get_av_uniqueness_from_triple_barrier)
from mllab.util.multiprocess import mp_pandas_obj

class SampleWeights:

    @staticmethod
    def _apply_weight_by_return(label_endtime, num_conc_events, close_series, molecule):
        """
        Advances in Financial Machine Learning, Snippet 4.10, page 69.

        Determination of Sample Weight by Absolute Return Attribution

        Derives sample weights based on concurrency and return. Works on a set of
        datetime index values (molecule). This allows the program to parallelize the processing.

        :param label_endtime: (pd.Series) Label endtime series (t1 for triple barrier events).
        :param num_conc_events: (pd.Series) Number of concurrent labels (output from num_concurrent_events function).
        :param close_series: (pd.Series) Close prices.
        :param molecule: (an array) A set of datetime index values for processing.
        :return: (pd.Series) Sample weights based on number return and concurrency for molecule.
        """
        weights = pd.Series(index=molecule)
        for dt in molecule:
            loc = slice(label_endtime.index.get_loc(dt), label_endtime.index.get_loc(label_endtime[dt]))
            returns = (close_series[label_endtime[dt]] / close_series[dt] - 1).abs()
            weights[dt] = returns / num_conc_events[loc].mean()
        return weights

    @staticmethod
    def get_weights_by_return(triple_barrier_events, close_series, num_threads=5, verbose=True):
        """
        Advances in Financial Machine Learning, Snippet 4.10(part 2), page 69.

        Determination of Sample Weight by Absolute Return Attribution

        This function is orchestrator for generating sample weights based on return using mp_pandas_obj.

        :param triple_barrier_events: (pd.DataFrame) Events from labeling.get_events().
        :param close_series: (pd.Series) Close prices.
        :param num_threads: (int) The number of threads concurrently used by the function.
        :param verbose: (bool) Flag to report progress on asynch jobs.
        :return: (pd.Series) Sample weights based on number return and concurrency.
        """
        num_conc_events = num_concurrent_events(triple_barrier_events["t1"], close_series.index)
        args = (triple_barrier_events["t1"], num_conc_events, close_series)
        weights = mp_pandas_obj(SampleWeights._apply_weight_by_return, pd.IndexSlice[:, "molecule"], args,
                                num_threads, verbose=verbose)
        return weights

    @staticmethod
    def get_weights_by_time_decay(triple_barrier_events, close_series, num_threads=5, decay=1, verbose=True):
        """
        Advances in Financial Machine Learning, Snippet 4.11, page 70.

        Implementation of Time Decay Factors.

        :param triple_barrier_events: (pd.DataFrame) Events from labeling.get_events().
        :param close_series: (pd.Series) Close prices.
        :param num_threads: (int) The number of threads concurrently used by the function.
        :param decay: (int) Decay factor
            - decay = 1 means there is no time decay;
            - 0 < decay < 1 means that weights decay linearly over time, but every observation still receives a strictly positive weight, regadless of how old;
            - decay = 0 means that weights converge linearly to zero, as they become older;
            - decay < 0 means that the oldest portion c of the observations receive zero weight (i.e they are erased from memory).
        :param verbose: (bool) Flag to report progress on asynch jobs.
        :return: (pd.Series) Sample weights based on time decay factors.
        """
        weights = pd.Series(index=triple_barrier_events.index)
        for idx, row in triple_barrier_events.iterrows():
            if decay > 0:
                weights[idx] = (1 - decay) + decay * (close_series[row["t1"]] / close_series[idx])
            elif decay == 0:
                weights[idx] = max(0, (close_series[row["t1"]] / close_series[idx] - 1))
            else:
                weights[idx] = 0
        return weights

    @staticmethod
    def get_stacked_weights_by_return(triple_barrier_events_dict: dict, close_series_dict: dict, num_threads: int = 5,
                                      verbose: bool = True) -> dict:
        """
        Get return based sample weights for multi-asset dataset. The function applies mlinlab's get_weights_by_return.
        function to multi-asset dataset.

        :param triple_barrier_events_dict: (dict) Dictionary of asset_name: triple barrier event series.
        :param close_series_dict: (dict) Dictionary of asset_name: close series used to form label events.
        :param num_threads: (int) Number of threads used to get sample weights.
        :param verbose: (bool) Flag to report progress on asynch jobs.
        :return: (dict) Dictionary of asset_name: sample weight series.
        """
        weights_dict = {}
        for asset, events in triple_barrier_events_dict.items():
            weights_dict[asset] = SampleWeights.get_weights_by_return(events, close_series_dict[asset],
                                                                      num_threads=num_threads, verbose=verbose)
        return weights_dict

    @staticmethod
    def get_stacked_weights_time_decay(triple_barrier_events_dict: dict, close_series_dict: dict, decay: int = 0.5,
                                       num_threads: int = 5,
                                       verbose: bool = True) -> dict:
        """
        Get return based sample weights for multi-asset dataset. The function applies mlinlab's get_weights_by_time_decay.
        function to multi-asset dataset.

        :param triple_barrier_events_dict: (dict) Dictionary of asset_name: triple barrier event series.
        :param close_series_dict: (dict) Dictionary of asset_name: close series used to form label events.
        :param decay: (int) Decay factor
            - decay = 1 means there is no time decay;
            - 0 < decay < 1 means that weights decay linearly over time, but every observation still receives a strictly positive weight, regadless of how old;
            - decay = 0 means that weights converge linearly to zero, as they become older;
            - decay < 0 means that the oldest portion c of the observations receive zero weight (i.e they are erased from memory).
        :param num_threads: (int) Number of threads used to get sample weights.
        :param verbose: (bool) Flag to report progress on asynch jobs.
        :return: (dict) Dictionary of asset_name: sample weight series.
        """
        weights_dict = {}
        for asset, events in triple_barrier_events_dict.items():
            weights_dict[asset] = SampleWeights.get_weights_by_time_decay(events, close_series_dict[asset], decay=decay,
                                                                         num_threads=num_threads, verbose=verbose)
        return weights_dict
