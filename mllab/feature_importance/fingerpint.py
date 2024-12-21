"""
Implementation of an algorithm described in Yimou Li, David Turkington, Alireza Yazdani
'Beyond the Black Box: An Intuitive Approach to Investment Prediction with Machine Learning'
(https://jfds.pm-research.com/content/early/2019/12/11/jfds.2019.1.023)
"""

from abc import ABC, abstractmethod
from typing import Tuple, Dict
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# pylint: disable=invalid-name
# pylint: disable=too-many-locals

class AbstractModelFingerprint(ABC):
    """
    Model fingerprint constructor.

    This is an abstract base class for the RegressionModelFingerprint and ClassificationModelFingerprint classes.
    """

    def __init__(self):
        """
        Model fingerprint constructor.
        """
        pass

    def fit(self, model: object, X: pd.DataFrame, num_values: int = 50, pairwise_combinations: list = None) -> None:
        """
        Get linear, non-linear and pairwise effects estimation.

        :param model: (object) Trained model.
        :param X: (pd.DataFrame) Dataframe of features.
        :param num_values: (int) Number of values used to estimate feature effect.
        :param pairwise_combinations: (list) Tuples (feature_i, feature_j) to test pairwise effect.
        """
        self.X_values = self._get_feature_values(X, num_values)
        self.partial_dependence = self._get_individual_partial_dependence(model, X)
        self.linear_effect = self._get_linear_effect(X)
        self.non_linear_effect = self._get_non_linear_effect(X)
        if pairwise_combinations:
            self.pairwise_effect = self._get_pairwise_effect(pairwise_combinations, model, X, num_values)
        else:
            self.pairwise_effect = {}

    def get_effects(self) -> dict[str, dict[str, dict | None] | dict[str, dict] | dict[str, dict]]:
        """
        Return computed linear, non-linear and pairwise effects. The model should be fit() before using this method.

        :return: (tuple) Linear, non-linear and pairwise effects, of type dictionary (raw values and normalized).
        """
        return {
            "linear_effect": {
                "raw": self.linear_effect,
                "normalized": self._normalize(self.linear_effect)
            },
            "non_linear_effect": {
                "raw": self.non_linear_effect,
                "normalized": self._normalize(self.non_linear_effect)
            },
            "pairwise_effect": {
                "raw": self.pairwise_effect,
                "normalized": self._normalize(self.pairwise_effect) if self.pairwise_effect else None
            }
        }

    def plot_effects(self) -> plt.figure:
        """
        Plot each effect (normalized) on a bar plot (linear, non-linear). Also plots pairwise effects if calculated.

        :return: (plt.figure) Plot figure.
        """
        effects = self.get_effects()
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].bar(effects['linear_effect']['normalized'].keys(),
                    effects['linear_effect']['normalized'].values())
        axes[0].set_title('Linear Effects')

        axes[1].bar(effects['non_linear_effect']['normalized'].keys(),
                    effects['non_linear_effect']['normalized'].values())
        axes[1].set_title('Non-Linear Effects')

        if effects['pairwise_effect']['normalized']:
            axes[2].bar(effects['pairwise_effect']['normalized'].keys(),
                        effects['pairwise_effect']['normalized'].values())
            axes[2].set_title('Pairwise Effects')

        plt.tight_layout()
        return fig

    def _get_feature_values(self, X: pd.DataFrame, num_values: int) -> pd.DataFrame:
        """
        Step 1 of the algorithm which generates possible feature values used in analysis.

        :param X: (pd.DataFrame) Dataframe of features.
        :param num_values: (int) Number of values used to estimate feature effect.
        :return: (pd.DataFrame) Generated feature values.
        """
        feature_values = {
            column: np.linspace(X[column].min(), X[column].max(), num_values)
            for column in X.columns
        }
        return pd.DataFrame(feature_values)

    def _get_individual_partial_dependence(self, model: object, X: pd.DataFrame) -> dict:
        """
        Get individual partial dependence function values for each column.

        :param model: (object) Trained model.
        :param X: (pd.DataFrame) Dataframe of features.
        :return: (dict) Partial dependence for each feature.
        """
        partial_dependence = {}
        for column in X.columns:
            X_temp = X.copy()
            for value in self.X_values[column]:
                X_temp[column] = value
                partial_dependence.setdefault(column, []).append(self._get_model_predictions(model, X_temp).mean())
        return partial_dependence

    def _get_linear_effect(self, X: pd.DataFrame) -> dict:
        """
        Get linear effect estimates as the mean absolute deviation of the linear predictions around their average value.

        :param X: (pd.DataFrame) Dataframe of features.
        :return: (dict) Linear effect estimates for each feature column.
        """
        return {
            column: np.mean(np.abs(self.partial_dependence[column] - np.mean(self.partial_dependence[column])))
            for column in X.columns
        }

    def _get_non_linear_effect(self, X: pd.DataFrame) -> dict:
        """
        Get non-linear effect estimates as the mean absolute deviation of the total marginal (single variable)
        effect around its corresponding linear effect.

        :param X: (pd.DataFrame) Dataframe of features.
        :return: (dict) Non-linear effect estimates for each feature column.
        """
        linear_effect = self._get_linear_effect(X)
        return {
            column: np.mean(np.abs(self.partial_dependence[column] - linear_effect[column]))
            for column in X.columns
        }

    def _get_pairwise_effect(self, pairwise_combinations: list, model: object, X: pd.DataFrame, num_values: int) -> dict:
        """
        Get pairwise effect estimates as the de-meaned joint partial prediction of the two variables minus the de-meaned
        partial predictions of each variable independently.

        :param pairwise_combinations: (list) Tuples (feature_i, feature_j) to test pairwise effect.
        :param model: (object) Trained model.
        :param X: (pd.DataFrame) Dataframe of features.
        :param num_values: (int) Number of values used to estimate feature effect.
        :return: (dict) Raw and normalized pairwise effects.
        """
        pairwise_effect = {}
        for (feature_i, feature_j) in pairwise_combinations:
            values_i = self.X_values[feature_i]
            values_j = self.X_values[feature_j]

            pairwise_values = []
            for val_i in values_i:
                for val_j in values_j:
                    X_temp = X.copy()
                    X_temp[feature_i] = val_i
                    X_temp[feature_j] = val_j

                    pred = self._get_model_predictions(model, X_temp).mean()
                    pairwise_values.append(pred)

            mean_i = np.mean(self.partial_dependence[feature_i])
            mean_j = np.mean(self.partial_dependence[feature_j])
            pairwise_effect[(feature_i, feature_j)] = np.mean(pairwise_values) - mean_i - mean_j
        return pairwise_effect

    @abstractmethod
    def _get_model_predictions(self, model: object, X_: pd.DataFrame):
        """
        Get model predictions based on problem type (predict for regression, predict_proba for classification).

        :param model: (object) Trained model.
        :param X_: (np.array) Feature set.
        :return: (np.array) Predictions.
        """
        pass

    @staticmethod
    def _normalize(effect: dict) -> dict:
        """
        Normalize effect values (sum equals 1).

        :param effect: (dict) Effect values.
        :return: (dict) Normalized effect values.
        """
        total = sum(effect.values())
        return {key: value / total for key, value in effect.items()} if total > 0 else effect


class RegressionModelFingerprint(AbstractModelFingerprint):
    """
    Regression Fingerprint class used for regression type of models.
    """

    def __init__(self):
        """
        Regression model fingerprint constructor.
        """
        super().__init__()

    def _get_model_predictions(self, model, X_):
        """
        Abstract method _get_model_predictions implementation.

        :param model: (object) Trained model.
        :param X_: (np.array) Feature set.
        :return: (np.array) Predictions.
        """
        return model.predict(X_)


class ClassificationModelFingerprint(AbstractModelFingerprint):
    """
    Classification Fingerprint class used for classification type of models.
    """

    def __init__(self):
        """
        Classification model fingerprint constructor.
        """
        super().__init__()

    def _get_model_predictions(self, model, X_):
        """
        Abstract method _get_model_predictions implementation.

        :param model: (object) Trained model.
        :param X_: (np.array) Feature set.
        :return: (np.array) Predictions.
        """
        return model.predict_proba(X_)[:, 1]
