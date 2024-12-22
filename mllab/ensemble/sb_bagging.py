"""
Implementation of Sequentially Bootstrapped Bagging Classifier using sklearn's library as base class
"""

import numpy as np
from sklearn.utils import check_random_state
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.ensemble._bagging import BaseBagging
from sklearn.base import ClassifierMixin, RegressorMixin
from sklearn.utils import check_X_y
from sklearn.utils._joblib import Parallel, delayed
from mllab.sampling.bootstrapping import seq_bootstrap, get_ind_matrix
from abc import ABC, abstractmethod

MAX_INT = np.iinfo(np.int32).max

def indices_to_mask(indices, mask_length):
    """
    Converts indices to a boolean mask.

    :param indices: (array-like) Indices to set to True in the mask.
    :param mask_length: (int) Length of the mask.
    :return: (np.ndarray) Boolean mask of length `mask_length`.
    """
    mask = np.zeros(mask_length, dtype=bool)
    mask[indices] = True
    return mask

def _generate_random_features(random_state, bootstrap, n_population, n_samples):
    """
    Draw randomly sampled indices for feature selection.

    :param random_state: (np.random.RandomState) Random state for reproducibility.
    :param bootstrap: (bool) Whether sampling is with replacement.
    :param n_population: (int) Total population size (e.g., number of features).
    :param n_samples: (int) Number of samples to draw.
    :return: (np.ndarray) Randomly sampled indices.
    """
    if bootstrap:
        return random_state.randint(0, n_population, n_samples)
    else:
        return np.random.choice(np.arange(n_population), size=n_samples, replace=False)

def _generate_bagging_indices(random_state, bootstrap_features, n_features, max_features, max_samples, ind_mat):
    """
    Randomly draw feature and sample indices for bagging.

    :param random_state: (np.random.RandomState) Random state for reproducibility.
    :param bootstrap_features: (bool) Whether features are drawn with replacement.
    :param n_features: (int) Number of features in the dataset.
    :param max_features: (int) Maximum number of features to draw.
    :param max_samples: (int) Maximum number of samples to draw.
    :param ind_mat: (np.ndarray) Indicator matrix for sequential bootstrapping.
    :return: (tuple) Indices for features and samples.
    """
    # Feature indices
    feature_indices = _generate_random_features(
        random_state=random_state,
        bootstrap=bootstrap_features,
        n_population=n_features,
        n_samples=max_features
    )

    # Sample indices using sequential bootstrapping
    sample_indices = seq_bootstrap(ind_mat, sample_length=max_samples, random_state=random_state)
    return feature_indices, sample_indices

def _parallel_build_estimators(n_estimators, ensemble, X, y, ind_mat, sample_weight,
                               seeds, total_n_estimators, verbose):
    """
    Private function used to build a batch of estimators within a job.

    :param n_estimators: (int) Number of estimators to build.
    :param ensemble: (BaseBagging) The ensemble object.
    :param X: (array-like) Feature matrix.
    :param y: (array-like) Target labels.
    :param ind_mat: (np.ndarray) Indicator matrix for sequential bootstrapping.
    :param sample_weight: (array-like or None) Sample weights.
    :param seeds: (array-like) Random seeds for each estimator.
    :param total_n_estimators: (int) Total number of estimators in the ensemble.
    :param verbose: (int) Verbosity level.
    :return: (list) Trained estimators and additional information.
    """
    estimators = []
    estimators_features = []
    estimators_samples = []

    for i in range(n_estimators):
        if verbose > 1:
            print(f"Building estimator {i + 1}/{total_n_estimators}")

        # Random state for the estimator
        random_state = check_random_state(seeds[i])

        # Generate bagging indices
        feature_indices, sample_indices = _generate_bagging_indices(
            random_state=random_state,
            bootstrap_features=ensemble.bootstrap_features,
            n_features=X.shape[1],
            max_features=ensemble.max_features_,
            max_samples=ensemble.max_samples_,
            ind_mat=ind_mat
        )

        # Train the estimator
        estimator = ensemble._make_estimator(append=False, random_state=random_state)
        estimator.fit(X[sample_indices][:, feature_indices], y[sample_indices])

        estimators.append(estimator)
        estimators_samples.append(sample_indices)
        estimators_features.append(feature_indices)

    return estimators, estimators_samples, estimators_features

class SequentiallyBootstrappedBaseBagging(BaseBagging, ABC):
    """
    Base class for Sequentially Bootstrapped Bagging Classifier and Regressor.

    Extends sklearn's BaseBagging to incorporate sequential bootstrapping.
    """

    @abstractmethod
    def __init__(self,
                 samples_info_sets,
                 price_bars,
                 estimator=None,
                 n_estimators=10,
                 max_samples=1.0,
                 max_features=1.0,
                 bootstrap_features=False,
                 oob_score=False,
                 warm_start=False,
                 n_jobs=None,
                 random_state=None,
                 verbose=0):
        super().__init__(
            estimator=estimator,
            n_estimators=n_estimators,
            bootstrap=True,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap_features=bootstrap_features,
            oob_score=oob_score,
            warm_start=warm_start,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=verbose)

        # Custom parameters for sequential bootstrapping
        self.samples_info_sets = samples_info_sets
        self.price_bars = price_bars
        self.ind_mat = get_ind_matrix(samples_info_sets, price_bars)

class SequentiallyBootstrappedBaggingClassifier(SequentiallyBootstrappedBaseBagging, BaggingClassifier, ClassifierMixin):
    """
    A Sequentially Bootstrapped Bagging Classifier.
    """

    def __init__(self, samples_info_sets, price_bars, **kwargs):
        super().__init__(samples_info_sets=samples_info_sets, price_bars=price_bars, **kwargs)

    def fit(self, X, y, sample_weight=None):
        """Fit the model."""
        X, y = check_X_y(X, y)
        return super().fit(X, y, sample_weight=sample_weight)

class SequentiallyBootstrappedBaggingRegressor(SequentiallyBootstrappedBaseBagging, BaggingRegressor, RegressorMixin):
    """
    A Sequentially Bootstrapped Bagging Regressor.
    """

    def __init__(self, samples_info_sets, price_bars, **kwargs):
        super().__init__(samples_info_sets=samples_info_sets, price_bars=price_bars, **kwargs)

    def fit(self, X, y, sample_weight=None):
        """Fit the model."""
        X, y = check_X_y(X, y, multi_output=True)
        return super().fit(X, y, sample_weight=sample_weight)
