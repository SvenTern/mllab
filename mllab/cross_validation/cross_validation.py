"""
Implements Сhapter 7 of AFML on Cross Validation for financial data.

Also Stacked Purged K-Fold cross validation and Stacked ml cross val score. These functions are used
for multi-asset datasets.
"""
# pylint: disable=too-many-locals, invalid-name, comparison-with-callable

from typing import Callable
import pandas as pd
import numpy as np
from joblib import Parallel, delayed

from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.base import ClassifierMixin, clone
from sklearn.model_selection import BaseCrossValidator
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve, auc
import matplotlib.pyplot as plt


def ml_get_train_times(samples_info_sets: pd.Series, test_times: pd.Series) -> pd.Series:
    """
    Advances in Financial Machine Learning, Snippet 7.1, page 106.

    Purging observations in the training set.

    This function find the training set indexes given the information on which each record is based
    and the range for the test set.
    Given test_times, find the times of the training observations.

    :param samples_info_sets: (pd.Series) The information range on which each record is constructed from
        *samples_info_sets.index*: Time when the information extraction started.
        *samples_info_sets.value*: Time when the information extraction ended.
    :param test_times: (pd.Series) Times for the test dataset.
    :return: (pd.Series) Training set.
    """

    pass


class PurgedKFold(KFold):
    """
    Extend KFold class to work with labels that span intervals.

    The train is purged of observations overlapping test-label intervals.
    Test set is assumed contiguous (shuffle=False), w/o training samples in between.
    """

    def __init__(self,
                 n_splits: int = 3,
                 samples_info_sets: pd.Series = None,
                 pct_embargo: float = 0.):
        """
        Initialize.

        :param n_splits: (int) The number of splits. Default to 3
        :param samples_info_sets: (pd.Series) The information range on which each record is constructed from
            *samples_info_sets.index*: Time when the information extraction started.
            *samples_info_sets.value*: Time when the information extraction ended.
        :param pct_embargo: (float) Percent that determines the embargo size.
        """

        pass

    def split(self,
              X: pd.DataFrame,
              y: pd.Series = None,
              groups=None) -> tuple:
        """
        The main method to call for the PurgedKFold class.

        :param X: (pd.DataFrame) Samples dataset that is to be split.
        :param y: (pd.Series) Sample labels series.
        :param groups: (array-like), with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        :return: (tuple) [train list of sample indices, and test list of sample indices].
        """

        pass


def ml_cross_val_score(
        classifier: ClassifierMixin,
        X: pd.DataFrame,
        y: pd.Series,
        cv_gen: BaseCrossValidator,
        sample_weight_train: np.ndarray = None,
        sample_weight_score: np.ndarray = None,
        scoring: Callable[[np.array, np.array], float] = log_loss,
        require_proba: bool = True,
        n_jobs_score: int = 1) -> np.array:
    """
    Advances in Financial Machine Learning, Snippet 7.4, page 110.
    Using the PurgedKFold Class.

    Function to run a cross-validation evaluation of the using sample weights and a custom CV generator.
    Note: This function is different to the book in that it requires the user to pass through a CV object. The book
    will accept a None value as a default and then resort to using PurgedCV, this also meant that extra arguments had to
    be passed to the function. To correct this we have removed the default and require the user to pass a CV object to
    the function.

    Example:

    .. code-block:: python

        cv_gen = PurgedKFold(n_splits=n_splits, samples_info_sets=samples_info_sets,
                             pct_embargo=pct_embargo)

        scores_array = ml_cross_val_score(classifier, X, y, cv_gen, sample_weight_train=sample_train,
                                          sample_weight_score=sample_score, scoring=accuracy_score)

    :param classifier: (ClassifierMixin) A sk-learn Classifier object instance.
    :param X: (pd.DataFrame) The dataset of records to evaluate.
    :param y: (pd.Series) The labels corresponding to the X dataset.
    :param cv_gen: (BaseCrossValidator) Cross Validation generator object instance.
    :param sample_weight_train: (np.array) Sample weights used to train the model for each record in the dataset.
    :param sample_weight_score: (np.array) Sample weights used to evaluate the model quality.
    :param scoring: (Callable) A metric scoring, can be custom sklearn metric.
    :param require_proba: (bool) Boolean flag indicating that scoring function requires probabilities.
    :param n_jobs_score: (int) Number of cores used in score function calculation.
    :return: (np.array) The computed score.
    """

    pass

def _score_model(
        classifier: ClassifierMixin,
        X: pd.DataFrame,
        y: pd.Series,
        train,
        test,
        sample_weight_train: np.ndarray,
        sample_weight_score: np.ndarray,
        scoring: Callable[[np.array, np.array], float],
        require_proba: bool) -> np.array:
    """
    Helper function used in multi-core ml_cross_val_score.

    :param classifier: (ClassifierMixin) A sk-learn Classifier object instance.
    :param X: (pd.DataFrame) The dataset of records to evaluate.
    :param y: (pd.Series) The labels corresponding to the X dataset.
    :param sample_weight_train: (np.array) Sample weights used to train the model for each record in the dataset.
    :param sample_weight_score: (np.array) Sample weights used to evaluate the model quality.
    :param scoring: (Callable) A metric scoring, can be custom sklearn metric.
    :param require_proba: (bool) Boolean flag indicating that scoring function requires probabilities.
    :return: (np.array) The computed score.
    """

    pass


class StackedPurgedKFold(KFold):
    """
    Extend KFold class to work with labels that span intervals in multi-asset datasets.

    The train is purged of observations overlapping test-label intervals.
    Test set is assumed contiguous (shuffle=False), w/o training samples in between.
    """

    def __init__(self,
                 n_splits: int = 3,
                 samples_info_sets_dict: dict = None,
                 pct_embargo: float = 0.):
        """
        Initialize.

        :param n_splits: (int) The number of splits. Default to 3.
        :param samples_info_sets_dict: (dict) Dictionary of asset: the information range on which each record is
            *constructed from samples_info_sets.index*: Time when the information extraction started.
            *samples_info_sets.value*: Time when the information extraction ended.
        :param pct_embargo: (float) Percent that determines the embargo size.
        """

        pass

    def split(self,
              X_dict: dict,
              y_dict: dict = None,
              groups=None) -> dict:
        # pylint: disable=arguments-differ, unused-argument
        """
        The main method to call for the StackedPurgedKFold class.

        :param X_dict: (dict) Dictionary of asset: X(features).
        :param y_dict: (dict) Dictionary of asset: y(features).
        :param groups: (array-like), with shape (n_samples,), optional
            Group labels for the samples used while splitting the dataset into
            train/test set.
        :return: (dict) Dictionary of asset: [train list of sample indices, and test list of sample indices].
        """

        pass


def stacked_ml_cross_val_score(
        classifier: ClassifierMixin,
        X_dict: dict,
        y_dict: dict,
        cv_gen: BaseCrossValidator,
        sample_weight_train_dict: dict = None,
        sample_weight_score_dict: dict = None,
        scoring: Callable[[np.array, np.array], float] = log_loss,
        require_proba: bool = True,
        n_jobs_score: int = 1) -> np.array:
    """
    Implements ml_cross_val_score (mllab.cross_validation.ml_cross_val_score) for multi-asset dataset.

    Function to run a cross-validation evaluation of the using sample weights and a custom CV generator.
    Note: This function is different to the book in that it requires the user to pass through a CV object. The book
    will accept a None value as a default and then resort to using PurgedCV, this also meant that extra arguments had to
    be passed to the function. To correct this we have removed the default and require the user to pass a CV object to
    the function.

    Example:

    .. code-block:: python

        cv_gen = PurgedKFold(n_splits=n_splits, samples_info_sets=samples_info_sets,
                             pct_embargo=pct_embargo)

        scores_array = ml_cross_val_score(classifier, X, y, cv_gen, sample_weight_train=sample_train,
                                          sample_weight_score=sample_score, scoring=accuracy_score)

    :param classifier: (ClassifierMixin) A sk-learn Classifier object instance.
    :param X_dict: (dict) Dictionary of asset : X_{asset}.
    :param y_dict: (dict) Dictionary of asset : y_{asset}
    :param cv_gen: (BaseCrossValidator) Cross Validation generator object instance.
    :param sample_weight_train_dict: Dictionary of asset: sample_weights_train_{asset}
    :param sample_weight_score_dict: Dictionary of asset: sample_weights_score_{asset}
    :param scoring: (Callable) A metric scoring, can be custom sklearn metric.
    :param require_proba: (bool) Boolean flag indicating that scoring function requires probabilities.
    :param n_jobs_score: (int) Number of cores used in score function calculation.
    :return: (np.array) The computed score.
    """

    pass


def stacked_dataset_from_dict(X_dict: dict, y_dict: dict, sample_weights_train_dict: dict,
                              sample_weights_score_dict: dict,
                              index: dict) -> tuple:
    """
    Helper function used to create appended dataset (X, y, sample weights train/score) using dictionary of train/test
    indices.

    :param X_dict: (dict) Dictionary of asset : X_{asset}.
    :param y_dict: (dict) Dictionary of asset : y_{asset}
    :param sample_weights_train_dict: Dictionary of asset: sample_weights_train_{asset}
    :param sample_weights_score_dict: Dictionary of asset: sample_weights_score_{asset}
    :param index: (dict) Dictionary of asset: int indices (it may be either train or test indices).
    :return: (tuple) Tuple of appended datasets: X, y, sample weights train, sample_weights score.
    """

    pass


def _stacked_score_model(classifier, X_dict, y_dict, train, test, sample_weight_train_dict, sample_weight_score_dict,
                         scoring, require_proba):
    """
    Helper function used in multi-core ml_cross_val_score.

    :param classifier: (ClassifierMixin) A sk-learn Classifier object instance.
    :param X_dict: (dict) Dictionary of asset : X_{asset}.
    :param y_dict: (dict) Dictionary of asset : y_{asset}
    :param sample_weight_train_dict: (np.array) Sample weights used to train the model for each record in the dataset.
    :param sample_weight_score_dict: (np.array) Sample weights used to evaluate the model quality.
    :param scoring: (Callable) A metric scoring, can be custom sklearn metric.
    :param require_proba: (bool) Boolean flag indicating that scoring function requires probabilities.
    :return: (np.array) The computed score.
    """

    pass


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np


def plot_roc_multiclass(actual, prediction):
    """
    Calculate and plot the Receiver Operating Characteristic (ROC) curve and
    the Area Under the Curve (AUC) for a multi-class classification problem.

    Parameters:
        actual (array-like): Ground truth values (-1, 0, or 1).
        prediction (array-like): Predicted scores or probabilities as a 1D array corresponding to the actual values.

    Returns:
        None: Displays the ROC plot and prints average AUC.
    """
    # Ensure prediction values are non-negative by flipping the sign of negative values
    prediction = np.abs(prediction)

    # Classes to evaluate (-1 and 1 only, skipping 0 in this example)
    classes = [-1, 1, 0]
    roc_aucs = {}

    plt.figure(figsize=(8, 6))

    for cls in classes:
        # Binarize the actual values for the current class
        actual_bin = (actual == cls).astype(int)

        # Predicted probabilities/scores for the current class
        pred_class = prediction

        # Compute ROC curve and AUC
        fpr, tpr, _ = roc_curve(actual_bin, pred_class, pos_label=1)
        roc_auc = auc(fpr, tpr)
        roc_aucs[cls] = roc_auc

        # Plot the ROC curve for the current class
        plt.plot(fpr, tpr, lw=2, label=f'Class {cls} ROC (AUC = {roc_auc:.2f})')

    # Plot a diagonal line for random guessing
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guessing')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Multi-class)')
    plt.legend(loc="lower right")
    plt.show()

    # Print average AUC
    avg_roc_auc = np.mean(list(roc_aucs.values()))
    print(f"Average AUC (Classes {classes}): {avg_roc_auc:.2f}")

def cost_metric_3class(confusion_matrix: np.ndarray):
    """
    Функция рассчитывает:
      1) Общую стоимость (Cost) классификации по заданной confusion matrix.
      2) Максимально возможную стоимость (Cost_max).
      3) Итоговый cost-based accuracy = 1 - Cost / Cost_max.

    Порядок классов:
      - индекс 0 = класс -1
      - индекс 1 = класс  0
      - индекс 2 = класс  1

    Параметры:
    ----------
    confusion_matrix : np.ndarray (shape=(3,3))
        Матрица неточностей по классам, где
        confusion_matrix[i, j] = количество случаев,
        когда истинный класс i предсказан как j.

    Возвращает:
    -----------
    (cost, cost_max, cost_based_accuracy) : (float, float, float)
    """

    # Задаём матрицу стоимостей C(i->j).
    # Индексы [0 -> -1, 1 -> 0, 2 -> 1].
    # cost_matrix[i, j] = штраф за предсказание класса j,
    # когда истинный класс i.
    cost_matrix = np.array([
        [0, 1, 3],  # Реальный класс -1
        [1, 0, 1],  # Реальный класс  0
        [3, 1, 0]   # Реальный класс  1
    ])

    cost_max = np.array([
        [1, 1, 3],  # Реальный класс -1
        [1, 0, 1],  # Реальный класс  0
        [3, 1, 1]  # Реальный класс  1
    ])

    # 1) Считаем фактическую суммарную стоимость (Cost)
    cost = 0.0
    for i in range(3):
        for j in range(3):
            cost += confusion_matrix[i, j] * cost_matrix[i, j]

    cost_max = 0.0
    for i in range(3):
        for j in range(3):
            cost_max += confusion_matrix[i, j] * cost_max[i, j]

    # 2) Считаем максимально возможную стоимость (Cost_max).
    #    Предположим, что "худшая" ошибка для каждого класса i —
    #    это предсказывать тот класс j, который даёт максимальный штраф cost_matrix[i, j].
    #cost_max = 0.0
    #for i in range(3):
    #    row_sum = np.sum(confusion_matrix[i, :])   # общее кол-во примеров реального класса i
    #    # Максимальный штраф в строке i (не обязательно на позиции j!=i,
    #    # ведь cost_matrix[i,i] = 0, а нам нужен худший вариант).
    #    costliest = np.max(cost_matrix[i, :])
    #    cost_max += row_sum * costliest

    # 3) Cost-based Accuracy
    #    Если cost_max = 0 (например, пустая матрица), чтобы избежать деления на ноль,
    #    зададим результат 0.0 или 1.0 (по ситуации).
    if cost_max > 0:
        cost_based_accuracy = 1 - (cost / cost_max)
    else:
        cost_based_accuracy = 0.0

    return cost_based_accuracy

def score_confusion_matrix(y_test, y_pred, y_pred_auc = None):
    """
    вывод точности предсказания модели классификации
    """
    # Расчет матрицы ошибок
    cm = confusion_matrix(y_test, y_pred)

    # Определение уникальных классов
    unique_classes = sorted(set(y_test) | set(y_pred))
    print("Уникальные классы:", unique_classes)

    # Оценка качества модели
    print("\nClassification Report:")
    # Преобразование уникальных классов в строки (если требуется)
    class_labels = [str(cls) for cls in unique_classes]
    report = classification_report(y_test, y_pred, target_names=class_labels, output_dict=True)
    print(report)

    # Визуализация Classification Report
    metrics = ['precision', 'recall', 'f1-score']
    x = np.arange(len(unique_classes))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, metric in enumerate(metrics):
        values = [report[cls][metric] for cls in class_labels]
        ax.bar(x + i * width, values, width, label=metric)

    ax.set_xticks(x + width)
    ax.set_xticklabels(class_labels)
    ax.set_ylabel('Score')
    ax.set_title('Classification Report Metrics by Class')
    ax.legend()

    plt.show()

    # Визуализация конфьюжен матрицы
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()

    if y_pred_auc is None:
        plot_roc_multiclass(y_test, y_pred)
    else:
        plot_roc_multiclass(y_test, y_pred_auc)

    cost_metric = cost_metric_3class(cm)
    print(f"Cost metric: {cost_metric:.2f}")

    return cost_metric
