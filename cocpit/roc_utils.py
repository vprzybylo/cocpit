import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from cocpit import config as config
from sklearn.metrics import confusion_matrix, auc
from matplotlib.colors import BoundaryNorm, ListedColormap
from numpy import float64, int64, ndarray
from typing import Dict, List, Tuple


def contingency_table(cnf_matrix: ndarray, c: int) -> Dict[str, int64]:
    """
    Multi-class contingency table.
    A miss is anything outside of the target class.

    Args:
        cnf_matrix (ndarray): sklearn.metrics confusion matrix (labels, y_preds_bi)
        c (int): class id

    Returns:
        table (Dict[str, int64]): contingency table for a specified class

    """
    TP = np.diag(cnf_matrix)
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)

    return {
        "num_true_positives": TP[c],
        "num_false_positives": FP[c],
        "num_false_negatives": FN[c],
        "num_true_negatives": TN[c],
    }


def probability_of_detection(table: Dict[str, int64]) -> float64:
    """
    Computes probability of detection (POD).

    Args:
        table (Dict[str, int64]): see contingency_table()

    Returns:
        probability_of_detection (float64): tp/(tp+fn)
    """
    return float(table["num_true_positives"]) / (
        table["num_true_positives"] + table["num_false_negatives"]
    )


def success_ratio(table: Dict[str, int64]) -> float64:
    """
    Computes success ratio.

    Args:
        table (Dict[str, int64]): see contingency_table()

    Returns:
        success_ratio (float64): tp/(tp+fp)
    """
    return float(table["num_true_positives"]) / (
        table["num_true_positives"] + table["num_false_positives"]
    )


def csi_from_sr_and_pod(success_ratio_array: ndarray, pod_array: ndarray) -> ndarray:
    """
    Computes CSI (critical success index) from success ratio and probability of detection

    Args:
        success_ratio_array (ndrray): Any length of success ratios
        pod_array (ndrray): array of POD values same length as success ratios
    Returns:
        csi_array (ndarry): critcal success index length of success ratio
    """
    return (success_ratio_array**-1 + pod_array**-1 - 1.0) ** -1


def probability_of_false_detection(table: Dict[str, int64]) -> float64:
    """
    Computes POFD (probability of false detection).

    Args:
        table (Dict[str, int64]): see contingency_table()
    Returns:
        probability_of_false_detection (float64): fp/(fp+tn)
    """

    return float(table["num_false_positives"]) / (
        table["num_false_positives"] + table["num_true_negatives"]
    )


def _pofd_pod_grid(
    pofd_spacing: float = 0.01, pod_spacing: float = 0.01
) -> List[ndarray]:
    """
    Creates grid in POFD-POD space.
    M = number of rows (unique POD values) in grid
    N = number of columns (unique POFD values) in grid

    Args:
        pofd_spacing (float): Spacing between grid cells in adjacent columns.
        pod_spacing (float): Spacing between grid cells in adjacent rows.

    Returns:
        pofd_matrix (ndarray): M-by-N numpy array of POFD values.
        pod_matrix (ndarray): M-by-N numpy array of POD values.
    """

    num_pofd_values = 1 + int(np.ceil(1.0 / pofd_spacing))
    num_pod_values = 1 + int(np.ceil(1.0 / pod_spacing))

    unique_pofd_values = np.linspace(0.0, 1.0, num=num_pofd_values)
    unique_pod_values = np.linspace(0.0, 1.0, num=num_pod_values)[::-1]
    return np.meshgrid(unique_pofd_values, unique_pod_values)


def get_area_under_roc_curve(pofd_by_threshold, pod_by_threshold):
    """
    Computes area under ROC curve.
    This calculation ignores NaN's.  If you use `sklearn.metrics.auc` without
    this wrapper, if either input array contains any NaN, the result will be NaN.

    Args:
        pofd_by_threshold (ndarray(len(threshs))): mean probability of false detection for a class across all thresholds
        pod_by_threshold (ndarray(len(threshs))): mean probability of detection for a class across all thresholds

    Returns:
        area_under_curve (float)
    """

    sort_indices = np.argsort(-pofd_by_threshold)
    pofd_by_threshold = pofd_by_threshold[sort_indices]
    pod_by_threshold = pod_by_threshold[sort_indices]

    nan_flags = np.logical_or(np.isnan(pofd_by_threshold), np.isnan(pod_by_threshold))
    if np.all(nan_flags):
        return np.nan

    real_indices = np.where(np.invert(nan_flags))[0]

    return auc(pofd_by_threshold[real_indices], pod_by_threshold[real_indices])


def frequency_bias_from_sr_and_pod(
    success_ratio_array: ndarray, pod_array: ndarray
) -> ndarray:
    """
    Computes frequency bias from success ratio and probability of detection.
    The frequency bias is the ratio of the frequency of ‘‘yes’’ forecasts to the frequency of ‘‘yes’’ observations.

    Args:
        success_ratio_array (ndarray): array of any number of success ratios.
        pod_array (ndarray): array of pod values (same length as success_ratio_array).

    Returns:
        frequency_bias_array (ndarray): array of frequency biases on a meshgrid.
    """
    return pod_array / success_ratio_array
