"""
Functions to compute model performance
Modified from Dr. Lagerquist and found originally in his gewitter repo (https://github.com/thunderhoser/GewitterGefahr).
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from cocpit import config as config
import tqdm
from sklearn.metrics import confusion_matrix
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


def _get_peirce_colour_scheme(
    peirce_levels: ndarray = np.linspace(0, 1, num=11, dtype=float)
) -> Tuple[ListedColormap, BoundaryNorm]:
    """
    Color scheme for Peirce score.

    Args:
        peirce_levels (ndarray): list of contour levels between 0 and 1

    Returns:
        colour_map_object (matplotlib.colors.ListedColormap): Colormap object generated from a list of colors.
        colour_norm_object (matplotlib.colors.BoundaryNorm): Color scheme for Peirce score and a colormap index based on discrete intervals
    """

    this_colour_map_object = plt.cm.Blues
    this_colour_norm_object = matplotlib.colors.BoundaryNorm(
        peirce_levels, this_colour_map_object.N
    )

    rgba_matrix = this_colour_map_object(this_colour_norm_object(peirce_levels))

    colour_list = [rgba_matrix[i, ..., :-1] for i in range(rgba_matrix.shape[0])]

    colour_map_object = matplotlib.colors.ListedColormap(colour_list)
    colour_map_object.set_under(np.array([1, 1, 1]))
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        peirce_levels, colour_map_object.N
    )

    return colour_map_object, colour_norm_object


def add_colour_bar(
    axes_object: plt.Axes,
    colour_map_object: matplotlib.pyplot.cm,
    values_to_colour: ndarray,
    min_colour_value: int,
    max_colour_value: int,
    colour_norm_object: matplotlib.colors.BoundaryNorm = None,
    orientation_string: str = "vertical",
    fraction_of_axis_length: float = 1.0,
):
    """Adds colour bar to existing axes.
        axes_object (matplotlib.axes._subplots.AxesSubplot): Existing axes
        colour_map_object (matplotlib.pyplot.cm): Color scheme as defined in _get_peirce_colour_scheme()
        values_to_colour (ndarray): numpy array of values to color
        min_colour_value (int): Minimum value in colour map.
        max_colour_value (int): Max value in colour map.
        colour_norm_object (matplotlib.colors.BoundaryNorm): Linearly normalizes data into the [0.0, 1.0] interval.
        orientation_string (str): Orientation of colour bar ("vertical" or "horizontal").
        fraction_of_axis_length (float): Fraction of axis length occupied by colour bar.

    Returns:
        colour_bar_object (matplotlib.pyplot.colorbar): Color bar
    """

    if colour_norm_object is None:
        colour_norm_object = matplotlib.colors.Normalize(
            vmin=min_colour_value, vmax=max_colour_value, clip=False
        )

    scalar_mappable_object = plt.cm.ScalarMappable(
        cmap=colour_map_object, norm=colour_norm_object
    )
    scalar_mappable_object.set_array(values_to_colour)

    padding = 0.075 if orientation_string == "horizontal" else 0.05
    colour_bar_object = plt.colorbar(
        ax=axes_object,
        mappable=scalar_mappable_object,
        orientation=orientation_string,
        pad=padding,
        shrink=fraction_of_axis_length,
    )

    colour_bar_object.ax.tick_params(labelsize=14)
    return colour_bar_object


def peirce_contour(
    ax: plt.Axes, peirce_levels: ndarray = np.linspace(0, 1, num=11, dtype=float)
) -> None:
    """
    Add peirce contour (pod-pofd) to ROC curve.

    Args:
        ax (plt.Axes): subplot axis for ROC curve
        peirce_levels (ndarray): contour levels from 0-1
    """
    pofd_matrix, pod_matrix = _pofd_pod_grid()
    peirce_score_matrix = pod_matrix - pofd_matrix

    colour_map_object, colour_norm_object = _get_peirce_colour_scheme()

    ax.contourf(
        pofd_matrix,
        pod_matrix,
        peirce_score_matrix,
        peirce_levels,
        cmap=colour_map_object,
        norm=colour_norm_object,
        vmin=0.0,
        vmax=1.0,
    )

    colour_bar_object = add_colour_bar(
        axes_object=ax,
        colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        values_to_colour=peirce_score_matrix,
        min_colour_value=0.0,
        max_colour_value=1.0,
        orientation_string="vertical",
    )

    colour_bar_object.set_label("Peirce score [POD-POFD]")


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


def csi_contour(ax: plt.Axes) -> None:
    """
    Add critical success index contour to performance diagram

    Args:
        ax (plt.Axes): axes to plot performance diagram
    """
    sr_array = np.linspace(0.001, 1, 200)
    pod_array = np.linspace(0.001, 1, 200)
    X, Y = np.meshgrid(sr_array, pod_array)
    csi_vals = csi_from_sr_and_pod(X, Y)
    pm = ax.contourf(X, Y, csi_vals, levels=np.arange(0, 1.1, 0.1), cmap="Blues")
    plt.colorbar(pm, ax=ax, label="CSI")


def plot_frequency_bias(ax: plt.Axes) -> None:
    """
    Add frequency bias dashed lines on performance diagram from meshgrid of success ratio and pod.

    Args:
        ax (plt.Axes): axes to plot performance diagram
    """
    sr_array = np.linspace(0.001, 1, 200)
    pod_array = np.linspace(0.001, 1, 200)
    X, Y = np.meshgrid(sr_array, pod_array)
    fb = frequency_bias_from_sr_and_pod(X, Y)
    bias = ax.contour(
        X,
        Y,
        fb,
        levels=[0.25, 0.5, 1, 1.5, 2, 3, 5],
        linestyles="--",
        colors="k",
    )
    plt.clabel(
        bias,
        inline=True,
        inline_spacing=5,
        fmt="%.2f",
        fontsize=10,
        colors="k",
    )


def plot_roc(
    ax1: plt.Axes, pofds: ndarray, pods: ndarray, markerfacecolor: str, label: str
) -> None:
    """
    Plot ROC curve for one class

    Args:
        ax1 (plt.Axes): subplot axis for ROC
        pofds (ndarray(len(threshs))): probability of false detection for a class across all thresholds
        pods (ndarray(len(threshs))): probability of detection for a class across all thresholds
        markerfacecolor (str): color of line for class
        label (str): class name
    """
    ax1.plot(
        pofds, pods, "-", color=markerfacecolor, markerfacecolor="w", lw=2, label=label
    )
    ax1.set_xlabel("POFD (probability of false detection)")
    ax1.set_ylabel("POD (probability of detection)")
    ax1.set_title("ROC Curve")
    ax1.set_ylim([0.0, 1.0])
    ax1.set_xlim([0.0, 1.0])
    ax1.plot([0, 1], [0, 1], "k--", lw=1)


def plot_performance(
    ax2: plt.Axes, srs: ndarray, pods: ndarray, markerfacecolor: str, label: str
) -> None:
    """
    Plot performance metrics (success ratio and probability of detection) for one class

    Args:
        ax2 (plt.Axes): subplot axis for performance diagram
        srs (ndarray(len(threshs))): success ratio for a class across all thresholds
        pods (ndarray(len(threshs))): probability of detection for a class across all thresholds
        markerfacecolor (str): color of line for class
        label (str): class name
    """
    ax2.plot(
        srs, pods, "-", color=markerfacecolor, markerfacecolor="w", lw=2, label=label
    )
    ax2.legend(loc="center left", bbox_to_anchor=(1.5, 0.5))
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.0])
    ax2.set_xlabel("Precision/Success Ratio")
    ax2.set_title("Performance Diagram")


def table_at_threshold(
    labels: ndarray, y_preds: ndarray, c: int, t: float64
) -> Dict[str, int64]:
    """
    Create a multiclass contingency table using a threshold value for when a prediciton is true.
    If a prediction for a given class is greater than the threshold value, assign the index to the class label

    Args:
        labels (ndarray(len(samples))): array of actual labels
        y_preds (ndarray(len(samples))): array of predicted probabilities for the given class
        c (int): class id
        t (float64): threshold value

    Returns:
        contingency table (Dict[str, int64]): contingency table after overriding label based on threshold for predicted probability
    """
    # make a dummy array full of a wrong prediction
    if c == 0:
        y_preds_bi = np.ones(len(y_preds), dtype=int)
    else:
        y_preds_bi = np.zeros(len(y_preds), dtype=int)
    # find where the prediction is greater than or equal to the threshold
    idx = np.where(y_preds >= t)
    # set those indices to class label
    y_preds_bi[idx] = c
    # get the contingency with overriden predicted labels
    cm = confusion_matrix(labels, y_preds_bi)
    return contingency_table(cm, c)


def performance_diagram(
    ax1: plt.Axes, ax2: plt.Axes, labels: ndarray, yhat_proba: ndarray
) -> None:
    """
    Plot a ROC curve and performance diagram.

    Args:
        ax1 (plt.Axes): subplot axis for ROC curve
        ax2 (plt.Axes): subplot axis for performance diagram
        labels (ndarray(len(samples)): array of actual labels
        yhat_proba (ndarray(len(samples), len(classes))): array of predicted probabilities in order of class number
    """

    threshs = np.linspace(0.0, 1.0, 100)
    pofds = np.zeros((len(config.CLASS_NAMES), len(threshs)))
    pods = np.zeros((len(config.CLASS_NAMES), len(threshs)))
    srs = np.zeros((len(config.CLASS_NAMES), len(threshs)))

    markerfacecolor = ["limegreen", "orange", "r"]
    for c, _ in enumerate(config.CLASS_NAMES):
        y_preds = yhat_proba[:, c]
        for i, t in enumerate(tqdm.tqdm(threshs)):
            table = table_at_threshold(labels, y_preds, c, t)
            # calculate pod, sr and csi
            pofds[c, i] = probability_of_false_detection(table)
            srs[c, i] = success_ratio(table)
            pods[c, i] = probability_of_detection(table)

        plot_roc(
            ax1,
            pofds[c, :],
            pods[c, :],
            markerfacecolor[c],
            label=config.CLASS_NAMES[c],
        )
        plot_performance(
            ax2, srs[c, :], pods[c, :], markerfacecolor[c], label=config.CLASS_NAMES[c]
        )
