"""
Functions to compute model performance
Modified from Dr. Lagerquist and found originally in his gewitter repo (https://github.com/thunderhoser/GewitterGefahr).
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from cocpit import config as config
from cocpit import roc_utils as roc_utils
from sklearn.metrics import confusion_matrix, auc
from matplotlib.colors import BoundaryNorm, ListedColormap
from numpy import float64, int64, ndarray
from typing import Dict, List, Tuple
import matplotlib.patheffects as path_effects


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
    pofd_matrix, pod_matrix = roc_utils._pofd_pod_grid()
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


def csi_contour(ax: plt.Axes) -> None:
    """
    Add critical success index contour to performance diagram

    Args:
        ax (plt.Axes): axes to plot performance diagram
    """
    sr_array = np.linspace(0.001, 1, 200)
    pod_array = np.linspace(0.001, 1, 200)
    X, Y = np.meshgrid(sr_array, pod_array)
    csi_vals = roc_utils.csi_from_sr_and_pod(X, Y)
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
    fb = roc_utils.frequency_bias_from_sr_and_pod(X, Y)
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
    ax1: plt.Axes,
    pofds_mean: ndarray,
    pods_mean: ndarray,
    pods_std: ndarray,
    markerfacecolor: str,
    label: str,
    text_height: float,
) -> None:
    """
    Plot ROC curve for one class

    Args:
        ax1 (plt.Axes): subplot axis for ROC
        pofds_mean (ndarray(len(threshs))): mean probability of false detection for a class across all thresholds
        pods_mean (ndarray(len(threshs))): mean probability of detection for a class across all thresholds
        pods_std (ndarray(len(threshs))): std of probability of detection for a class across all thresholds
        markerfacecolor (str): color of line for class
        label (str): class name
        text_height (float): label distance above y-axis
    """
    ax1.plot(
        pofds_mean,
        pods_mean,
        "-",
        color=markerfacecolor,
        markerfacecolor="w",
        lw=2,
        label=label,
    )
    auc_val = np.round(roc_utils.get_area_under_roc_curve(pofds_mean, pods_mean), 2)
    print(f"AUC: {auc_val}")
    ax1.text(
        0.38,
        text_height,
        f"AUC = {auc_val}: ",
        color="k",
        fontsize=12,
    )
    ax1.text(
        0.65,
        text_height,
        f"{label}",
        color=markerfacecolor,
        fontsize=12,
    )
    ax1.arrow(0.5, 0.55, -0.1, 0.1, facecolor="k", zorder=2, width=0.007)
    ax1.text(0.42, 0.66, "Better", rotation=45, color="k", fontsize=16)
    ax1.fill_between(pofds_mean, pods_mean - pods_std, pods_mean + pods_std, alpha=0.6)
    ax1.set_xlabel("POFD (probability of false detection)")
    ax1.set_ylabel("POD (probability of detection)")
    ax1.set_title("ROC Curve")
    ax1.set_ylim([0.0, 1.0])
    ax1.set_xlim([0.0, 1.0])
    ax1.plot([0, 1], [0, 1], "k--", lw=1)


def plot_performance(
    ax2: plt.Axes,
    srs_mean: ndarray,
    pods_mean: ndarray,
    pods_std: ndarray,
    markerfacecolor: str,
    label: str,
) -> None:
    """
    Plot performance metrics (success ratio and probability of detection) for one class

    Args:
        ax2 (plt.Axes): subplot axis for performance diagram
        srs_mean (ndarray(len(threshs))): mean success ratio for a class across all thresholds
        pods_mean (ndarray(len(threshs))): mean probability of detection for a class across all thresholds
        pods_std (ndarray(len(threshs))): std of probability of detection for a class across all thresholds
        markerfacecolor (str): color of line for class
        label (str): class name
    """

    ax2.plot(
        srs_mean,
        pods_mean,
        "-",
        color=markerfacecolor,
        markerfacecolor="w",
        lw=2,
        label=label,
    )
    ax2.fill_between(srs_mean, pods_mean - pods_std, pods_mean + pods_std, alpha=0.6)
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
    return roc_utils.contingency_table(cm, c)


def performance_diagram(
    ax1: plt.Axes, ax2: plt.Axes, labels: ndarray, yhat_proba: ndarray
) -> None:
    """
    Plot a ROC curve and performance diagram.

    Args:
        ax1 (plt.Axes): subplot axis for ROC curve
        ax2 (plt.Axes): subplot axis for performance diagram
        labels (ndarray(len(folds), len(samples)): array of actual labels
        yhat_proba (ndarray(len(folds), len(samples), len(classes))): array of predicted probabilities in order of class number
    """

    threshs = np.linspace(0.0, 1.0, 100)
    pofds = np.zeros(((config.KFOLD + 1), len(config.CLASS_NAMES), len(threshs)))
    pods = np.zeros(((config.KFOLD + 1), len(config.CLASS_NAMES), len(threshs)))
    srs = np.zeros(((config.KFOLD + 1), len(config.CLASS_NAMES), len(threshs)))
    csi = np.zeros(((config.KFOLD + 1), len(config.CLASS_NAMES), len(threshs)))
    markerfacecolor = ["limegreen", "orange", "r"]
    text_height = [0.05, 0.12, 0.19]

    for c, _ in enumerate(config.CLASS_NAMES):
        for f in range(config.KFOLD + 1):
            y_preds = yhat_proba[f, :, c]

            for i, t in enumerate(threshs):
                table = table_at_threshold(labels[f, :], y_preds, c, t)
                # calculate pod, sr and csi
                pofds[f, c, i] = roc_utils.probability_of_false_detection(table)
                srs[f, c, i] = roc_utils.success_ratio(table)
                pods[f, c, i] = roc_utils.probability_of_detection(table)
                csi[f, c, i] = roc_utils.csi_from_sr_and_pod(
                    srs[f, c, i], pods[f, c, i]
                )

        plot_roc(
            ax1,
            np.mean(pofds[:, c, :], axis=0),
            np.mean(pods[:, c, :], axis=0),
            np.std(pods[:, c, :], axis=0),
            markerfacecolor[c],
            label=config.CLASS_NAMES[c],
            text_height=text_height[c],
        )
        plot_performance(
            ax2,
            np.mean(srs[:, c, :], axis=0),
            np.mean(pods[:, c, :], axis=0),
            np.std(pods[:, c, :], axis=0),
            markerfacecolor[c],
            label=config.CLASS_NAMES[c],
        )
