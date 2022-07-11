"""
Functions to compute model performance
Extended from Dr. Lagerquist and found originally in his gewitter repo (https://github.com/thunderhoser/GewitterGefahr).
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from cocpit import config as config
import tqdm
from sklearn.metrics import confusion_matrix


def contingency_table(cnf_matrix, c):
    """
    Multi-class contingency table.
    A miss is anything outside of the target class.

    Args:

    Returns:
        contingency table for a specified class (dict)

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


def get_contingency_table(num_classes, id, ol, pl):
    wrong_labels = np.arange(0, num_classes)
    wrong_labels = wrong_labels[wrong_labels != id]
    tp = len(np.where(np.logical_and(ol == id, pl == id))[0])
    fp = len(
        np.where(
            np.logical_and(
                np.logical_or(ol == wrong_labels[0], ol == wrong_labels[1]), pl == id
            )
        )[0]
    )
    fn = len(
        np.where(
            np.logical_and(
                np.logical_or(pl == wrong_labels[0], pl == wrong_labels[1]), ol == id
            )
        )[0]
    )
    tn = sum(
        [
            len(
                np.where(np.logical_and(ol == wrong_labels[0], pl == wrong_labels[1]))[
                    0
                ]
            ),
            len(
                np.where(np.logical_and(ol == wrong_labels[0], pl == wrong_labels[0]))[
                    0
                ]
            ),
            len(
                np.where(np.logical_and(ol == wrong_labels[1], pl == wrong_labels[1]))[
                    0
                ]
            ),
            len(
                np.where(np.logical_and(ol == wrong_labels[1], pl == wrong_labels[0]))[
                    0
                ]
            ),
        ]
    )

    return {
        "num_true_positives": tp,
        "num_false_positives": fp,
        "num_false_negatives": fn,
        "num_true_negatives": tn,
    }


def probability_of_detection(contingency_table_as_dict):
    """Computes POD (probability of detection).
    Args:
        contingency_table_as_dict: Dictionary created by contingency_table.
    Returns:
        probability_of_detection (float)
    """

    denominator = (
        contingency_table_as_dict["num_true_positives"]
        + contingency_table_as_dict["num_false_negatives"]
    )

    if denominator == 0:
        return np.nan

    numerator = float(contingency_table_as_dict["num_true_positives"])
    return numerator / denominator


def success_ratio(contingency_table_as_dict):
    """Computes success ratio.
    :param contingency_table_as_dict: Dictionary created by
        contingency_table.
    :return: success_ratio: Success ratio.
    """

    denominator = (
        contingency_table_as_dict["num_true_positives"]
        + contingency_table_as_dict["num_false_positives"]
    )

    if denominator == 0:
        return np.nan

    # tp/(tp+fp) #performance
    # fp/(fp+tn)    # ROC
    numerator = float(contingency_table_as_dict["num_true_positives"])
    return numerator / denominator


def csi_from_sr_and_pod(success_ratio_array, pod_array):
    """Computes CSI (critical success index) from success ratio and POD.
    POD = probability of detection
    :param success_ratio_array: np array (any shape) of success ratios.
    :param pod_array: np array (same shape) of POD values.
    :return: csi_array: np array (same shape) of CSI values.
    """
    return (success_ratio_array**-1 + pod_array**-1 - 1.0) ** -1


def probability_of_false_detection(contingency_table_as_dict):
    """Computes POFD (probability of false detection).
    :param contingency_table_as_dict: Dictionary created by
        contingency_table.
    :return: probability_of_false_detection: POFD.
    """
    # fp/(fp+tn)
    denominator = (
        contingency_table_as_dict["num_false_positives"]
        + contingency_table_as_dict["num_true_negatives"]
    )

    if denominator == 0:
        return np.nan

    numerator = float(contingency_table_as_dict["num_false_positives"])
    return numerator / denominator


def _probability_of_false_detection_pod_grid(pofd_spacing=0.01, pod_spacing=0.01):
    """Creates grid in POFD-POD space.
    M = number of rows (unique POD values) in grid
    N = number of columns (unique POFD values) in grid
    :param pofd_spacing: Spacing between grid cells in adjacent columns.
    :param pod_spacing: Spacing between grid cells in adjacent rows.
    :return: pofd_matrix: M-by-N numpy array of POFD values.
    :return: pod_matrix: M-by-N numpy array of POD values.
    """

    num_pofd_values = 1 + int(np.ceil(1.0 / pofd_spacing))
    num_pod_values = 1 + int(np.ceil(1.0 / pod_spacing))

    unique_pofd_values = np.linspace(0.0, 1.0, num=num_pofd_values)
    unique_pod_values = np.linspace(0.0, 1.0, num=num_pod_values)[::-1]
    return np.meshgrid(unique_pofd_values, unique_pod_values)


def _get_peirce_colour_scheme(peirce_levels=np.linspace(0, 1, num=11, dtype=float)):
    """Returns colour scheme for Peirce score.
    :return: colour_map_object: Colour scheme (instance of
        `matplotlib.colors.ListedColormap`).
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`,
        defining the scale of the colour map.
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
    axes_object,
    colour_map_object,
    values_to_colour,
    min_colour_value,
    max_colour_value,
    colour_norm_object=None,
    orientation_string="vertical",
    extend_min=True,
    extend_max=True,
    fraction_of_axis_length=1.0,
):
    """Adds colour bar to existing axes.
    :param axes_object: Existing axes (instance of
        `matplotlib.axes._subplots.AxesSubplot`).
    :param colour_map_object: Colour scheme (instance of
        `matplotlib.pyplot.cm`).
    :param values_to_colour: numpy array of values to colour.
    :param min_colour_value: Minimum value in colour map.
    :param max_colour_value: Max value in colour map.
    :param colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`,
        defining the scale of the colour map.  If `colour_norm_object is None`,
        will assume that scale is linear.
    :param orientation_string: Orientation of colour bar ("vertical" or
        "horizontal").
    :param extend_min: Boolean flag.  If True, the bottom of the colour bar will
        have an arrow.  If False, it will be a flat line, suggesting that lower
        values are not possible.
    :param extend_max: Same but for top of colour bar.
    :param fraction_of_axis_length: Fraction of axis length (y-axis if
        orientation is "vertical", x-axis if orientation is "horizontal")
        occupied by colour bar.
    :param font_size: Font size for labels on colour bar.
    :return: colour_bar_object: Colour bar (instance of
        `matplotlib.pyplot.colorbar`) created by this method.
    """

    if colour_norm_object is None:
        colour_norm_object = matplotlib.colors.Normalize(
            vmin=min_colour_value, vmax=max_colour_value, clip=False
        )

    scalar_mappable_object = plt.cm.ScalarMappable(
        cmap=colour_map_object, norm=colour_norm_object
    )
    scalar_mappable_object.set_array(values_to_colour)

    if extend_min and extend_max:
        extend_string = "both"
    elif extend_min:
        extend_string = "min"
    elif extend_max:
        extend_string = "max"
    else:
        extend_string = "neither"

    padding = 0.075 if orientation_string == "horizontal" else 0.05
    colour_bar_object = plt.colorbar(
        ax=axes_object,
        mappable=scalar_mappable_object,
        orientation=orientation_string,
        pad=padding,
        extend=extend_string,
        shrink=fraction_of_axis_length,
    )

    colour_bar_object.ax.tick_params(labelsize=14)
    return colour_bar_object


def peirce_contour(ax, peirce_levels=np.linspace(0, 1, num=11, dtype=float)):
    pofd_matrix, pod_matrix = _probability_of_false_detection_pod_grid()
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
        extend_min=False,
        extend_max=False,
    )

    colour_bar_object.set_label("Peirce score [POD-POFD]")


def frequency_bias_from_sr_and_pod(success_ratio_array, pod_array):
    """Computes frequency bias from success ratio and POD.
    POD = probability of detection
    :param success_ratio_array: np array (any shape) of success ratios.
    :param pod_array: np array (same shape) of POD values.
    :return: frequency_bias_array: np array (same shape) of frequency biases.
    """
    return pod_array / success_ratio_array


def make_axis(ax=None, figsize=(15, 5), CSIBOOL=True, FBBOOL=True, csi_cmap="Blues"):
    if ax is None:
        fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        fig.set_facecolor("w")

    if CSIBOOL:
        sr_array = np.linspace(0.001, 1, 200)
        pod_array = np.linspace(0.001, 1, 200)
        X, Y = np.meshgrid(sr_array, pod_array)
        csi_vals = csi_from_sr_and_pod(X, Y)
        pm = ax2.contourf(X, Y, csi_vals, levels=np.arange(0, 1.1, 0.1), cmap=csi_cmap)
        plt.colorbar(pm, ax=ax2, label="CSI")

    if FBBOOL:
        fb = frequency_bias_from_sr_and_pod(X, Y)
        bias = ax2.contour(
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

    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.0])
    ax2.set_xlabel("Precision/Success Ratio")
    ax2.set_title("Performance Diagram")
    return ax1, ax2


def table_at_threshold(labels, y_preds, c, t):
    # make a dummy array full of a wrong prediction
    if c == 0:
        y_preds_bi = np.ones(len(y_preds), dtype=int)
    else:
        y_preds_bi = np.zeros(len(y_preds), dtype=int)
    # find where the prediction is greater than or equal to the threshold
    idx = np.where(y_preds >= t)
    # set those indices to class label
    y_preds_bi[idx] = c
    # get the contingency with overriden predictesd labels
    cm = confusion_matrix(labels, y_preds_bi)
    return contingency_table(cm, c)


def plot_roc(ax1, pofds, pods, markerfacecolor, label):
    ax1.plot(
        pofds, pods, "-", color=markerfacecolor, markerfacecolor="w", lw=2, label=label
    )
    ax1.set_xlabel("POFD (probability of false detection)")
    ax1.set_ylabel("POD (probability of detection)")
    ax1.set_title("ROC Curve")
    ax1.set_ylim([0.0, 1.0])
    ax1.set_xlim([0.0, 1.0])
    ax1.plot([0, 1], [0, 1], "k--", lw=1)


def plot_performance(ax2, srs, pods, markerfacecolor, label):
    ax2.plot(
        srs, pods, "-", color=markerfacecolor, markerfacecolor="w", lw=2, label=label
    )
    ax2.legend(loc="center left", bbox_to_anchor=(1.5, 0.5))


def performance_diagram(labels, yhat_proba):
    """plot performance diagram and roc curve in subplots"""

    threshs = np.linspace(0.0, 1.0, 100)
    pofds = np.zeros((len(config.CLASS_NAMES), len(threshs)))
    pods = np.zeros((len(config.CLASS_NAMES), len(threshs)))
    srs = np.zeros((len(config.CLASS_NAMES), len(threshs)))

    ax1, ax2 = make_axis()
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
    peirce_contour(ax1)
