"""
Description:
-----------
This script hosts many helper functions to make notebooks cleaner. The hope is to not distract users with ugly code.

Alot of these were sourced from Dr. Lagerquist and found originally in his gewitter repo (https://github.com/thunderhoser/GewitterGefahr).

"""

# additional libraries needed here
import scipy.stats as st
import copy
import sklearn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from cocpit import config as config

############# Global default variables ####################
# if you want to see the functions go see after line 92

NUM_TRUE_POSITIVES_KEY = "num_true_positives"
NUM_FALSE_POSITIVES_KEY = "num_false_positives"
NUM_FALSE_NEGATIVES_KEY = "num_false_negatives"
NUM_TRUE_NEGATIVES_KEY = "num_true_negatives"

FREQ_BIAS_COLOUR = np.full(3, 152.0 / 255)
FREQ_BIAS_WIDTH = 2.0
FREQ_BIAS_STRING_FORMAT = "%.2f"
FREQ_BIAS_PADDING = 5

TOLERANCE = 1e-6
DUMMY_TARGET_NAME = "tornado_lead-time=0000-3600sec_distance=00000-10000m"

MIN_OPTIMIZATION_STRING = "min"
MAX_OPTIMIZATION_STRING = "max"
VALID_OPTIMIZATION_STRINGS = [MIN_OPTIMIZATION_STRING, MAX_OPTIMIZATION_STRING]
LEVELS_FOR_PEIRCE_CONTOURS = np.linspace(0, 1, num=11, dtype=float)

POD_BY_THRESHOLD_KEY = "pod_by_threshold"
POFD_BY_THRESHOLD_KEY = "pofd_by_threshold"
SR_BY_THRESHOLD_KEY = "success_ratio_by_threshold"
MEAN_FORECAST_BY_BIN_KEY = "mean_forecast_by_bin"
EVENT_FREQ_BY_BIN_KEY = "event_frequency_by_bin"

POD_KEY = "pod"
POFD_KEY = "pofd"
SUCCESS_RATIO_KEY = "success_ratio"
FOCN_KEY = "focn"
ACCURACY_KEY = "accuracy"
CSI_KEY = "csi"
FREQUENCY_BIAS_KEY = "frequency_bias"
PEIRCE_SCORE_KEY = "peirce_score"
HEIDKE_SCORE_KEY = "heidke_score"
AUC_KEY = "auc"
AUPD_KEY = "aupd"

EVALUATION_TABLE_COLUMNS = [
    NUM_TRUE_POSITIVES_KEY,
    NUM_FALSE_POSITIVES_KEY,
    NUM_FALSE_NEGATIVES_KEY,
    NUM_TRUE_NEGATIVES_KEY,
    POD_KEY,
    POFD_KEY,
    SUCCESS_RATIO_KEY,
    FOCN_KEY,
    ACCURACY_KEY,
    CSI_KEY,
    FREQUENCY_BIAS_KEY,
    PEIRCE_SCORE_KEY,
    HEIDKE_SCORE_KEY,
    POD_BY_THRESHOLD_KEY,
    POFD_BY_THRESHOLD_KEY,
    AUC_KEY,
    SR_BY_THRESHOLD_KEY,
    AUPD_KEY,
    MEAN_FORECAST_BY_BIN_KEY,
    EVENT_FREQ_BY_BIN_KEY,
    RELIABILITY_KEY,
    RESOLUTION_KEY,
    BSS_KEY,
]

EVALUATION_DICT_KEYS = [
    FORECAST_PROBABILITIES_KEY,
    OBSERVED_LABELS_KEY,
    BEST_THRESHOLD_KEY,
    ALL_THRESHOLDS_KEY,
    NUM_EXAMPLES_BY_BIN_KEY,
    DOWNSAMPLING_DICT_KEY,
    EVALUATION_TABLE_KEY,
]

MIN_BINARIZATION_THRESHOLD = 0.0
MAX_BINARIZATION_THRESHOLD = 1.0 + TOLERANCE

DEFAULT_NUM_RELIABILITY_BINS = 20
DEFAULT_FORECAST_PRECISION = 1e-4
THRESHOLD_ARG_FOR_UNIQUE_FORECASTS = "unique_forecasts"

###########################################################

################### Helper functions ######################


def get_contingency_table(id, ol, pl):
    """

    """

    wrong_labels = np.arange(0, len(config.CLASS_NAMES))
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
        NUM_TRUE_POSITIVES_KEY: tp,
        NUM_FALSE_POSITIVES_KEY: fp,
        NUM_FALSE_NEGATIVES_KEY: fn,
        NUM_TRUE_NEGATIVES_KEY: tn,
    }


def get_contingency_table_binary(forecast_labels, observed_labels):
    """Computes contingency table.
    N = number of forecasts
    :param forecast_labels: See documentation for
        _check_forecast_and_observed_labels.
    :param observed_labels: See doc for _check_forecast_and_observed_labels.
    :return: contingency_table_as_dict: Dictionary with the following keys.
    contingency_table_as_dict['num_true_positives']: Number of true positives.
    contingency_table_as_dict['num_false_positives']: Number of false positives.
    contingency_table_as_dict['num_false_negatives']: Number of false negatives.
    contingency_table_as_dict['num_true_negatives']: Number of true negatives.
    """
    tp = np.where(np.logical_and(
        forecast_labels == 1, observed_labels == 1
    ))[0]
    fp = np.where(np.logical_and(
        forecast_labels == 1, observed_labels == 0
    ))[0]
    fn = np.where(np.logical_and(
        forecast_labels == 0, observed_labels == 1
    ))[0]
    tn = np.where(np.logical_and(
        forecast_labels == 0, observed_labels == 0
    ))[0]

    return {
        NUM_TRUE_POSITIVES_KEY: len(tp),
        NUM_FALSE_POSITIVES_KEY: len(fp),
        NUM_FALSE_NEGATIVES_KEY: len(fn),
        NUM_TRUE_NEGATIVES_KEY: len(tn)
    }


def get_pod(contingency_table_as_dict):
    """Computes POD (probability of detection).
    :param contingency_table_as_dict: Dictionary created by
        get_contingency_table.
    :return: probability_of_detection: POD.
    """

    denominator = (
        contingency_table_as_dict[NUM_TRUE_POSITIVES_KEY]
        + contingency_table_as_dict[NUM_FALSE_NEGATIVES_KEY]
    )

    if denominator == 0:
        return np.nan

    numerator = float(contingency_table_as_dict[NUM_TRUE_POSITIVES_KEY])
    return numerator / denominator


def get_sr(contingency_table_as_dict):
    """Computes success ratio.
    :param contingency_table_as_dict: Dictionary created by
        get_contingency_table.
    :return: success_ratio: Success ratio.
    """

    denominator = (
        contingency_table_as_dict[NUM_TRUE_POSITIVES_KEY]
        + contingency_table_as_dict[NUM_FALSE_POSITIVES_KEY]
    )

    if denominator == 0:
        return np.nan

    # tp/(tp+fp) #performance
    # fp/(fp+tn)    # ROC
    numerator = float(contingency_table_as_dict[NUM_TRUE_POSITIVES_KEY])
    return numerator / denominator


def get_acc(contingency_table_as_dict):
    """Computes accuracy.
    :param contingency_table_as_dict: Dictionary created by
        get_contingency_table.
    :return: accuracy: accuracy.
    """
    denominator = (
        contingency_table_as_dict[NUM_TRUE_POSITIVES_KEY]
        + contingency_table_as_dict[NUM_FALSE_POSITIVES_KEY]
        + contingency_table_as_dict[NUM_FALSE_NEGATIVES_KEY]
        + contingency_table_as_dict[NUM_TRUE_NEGATIVES_KEY]
    )
    numerator = (
        contingency_table_as_dict[NUM_TRUE_POSITIVES_KEY]
        + contingency_table_as_dict[NUM_TRUE_NEGATIVES_KEY]
    )

    return 100 * (numerator / denominator)


def csi_from_sr_and_pod(success_ratio_array, pod_array):
    """Computes CSI (critical success index) from success ratio and POD.
    POD = probability of detection
    :param success_ratio_array: np array (any shape) of success ratios.
    :param pod_array: np array (same shape) of POD values.
    :return: csi_array: np array (same shape) of CSI values.
    """
    return (success_ratio_array ** -1 + pod_array ** -1 - 1.0) ** -1


def frequency_bias_from_sr_and_pod(success_ratio_array, pod_array):
    """Computes frequency bias from success ratio and POD.
    POD = probability of detection
    :param success_ratio_array: np array (any shape) of success ratios.
    :param pod_array: np array (same shape) of POD values.
    :return: frequency_bias_array: np array (same shape) of frequency biases.
    """
    return pod_array / success_ratio_array


def get_far(contingency_table_as_dict):
    """Computes FAR (false-alarm rate).
    :param contingency_table_as_dict: Dictionary created by
        get_contingency_table.
    :return: false_alarm_rate: FAR.
    """
    return 1.0 - get_sr(contingency_table_as_dict)


def get_pofd(contingency_table_as_dict):
    """Computes POFD (probability of false detection).
    :param contingency_table_as_dict: Dictionary created by
        get_contingency_table.
    :return: probability_of_false_detection: POFD.
    """

    denominator = (
        contingency_table_as_dict[NUM_FALSE_POSITIVES_KEY]
        + contingency_table_as_dict[NUM_TRUE_NEGATIVES_KEY]
    )

    if denominator == 0:
        return np.nan
    # fp/(fp+tn)
    numerator = float(contingency_table_as_dict[NUM_FALSE_POSITIVES_KEY])
    return numerator / denominator

def _get_pofd_pod_grid(pofd_spacing=0.01, pod_spacing=0.01):
    """Creates grid in POFD-POD space.
    M = number of rows (unique POD values) in grid
    N = number of columns (unique POFD values) in grid
    :param pofd_spacing: Spacing between grid cells in adjacent columns.
    :param pod_spacing: Spacing between grid cells in adjacent rows.
    :return: pofd_matrix: M-by-N numpy array of POFD values.
    :return: pod_matrix: M-by-N numpy array of POD values.
    """

    num_pofd_values = 1 + int(np.ceil(1. / pofd_spacing))
    num_pod_values = 1 + int(np.ceil(1. / pod_spacing))

    unique_pofd_values = np.linspace(0., 1., num=num_pofd_values)
    unique_pod_values = np.linspace(0., 1., num=num_pod_values)[::-1]
    return np.meshgrid(unique_pofd_values, unique_pod_values)


def _get_peirce_colour_scheme():
    """Returns colour scheme for Peirce score.
    :return: colour_map_object: Colour scheme (instance of
        `matplotlib.colors.ListedColormap`).
    :return: colour_norm_object: Instance of `matplotlib.colors.BoundaryNorm`,
        defining the scale of the colour map.
    """

    this_colour_map_object = plt.cm.Blues
    this_colour_norm_object = matplotlib.colors.BoundaryNorm(
        LEVELS_FOR_PEIRCE_CONTOURS, this_colour_map_object.N)

    rgba_matrix = this_colour_map_object(this_colour_norm_object(
        LEVELS_FOR_PEIRCE_CONTOURS
    ))

    colour_list = [
        rgba_matrix[i, ..., :-1] for i in range(rgba_matrix.shape[0])
    ]

    colour_map_object = matplotlib.colors.ListedColormap(colour_list)
    colour_map_object.set_under(np.array([1, 1, 1]))
    colour_norm_object = matplotlib.colors.BoundaryNorm(
        LEVELS_FOR_PEIRCE_CONTOURS, colour_map_object.N)

    return colour_map_object, colour_norm_object

def add_colour_bar(
        axes_object, colour_map_object, values_to_colour, min_colour_value,
        max_colour_value, colour_norm_object=None,
        orientation_string='vertical', extend_min=True, extend_max=True,
        fraction_of_axis_length=1.):
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
            vmin=min_colour_value, vmax=max_colour_value, clip=False)

    scalar_mappable_object = plt.cm.ScalarMappable(
        cmap=colour_map_object, norm=colour_norm_object)
    scalar_mappable_object.set_array(values_to_colour)

    if extend_min and extend_max:
        extend_string = 'both'
    elif extend_min:
        extend_string = 'min'
    elif extend_max:
        extend_string = 'max'
    else:
        extend_string = 'neither'

    padding = 0.075 if orientation_string == 'horizontal' else 0.05
    colour_bar_object = plt.colorbar(
        ax=axes_object, mappable=scalar_mappable_object,
        orientation=orientation_string, pad=padding, extend=extend_string,
        shrink=fraction_of_axis_length)

    colour_bar_object.ax.tick_params(labelsize=14)
    return colour_bar_object

def peirce_contour(ax):
    pofd_matrix, pod_matrix = _get_pofd_pod_grid()
    peirce_score_matrix = pod_matrix - pofd_matrix

    colour_map_object, colour_norm_object = _get_peirce_colour_scheme()

    ax.contourf(
        pofd_matrix, pod_matrix, peirce_score_matrix,
        LEVELS_FOR_PEIRCE_CONTOURS, cmap=colour_map_object,
        norm=colour_norm_object, vmin=0., vmax=1.)

    # TODO(thunderhoser): Calling private method is a HACK.
    colour_bar_object = add_colour_bar(
        axes_object=ax, colour_map_object=colour_map_object,
        colour_norm_object=colour_norm_object,
        values_to_colour=peirce_score_matrix, min_colour_value=0.,
        max_colour_value=1., orientation_string='vertical',
        extend_min=False, extend_max=False)

    colour_bar_object.set_label('Peirce score')

def get_points_in_roc_curve(
    id, ol, pl,
    forecast_probabilities=None,
    threshold_arg=None,
    forecast_precision=DEFAULT_FORECAST_PRECISION,
):
    """Determines points in ROC (receiver operating characteristic) curve.
    N = number of forecasts
    T = number of binarization thresholds
    :param forecast_probabilities: See documentation for
        `_check_forecast_probs_and_observed_labels`.
    :param observed_labels: See doc for
        `_check_forecast_probs_and_observed_labels`.
    :param threshold_arg: See documentation for get_binarization_thresholds.
    :param forecast_precision: See doc for get_binarization_thresholds.
    :return: pofd_by_threshold: length-T np array of POFD values, to be
        plotted on the x-axis.
    :return: pod_by_threshold: length-T np array of POD values, to be plotted
        on the y-axis.
    """

    binarization_thresholds = get_binarization_thresholds(
        threshold_arg=threshold_arg,
        forecast_probabilities=forecast_probabilities,
        forecast_precision=forecast_precision,
    )

    num_thresholds = len(binarization_thresholds)
    pofd_by_threshold = np.full(num_thresholds, np.nan)
    pod_by_threshold = np.full(num_thresholds, np.nan)

    for i in range(num_thresholds):
        these_forecast_labels = binarize_forecast_probs(
            forecast_probabilities, binarization_thresholds[i]
        )
        this_contingency_table_as_dict = get_contingency_table(
            id, ol, pl
        )

        pofd_by_threshold[i] = get_pofd(this_contingency_table_as_dict)
        pod_by_threshold[i] = get_pod(this_contingency_table_as_dict)

    return pofd_by_threshold, pod_by_threshold


def get_binarization_thresholds(
    threshold_arg,
    forecast_probabilities=None,
    forecast_precision=DEFAULT_FORECAST_PRECISION,
):
    """Returns list of binarization thresholds.
    To understand the role of binarization thresholds, see
    binarize_forecast_probs.
    :param threshold_arg: Main threshold argument.  May be in one of 3 formats.
    [1] threshold_arg = "unique_forecasts".  In this case all unique forecast
        probabilities will become binarization thresholds.
    [2] 1-D np array.  In this case threshold_arg will be treated as an array
        of binarization thresholds.
    [3] Positive integer.  In this case threshold_arg will be treated as the
        number of binarization thresholds, equally spaced from 0...1.
    :param forecast_probabilities:
        [used only if threshold_arg = "unique_forecasts"]
        1-D np array of forecast probabilities to binarize.
    :param forecast_precision:
        [used only if threshold_arg = "unique_forecasts"]
        Before computing unique forecast probabilities, they will all be rounded
        to the nearest `forecast_precision`.  This prevents the number of
        thresholds from becoming ridiculous (millions).
    :return: binarization_thresholds: 1-D np array of binarization
        thresholds.
    :raises: ValueError: if threshold_arg cannot be interpreted.
    """

    if isinstance(threshold_arg, str):
        if threshold_arg != THRESHOLD_ARG_FOR_UNIQUE_FORECASTS:
            error_string = (
                'If string, threshold_arg must be "{0:s}".  Instead, got ' '"{1:s}".'
            ).format(THRESHOLD_ARG_FOR_UNIQUE_FORECASTS, threshold_arg)

            raise ValueError(error_string)

        binarization_thresholds = np.unique(
            rounder.round_to_nearest(forecast_probabilities + 0.0, forecast_precision)
        )

    elif isinstance(threshold_arg, np.ndarray):
        binarization_thresholds = copy.deepcopy(threshold_arg)
    else:
        num_thresholds = copy.deepcopy(threshold_arg)

        binarization_thresholds = np.linspace(0, 1, num=num_thresholds, dtype=float)

    return _pad_binarization_thresholds(binarization_thresholds)


def _pad_binarization_thresholds(thresholds):
    """Pads an array of binarization thresholds.
    Specifically, this method ensures that the array contains 0 and a number
        slightly greater than 1.  This ensures that:
    [1] For the lowest threshold, POD = POFD = 1, which is the top-right corner
        of the ROC curve.
    [2] For the highest threshold, POD = POFD = 0, which is the bottom-left
        corner of the ROC curve.
    :param thresholds: 1-D np array of binarization thresholds.
    :return: thresholds: 1-D np array of binarization thresholds (possibly
        with new elements).
    """

    thresholds = np.sort(thresholds)

    if thresholds[0] > MIN_BINARIZATION_THRESHOLD:
        thresholds = np.concatenate(
            (np.array([MIN_BINARIZATION_THRESHOLD]), thresholds)
        )

    if thresholds[-1] < MAX_BINARIZATION_THRESHOLD:
        thresholds = np.concatenate(
            (thresholds, np.array([MAX_BINARIZATION_THRESHOLD]))
        )

    return thresholds


def binarize_forecast_probs(forecast_probabilities, binarization_threshold):
    """Binarizes probabilistic forecasts, turning them into deterministic ones.
    N = number of forecasts
    :param forecast_probabilities: length-N numpy array with forecast
        probabilities of some event (e.g., tornado).
    :param binarization_threshold: Binarization threshold (f*).  All forecasts
        >= f* will be turned into "yes" forecasts; all forecasts < f* will be
        turned into "no".
    :return: forecast_labels: length-N integer numpy array of deterministic
        forecasts (1 for "yes", 0 for "no").
    """

    forecast_labels = np.full(len(forecast_probabilities), 0, dtype=int)
    forecast_labels[forecast_probabilities >= binarization_threshold] = 1

    return forecast_labels


def get_area_under_roc_curve(pofd_by_threshold, pod_by_threshold):
    """Computes area under ROC curve.
    This calculation ignores NaN's.  If you use `sklearn.metrics.auc` without
    this wrapper, if either input array contains any NaN, the result will be
    NaN.
    T = number of binarization thresholds
    :param pofd_by_threshold: length-T numpy array of POFD values.
    :param pod_by_threshold: length-T numpy array of corresponding POD values.
    :return: area_under_curve: Area under ROC curve.
    """

    sort_indices = np.argsort(-pofd_by_threshold)
    pofd_by_threshold = pofd_by_threshold[sort_indices]
    pod_by_threshold = pod_by_threshold[sort_indices]

    nan_flags = np.logical_or(np.isnan(pofd_by_threshold), np.isnan(pod_by_threshold))
    if np.all(nan_flags):
        return np.nan

    real_indices = np.where(np.invert(nan_flags))[0]

    return sklearn.metrics.auc(
        pofd_by_threshold[real_indices], pod_by_threshold[real_indices]
    )


def make_performance_diagram_axis(
    ax=None, figsize=(15, 5), CSIBOOL=True, FBBOOL=True, csi_cmap="Blues"
):
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
            inline_spacing=FREQ_BIAS_PADDING,
            fmt=FREQ_BIAS_STRING_FORMAT,
            fontsize=10,
            colors="k",
        )

    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.0])
    ax2.set_xlabel("Precision/Success Ratio")
    ax2.set_title('Performance Diagram')
    return ax1, ax2

