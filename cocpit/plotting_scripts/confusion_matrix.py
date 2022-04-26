import copy
import matplotlib as mpl
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
import cocpit.config as config  # isort: split
import seaborn as sns
import matplotlib.pyplot as plt
from typing import List


def mask_cm_small_values(cm, value=0.005):
    """
    Mask small values to plot white

    Args:
        value (float): value to mask below
    Return:
        cm (sklearn.metrics.confusion_matrix): masked cm
    """
    cm = cm.astype(float)
    cm[cm < value] = np.nan
    return cm


def heatmap(cm: sklearn.metrics.confusion_matrix) -> sns.heatmap:
    """
    Create seaborn heatmap as confusion matrix
    Args:
        cm (sklearn.metrics.confusion_matrix): confusion matrix
    Return:
        sns.heatmap
    """
    return sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        linewidths=1,
        linecolor="k",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        cmap=cmap(),
        annot_kws={"size": 16},
    )


def cmap() -> mpl.cm:
    """Color map from matplotlib

    Return:
        mpl.cm: masking out nans to white on red colormap
    """
    cmap = copy.copy(mpl.cm.get_cmap("Reds"))
    return cmap.set_bad(color="white")


def heatmap_axes(hm: sns.heatmap) -> None:
    """
    Confusion matrix axis labels, colorbar, and tick marks

    Args:
        hm (sns.heatmap): heatmap of confusion matrix
    """
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    plt.xlabel("Predicted Labels", fontsize=22)
    plt.ylabel("Actual Labels", fontsize=22)
    hm.set_xticklabels(hm.get_xticklabels(), rotation=90, fontsize=20)
    hm.set_yticklabels(hm.get_xticklabels(), rotation=0, fontsize=20)


def conf_matrix(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    save_name: str,
    norm=None,
    save_fig=False,
) -> sklearn.metrics.confusion_matrix:
    """
    Plot and save a confusion matrix from a saved validation dataloader

    Args:
        all_labels (np.ndarray): actual labels (correctly hand labeled)
        all_preds (np.ndarray): list of predictions from the model for all batches
        norm (str): 'true', 'pred', or None.
                Normalizes confusion matrix over the true (rows),
                predicted (columns) conditions or all the population.
                If None, confusion matrix will not be normalized.
        save_name (str): plot filename to save as
        save_fig (bool): save the conf matrix to file
    """
    cm = confusion_matrix(all_labels, all_preds, normalize=norm)
    cm = mask_cm_small_values(cm)
    hm = heatmap(cm)
    heatmap_axes(hm)
    if norm:
        plt.title("Normalized", fontsize=18)
    else:
        plt.title("Unweighted", fontsize=18)

    if save_fig:
        plt.savefig(save_name, bbox_inches="tight")
    return cm
