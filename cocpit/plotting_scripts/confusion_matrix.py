import copy
import matplotlib as mpl
import numpy as np
import sklearn
from sklearn.metrics import confusion_matrix
import cocpit.config as config  # isort: split
import seaborn as sns
import matplotlib.pyplot as plt


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
        ax (sns.heatmap): heatmap
    """
    ax = sns.heatmap(
        cm,
        annot=True,
        fmt=".2f",
        linewidths=1,
        linecolor="k",
        xticklabels=config.CLASS_NAMES,
        yticklabels=config.CLASS_NAMES,
        cmap=change_cmap(),
        annot_kws={"size": 16},
    )
    b, t = plt.ylim()  # discover the values for bottom and top
    l, r = plt.xlim()
    b += 0.1  # Add 0.5 to the bottom
    t -= 0.1  # Subtract 0.5 from the top
    l -= 0.1
    r += 0.1
    plt.ylim(b, t)  # update the ylim(bottom, top) values
    plt.xlim(l, r)
    plt.show()
    return ax


def change_cmap() -> mpl.cm:
    """Color map from matplotlib

    Return:
        mpl.cm: masking out nans to white on red colormap
    """
    return copy.copy(mpl.cm.get_cmap("Reds"))


def heatmap_axes(hm: sns.heatmap, ax: plt.Axes) -> None:
    """
    Confusion matrix axis labels, colorbar, and tick marks

    Args:
        hm (sns.heatmap): heatmap of confusion matrix
        ax (plt.Axes): conf matrix axis
    """
    cbar = hm.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    ax.set_xlabel("Predicted Labels", fontsize=22)
    ax.set_ylabel("Actual Labels", fontsize=22)
    hm.set_xticklabels(hm.get_xticklabels(), rotation=20, fontsize=20)
    hm.set_yticklabels(hm.get_xticklabels(), rotation=0, fontsize=20)


def conf_matrix(
    all_labels: np.ndarray,
    all_preds: np.ndarray,
    norm=None,
    save_fig=True,
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
        save_fig (bool): save the conf matrix to file

    Returns:
        cm (sklearn.metrics.confusion_matrix): confusion matrix
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    cm = confusion_matrix(all_labels, all_preds, normalize=norm)
    # cm = mask_cm_small_values(cm)
    hm = heatmap(cm)
    heatmap_axes(hm, ax)
    if norm:
        ax.set_title("Normalized", fontsize=18)
    else:
        ax.set_title("Unweighted", fontsize=18)

    if save_fig:
        fig.savefig(config.CONF_MATRIX_SAVENAME, bbox_inches="tight")
    return cm
