"""
calculation and plotting functions for reporting performance metrics
"""
import copy

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import cocpit.plotting_scripts.grid_shader as grid_shader
import cocpit.config as config  # isort: split


def conf_matrix(all_labels, all_preds, save_name, norm='true', save_fig=False):
    """
    Plot and save a confusion matrix from a saved validation dataloader
    Params
    ------
    - all_labels (list): actual labels (correctly hand labeled)
    - all_preds (list): list of predictions from the model for all batches
    - norm (str): 'true', 'pred', or None.
                Normalizes confusion matrix over the true (rows),
                predicted (columns) conditions or all the population.
                If None, confusion matrix will not be normalized.
    - save_name (str): plot filename to save as
    - save_fig (bool): save the conf matrix to file
    """

    fig, ax = plt.subplots(figsize=(10, 7))
    # all_preds[all_preds == 0] = np.nan
    # all_labels[all_labels == 0] = np.nan
    cm = confusion_matrix(all_labels, all_preds)

    cmap = copy.copy(mpl.cm.get_cmap("Reds"))
    cmap.set_bad(color='white')

    if norm is not None:
        cmn = confusion_matrix(all_labels, all_preds, normalize=norm)
        cmn[cmn < 0.005] = np.nan

        heat = sns.heatmap(
            cmn,
            annot=True,
            fmt=".2f",
            linewidths=1,
            linecolor='k',
            xticklabels=config.CLASS_NAMES,
            yticklabels=config.CLASS_NAMES,
            cmap=cmap,
            annot_kws={"size": 16},
        )

        plt.title("Normalized", fontsize=18)
    else:
        # cm = np.ma.masked_where(cm < 0.01, cm)
        cm = cm.astype(float)
        cm[cm < 0.005] = np.nan

        heat = sns.heatmap(
            cm,
            annot=True,
            linewidths=1,
            linecolor='k',
            xticklabels=config.CLASS_NAMES,
            yticklabels=config.CLASS_NAMES,
            cmap=cmap,
            fmt='.0f',
            annot_kws={"size": 18},
        )
        plt.title("Unweighted", fontsize=18)

    cbar = heat.collections[0].colorbar
    cbar.ax.tick_params(labelsize=20)
    plt.xlabel("Predicted Labels", fontsize=22)
    plt.ylabel("Actual Labels", fontsize=22)
    heat.set_xticklabels(heat.get_xticklabels(), rotation=90, fontsize=20)
    heat.set_yticklabels(heat.get_xticklabels(), rotation=0, fontsize=20)
    if save_fig:
        plt.savefig(save_name, bbox_inches="tight")


def model_metric_folds(
    metric_filename, convert_names, save_name, avg="folds", save_fig=False
):
    """
    Plot each model w.r.t. precision, recall, and f1-score
    Params
    ------
    metric_filename (str): holds the csv file of metric scores per fold and model
    convert_names (dict): keys: model names used during training,
                    values: model names used for publication (capitalized and hyphenated)
    avg (str): 'classes': plot variability across folds (avg across classes)
                'folds': plot variability across classes (avg across folds)
                'none': plot variability including all folds and classes
    - save_name (str): plot filename to save as
    - save_fig (bool): save the figure to file
    """

    fig, ax = plt.subplots(figsize=(9, 6))
    df = pd.read_csv(metric_filename)
    df.columns.values[0] = "class"
    df.replace(convert_names, inplace=True)

    if avg == "classes":
        # average across classes, include all folds
        df = df[(df["class"] == "macro avg")]
        title = 'Averaging across Classes \n Variation in Folds'
    elif avg == "folds":
        # first don't include class averages
        df = df[
            (df["class"] != "accuracy")
            & (df["class"] != "macro avg")
            & (df["class"] != "weighted avg")
        ]
        # average across folds, include all classes
        df = df.groupby(["model", "class"]).mean().reset_index()
        title = 'Averaging across Folds \n Variation in Classes'

    else:
        # include all classes and folds
        df = df[
            (df["class"] != "accuracy")
            & (df["class"] != "macro avg")
            & (df["class"] != "weighted avg")
        ]
        title = 'No Averaging \n Variation in Folds and Classes'

    dd = pd.melt(
        df,
        id_vars=["model"],
        value_vars=["precision", "recall", "f1-score"],
        var_name="Metric",
    )

    dd.sort_values(["model", "Metric"], inplace=True)

    g = sns.boxplot(x="model", y="value", data=dd, hue="Metric")
    grid_shader.GridShader(ax, facecolor="lightgrey", first=False, alpha=0.7)
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    g.set_xlabel("Model")
    g.set_ylabel("Value")
    plt.legend(loc="lower right")
    plt.setp(ax.get_legend().get_texts(), fontsize="14")  # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize="16")  # for legend title

    g.yaxis.grid(True, linestyle="-", which="major", color="lightgrey", alpha=0.5)
    g.set_ylim(0.75, 1.00)
    g.set_title(title)
    if save_fig:
        plt.savefig(save_name, dpi=300, bbox_inches="tight")


def classification_report_classes(clf_report, save_name, save_fig=False):
    """
    plot precision, recall, and f1-score for each class from 1 model
    average across folds
    also includes accuracy, macro avg, and weighted avg total

    Params
    ------
    - clf_report: classification report from sklearn
        or from metrics_report() above
    - save_name (str): plot filename to save as
    - save_fig (bool): save the figure to file
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    # .iloc[:-1, :] to exclude support
    clf_report = pd.DataFrame(clf_report).iloc[:-1, :]

    sns.heatmap(
        clf_report,
        annot=True,
        fmt=".1%",
        cmap="coolwarm",
        linecolor="k",
        linewidths=1,
        annot_kws={"fontsize": 14},
        vmin=0.90,
        vmax=1.00,
    )
    ax.set_title('Weighted')
    if save_fig:
        plt.savefig(save_name, dpi=300, bbox_inches="tight")
