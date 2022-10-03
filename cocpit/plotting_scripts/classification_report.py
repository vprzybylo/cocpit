"""
calculation and plotting functions for reporting performance metrics
"""

import matplotlib.pyplot as plt

import pandas as pd
import seaborn as sns

import cocpit.plotting_scripts.grid_shader as grid_shader
from typing import Dict, Optional


# plt_params = {
#     "axes.labelsize": "x-large",
#     "axes.titlesize": "x-large",
#     "xtick.labelsize": "x-large",
#     "ytick.labelsize": "x-large",
#     "legend.title_fontsize": 12,
# }
plt.rcParams["font.family"] = "serif"
# plt.rcParams.update(plt_params)


def manipulate_df(df: pd.DataFrame, avg: Optional[str]):
    """
    Eliminate some metrics based on averaging

    Args:
        df (pd.DataFrame): df holding classification report
        avg (Optional[str]): how to average - across folds, classes, or none
    """
    if avg == "classes":
        # average across classes, include all folds
        df = df[(df["class"] == "macro avg")]
        title = "Averaging across Classes \n Variation in Folds"
    elif avg == "folds":
        # first don't include class averages
        df = df[
            (df["class"] != "accuracy")
            & (df["class"] != "macro avg")
            & (df["class"] != "weighted avg")
        ]
        # average across folds, include all classes
        df = df.groupby(["model", "class"]).mean().reset_index()
        title = "Averaging across Folds \n Variation in Classes"

    else:
        # include all classes and folds
        df = df[
            (df["class"] != "accuracy")
            & (df["class"] != "macro avg")
            & (df["class"] != "weighted avg")
        ]
        title = "No Averaging \n Variation in Folds and Classes"
    return df, title


def model_metric_folds(
    metric_filename: str,
    convert_names: Dict[str, str],
    save_name: str,
    avg: str = "folds",
    save_fig: bool = False,
) -> None:
    """
    Plot each model w.r.t. precision, recall, and f1-score

    Args:
        metric_filename (str): holds the csv file of metric scores per fold and model
        convert_names (Dict[str, str]): keys: model names used during training,
                    values: model names used for publication (capitalized and hyphenated)
        avg (str): 'classes': plot variability across folds (avg across classes)
                'folds': plot variability across classes (avg across folds)
                'none': plot variability including all folds and classes
        save_name (str): plot filename to save as
        save_fig (bool): save the figure to file
    """

    _, ax = plt.subplots(figsize=(9, 6))
    df = pd.read_csv(metric_filename)
    df.columns.values[0] = "class"
    df.replace(convert_names, inplace=True)
    df, title = manipulate_df(df, avg=avg)

    dd = pd.melt(
        df,
        id_vars=["model"],
        value_vars=["precision", "recall", "f1-score"],
        var_name="Metric",
    )
    convert_metric_names = {
        "f1-score": "F1-score",
        "precision": "Precision",
        "recall": "Recall",
    }
    dd = dd.replace(convert_metric_names).sort_values(["model", "Metric"])
    plot_dd(ax, dd, title, save_name, save_fig)


def plot_dd(
    ax: plt.Axes,
    dd: pd.DataFrame,
    title: str,
    save_name: str,
    save_fig: bool = False,
):
    """
    Plot model with respect to F1-score, precision, and recall

    Args:
        ax (plt.Axes): axes to plot
        dd (pd.DataFrame): dataframe holding model and performance metrics
        title (str): title for figure based on averaging
        save_name (str): name of the file to save
        save_fig (bool): whether to save the figure
    """
    g = sns.boxplot(x="model", y="value", data=dd, hue="Metric")
    grid_shader.GridShader(ax, facecolor="lightgrey", first=False, alpha=0.7)
    g.set_xticklabels(g.get_xticklabels(), rotation=90)
    g.set_xlabel("Model")
    g.set_ylabel("Value")
    plt.legend(loc="lower right")
    plt.setp(ax.get_legend().get_texts(), fontsize="14")  # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize="16")  # for legend title

    g.yaxis.grid(
        True, linestyle="-", which="major", color="lightgrey", alpha=0.5
    )
    g.set_ylim(0.75, 1.00)
    g.set_title(title)
    if save_fig:
        plt.savefig(save_name, dpi=300, bbox_inches="tight")


def classification_report_classes(
    clf_report, save_name, save_fig=False
) -> None:
    """
    Plot precision, recall, and f1-score for each class from 1 model
    Average across folds and include accuracy, macro avg, and weighted avg total

    Args:
        clf_report: classification report from sklearn
        or from metrics_report() above
        save_name (str): plot filename to save as
        save_fig (bool): whether to save the figure
    """
    _, ax = plt.subplots(figsize=(9, 7))
    # .iloc[:-1, :] to exclude support
    clf_report = pd.DataFrame(clf_report).iloc[:-1, :]
    sns.set(font="Serif")

    ax = sns.heatmap(
        clf_report,
        annot=True,
        fmt=".1%",
        cmap="coolwarm",
        cbar_kws={"shrink": 0.82},
        linecolor="k",
        linewidths=1,
        vmin=0.80,
        vmax=1.00,
    )
    ax.figure.axes[-1].set_ylabel(" ")
    plt.setp(ax.get_yticklabels(), rotation=0)
    sns.set(font_scale=4)
    # ax.set_title("Weighted")
    if save_fig:
        plt.savefig(save_name, dpi=300, bbox_inches="tight")
