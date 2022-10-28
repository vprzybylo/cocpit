"""
plot classification report, confusion matrix, and uncertainty histogram after validation
"""
import itertools
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional, Any
from sklearn.metrics import classification_report
import pandas as pd
from cocpit import config as config
from cocpit.plotting_scripts import classification_report as cr
from cocpit.plotting_scripts import confusion_matrix as cm
import seaborn as sns

def flatten(var):
    """flatten vars from batches"""
    return (
        [item for sublist in list(itertools.chain(*var)) for item in sublist],
    )


def conf_matrix(
    labels,
    preds,
    norm: Optional[str] = None,
) -> None:
    """
    log a confusion matrix to comet ml after the last epoch
        - found under the graphics tab
    if using kfold, it will concatenate all validation dataloaders
    if not using kfold, it will only plot the validation dataset (e.g, 20%)

    Args:
        labels (np.ndarray[Any, Any]): nested list of truth labels across batches from the last epoch
        preds (np.ndarray[Any, Any]): nested list of predicted labels across batches from the last epoch
        norm (str): 'true', 'pred', or None.
            Normalizes confusion matrix over the true (rows),
            predicted (columns) conditions or all the population.
            If None, confusion matrix will not be normalized.
    """
    # needed to flatten across batches
    _ = cm.conf_matrix(
        labels,
        preds,
        norm=norm,
        save_fig=True,
    )

    # log to comet
    if config.LOG_EXP:
        config.experiment.log_image(
            config.CONF_MATRIX_SAVENAME,
            name="confusion matrix",
            image_format="pdf",
        )


def class_report(model_name, labels, preds, fold: int) -> None:
    """
    create classification report from sklearn
    add model name and fold iteration to the report

    Args:
        model_name (str): name of model architecture
        labels (List[List]): nested list of truth labels across batches from the last epoch
        preds (List[List]): nested list of predicted labels across batches from the last epoch
        fold (int): which fold to use in resampling procedure
    """

    # needed to flatten across batches and epochs
    clf_report = classification_report(
        labels[0],
        preds[0],
        digits=3,
        target_names=config.CLASS_NAMES,
        output_dict=True,
    )
    clf_report_df = pd.DataFrame(clf_report)
    if config.SAVE_ACC:
        clf_report_df.to_csv(config.METRICS_SAVENAME, mode="a")

    # transpose classes as columns and convert to df
    clf_report = pd.DataFrame(clf_report).iloc[:-1, :].T

    cr.classification_report_classes(
        clf_report,
        save_name=config.CLASSIFICATION_REPORT_SAVENAME,
        save_fig=True,
    )

    # add fold iteration and model name
    clf_report["fold"] = fold
    clf_report["model"] = model_name

    if config.SAVE_ACC:
        clf_report.to_csv(config.METRICS_SAVENAME, mode="a")

    # log to comet
    if config.LOG_EXP:
        config.experiment.log_image(
            config.CLASSIFICATION_REPORT_SAVENAME,
            name="classification report",
            image_format="pdf",
        )


def uncertainty_prob_scatter(u, p):
    """Plot probabilities vs uncertainties"""
    _, ax = plt.subplots()
    max_probs = [torch.Tensor(t).cpu() for t in p][0].numpy()
    uncertainty = [torch.Tensor(t).cpu() for t in u][0].numpy()
    ax.scatter(max_probs,uncertainty)
    # ax.set_ylim([0.0,1.0])
    # ax.set_xlim([0.5, 1.0])
    ax.set_xlabel("Softmax Probability", fontsize=16)
    ax.set_ylabel("Evidential Uncertainty", fontsize=16)
    ax.set_title(f'Validation Dataset: $n$ = {len(max_probs)}', fontsize=18)
    plt.savefig("/ai2es/plots/uncertainty_probability_scatter.png")

    _, ax = plt.subplots()
    sns.kdeplot(max_probs, uncertainty)
    plt.savefig("/ai2es/plots/uncertainty_probability_kde.png")


def hist(var):
    """plot histogram of uncertainty or probability"""
    _, ax = plt.subplots(1, 1)
    _ = ax.hist(
        [np.array(t) for t in torch.Tensor(var).cpu()],
        alpha=0.5,
        label="Uncertainty",
    )
    plt.savefig("/ai2es/plots/histogram.png")
    plt.legend(fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    # ax.set_xlabels()
