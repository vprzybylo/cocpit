"""
plot classification report, confusion matrix, and uncertainty histogram after validation
"""
from cocpit import config as config
from cocpit.plotting_scripts import classification_report as cr
from cocpit.plotting_scripts import confusion_matrix as cm
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Optional, Any, List
from sklearn.metrics import classification_report
import pandas as pd
import seaborn as sns
import pickle
from dataclasses import dataclass


@dataclass
class Report:
    """
    Performs all reporting and plotting of validation predictions, accuracies, and uncertainties
    See validation.py for definitions

    Args:
        uncertainties (List[Any]): validation uncertainties from evidential deep learning model (output of relu) from last epoch.
        probs (List[Any]): validation softmax output probabilities for each class from the last epoch
        labels (List[Any]): list of truth labels across batches from the last epoch
        preds (List[Any]): list of predicted labels across batches from the last epoch
    """

    uncertainties: List[Any]
    probs: List[List[Any]]
    labels: List[Any]
    preds: List[Any]

    def conf_matrix(
        self,
        norm: Optional[str] = None,
    ) -> None:
        """
        log a confusion matrix to comet ml after the last epoch
            - found under the graphics tab
        if using kfold, it will concatenate all validation dataloaders
        if not using kfold, it will only plot the validation dataset (e.g, 20%)

        Args:

            norm (str): 'true', 'pred', or None.
                Normalizes confusion matrix over the true (rows),
                predicted (columns) conditions or all the population.
                If None, confusion matrix will not be normalized.
        """
        _ = cm.conf_matrix(
            self.labels,
            self.preds,
            norm=norm,
            save_fig=True,
        )

    # log to comet
    if config.LOG_EXP:
        config.experiment.log_image(
            f"{config.PLOT_DIR}/conf_matrix.png",
            name="confusion matrix",
            image_format="pdf",
        )

    def class_report(self, model_name: str, fold: int) -> None:
        """
        create classification report from sklearn
        add model name and fold iteration to the report

        Args:
            model_name (str): name of model architecture
            fold (int): which fold to use in resampling procedure
        """

        # needed to flatten across batches and epochs
        clf_report = classification_report(
            self.labels[0],
            self.preds[0],
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
            save_name=f"{config.PLOT_DIR}/classification_report.png",
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
                f"{config.PLOT_DIR}/classification_report.png",
                name="classification report",
                image_format="pdf",
            )

    def pickle_uncertainties(self) -> None:
        """save uncertainties, predictions, and labels"""
        with open("val_probs.pickle", "wb") as handle:
            pickle.dump(self.probs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("val_uncertainties.pickle", "wb") as handle:
            pickle.dump(
                self.uncertainties, handle, protocol=pickle.HIGHEST_PROTOCOL
            )
        with open("val_labels.pickle", "wb") as handle:
            pickle.dump(self.labels, handle, protocol=pickle.HIGHEST_PROTOCOL)
        with open("val_preds.pickle", "wb") as handle:
            pickle.dump(self.preds, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def uncertainty_prob_scatter(self) -> None:
        """Plot probabilities vs uncertainties"""
        _, ax = plt.subplots()
        max_probs = [torch.Tensor(t).cpu() for t in self.probs][0].numpy()
        uncertainty = [torch.Tensor(t).cpu() for t in self.uncertainties][
            0
        ].numpy()
        ax.scatter(max_probs, uncertainty)
        # ax.set_ylim([0.0,1.0])
        # ax.set_xlim([0.5, 1.0])
        ax.set_xlabel("Softmax Probability", fontsize=16)
        ax.set_ylabel("Evidential Uncertainty", fontsize=16)
        ax.set_title(
            f"Validation Dataset: $n$ = {len(max_probs)}", fontsize=18
        )
        plt.savefig(f"{config.PLOT_DIR}/uncertainty_probability_scatter.png")

        _, ax = plt.subplots()
        sns.kdeplot(max_probs, uncertainty)
        plt.savefig(f"{config.PLOT_DIR}/uncertainty_probability_kde.png")


def hist(var: torch.Tensor, savename: str) -> None:
    """plot histogram of uncertainty or probability"""
    _, ax = plt.subplots(1, 1)
    _ = ax.hist(
        [np.array(t) for t in torch.Tensor(var).cpu()],
        alpha=0.5,
        label="Uncertainty",
    )
    plt.savefig(savename)
    plt.legend(fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    # ax.set_xlabels()
