"""
train the CNN model(s)
"""
import itertools
import time
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report

import cocpit
import cocpit.config as config  # isort:split
from cocpit.plotting_scripts import confusion_matrix as confusion_matrix

plt_params = {
    "axes.labelsize": "x-large",
    "axes.titlesize": "large",
    "xtick.labelsize": "large",
    "ytick.labelsize": "large",
}
plt.rcParams["font.family"] = "serif"
plt.rcParams.update(plt_params)


def determine_phases() -> List[str]:
    """determine if there is both a training and validation phase

    Returns:
        List[str]: either training or training and validation phase"""
    return ["train"] if config.VALID_SIZE < 0.1 else ["train", "val"]


def conf_matrix(labels, preds, norm: Optional[str] = None) -> None:
    """
    log a confusion matrix to comet ml after the last epoch
        - found under the graphics tab
    if using kfold, it will concatenate all validation dataloaders
    if not using kfold, it will only plot the validation dataset (e.g, 20%)

    Args:
        labels (List[List]): nested list of truth labels across batches from the last epoch
        preds (List[List]): nested list of predicted labels across batches from the last epoch
        norm (str): 'true', 'pred', or None.
            Normalizes confusion matrix over the true (rows),
            predicted (columns) conditions or all the population.
            If None, confusion matrix will not be normalized.
    """
    # needed to flatten across batches
    _ = confusion_matrix.conf_matrix(
        [item for sublist in list(itertools.chain(*labels)) for item in sublist],
        [item for sublist in list(itertools.chain(*preds)) for item in sublist],
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
        [item for sublist in list(itertools.chain(*labels)) for item in sublist],
        [item for sublist in list(itertools.chain(*preds)) for item in sublist],
        digits=3,
        target_names=config.CLASS_NAMES,
        output_dict=True,
    )
    clf_report_df = pd.DataFrame(clf_report)
    if config.SAVE_ACC:
        clf_report_df.to_csv(config.METRICS_SAVENAME, mode="a")

    # transpose classes as columns and convert to df
    clf_report = pd.DataFrame(clf_report).iloc[:-1, :].T

    cocpit.plotting_scripts.classification_report.classification_report_classes(
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


def main(
    f: cocpit.fold_setup.FoldSetup,
    c: cocpit.model_config.ModelConfig,
    model_name: str,
    epochs: int,
    kfold: int,
) -> None:
    """
    Trains and validates a model across epochs, phases, and batches

    Args:
        f (cocpit.fold_setup.FoldSetup): instance of FoldSetup class
        c (model_config.ModelConfig): instance of ModelConfig class
        model_name (str): name of model architecture
        epochs (int): total epochs for training loop
        kfold (int): number of folds use in k-fold cross validation
    """

    since_total = time.time()
    val_best_acc = 0.0
    val_labels = []
    val_preds = []
    for epoch in range(epochs):
        since_epoch = time.time()
        t = cocpit.timing.EpochTime(since_total, since_epoch)
        for phase in determine_phases():
            # put model in correct mode based on if training or validating
            c.model.train() if phase == "train" else c.model.eval()
            if phase == "train":
                train = cocpit.train.Train(f, epoch, epochs, model_name, kfold, c)
                train.run()
            else:
                val = cocpit.validate.Validation(
                    f, epoch, epochs, model_name, kfold, val_best_acc, c
                )
                val_best_acc = val.run()
                if epoch == epochs - 1:
                    val_labels.append(val.epoch_labels)
                    val_preds.append(val.epoch_preds)
            t.print_time_one_epoch()

    conf_matrix(val_labels, val_preds)
    class_report(model_name, val_labels, val_preds, kfold)

    try:
        t.print_time_all_epochs()
        t.write_times(model_name, kfold)
    except NameError:
        print("Number of epochs needs to increase")
