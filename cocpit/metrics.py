"""
- holds epoch and batch metrics for both the training and validation datasets
- called in train_model.py
- updates and resets totals within training loop
- logs metrics to console and/or comet-ml interface (see config.py to turn on)
- writes metrics to csv's defined in config.py
- creates a sklearn classification report using the metrics
"""

import csv
import itertools

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report

import cocpit.config as config  # isort:split
import cocpit.plotting_scripts.plot_metrics as plot_metrics
from cocpit.auto_str import auto_str


@auto_str
class Metrics:
    """
    calculates batch and epoch metrics for
    training and validation datasets
    """

    def __init__(self):
        self.loss = 0.0
        self.totals = 0.0
        self.running_loss = 0.0
        self.running_corrects = 0.0
        self.batch_loss = 0.0
        self.batch_corrects = 0.0
        self.epoch_loss = 0.0
        self.epoch_acc = 0.0
        # validation preds and labels only
        self.all_preds = []
        self.all_labels = []

    def update_batch_metrics(self, loss, inputs, preds, labels):
        """
        Calculate loss and accuracy for each batch in dataloader
        """
        # Batch accuracy and loss statistics
        self.batch_loss = loss.item() * inputs.size(0)
        self.batch_corrects = torch.sum(preds == labels.data)

        # for accuracy and loss statistics overall
        self.running_loss += loss.item() * inputs.size(0)
        self.running_corrects += torch.sum(preds == labels.data)
        self.totals += labels.size(0)

    def print_batch_metrics(self, labels, batch, phase, dataloaders_dict):
        """
        outputs batch iteration, loss, and accuracy to terminal or log file
        """

        loss = self.batch_loss / labels.size(0)
        acc = float(self.batch_corrects) / labels.size(0)

        print(
            f"{phase}, Batch {batch + 1}/{len(dataloaders_dict[phase])},\
            Loss: {loss:.3f}, Accuracy: {acc:.3f}"
        )

    def epoch_metrics(self):
        """
        Calculate loss and accuracy after each epoch (iteration across all batches)
        """
        self.epoch_loss = self.running_loss / self.totals
        self.epoch_acc = self.running_corrects.double() / self.totals

    def print_epoch_metrics(self, epoch, epochs, phase):
        """
        outputs epoch iteration, loss, and accuracy to terminal or log file
        """

        print(
            f"{phase} Epoch {epoch + 1}/{epochs},\
            Loss: {self.epoch_loss:.3f},\
            Accuracy: {self.epoch_acc:.3f}"
        )


##############


def log_metrics(
    metric_instance,
    kfold,
    batch_size,
    model_name,
    epoch,
    epochs,
    phase,
):
    """
    calculate the accuracy and loss per epoch for validation data
    log the results to comet
    write the accuracy and loss metrics to file and print to console
    """

    # calculate acc and loss for validation data
    metric_instance.epoch_metrics()

    # log to comet
    if config.LOG_EXP:
        config.experiment.log_metric(
            f"epoch_acc_{phase}", metric_instance.epoch_acc * 100
        )
        config.experiment.log_metric(f"epoch_loss_{phase}", metric_instance.epoch_loss)

    # write acc and loss to file within epoch iteration
    acc_savename = (
        config.ACC_SAVENAME_VAL if phase == "val" else config.ACC_SAVENAME_TRAIN
    )
    if config.SAVE_ACC:
        with open(acc_savename, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    model_name,
                    epoch,
                    kfold,
                    batch_size,
                    metric_instance.epoch_acc.cpu().numpy(),
                    metric_instance.epoch_loss,
                ]
            )
            file.close()

    # print output
    metric_instance.print_epoch_metrics(epoch, epochs, phase)


def sklearn_report(val_metrics, fold, model_name):
    """
    create classification report from sklearn
    add model name and fold iteration to the report

    Params
    - fold (int): kfold iteration
    - model_name (str): name of model being trained (e.g., VGG-16)
    """
    all_labels = np.asarray(list(itertools.chain(*val_metrics.all_labels)))
    all_preds = np.asarray(list(itertools.chain(*val_metrics.all_preds)))
    clf_report = classification_report(
        all_labels,
        all_preds,
        digits=3,
        target_names=config.CLASS_NAMES,
        output_dict=True,
    )

    # transpose classes as columns and convert to df
    clf_report = pd.DataFrame(clf_report).iloc[:-1, :].T

    # add fold iteration and model name
    clf_report["fold"] = fold
    clf_report["model"] = model_name

    if config.SAVE_ACC:
        clf_report.to_csv(config.METRICS_SAVENAME, mode="a")


def log_confusion_matrix(val_metrics):
    """
    log a confusion matrix to comet ml after the last epoch
    found under the graphics tab
    if using kfold, it will concatenate all validation dataloaders
    if not using kfold, it will only plot the validation dataset (e.g, 20%)
    """
    all_labels = np.asarray(list(itertools.chain(*val_metrics.all_labels)))
    all_preds = np.asarray(list(itertools.chain(*val_metrics.all_preds)))

    plot_metrics.conf_matrix(
        all_labels,
        all_preds,
        save_name=config.CONF_MATRIX_SAVENAME,
        save_fig=True,
    )

    # log to comet
    if config.LOG_EXP:
        config.experiment.log_image(
            config.CONF_MATRIX_SAVENAME, name="confusion matrix", image_format="pdf"
        )

    # unnormalized matrix
    plot_metrics.conf_matrix(
        all_labels,
        all_preds,
        norm=None,
        save_name=config.CONF_MATRIX_SAVENAME,
        save_fig=True,
    )

    # log to comet
    if config.LOG_EXP:
        config.experiment.log_image(
            config.CONF_MATRIX_SAVENAME, name="confusion matrix", image_format="pdf"
        )
