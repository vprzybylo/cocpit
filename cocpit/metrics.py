import csv
import itertools

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report

import cocpit.config as config


class Metrics:
    """
    calculates batch and epoch metrics for
    training and validation datasets
    """

    def __init__(self):

        self.best_acc = 0.0
        self.loss = 0.0
        self.totals = 0.0
        self.running_loss = 0.0
        self.running_corrects = 0.0
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
    model,
    batch_size,
    model_name,
    epoch,
    epochs,
    scheduler,
    phase,
    acc_savename,
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
        config.experiment.log_metric("epoch_acc_val", metric_instance.epoch_acc * 100)
        config.experiment.log_metric("epoch_loss_val", metric_instance.epoch_loss)

    # write acc and loss to file within epoch iteration
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


def sklearn_report(metric_instance, fold, model_name):
    """
    create classification report from sklearn
    add model name and fold iteration to the report

    Params
    ------
    - metric_instance (obj): Metric() class obj created in train_model.py as val_metrics
    - fold (int): kfold iteration
    - model_name (str): name of model being trained (e.g., VGG-16)
    """

    # flatten from appending in batches
    all_preds = np.asarray(list(itertools.chain(*metric_instance.all_preds)))
    all_labels = np.asarray(list(itertools.chain(*metric_instance.all_labels)))

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
