"""
- holds epoch and batch metrics for both the training and validation datasets
- called in train_model.py
- logs metrics to console and/or comet-ml interface (see config.py to turn on)
- writes metrics to csv's defined in config.py
- creates a sklearn classification report using the metrics
"""

import csv
import itertools
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report

import cocpit
import cocpit.config as config  # isort:split
import cocpit.plotting_scripts.plot_metrics as plot_metrics
from dataclasses import dataclass
from typing import List, Any


@dataclass(kw_only=True)
class Metrics(cocpit.train_model.Train):
    """
    calculates batch and epoch metrics for
    training and validation datasets
    """

    # validation preds and labels only
    all_preds = List[int]
    all_labels = List[int]
    totals: float = 0.0
    running_loss: Any = 0.0
    running_corrects: Any = 0.0
    batch_loss: Any = 0.0
    batch_corrects: Any = 0.0
    epoch_loss: Any = 0.0
    epoch_acc: Any = 0.0

    def update_batch_metrics(self):
        """
        Calculate loss and accuracy for each batch in dataloader
        """
        # Batch accuracy and loss statistics
        self.batch_loss = self.loss.item() * self.inputs.size(0)
        self.batch_corrects = torch.sum(self.preds == self.labels.data)

        # for accuracy and loss statistics overall
        self.running_loss += self.loss.item() * self.inputs.size(0)
        self.running_corrects += torch.sum(self.preds == self.labels.data)
        self.totals += self.labels.size(0)

    def print_batch_metrics(self):
        """
        outputs batch iteration, loss, and accuracy to terminal or log file
        """

        loss = self.batch_loss / self.labels.size(0)
        acc = float(self.batch_corrects) / self.labels.size(0)

        print(
            f"{self.phase}, Batch {self.batch + 1}/{len(self.dataloaders[self.phase])},\
            Loss: {loss:.3f}, Accuracy: {acc:.3f}"
        )

    def calculate_epoch_metrics(self):
        """
        Calculate loss and accuracy after each epoch (iteration across all batches)
        """
        self.epoch_loss = self.running_loss / self.totals
        self.epoch_acc = self.running_corrects.double() / self.totals

    def print_epoch_metrics(self):
        """
        outputs epoch iteration, loss, and accuracy to terminal or log file
        """

        print(
            f"{self.phase} Epoch {self.epoch + 1}/{self.epochs},\
            Loss: {self.epoch_loss:.3f},\
            Accuracy: {self.epoch_acc:.3f}"
        )

    def log_epoch_metrics(self, metrics, best_acc):
        """log epoch metrics to comet and write to file
        also saves model if acc improves"""

        # log to comet
        if config.LOG_EXP:
            config.experiment.log_metric(
                f"epoch_acc_{self.phase}", self.epoch_acc * 100
            )
            config.experiment.log_metric(f"epoch_loss_{self.phase}", self.epoch_loss)

        # write acc and loss to file within epoch iteration
        acc_savename = (
            config.ACC_SAVENAME_VAL
            if self.phase == "val"
            else config.ACC_SAVENAME_TRAIN
        )
        if config.SAVE_ACC:
            with open(acc_savename, "a", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        self.model_name,
                        self.epoch,
                        self.kfold,
                        self.batch_size,
                        self.epoch_acc.cpu().numpy(),
                        self.epoch_loss,
                    ]
                )
                file.close()

        # print output
        self.print_epoch_metrics(self.epoch)

        if metrics.epoch_acc > best_acc and config.SAVE_MODEL:
            best_acc = metrics.epoch_acc
            # save/load best model weights
            if not os.path.exists(config.MODEL_SAVE_DIR):
                os.makedirs(config.MODEL_SAVE_DIR)
            torch.save(self.model, config.MODEL_SAVENAME)
        return best_acc, metrics.epoch_acc

    def confusion_matrix(self):
        """
        log a confusion matrix to comet ml after the last epoch
        found under the graphics tab
        if using kfold, it will concatenate all validation dataloaders
        if not using kfold, it will only plot the validation dataset (e.g, 20%)
        """

        if (
            self.epoch == self.epochs - 1
            and (config.KFOLD != 0 and self.kfold == config.KFOLD - 1)
            or (config.KFOLD == 0)
        ):
            all_labels = np.asarray(list(itertools.chain(*self.val_metrics.all_labels)))
            all_preds = np.asarray(list(itertools.chain(*self.val_metrics.all_preds)))

            plot_metrics.conf_matrix(
                all_labels,
                all_preds,
                save_name=config.CONF_MATRIX_SAVENAME,
                save_fig=True,
            )

            # log to comet
            if config.LOG_EXP:
                config.experiment.log_image(
                    config.CONF_MATRIX_SAVENAME,
                    name="confusion matrix",
                    image_format="pdf",
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
                    config.CONF_MATRIX_SAVENAME,
                    name="confusion matrix",
                    image_format="pdf",
                )

    def classification_report(self):
        """
        create classification report from sklearn
        add model name and fold iteration to the report

        Params
        - fold (int): kfold iteration
        - model_name (str): name of model being trained (e.g., VGG-16)
        """
        if self.epoch == self.epochs - 1:
            all_labels = np.asarray(list(itertools.chain(*self.val_metrics.all_labels)))
            all_preds = np.asarray(list(itertools.chain(*self.val_metrics.all_preds)))
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
            clf_report["fold"] = self.fold
            clf_report["model"] = self.model_name

            if config.SAVE_ACC:
                clf_report.to_csv(config.METRICS_SAVENAME, mode="a")
