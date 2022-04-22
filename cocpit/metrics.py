"""
- holds epoch and batch metrics for both the training and validation datasets
- called in train_model.py
- logs metrics to console and/or comet-ml interface (see config.py to turn on)
- writes metrics to csv's defined in config.py
- creates a sklearn classification report using the metrics
"""

import itertools
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
import cocpit
import cocpit.config as config  # isort:split
import cocpit.plotting_scripts.plot_metrics as plot_metrics
from dataclasses import dataclass
from typing import Any
from torch.optim.lr_scheduler import ReduceLROnPlateau


@dataclass
class Metrics(cocpit.train.Train, cocpit.validate.Validation):
    """
    calculates batch and epoch metrics for
    training and validation datasets
    """

    totals: float = 0.0
    running_loss: Any = 0.0
    running_corrects: Any = 0.0
    batch_loss: Any = 0.0
    batch_corrects: Any = 0.0
    epoch_loss: Any = 0.0
    epoch_acc: Any = 0.0
    val_best_acc: float = 0.0

    def reduce_lr(self):
        """reduce learning rate upon plateau in epoch validation accuracy"""
        scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=0, verbose=True, eps=1e-04
        )
        scheduler.step(self.epoch_acc)

    def calculate_batch_metrics(self) -> None:
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
        if (self.batch + 1) % 5 == 0:
            self.print_batch_metrics()

    def print_batch_metrics(self) -> None:
        """
        outputs batch iteration, loss, and accuracy to terminal or log file
        """

        loss = self.batch_loss / self.labels.size(0)
        acc = float(self.batch_corrects) / self.labels.size(0)

        print(
            f"Validation, Batch {self.batch + 1}/{len(self.dataloaders['val'])},\
            Loss: {loss:.3f}, Accuracy: {acc:.3f}"
        )

    def calculate_epoch_metrics(self):
        """
        Calculate loss and accuracy after each epoch (iteration across all batches)
        """
        self.epoch_loss = self.running_loss / self.totals
        self.epoch_acc = self.running_corrects.double() / self.totals

    def confusion_matrix(self, norm=None):
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
            plot_metrics.conf_matrix(
                np.asarray(list(itertools.chain(*self.val_metrics.all_labels))),
                np.asarray(list(itertools.chain(*self.val_metrics.all_preds))),
                norm=norm,
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
            clf_report = classification_report(
                np.asarray(list(itertools.chain(*self.all_labels))),
                np.asarray(list(itertools.chain(*self.all_preds))),
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
