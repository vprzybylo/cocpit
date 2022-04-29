import os
import torch
import itertools
import numpy as np
from cocpit.plotting_scripts import confusion_matrix as confusion_matrix
from sklearn.metrics import classification_report
import pandas as pd
from cocpit.performance_metrics import Metrics
from cocpit import config as config
from dataclasses import dataclass
from typing import Optional
from torch.optim.lr_scheduler import ReduceLROnPlateau


@dataclass
class Validation(Metrics):
    """Perform validation methods on batched dataset

    Args:
        val_best_acc: (float): best validation accuracy of the model up to the given batch
    """

    all_preds = []  # validation preds for 1 epoch for plotting
    all_labels = []  # validation labels for 1 epoch for plotting

    def __post_init__(self):
        super().__init__()

    def predict(self) -> None:
        """make predictions"""

        with torch.no_grad():
            outputs = self.model(self.inputs)
            self.loss = self.criterion(outputs, self.labels)
            _, self.preds = torch.max(outputs, 1)

    def append_preds(self) -> None:
        """save each batch prediction and labels for plots"""
        self.all_preds.append(self.preds.cpu().numpy())
        self.all_labels.append(self.labels.cpu().numpy())

    def save_model(self) -> None:
        """save/load best model weights after improvement in val accuracy"""
        if self.epoch_acc > self.val_best_acc and config.SAVE_MODEL:
            self.val_best_acc = self.epoch_acc

            if not os.path.exists(config.MODEL_SAVE_DIR):
                os.makedirs(config.MODEL_SAVE_DIR)
            torch.save(self.model, config.MODEL_SAVENAME)

    def reduce_lr(self) -> None:
        """reduce learning rate upon plateau in epoch validation accuracy"""
        scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=0, verbose=True, eps=1e-04
        )
        scheduler.step(self.epoch_acc)

    def iterate_batches(self) -> None:
        """iterate over a batch in a dataloader and make predictions"""
        for self.batch, ((inputs, labels, _), _) in enumerate(self.dataloaders["val"]):

            self.inputs = inputs.to(config.DEVICE)
            self.labels = labels.to(config.DEVICE)

            # zero the parameter gradients
            self.optimizer.zero_grad()
            self.predict()
            self.batch_metrics()
            self.append_preds()
            if (self.batch + 1) % 5 == 0:
                self.print_batch_metrics("val")

    def confusion_matrix(self, norm: Optional[str] = None) -> None:
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
        _ = confusion_matrix.conf_matrix(
            np.asarray(list(itertools.chain(*self.all_labels))),
            np.asarray(list(itertools.chain(*self.all_preds))),
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

    def classification_report(self, fold: int, model_name: str) -> None:
        """
        create classification report from sklearn
        add model name and fold iteration to the report

        Args:
            fold (int): which fold to use in resampling procedure
            model_name (str): name of the models
        """

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
        clf_report["fold"] = fold
        clf_report["model"] = model_name

        if config.SAVE_ACC:
            clf_report.to_csv(config.METRICS_SAVENAME, mode="a")

    def run(self) -> None:
        """
        Run model on validation data and calculate metrics
        Reset acc, loss, labels, and predictions for each epoch, model, phase, and fold
        """
        self.iterate_batches()
        self.epoch_metrics()
        self.reduce_lr()
        self.save_model()

        # confusion matrix
        if (
            self.epoch == self.epochs - 1
            and (config.KFOLD != 0 and self.kfold == config.KFOLD - 1)
            or (config.KFOLD == 0)
        ):
            self.confusion_matrix()

        # classification report
        if self.epoch == self.epochs - 1:
            self.classification_report(self.kfold, self.model_name)

        self.log_epoch_metrics("epoch_acc_val", "epoch_loss_val")
        self.print_epoch_metrics("Validation")
        self.write_output(config.ACC_SAVENAME_VAL)
