import os
import torch
import itertools
import numpy as np
from cocpit.plotting_scripts import plot_metrics as plot_metrics
from sklearn.metrics import classification_report
import pandas as pd
from cocpit.performance_metrics import Metrics
from cocpit.runner import Runner
from cocpit import config as config
from dataclasses import dataclass, field
from typing import Any, List, Optional
from torch.optim.lr_scheduler import ReduceLROnPlateau


@dataclass
class Validation(Metrics):
    """Perform validation methods on batched dataset"""

    all_preds = []  # validation preds for 1 epoch for plotting
    all_labels = []  # validation labels for 1 epoch for plotting

    def predict(self) -> None:
        """make predictions"""
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
            if self.batch % 5:
                self.print_batch_metrics("Val")

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

        plot_metrics.conf_matrix(
            np.asarray(list(itertools.chain(*self.all_labels))),
            np.asarray(list(itertools.chain(*self.all_preds))),
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
