"""validate on batches"""
import os
from typing import List

import torch
from cocpit.performance_metrics import Metrics
from cocpit import config as config
from cocpit import fold_setup, model_config, loss
from torch.optim.lr_scheduler import ReduceLROnPlateau
import csv
from ray import tune
import torch.nn.functional as F
import torch.nn as nn
import numpy as np


class Validation(Metrics):
    """Perform validation methods on batched dataset

    Args:
        f (cocpit.fold_setup.FoldSetup): instance of FoldSetup class
        epoch (int): epoch index in training loop
        epochs (int): total epochs for training loop
        model_name (str): name of model architecture
        kfold (int): number of folds use in k-fold cross validation
        val_best_acc (float): highest validation accuracy across epochs
        c (cocpit.model_config.ModelConfig): instance of ModelConfig class
        epoch_preds (List): predictions within an epoch
        epoch_labels (List): labels within an epoch
    """

    def __init__(
        self,
        f: fold_setup.FoldSetup,
        epoch: int,
        epochs: int,
        model_name: str,
        kfold: int,
        val_best_acc: torch.Tensor,
        c: model_config.ModelConfig,
        epoch_preds: List = [],
        epoch_labels: List = [],
    ):
        super().__init__(f, epoch, epochs)
        self.model_name = model_name
        self.kfold = kfold
        self.val_best_acc = val_best_acc
        self.c = c
        self.epoch_preds = epoch_preds  # validation preds for 1 epoch for plotting
        self.epoch_labels = epoch_labels  # validation labels for 1 epoch for plotting
        self.epoch_probs = (
            []
        )  # validation probabilities for 1 epoch for plotting
        self.epoch_uncertainties = (
            []
        )  # validation uncertainty for 1 epoch for plotting

    def predict(self) -> None:
        """Make predictions"""

        with torch.no_grad():

            outputs = self.c.model(self.inputs)
            if config.EVIDENTIAL:
                y = torch.eye(len(config.CLASS_NAMES)).to(config.DEVICE)
                y = y[self.labels]
                self.loss = loss.edl_digamma_loss(
                    outputs, y.float(), self.epochs
                )
            else:
                self.criterion = nn.CrossEntropyLoss()
                self.loss = self.criterion(outputs, self.labels)
            _, self.preds = torch.max(outputs, 1)
            self.probs = F.softmax(outputs, dim=1).max(dim=1).values
            self.uncertainty(outputs)

    def uncertainty(self, outputs: torch.Tensor) -> None:
        """
        Calculate uncertainty, which is inversely proportional to the total evidence
        Model is more confident the more evidence output by relu activation

        Args:
            outputs (torch.Tensor): model outputs
        """
        evidence = F.relu(outputs)
        alpha = evidence + 1
        # uncertainty
        self.u = len(config.CLASS_NAMES) / torch.sum(
            alpha, dim=1, keepdim=True
        )
        # print("uncertainty", self.u)

    def append_preds(self) -> None:
        """Save each batch prediction and labels for plots"""
        self.epoch_preds.append(self.preds.cpu().tolist())
        self.epoch_labels.append(self.labels.cpu().tolist())
        self.epoch_probs.append(self.probs)
        self.epoch_uncertainties.append(self.u)

    def save_model(self) -> float:
        """Save/load best model weights after improvement in val accuracy"""
        if self.epoch_acc > self.val_best_acc and config.SAVE_MODEL:
            print(
                f"Epoch acc:{self.epoch_acc} > best acc: {self.val_best_acc}."
                " Saving model."
            )
            self.val_best_acc = self.epoch_acc
            if not os.path.exists(config.MODEL_SAVE_DIR):
                os.makedirs(config.MODEL_SAVE_DIR)
            torch.save(self.c.model, config.MODEL_SAVENAME)
        return self.val_best_acc

    def reduce_lr(self) -> None:
        """Reduce learning rate upon plateau in epoch validation accuracy"""
        scheduler = ReduceLROnPlateau(
            self.c.optimizer,
            mode="max",
            factor=0.5,
            patience=0,
            verbose=True,
            eps=1e-04,
        )
        scheduler.step(self.epoch_acc)

    def iterate_batches(self) -> None:
        """Iterate over a batch in a dataloader and make predictions"""
        for self.batch, ((inputs, labels, _), _) in enumerate(
            self.f.dataloaders["val"]
        ):

            self.inputs = inputs.to(config.DEVICE)
            self.labels = labels.to(config.DEVICE)

            # zero the parameter gradients
            self.c.optimizer.zero_grad()
            self.predict()
            self.batch_metrics()
            self.append_preds()
            if (self.batch + 1) % 5 == 0:
                self.print_batch_metrics("val")

    def write_output(self) -> None:
        """
        Write acc and loss to csv file within model, epoch, kfold iteration
        """

        with open(config.ACC_SAVENAME_VAL, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    self.model_name,
                    self.epoch,
                    self.kfold,
                    self.f.batch_size,
                    self.epoch_acc.cpu().numpy(),
                    self.epoch_loss,
                ]
            )
            file.close()

    def run(self) -> torch.Tensor:
        """
        Run model on validation data and calculate metrics
        Reset acc, loss, labels, and predictions for each epoch, model, phase, and fold

        Returns:
            val_best_acc (torch.Tensor): best validation accuracy for the epoch
        """
        self.iterate_batches()
        self.epoch_metrics()
        if not config.TUNE:
            self.reduce_lr()
        val_best_acc = self.save_model()
        if config.TUNE:
            tune.report(loss=self.epoch_loss, accuracy=self.epoch_acc)
        self.log_epoch_metrics("epoch_acc_val", "epoch_loss_val")
        self.print_epoch_metrics("Validation")
        if config.SAVE_ACC:
            self.write_output()
        return val_best_acc
