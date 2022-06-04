"""
Holds epoch and batch metrics for both the training and validation datasets
Logs metrics to console and/or comet-ml interface (see config.py to turn on)
"""

from dataclasses import dataclass, field
from typing import Dict

import torch

from cocpit import config as config
import cocpit


class Metrics:
    """
    Calculate batch and epoch metrics for training and validation datasets
    Inherited in train and validate

    Args:
        f (cocpit.fold_setup.FoldSetup): instance of FoldSetup class
        epoch (int): epoch index in training loop
        epochs (int): total epochs for training loop

        loss (torch.Tensor): penalty for a bad prediction or a number indicating how bad the model's prediction was on a single example
        preds (torch.Tensor): probabilities for each class
        inputs (torch.Tensor): batch of images
        labels (torch.Tensor): labels for a batch
        batch (int): index of the batch of images

        totals (int): number of images seen
        running_loss (float): cumulative penalties for bad predictions
        running_corrects (torch.Tensor): cumulative accuracy over batches
        batch_acc (float): accuracy over a given batch
        epoch_loss (float): loss at the end of an epoch
        epoch_acc (torch.Tensor): accuracy at the end of an epoch
    """

    def __init__(self, f, epoch, epochs):
        # used in runner.py
        self.f: cocpit.fold_setup.FoldSetup = f
        self.epoch: int = epoch
        self.epochs: int = epochs

        default_torch_type = torch.tensor([0.0], device=config.DEVICE)
        self.loss: torch.Tensor = default_torch_type
        self.preds: torch.Tensor = default_torch_type
        self.inputs: torch.Tensor = default_torch_type
        self.labels: torch.Tensor = default_torch_type
        self.batch: int = 0

        self.totals: int = 0
        self.running_loss: float = 0.0
        self.running_corrects: torch.Tensor = default_torch_type
        self.batch_acc: float = 0.0
        self.epoch_loss: float = 0.0
        self.epoch_acc: torch.Tensor = default_torch_type

    def batch_metrics(self) -> None:
        """
        Calculate loss and accuracy for each batch in dataloader
        """
        # batch accuracy
        self.batch_acc = float(
            torch.sum(self.preds == self.labels.data)
        ) / self.labels.size(0)

        # for accuracy and loss statistics overall
        self.running_loss += self.loss.item() * self.inputs.size(0)
        self.running_corrects += torch.sum(self.preds == self.labels.data)
        self.totals += self.labels.size(0)

    def epoch_metrics(self) -> None:
        """
        Calculate loss and accuracy after each epoch (iteration across all batches)
        """
        self.epoch_loss = self.running_loss / self.totals
        self.epoch_acc = self.running_corrects / self.totals

    def log_epoch_metrics(self, acc_name: str, loss_name: str) -> None:
        """
        Log epoch metrics to comet

        Args:
            acc_name (str): name to log for accuracy (e.g., "epoch_acc_val")
            loss_name (str): name to log for loss (e.g., "epoch_loss_val")
        """
        if config.LOG_EXP:
            config.experiment.log_metric(acc_name, self.epoch_acc * 100)
            config.experiment.log_metric(loss_name, self.epoch_loss)

    def print_batch_metrics(self, phase: str) -> None:
        """
        Outputs batch iteration, loss, and accuracy

        Args:
            phase (str): "Train" or "Validation"
        """
        print(
            f"{phase}, Batch {self.batch + 1}/{len(self.f.dataloaders[phase])},\
            Loss: {self.loss.item():.3f}, Accuracy: {self.batch_acc:.3f}"
        )

    def print_epoch_metrics(self, phase: str) -> None:
        """
        Print epoch loss and accuracy based on phase

        Args:
            phase (str): "train" or "val"
        """

        print(
            f"{phase} Epoch {self.epoch + 1}/{self.epochs},\
                Loss: {self.epoch_loss:.3f},\
                Accuracy: {self.epoch_acc.cpu().item():.3f}"
        )
