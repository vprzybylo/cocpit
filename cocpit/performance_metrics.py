"""
Holds epoch and batch metrics for both the training and validation datasets
Logs metrics to console and/or comet-ml interface (see config.py to turn on)
"""

import csv
from dataclasses import dataclass, field
from typing import Dict

import torch
from torch import nn

from cocpit import config as config


@dataclass
class Metrics:
    """
    Calculate batch and epoch metrics for training and validation datasets
    Inherited in train and validate

    Args:
        dataloaders (dict[str, torch.utils.data.DataLoader]): training and validation dict that loads images with sampling procedure
        optimizer (torch.optim.sgd.SGD): an algorithm that modifies the attributes of the neural network
        model (torch.nn.parallel.data_parallel.DataParallel): saved and loaded model

        model_name (str): name of model architecture
        epoch (int): epoch index in training loop
        epochs (int): max epochs
        kfold (int): number of k-folds used in resampling procedure
        batch_size (int): number of images read into memory at a time

        totals: (float): number of images seen
        running_loss: (float): cumulative penalties for bad predictions
        running_corrects:(torch.Tensor): cumulative accuracy over batches
        epoch_loss: (float): loss at the end of an epoch
        epoch_acc: (float): accuracy at the end of an epoch

    """

    dataloaders: Dict[str, torch.utils.data.DataLoader]
    optimizer: torch.optim.SGD
    model: torch.nn.parallel.DataParallel = field(repr=False)

    # used in runner.py
    model_name: str
    epoch: int
    epochs: int
    kfold: int
    batch_size: int
    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()

    def __init__(self):

        default_torch_type = torch.tensor([0.0], device=config.DEVICE)
        self.loss: torch.Tensor = default_torch_type
        self.preds: torch.Tensor = default_torch_type
        self.inputs: torch.Tensor = default_torch_type
        self.labels: torch.Tensor = default_torch_type
        self.batch: int = 0
        self.val_best_acc: torch.Tensor = default_torch_type

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
            f"{phase}, Batch {self.batch + 1}/{len(self.dataloaders[phase])},\
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

    def write_output(self, filename: str) -> None:
        """
        Write acc and loss to csv file within model, epoch, kfold iteration

        Args:
            filename: config.ACC_SAVENAME_TRAIN or config.ACC_SAVENAME_VAL depending on phase
        """
        if config.SAVE_ACC:
            with open(filename, "a", newline="") as file:
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
