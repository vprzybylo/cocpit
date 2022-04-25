"""
Holds epoch and batch metrics for both the training and validation datasets
Logs metrics to console and/or comet-ml interface (see config.py to turn on)
"""

import torch
from torch import nn
from dataclasses import dataclass, field
from cocpit import config as config
from typing import Dict
import csv


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
        val_best_acc: (float): best validation accuracy of the model up to the given batch
    """

    dataloaders: Dict[str, torch.utils.data.DataLoader]
    optimizer: torch.optim.SGD
    model: torch.nn.parallel.DataParallel

    # used in runner.py
    model_name: str
    epoch: int
    epochs: int
    kfold: int
    batch_size: int

    totals: int = field(init=False, default=0)
    running_loss: float = field(init=False, default=0.0)
    running_corrects: torch.Tensor = field(
        init=False, default=torch.tensor([0], device=config.DEVICE)
    )
    batch_acc: float = field(init=False, default=0.0)
    epoch_loss: float = field(init=False, default=0.0)
    epoch_acc: torch.Tensor = field(
        init=False, default=torch.tensor([0], device=config.DEVICE)
    )
    val_best_acc: torch.Tensor = field(
        init=False, default=torch.tensor([0], device=config.DEVICE)
    )

    # calculated in train.py and validate.py
    loss: torch.Tensor = field(
        init=False, default=torch.tensor([0], device=config.DEVICE)
    )
    preds: torch.Tensor = field(
        init=False, default=torch.tensor([0], device=config.DEVICE)
    )
    inputs: torch.Tensor = field(
        init=False, default=torch.tensor([0], device=config.DEVICE)
    )
    labels: torch.Tensor = field(
        init=False, default=torch.tensor([0], device=config.DEVICE)
    )
    batch: int = field(init=False, default=0)
    criterion: nn.CrossEntropyLoss = nn.CrossEntropyLoss()

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
            phase (str): "Train" or "Validation"
        """

        print(self.epoch_acc, self.epoch_acc.cpu().numpy(), self.epoch_loss)

        print(
            f"{phase} Epoch {self.epoch + 1}/{self.epochs},\
                Loss: {self.epoch_loss:.3f},\
                Accuracy: {self.epoch_acc.cpu().numpy(),:.3f}"
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
