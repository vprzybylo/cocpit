"""
- holds epoch and batch metrics for both the training and validation datasets
- called in train_model.py
- logs metrics to console and/or comet-ml interface (see config.py to turn on)
- writes metrics to csv's defined in config.py
- creates a sklearn classification report using the metrics
"""

import torch
from torch import nn
from dataclasses import dataclass, field
from typing import Any
import cocpit.config as config  # isort:split


@dataclass
class Metrics:
    """
    calculates batch and epoch metrics for
    training and validation datasets
    """

    dataloaders: Any
    optimizer: Any
    model: Any

    totals: float = 0.0
    running_loss: Any = 0.0
    running_corrects: Any = 0.0
    batch_loss: Any = 0.0
    epoch_loss: Any = 0.0
    epoch_acc: Any = 0.0
    val_best_acc: float = 0.0

    # calculated in train.py and validate.py
    loss: Any = field(default=None, init=False)
    preds: Any = field(default=None, init=False)
    labels: Any = field(default=None, init=False)
    inputs: Any = field(default=None, init=False)
    batch: int = field(default_factory=int, init=False)
    criterion: Any = nn.CrossEntropyLoss()

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

    def epoch_metrics(self):
        """
        Calculate loss and accuracy after each epoch (iteration across all batches)
        """
        self.epoch_loss = self.running_loss / self.totals
        self.epoch_acc = self.running_corrects.double() / self.totals

    def log_epoch_metrics(
        self, acc_name: str, loss_name: str, epoch_acc: float, epoch_loss: float
    ) -> None:
        """log epoch metrics to comet"""
        if config.LOG_EXP:
            config.experiment.log_metric(acc_name, epoch_acc * 100)
            config.experiment.log_metric(loss_name, epoch_loss)
