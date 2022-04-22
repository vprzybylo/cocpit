import os
import torch
from torch import nn
import cocpit
from cocpit import config as config
from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class Validation:

    dataloaders: Any
    optimizer: Any = field(default=None)
    model: Any = field(default=None)
    loss: Any = field(default=None, init=False)
    preds: Any = field(default=None, init=False)
    labels: Any = field(default=None, init=False)
    inputs: Any = field(default=None, init=False)
    batch: int = field(default_factory=int, init=False)
    criterion: Any = nn.CrossEntropyLoss()
    all_preds = []  # validation preds for 1 epoch for plotting
    all_labels = []  # validation labels for 1 epoch for plotting

    def predict(self) -> None:
        """make predictions"""
        outputs = self.model(self.inputs)
        self.loss = self.criterion(outputs, self.labels)
        _, self.preds = torch.max(outputs, 1)

    def iterate_batches(self) -> None:
        """iterate over a batch in a dataloader and make predictions"""
        for self.batch, ((inputs, labels, _), _) in enumerate(self.dataloaders["val"]):

            self.inputs = inputs.to(config.DEVICE)
            self.labels = labels.to(config.DEVICE)

            # zero the parameter gradients
            self.optimizer.zero_grad()
            self.predict()
            self.calculate_batch_metrics()
            self.append_preds()

    def append_preds(self) -> None:
        """append batch prediction and labels for plots"""
        self.all_preds.append(self.preds.cpu().numpy())
        self.all_labels.append(self.labels.cpu().numpy())

    def save_model(self) -> None:
        """save/load best model weights after improvement in val accuracy"""
        if self.epoch_acc > self.val_best_acc and config.SAVE_MODEL:
            self.val_best_acc = self.epoch_acc

            if not os.path.exists(config.MODEL_SAVE_DIR):
                os.makedirs(config.MODEL_SAVE_DIR)
            torch.save(self.model, config.MODEL_SAVENAME)
