import operator
import numpy as np
import torch
from torch import nn
import cocpit
from cocpit import config as config
from dataclasses import dataclass, field
from typing import Any, List


@dataclass
class Train:

    dataloaders: Any
    optimizer: Any
    model: Any
    loss: Any = field(default=None, init=False)
    preds: Any = field(default=None, init=False)
    labels: Any = field(default=None, init=False)
    inputs: Any = field(default=None, init=False)
    batch: int = field(default_factory=int, init=False)
    criterion: Any = nn.CrossEntropyLoss()

    def label_counts(self, i, label_cnts, labels):
        """
        Calculate the # of labels per batch to ensure
        weighted random sampler is correct
        """
        for n, _ in enumerate(config.CLASS_NAMES):
            label_cnts[n] += len(np.where(labels.numpy() == n)[0])
        print("LABEL COUNT = ", label_cnts)

        return label_cnts

    def print_label_count(self, label_cnts_total, index, labels):
        """print cumulative sum of images per class, per batch to
        ensure weighted sampler is working properly"""
        label_cnts = self.label_counts(index, label_cnts_total, labels)
        label_cnts_total = list(map(operator.add, label_cnts, label_cnts_total))

    def forward(self):
        """perform forward operator"""
        with torch.set_grad_enabled(True):
            outputs = self.model(self.inputs)
            self.loss = self.criterion(outputs, self.labels)
            _, self.preds = torch.max(outputs, 1)
            self.loss.backward()  # compute updates for each parameter
            self.optimizer.step()  # make the updates for each parameter

    def iterate_batches(self, print_label_count=False):
        """iterate over a batch in a dataloader and train"""

        label_cnts_total = np.zeros(len(config.CLASS_NAMES))
        for self.batch, ((inputs, labels, _), index) in enumerate(
            self.dataloaders["train"]
        ):
            if print_label_count:
                self.print_label_count(label_cnts_total, index, labels)

            self.inputs = inputs.to(config.DEVICE)
            self.labels = labels.to(config.DEVICE)

            # zero the parameter gradients
            self.optimizer.zero_grad()
            self.forward()
            self.calculate_batch_metrics()
