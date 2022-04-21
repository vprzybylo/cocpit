"""
train the CNN model(s)
"""

import operator
import time

import numpy as np
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from dataclasses import dataclass, field
from typing import Any, List
import cocpit
import cocpit.config as config  # isort:split
import collections
import collections.abc


@dataclass
class Train:
    dataloaders: Any = field(default=None)
    optimizer: Any = field(default=None)
    model: Any = field(default=None)
    loss: Any = field(default=None, init=False)
    preds: Any = field(default=None, init=False)
    labels: Any = field(default=None, init=False)
    inputs: Any = field(default=None, init=False)
    batch: int = field(default_factory=int, init=False)
    phases: List[str] = field(default_factory=list, init=False)
    train_best_acc: float = 0.0
    val_best_acc: float = 0.0
    criterion: Any = nn.CrossEntropyLoss()
    phase: str = "train"

    # train_metrics: cocpit.metrics.Metrics = field(
    #     default_factory=cocpit.metrics.Metrics, init=False
    # )
    # val_metricss: cocpit.metrics.Metrics = field(
    #     default_factory=cocpit.metrics.Metrics, init=False
    # )

    def determine_phases(self):
        """determine if there is both a training and validation phase"""
        self.phases = ["train"] if config.VALID_SIZE < 0.1 else ["train", "val"]

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
        if self.phase == "train":
            label_cnts = self.label_counts(index, label_cnts_total, labels)
            label_cnts_total = list(map(operator.add, label_cnts, label_cnts_total))

    def forward(self):
        """perform forward operator"""
        with torch.set_grad_enabled(self.phase == "train"):
            outputs = self.model(self.inputs)
            self.loss = self.criterion(outputs, self.labels)
            _, self.preds = torch.max(outputs, 1)

            if self.phase == "train":
                self.loss.backward()  # compute updates for each parameter
                self.optimizer.step()  # make the updates for each parameter

    def batch_metrics(self):
        """calculate and log batch metrics"""
        if self.phase == "train" or config.VALID_SIZE < 0.01:

            self.train_metrics.update_batch_metrics()
            # print train batch metrics
            if (self.batch + 1) % 5 == 0:
                self.train_metrics.print_batch_metrics()

        else:
            self.val_metrics.update_batch_metrics()
            # print val batch metrics
            if (self.batch + 1) % 5 == 0:
                self.val_metrics.print_batch_metrics()
                # append batch prediction and labels for plots
                self.val_metrics.all_preds.append(self.preds.cpu().numpy())
                self.val_metrics.all_labels.append(self.labels.cpu().numpy())

    def iterate_batches(self, print_label_count=False):
        """iterate over a batch in a dataloader and train or evaluate"""

        label_cnts_total = np.zeros(len(config.CLASS_NAMES))
        for self.batch, ((inputs, labels, _), index) in enumerate(
            self.dataloaders[self.phase]
        ):
            if print_label_count:
                self.print_label_count(label_cnts_total, index, labels)

            self.inputs = inputs.to(config.DEVICE)
            self.labels = labels.to(config.DEVICE)

            # zero the parameter gradients
            self.optimizer.zero_grad()
            self.forward()
            self.batch_metrics()

    def reset_metrics(self):
        """reset acc, loss, labels, and predictions for each epoch and each phase"""
        self.train_metrics = cocpit.metrics.Metrics()
        self.val_metrics = cocpit.metrics.Metrics()

    def epoch_metrics(self):
        """call epoch metrics"""
        if self.phase == "train" or config.VALID_SIZE < 0.01:
            # calculate acc and loss for validation data
            self.train_metrics.calculate_epoch_metrics()
            (
                self.train_best_acc,
                self.train_metrics.epoch_acc,
            ) = self.train_metrics.log_epoch_metrics(
                self.train_metrics, self.train_best_acc
            )
        else:
            (
                self.val_best_acc,
                self.val_metrics.epoch_acc,
            ) = self.val_metrics.log_epoch_metrics(self.val_metrics, self.val_best_acc)

    def reduce_lr(self):
        """reduce learning rate upon plateau in epoch validation accuracy"""
        scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=0, verbose=True, eps=1e-04
        )
        scheduler.step(self.val_metrics.epoch_acc)

    def normalization_values(self, phase):
        """
        Get mean and standard deviation of pixel values
        across all batches
        """
        mean = 0.0
        std = 0.0
        nb_samples = 0.0
        for ((inputs, labels, paths), index) in self.dataloaders_dict[phase]:
            batch_samples = inputs.size(0)
            data = inputs.view(batch_samples, inputs.size(1), -1)
            mean += data.mean(2).sum(0)
            std += data.std(2).sum(0)
            nb_samples += batch_samples

        mean /= nb_samples
        std /= nb_samples
        print(mean, std)
        return mean, std

    def iterate_phase(self, norm_values=False):
        print("-" * 20)
        self.determine_phases()
        for self.phase in self.phases:
            print(f"Phase: {self.phase}")
            self.model.train() if self.phase == "train" else self.model.eval()
            self.reset_metrics()
            # get transformation normalization values per channel
            if norm_values:
                mean, std = self.normalization_values(self.phase)

            self.iterate_batches()
            self.epoch_metrics()
            if self.phase == "val":
                self.val_metrics.confusion_matrix()
                self.val_metrics.classification_report()


def main(dataloaders, epochs, optimizer, model):
    """calls above methods to train across epochs and batches"""
    since_total = time.time()
    t = Train(dataloaders, optimizer, model)
    for epoch in range(epochs):
        since_epoch = time.time()
        t.iterate_phase()
        t.reduce_lr()
        timing = cocpit.timing.Time(epoch, since_total, since_epoch)
        timing.print_time_one_epoch(since_epoch)
    timing.print_time_all_epochs()
    timing.write_times(epoch)
