"""
train the CNN model(s)
"""
import csv
import operator
import os
import time

import numpy as np
import torch
from torch import optim, nn
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataclasses import dataclass, field
from typing import Any, List
import cocpit
import cocpit.config as config  # isort:split


@dataclass
class Train:
    loss: Any = field(init=False)
    preds: Any = field(init=False)
    labels: Any = field(init=False)
    inputs: Any = field(init=False)
    batch: int = field(init=False)
    train_best_acc: float = field(default=0.0, init=False)
    val_best_acc: float = field(default=0.0, init=False)
    phases: List[str] = field(default_factory=list, init=False)
    phase: str = field(default="train", init=False)

    # train_metrics: cocpit.metrics.Metrics = field(
    #     default_factory=cocpit.metrics.Metrics, init=False
    # )
    # val_metricss: cocpit.metrics.Metrics = field(
    #     default_factory=cocpit.metrics.Metrics, init=False
    # )

    def model_config(self):
        """model configurations"""
        params_to_update = cocpit.model_config.update_params(self.model)
        self.optimizer = optim.SGD(
            params_to_update, lr=0.01, momentum=0.9, nesterov=True
        )
        self.criterion = nn.CrossEntropyLoss()  # Loss function
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=0, verbose=True, eps=1e-04
        )

    def determine_phases(self):
        """determine if there is both a training and validation phase"""
        if config.VALID_SIZE < 0.1:
            self.phases = ["train"]
        else:
            self.phases = ["train", "val"]

    def print_label_count(self, label_cnts_total, index, labels):
        """print cumulative sum of images per class, per batch to
        ensure weighted sampler is working properly"""
        if self.phase == "train":
            label_cnts = cocpit.model_config.label_counts(
                index, label_cnts_total, labels
            )
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

    def epoch_metrics(self, epoch):
        """call epoch metrics"""
        if self.phase == "train" or config.VALID_SIZE < 0.01:
            # calculate acc and loss for validation data
            self.train_metrics.calculate_epoch_metrics()
            (
                self.train_best_acc,
                self.train_metrics.epoch_acc,
            ) = self.train_metrics.log_epoch_metrics(
                self.train_metrics, self.train_best_acc, epoch
            )
        else:
            (
                self.val_best_acc,
                self.val_metrics.epoch_acc,
            ) = self.val_metrics.log_epoch_metrics(
                self.val_metrics, self.val_best_acc, epoch
            )

    def write_times(self, epoch, since_total):
        """write out time to train to file"""
        time_elapsed = time.time() - since_total
        with open(
            "/data/data/saved_timings/model_timing_only_cpu.csv", "a", newline=""
        ) as file:
            writer = csv.writer(file)
            writer.writerow([self.model_name, epoch, self.kfold, time_elapsed])

    def print_time_all_epochs(self, since_total):
        """print time it took for all epochs to train"""
        time_elapsed = time.time() - since_total
        print(
            "All epochs comlete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

    def print_time_one_epoch(self, since_epoch):
        """print time for one epoch"""
        time_elapsed = time.time() - since_epoch
        print(
            "Epoch complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

    def reduce_lr(self):
        """reduce learning rate upon plateau in epoch validation accuracy"""
        self.scheduler.step(self.val_metrics.epoch_acc)

    def iterate_phase(self, epoch, norm_values=False):
        print("-" * 20)
        self.determine_phases()
        for self.phase in self.phases:
            print(f"Phase: {self.phase}")
            self.model.train() if self.phase == "train" else self.model.eval()
            self.reset_metrics()
            # get transformation normalization values per channel
            if norm_values:
                mean, std = cocpit.model_config.normalization_values(self.phase)

            self.iterate_batches()
            self.epoch_metrics(epoch)
            if self.phase == "val":
                self.val_metrics.confusion_matrix(epoch)
                self.val_metrics.classification_report(epoch)

    def train_model(self):
        """calls above methods to train across epochs and batches"""

        self.train_best_acc = 0.0
        self.val_best_acc = 0.0
        since_total = time.time()

        for epoch in range(self.epochs):
            since_epoch = time.time()
            self.iterate_phase(epoch)
            self.reduce_lr()
            self.print_time_one_epoch(since_epoch)
        self.print_time_all_epochs(since_total)
        self.write_times(epoch, since_total)
