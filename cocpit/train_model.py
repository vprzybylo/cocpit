"""
train the CNN model(s)
"""
import csv
import operator
import os
import time

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import cocpit
import cocpit.config as config  # isort:split


class Train:
    def __init__(self, kfold, model, batch_size, model_name, epochs, dataloaders_dict):
        self.kfold = kfold
        self.model = model
        self.batch_size = batch_size
        self.model_name = model_name
        self.epochs = epochs
        self.dataloaders_dict = dataloaders_dict

        self.phases = None  # train/val
        self.phase = None
        self.loss = None
        self.preds = None
        self.batch = None
        self.kfold = None
        self.batch_size = None

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

    def log_epoch_metrics(self, metrics, best_acc, epoch):
        """log epoch metrics to comet and write to file
        also saves model if acc improves"""
        cocpit.metrics.log_metrics(
            metrics,
            self.kfold,
            self.batch_size,
            self.model_name,
            epoch,
            self.epochs,
            self.phase,
            acc_savename=config.ACC_SAVENAME_TRAIN,
        )
        if metrics.epoch_acc > best_acc and config.SAVE_MODEL:
            best_acc = metrics.epoch_acc
            # save/load best model weights
            if not os.path.exists(config.MODEL_SAVE_DIR):
                os.makedirs(config.MODEL_SAVE_DIR)
            torch.save(self.model, config.MODEL_SAVENAME)
        return best_acc, metrics.epoch_acc

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

            self.train_metrics.update_batch_metrics(
                self.loss, self.inputs, self.preds, self.labels
            )
            # print train batch metrics
            if (self.batch + 1) % 5 == 0:
                self.train_metrics.print_batch_metrics(
                    self.labels, self.batch, self.phase, self.dataloaders_dict
                )

        else:
            self.val_metrics.update_batch_metrics(
                self.loss, self.inputs, self.preds, self.labels
            )
            # print val batch metrics
            if (self.batch + 1) % 5 == 0:
                self.val_metrics.print_batch_metrics(
                    self.labels, self.batch, self.phase, self.dataloaders_dict
                )
                # append batch prediction and labels for plots
                self.val_metrics.all_preds.append(self.preds.cpu().numpy())
                self.val_metrics.all_labels.append(self.labels.cpu().numpy())

    def iterate_batches(self, print_label_count=False):
        """iterate over a batch in a dataloader and train or evaluate"""

        label_cnts_total = np.zeros(len(config.CLASS_NAMES))
        for self.batch, ((inputs, labels, paths), index) in enumerate(
            self.dataloaders_dict[self.phase]
        ):
            if print_label_count:
                self.print_label_count(label_cnts_total, index, labels)

            self.inputs = inputs.to(config.DEVICE)
            self.labels = labels.to(config.DEVICE)

            # zero the parameter gradients
            self.optimizer.zero_grad()
            self.forward()
            self.batch_metrics()

    def epoch_metrics(self, epoch):
        """call epoch metrics"""
        if self.phase == "train" or config.VALID_SIZE < 0.01:
            (
                self.train_best_acc,
                self.train_metrics.epoch_acc,
            ) = self.log_epoch_metrics(self.train_metrics, self.train_best_acc, epoch)
        else:
            self.val_best_acc, self.val_metrics.epoch_acc = self.log_epoch_metrics(
                self.val_metrics, self.val_best_acc, epoch
            )

    def confusion_matrix(self, epoch):
        """log confusion matrix"""
        if (
            epoch == self.epochs - 1
            and (config.KFOLD != 0 and self.kfold == config.KFOLD - 1)
            or (config.KFOLD == 0)
        ):
            cocpit.metrics.log_confusion_matrix(self.val_metrics)

    def classification_report(self, epoch):
        """save classification report"""
        if epoch == self.epochs - 1:
            cocpit.metrics.sklearn_report(
                self.val_metrics,
                self.kfold,
                self.model_name,
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

    def train_model(self, norm_values=False):
        """calls above methods to train across epochs and batches"""
        self.train_best_acc = 0.0
        self.val_best_acc = 0.0
        since_total = time.time()

        for epoch in range(self.epochs):
            since_epoch = time.time()
            print("-" * 20)

            self.determine_phases()
            for self.phase in self.phases:
                print("Phase: {}".format(self.phase))
                if self.phase == "train":
                    self.model.train()
                else:
                    self.model.eval()

                # reset acc, loss, labels, and predictions for each epoch and each phase
                self.train_metrics = cocpit.metrics.Metrics()
                self.val_metrics = cocpit.metrics.Metrics()

                # get transformation normalization values per channel
                if norm_values:
                    mean, std = cocpit.model_config.normalization_values(
                        self.dataloaders_dict, phase
                    )

                self.iterate_batches()
                self.epoch_metrics(epoch)
                if self.phase == "val":
                    self.confusion_matrix(epoch)
                    self.classification_report(epoch)
            self.reduce_lr()
            self.print_time_one_epoch(since_epoch)
        self.print_time_all_epochs(since_total)
        self.write_times(epoch, since_total)
