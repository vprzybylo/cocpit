"""
train the CNN model(s)
"""
import csv
import time
import os
import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import operator
import cocpit
import cocpit.config as config  # isort:split

class Train():
    def __init__(self, kfold, model, batch_size, model_name, epochs, dataloaders_dict):
        self.kfold=kfold
        self.model = model
        self.batch_size = batch_size
        self.model_name= model_name
        self.epochs = epochs
        self.dataloaders_dict = dataloaders_dict

    def model_config(self):
        '''model configurations'''
        self.model = cocpit.model_config.to_device(self.model)
        params_to_update = cocpit.model_config.update_params(self.model)
        self.optimizer = optim.SGD(params_to_update, lr=0.01, momentum=0.9, nesterov=True)
        self.criterion = nn.CrossEntropyLoss()  # Loss function
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=0, verbose=True, eps=1e-04
        )

    def phase(self):
        '''determine if there is both a training and validation phase'''
        if config.VALID_SIZE < 0.1:
            self.phases = ["train"]
        else:
            self.phases = ["train", "val"]

    def print_label_count(self, label_cnts_total, index, labels):
        '''print cumulative sum of images per class, per batch to
        ensure weighted sampler is working properly'''
        if self.phase == 'train':
            label_cnts = cocpit.model_config.label_counts(index, label_cnts_total, labels)
            label_cnts_total = list(map(operator.add, label_cnts, label_cnts_total))
            print(label_cnts_total)

    def batch_metrics(self, metrics):
        metrics.update_batch_metrics(self.loss, self.inputs, self.preds, self.labels)
        if (self.batch + 1) % 5 == 0:
            metrics.print_batch_metrics(
                self.labels, self.batch, self.phase, self.dataloaders_dict
            )

    def forward(self):
        ''' perform forward operator'''
        with torch.set_grad_enabled(self.phase == "train"):
            outputs = self.model(self.inputs)
            self.loss = self.criterion(outputs, self.labels)
            _, self.preds = torch.max(outputs, 1)

            if self.phase == "train":
                self.loss.backward()  # compute updates for each parameter
                self.optimizer.step()  # make the updates for each parameter

    def epoch_metrics(self, metrics, best_acc, epoch):
        '''log epoch metrics to comet and write to file
        also saves model if acc improves'''
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
        if (
            metrics.epoch_acc > best_acc
            and config.SAVE_MODEL
        ):
            best_acc = metrics.epoch_acc
            # save/load best model weights
            if not os.path.exists(config.MODEL_SAVE_DIR):
                os.makedirs(config.MODEL_SAVE_DIR)
            torch.save(self.model, config.MODEL_SAVENAME)
        return best_acc

    def train_batch(self, print_label_count=True):
        '''iterate over a batch in a dataloader and train'''

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

    def confusion_matrix(self, epoch):
        '''log confusion matrix'''
        if (
            epoch == self.epochs - 1
            and (config.KFOLD != 0 and self.kfold == config.KFOLD - 1)
            or (config.KFOLD == 0)
        ):
            cocpit.metrics.log_confusion_matrix(self.val_metrics)

    def classification_report(self, epoch):
        '''save classification report'''
        if epoch == self.epochs - 1:
            cocpit.metrics.sklearn_report(
                self.val_metrics,
                self.kfold,
                self.model_name,
        )

    def write_times(self, epoch, since_total):
        '''write out time to train to file'''
        time_elapsed = time.time() - since_total
        with open(
            "/data/data/saved_timings/model_timing_only_cpu.csv", "a", newline=""
        ) as file:
            writer = csv.writer(file)
            writer.writerow([self.model_name, epoch, self.kfold, time_elapsed])

    def print_time_all_epochs(self, since_total):
        '''print time it took for all epochs to train'''
        time_elapsed = time.time() - since_total
        print(
            "All epochs comlete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

    def print_time_one_epoch(self, since_epoch):
        '''print time for one epoch'''
        time_elapsed = time.time() - since_epoch
        print(
                "Epoch complete in {:.0f}m {:.0f}s".format(
                    time_elapsed // 60, time_elapsed % 60
                )
            )
    def reduce_lr(self):
        '''reduce learning rate upon plateau in epoch validation accuracy'''
        self.scheduler.step(self.val_metrics.epoch_acc)

    def train_model(self, norm_values=False):

        self.train_best_acc = 0.0
        self.val_best_acc = 0.0
        since_total = time.time()

        for epoch in range(self.epochs):
            since_epoch = time.time()
            print("-" * 20)

            self.phase()
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
                    mean, std = cocpit.model_config.normalization_values(self.dataloaders_dict, phase)

                # BATCH METRICS
                self.train_batch()
                if self.phase == 'train' or config.VALID_SIZE < 0.01:
                    self.batch_metrics(self.train_metrics)
                else:
                    self.batch_metrics()
                    # append batch prediction and labels for plots
                    self.val_metrics.all_preds.append(self.preds.cpu().numpy())
                    self.val_metrics.all_labels.append(self.labels.cpu().numpy())

                # EPOCH METRICS
                if (self.phase == "train" or config.VALID_SIZE < 0.01):
                    self.train_best_acc = self.epoch_metrics(self.train_metrics, self.train_best_acc, epoch)
                else:
                    self.val_best_acc = self.epoch_metrics(self.val_metrics, self.val_best_acc, epoch)

                self.reduce_lr()
            self.print_time_one_epoch(since_epoch)
        self.print_time_all_epochs(since_total)
        self.write_times(epoch, since_total)