"""Train model across batches"""
import numpy as np
import torch

import cocpit
from cocpit import config as config
from cocpit.performance_metrics import Metrics
from cocpit import config as config
import csv
import torch.nn.functional as F
import torch.nn as nn
from typing import Any
import os

class Train(Metrics):
    """Perform training methods on batched dataset

    Args:
        f (cocpit.fold_setup.FoldSetup): instance of FoldSetup class
        epoch (int): epoch index in training loop
        epochs (int): total epochs for training loop
        model_name (str): name of model architecture
        kfold (int): number of folds use in k-fold cross validation
        c (model_config.ModelConfig): instance of ModelConfig class
    """

    def __init__(
        self,
        f: cocpit.fold_setup.FoldSetup,
        epoch: int,
        epochs: int,
        model_name: str,
        kfold: int,
        c: cocpit.model_config.ModelConfig,
    ):
        super().__init__(f, epoch, epochs)
        self.model_name = model_name
        self.kfold = kfold
        self.c = c

    def label_counts(
        self, label_cnts: np.ndarray, labels: torch.Tensor
    ) -> np.ndarray:
        """
        Calculate the # of labels per batch to ensure weighted random sampler is correct

        Args:
            label_cnts (np.ndarray): number of labels per class from all batches before
            labels (torch.Tensor): class/label names
        Returns:
            label_cnts (np.ndarray): sum of label counts from prior batches plus current batch
        """

        for n, _ in enumerate(config.CLASS_NAMES):
            label_cnts[n] += len(np.where(labels.numpy() == n)[0])
        print("LABEL COUNT = ", label_cnts)
        return label_cnts

    def forward(self) -> None:
        """Perform forward operator and make predictions"""
        with torch.set_grad_enabled(True):
            outputs = self.c.model(self.inputs)
            if config.EVIDENTIAL:
                y = torch.eye(len(config.CLASS_NAMES)).to(config.DEVICE)
                y = y[self.labels]
                self.loss = cocpit.loss.edl_digamma_loss(
                    outputs, y.float(), self.epochs
                )
            else:
                self.criterion = nn.CrossEntropyLoss()
                self.loss = self.criterion(outputs, self.labels)
            _, self.preds = torch.max(outputs, 1)
            # self.probs = F.softmax(outputs, dim=1).max(dim=1)
            # self.uncertainty(outputs)
            self.loss.backward()  # compute updates for each parameter
            self.c.optimizer.step()  # make the updates for each parameter

    def uncertainty(self, outputs: torch.Tensor) -> None:
        """
        Calculate uncertainty, which is inversely proportional to the total evidence
        Model more confident the more evidence output by relu activation

        Args:
            outputs (torch.Tensor): output from model
        """
        evidence = F.relu(outputs)
        alpha = (
            evidence + 1
        )  # alpha summed over classes is the Dirichlet strength
        # uncertainty
        self.u = len(config.CLASS_NAMES) / torch.sum(
            alpha, dim=1, keepdim=True
        )

    def iterate_batches(self, print_label_count: bool = False) -> None:
        """Iterate over a batch in a dataloader and train

        Args:
            print_label_count (bool): if True print class counts when iterating batches
        """

        label_cnts_total = np.zeros(len(config.CLASS_NAMES))
        for self.batch, ((inputs, labels, _), _) in enumerate(
            self.f.dataloaders["train"]
        ):
            if print_label_count:
                self.label_counts(label_cnts_total, labels)

            self.inputs = inputs.to(config.DEVICE)
            self.labels = labels.to(config.DEVICE)

            # zero the parameter gradients
            self.c.optimizer.zero_grad()
            self.forward()
            self.batch_metrics()
            if (self.batch + 1) % 5 == 0:
                self.print_batch_metrics("train")

    def write_output(self) -> None:
        """
        Write acc and loss to csv file within model, epoch, kfold iteration
        """
        
        with open(config.ACC_SAVENAME_TRAIN, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    self.model_name,
                    self.epoch,
                    self.kfold,
                    self.f.batch_size,
                    self.epoch_acc.cpu().numpy(),
                    self.epoch_loss,
                ]
            )
            file.close()

    def run(self) -> None:
        """Train model and save output"""
        self.iterate_batches()
        self.epoch_metrics()
        self.log_epoch_metrics("epoch_acc_train", "epoch_loss_train")
        self.print_epoch_metrics("Train")
        if config.SAVE_ACC:
            if not os.path.exists(config.ACC_SAVE_DIR):
                os.makedirs(config.ACC_SAVE_DIR)
            self.write_output()
