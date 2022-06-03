import numpy as np
import torch
from cocpit.performance_metrics import Metrics
from dataclasses import dataclass
from typing import Dict
from cocpit import config as config


class Train(Metrics):
    """Perform training methods on batched dataset"""

    def __init__(
        self,
        dataloaders,
        optimizer,
        model,
        model_name,
        epoch,
        epochs,
        kfold,
        batch_size,
    ):

        super().__init__(
            dataloaders, optimizer, model, model_name, epoch, epochs, kfold, batch_size
        )

    def label_counts(self, label_cnts: np.ndarray, labels: torch.Tensor):
        """
        Calculate the # of labels per batch to ensure weighted random sampler is correct

        Args:
            label_cnts (np.ndarray): number of labels per class from all batches before
            labels (torch.Tensor): class/label names
        Returns:
            label_cnts (List[int]): sum of label counts from prior batches plus current batch
        """

        for n, _ in enumerate(config.CLASS_NAMES):
            label_cnts[n] += len(np.where(labels.numpy() == n)[0])
        print("LABEL COUNT = ", label_cnts)
        return label_cnts

    def forward(self) -> None:
        """perform forward operator and make predictions"""
        with torch.set_grad_enabled(True):
            outputs = self.model(self.inputs)
            self.loss = self.criterion(outputs, self.labels)
            _, self.preds = torch.max(outputs, 1)
            self.loss.backward()  # compute updates for each parameter
            self.optimizer.step()  # make the updates for each parameter

    def iterate_batches(self, print_label_count: bool = False) -> None:
        """iterate over a batch in a dataloader and train

        Args:
            print_label_count (bool): if True print class counts when iterating batches
        """

        label_cnts_total = np.zeros(len(config.CLASS_NAMES))
        for self.batch, ((inputs, labels, _), _) in enumerate(
            self.dataloaders["train"]
        ):
            if print_label_count:
                self.label_counts(label_cnts_total, labels)

            self.inputs = inputs.to(config.DEVICE)
            self.labels = labels.to(config.DEVICE)

            # zero the parameter gradients
            self.optimizer.zero_grad()
            self.forward()
            self.batch_metrics()
            if (self.batch + 1) % 5 == 0:
                self.print_batch_metrics("train")

    def run(self):
        self.iterate_batches()
        self.epoch_metrics()
        self.log_epoch_metrics("epoch_acc_train", "epoch_loss_train")
        self.print_epoch_metrics("Train")
        self.write_output(config.ACC_SAVENAME_TRAIN)
