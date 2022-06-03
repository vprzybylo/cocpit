import numpy as np
import torch
from cocpit.performance_metrics import Metrics
from dataclasses import dataclass
from typing import Dict
from cocpit import config as config
import csv


class Train(Metrics):
    """Perform training methods on batched dataset

    Args:
        f (cocpit.fold_setup.FoldSetup): instance of FoldSetup class
        model_name (str): name of model architecture
        c (model_config.ModelConfig): instance of ModelConfig class
        epoch (int): epoch index in training loop
        epochs (int): total epochs for training loop
    """

    def __init__(self, f, epoch, epochs, model_name, kfold, batch_size, c):

        super().__init__(f, epoch, epochs)
        self.model_name: str = model_name
        self.kfold = kfold
        self.batch_size = batch_size
        self.c = c

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
            outputs = self.c.model(self.inputs)
            self.loss = self.c.criterion(outputs, self.labels)
            _, self.preds = torch.max(outputs, 1)
            self.loss.backward()  # compute updates for each parameter
            self.c.optimizer.step()  # make the updates for each parameter

    def iterate_batches(self, print_label_count: bool = False) -> None:
        """iterate over a batch in a dataloader and train

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

    def run(self):
        self.iterate_batches()
        self.epoch_metrics()
        self.log_epoch_metrics("epoch_acc_train", "epoch_loss_train")
        self.print_epoch_metrics("Train")
        self.write_output(config.ACC_SAVENAME_TRAIN)
