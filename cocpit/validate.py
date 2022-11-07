"""validate on batches"""
import os
import torch
from cocpit.performance_metrics import Metrics
from cocpit import config as config
import cocpit
from torch.optim.lr_scheduler import ReduceLROnPlateau
import csv
from ray import tune
import torch.nn.functional as F
from torch import nn


class Validation(Metrics):
    """Perform validation methods on batched dataset

    Args:
        f (cocpit.fold_setup.FoldSetup): instance of FoldSetup class
        epoch (int): epoch index in training loop
        epochs (int): total epochs for training loop
        model_name (str): name of model architecture
        k_outer (int): index of outer k-fold cross validation (test/train)
        k_inner (int): index of inner k-fold cross validation (val/train)
        val_best_acc (float): highest validation accuracy across epochs
        c (model_config.ModelConfig): instance of ModelConfig class
    """

    def __init__(
        self,
        f,
        epoch,
        epochs,
        model_name,
        k_outer,
        k_inner,
        val_best_acc,
        c,
    ):
        super().__init__(f, epoch, epochs)
        self.model_name = model_name
        self.k_outer = k_outer
        self.k_inner = k_inner
        self.val_best_acc = val_best_acc
        self.c = c
        self.epoch_preds = []  # validation preds for 1 epoch for plotting
        self.epoch_labels = []  # validation labels for 1 epoch for plotting
        self.epoch_probs = (
            []
        )  # validation probabilities for 1 epoch for plotting
        self.epoch_uncertainties = (
            []
        )  # validation uncertainty for 1 epoch for plotting

    def predict(self) -> None:
        """make predictions"""

        with torch.no_grad():

            outputs = self.c.model(self.inputs)
            if config.EVIDENTIAL:
                self.loss = cocpit.loss.categorical_evidential_loss(
                    outputs, self.labels, self.epochs
                )
            else:
                self.criterion = nn.CrossEntropyLoss()
                self.loss = self.criterion(outputs, self.labels)
            _, self.preds = torch.max(outputs, 1)
            self.probs = F.softmax(outputs, dim=1).max(dim=1).values
            self.uncertainty(outputs)

    def uncertainty(self, outputs) -> None:
        """
        Calculate uncertainty, which is inversely proportional to the total evidence
        Model more confident the more evidence output by relu activation
        """
        evidence = F.relu(outputs)
        alpha = evidence + 1
        # uncertainty
        self.u = len(config.CLASS_NAMES) / torch.sum(
            alpha, dim=1, keepdim=True
        )
        # print("uncertainty", self.u)

    def append_preds(self) -> None:
        """save each batch prediction and labels for plots"""
        self.epoch_preds.append(self.preds.cpu().tolist())
        self.epoch_labels.append(self.labels.cpu().tolist())
        self.epoch_probs.append(self.probs)
        self.epoch_uncertainties.append(self.u)

    def save_model(self) -> float:
        """save/load best model weights after improvement in val accuracy"""
        if self.epoch_acc > self.val_best_acc and config.SAVE_MODEL:
            print(
                f"Epoch acc:{self.epoch_acc} > best acc: {self.val_best_acc}."
                " Saving model."
            )
            self.val_best_acc = self.epoch_acc

            torch.save(self.c.model, config.MODEL_SAVENAME)
        if config.TUNE:
            tune.report(loss=self.epoch_loss, accuracy=self.epoch_acc)
        return self.val_best_acc

    def reduce_lr(self) -> None:
        """reduce learning rate upon plateau in epoch validation accuracy"""
        scheduler = ReduceLROnPlateau(
            self.c.optimizer,
            mode="max",
            factor=0.5,
            patience=0,
            verbose=True,
            eps=1e-04,
        )
        scheduler.step(self.epoch_acc)

    def iterate_batches(self, batch_size: int) -> None:
        """
        iterate over a batch in a dataloader and make predictions

        Args:
            batch_size (int): size of the samples fed into memory
        """
        self.f.create_dataloaders(batch_size)
        for self.batch, ((inputs, labels, _), _) in enumerate(
            self.f.dataloaders["val"]
        ):

            self.inputs = inputs.to(config.DEVICE)
            self.labels = labels.to(config.DEVICE)

            # zero the parameter gradients
            self.c.optimizer.zero_grad()
            self.predict()
            self.batch_metrics()
            self.append_preds()
            if (self.batch + 1) % 5 == 0:
                self.print_batch_metrics("val")

    def write_output(self, filename: str, batch_size: int) -> None:
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
                        self.k_outer,
                        self.k_inner,
                        batch_size,
                        self.epoch_acc.cpu().numpy(),
                        self.epoch_loss,
                    ]
                )
                file.close()

    def run(self, batch_size: int) -> None:
        """
        Run model on validation data and calculate metrics
        Reset acc, loss, labels, and predictions for each epoch, model, phase, and fold
        """
        self.iterate_batches(batch_size)
        self.epoch_metrics()
        if not config.TUNE:
            self.reduce_lr()
        val_best_acc = self.save_model()

        self.log_epoch_metrics("epoch_acc_val", "epoch_loss_val")
        self.print_epoch_metrics("Validation")
        self.write_output(config.ACC_SAVENAME_VAL, batch_size)
        return val_best_acc
