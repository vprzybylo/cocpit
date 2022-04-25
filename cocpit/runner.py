"""
train the CNN model(s)
"""
import time
import csv
import cocpit
import cocpit.config as config  # isort:split
from typing import List
from dataclasses import dataclass
from cocpit.performance_metrics import Metrics
import torch


@dataclass
class Runner(Metrics):
    """Iterate through epochs, batch sizes, models, kfolds, phases and train/validate"""

    def model_eval(self, phase: str) -> None:
        """
        Put model in evaluation mode if predicting on validation data

        Args:
            phase (str): "train" or "val"
        """
        self.model.train() if phase == "train" else self.model.eval()

    def determine_phases(self) -> List[str]:
        """determine if there is both a training and validation phase"""
        return ["train"] if config.VALID_SIZE < 0.1 else ["train", "val"]

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

    def run_train(self) -> None:
        """
        Run training loop and calculate metrics
        Reset acc, loss, labels, and predictions for each epoch, model, phase, and fold
        """
        t = cocpit.train.Train(
            self.dataloaders,
            self.optimizer,
            self.model,
            self.model_name,
            self.epoch,
            self.epochs,
            self.kfold,
            self.batch_size,
        )
        t.iterate_batches()
        t.epoch_metrics()
        t.log_epoch_metrics("epoch_acc_train", "epoch_loss_train")
        self.print_epoch_metrics("Train", self.epoch, self.epochs)
        self.write_output(config.ACC_SAVENAME_TRAIN)

    def run_val(self) -> None:
        """
        Run model on validation data and calculate metrics
        Reset acc, loss, labels, and predictions for each epoch, model, phase, and fold
        """
        v = cocpit.validate.Validation(
            self.dataloaders,
            self.optimizer,
            self.model,
            self.model_name,
            self.epoch,
            self.epochs,
            self.kfold,
            self.batch_size,
        )
        v.iterate_batches()
        v.epoch_metrics()
        v.reduce_lr()
        v.save_model()

        # confusion matrix
        if (
            self.epoch == self.epochs - 1
            and (config.KFOLD != 0 and self.kfold == config.KFOLD - 1)
            or (config.KFOLD == 0)
        ):
            v.confusion_matrix()

        # classification report
        if self.epoch == self.epochs - 1:
            v.classification_report(self.kfold, self.model_name)

        v.log_epoch_metrics("epoch_acc_val", "epoch_loss_val")
        self.print_epoch_metrics("Validation", self.epoch, self.epochs)
        self.write_output(config.ACC_SAVENAME_VAL)


def main(
    dataloaders: torch.utils.data.DataLoader,
    optimizer: torch.optim.SGD,
    model: torch.nn.parallel.DataParallel,
    epochs: int,
    model_name: str,
    batch_size: int,
    kfold: int = 0,
) -> None:
    """
    Calls above methods to train and validate across epochs and batches

    Args:
        dataloaders (torch.utils.data.DataLoader): training and validation dict that loads images with sampling procedure
        optimizer (torch.optim.SGD): an algorithm that modifies the attributes of the neural network
        model (torch.nn.parallel.data_parallel.DataParallel): saved and loaded model
        epochs (int): max number of epochs to train and validate
        model_name (str): name of model architecture
        batch_size (int): number of images read into memory at a time
        kfold (int): number of k-folds used in resampling procedure
    """
    since_total = time.time()
    for epoch in range(epochs):
        since_epoch = time.time()
        r = Runner(
            dataloaders, optimizer, model, model_name, epoch, epochs, kfold, batch_size
        )
        for phase in r.determine_phases():
            # put model in correct mode based on if training or validating
            r.model_eval(phase)
            if phase == "train" or config.VALID_SIZE < 0.01:
                r.run_train()
            else:
                r.run_val()
            timing = cocpit.timing.Time(since_total, since_epoch)
            timing.print_time_one_epoch()
    timing.print_time_all_epochs()
    timing.write_times(model_name, kfold)
