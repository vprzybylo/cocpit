"""
train the CNN model(s)
"""
import time
import csv
import cocpit
import cocpit.config as config  # isort:split
from typing import Any, List


def model_eval(model, phase: str) -> None:
    """put model in evaluation mode if predicting on validation data"""
    model.train() if phase == "train" else model.eval()


def determine_phases() -> List[str]:
    """determine if there is both a training and validation phase"""
    return ["train"] if config.VALID_SIZE < 0.1 else ["train", "val"]


def write_output(
    filename: str,
    model_name: str,
    epoch: int,
    kfold: int,
    batch_size: int,
    epoch_acc: float,
    epoch_loss: float,
) -> None:
    """write acc and loss to file within epoch iteration"""
    if config.SAVE_ACC:
        with open(filename, "a", newline="") as file:
            writer = csv.writer(file)
            writer.writerow(
                [
                    model_name,
                    epoch,
                    kfold,
                    batch_size,
                    epoch_acc.cpu().numpy(),
                    epoch_loss,
                ]
            )
            file.close()


def run_train(
    dataloaders,
    optimizer,
    model,
    model_name: str,
    epoch: int,
    epochs: int,
    kfold: int,
    batch_size: int,
) -> None:
    """reset acc, loss, labels, and predictions for each epoch and each phase
    call methods in train.py"""
    t = cocpit.train.Train(dataloaders, optimizer, model)
    t.iterate_batches()
    t.epoch_metrics()
    t.log_epoch_metrics(
        "epoch_acc_train", "epoch_loss_train", t.epoch_acc, t.epoch_loss
    )
    print(
        f"Train Epoch {epoch + 1}/{epochs},\
        Loss: {t.epoch_loss:.3f},\
        Accuracy: {t.epoch_acc:.3f}"
    )
    write_output(
        config.ACC_SAVENAME_TRAIN,
        model_name,
        epoch,
        kfold,
        batch_size,
        t.epoch_acc,
        t.epoch_loss,
    )


def run_val(
    dataloaders,
    optimizer,
    model,
    model_name: str,
    epoch: int,
    epochs: int,
    kfold: int,
    batch_size: int,
) -> None:
    """reset acc, loss, labels, and predictions for each epoch and each phase
    call methods in validate.py"""
    v = cocpit.validate.Validation(dataloaders, optimizer, model)
    v.iterate_batches()
    v.epoch_metrics()
    v.reduce_lr()
    v.save_model()

    # make confusion matrix
    if (
        epoch == epochs - 1
        and (config.KFOLD != 0 and kfold == config.KFOLD - 1)
        or (config.KFOLD == 0)
    ):
        v.confusion_matrix()

    # make classification report
    if epoch == epochs - 1:
        v.classification_report(kfold, model_name)

    v.log_epoch_metrics("epoch_acc_val", "epoch_loss_val", v.epoch_acc, v.epoch_loss)
    print(
        f"Validation Epoch {epoch + 1}/{epochs},\
        Loss: {v.epoch_loss:.3f},\
        Accuracy: {v.epoch_acc:.3f}"
    )

    # output results to file for each model, epoch, fold, etc.
    write_output(
        config.ACC_SAVENAME_VAL,
        model_name,
        epoch,
        kfold,
        batch_size,
        v.epoch_acc,
        v.epoch_loss,
    )


def main(dataloaders, epochs, optimizer, model, model_name, batch_size, kfold=0):
    """calls above methods to train and validate across epochs and batches"""
    since_total = time.time()
    for epoch in range(epochs):
        since_epoch = time.time()
        for phase in determine_phases():
            print(f"Phase: {phase}")
            # put model in correct mode based on if training or validating
            model_eval(model, phase)

            if phase == "train" or config.VALID_SIZE < 0.01:
                run_train(
                    dataloaders,
                    optimizer,
                    model,
                    model_name,
                    epoch,
                    epochs,
                    kfold,
                    batch_size,
                )
            else:
                run_val(
                    dataloaders,
                    optimizer,
                    model,
                    model_name,
                    epoch,
                    epochs,
                    kfold,
                    batch_size,
                )

            timing = cocpit.timing.Time(epoch, since_total, since_epoch)
            timing.print_time_one_epoch(since_epoch)
    timing.print_time_all_epochs()
    timing.write_times(epoch)
