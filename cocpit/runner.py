"""
train the CNN model(s)
"""
import time
import cocpit
import cocpit.config as config  # isort:split
from typing import List, Dict
import torch


def determine_phases() -> List[str]:
    """determine if there is both a training and validation phase

    Returns:
        List[str]: either training or training and validation phase"""
    return ["train"] if config.VALID_SIZE < 0.1 else ["train", "val"]


def main(
    f: cocpit.fold_setup.FoldSetup,
    c: cocpit.model_config.ModelConfig,
    model_name: str,
    epochs: int,
    kfold: int,
    batch_size: int,
) -> None:
    """
    Trains and validates a model across epochs, phases, and batches

    Args:
        f (cocpit.fold_setup.FoldSetup): instance of FoldSetup class
        c (model_config.ModelConfig): instance of ModelConfig class
        model_name (str): name of model architecture
        epochs (int): total epochs for training loop
        batch_size (int): number of images read into memory at a time

    """
    since_total = time.time()
    val_best_acc = 0.0
    for epoch in range(epochs):
        since_epoch = time.time()
        t = cocpit.timing.EpochTime(since_total, since_epoch)
        for phase in determine_phases():
            # put model in correct mode based on if training or validating
            c.model.train() if phase == "train" else c.model.eval()
            if phase == "train":
                train = cocpit.train.Train(
                    f, epoch, epochs, model_name, kfold, batch_size, c
                )
                train.run()
            else:
                val = cocpit.validate.Validation(
                    f,
                    epoch,
                    epochs,
                    kfold,
                    val_best_acc,
                )
                val_best_acc = val.run()
            t.print_time_one_epoch()
    try:
        t.print_time_all_epochs()
        t.write_times(f.model_name, f.kfold)
    except NameError:
        print("Number of epochs needs to increase")
