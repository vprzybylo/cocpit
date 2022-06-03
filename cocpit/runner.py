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
    dataloaders: Dict[str, torch.utils.data.DataLoader],
    optimizer: torch.optim.SGD,
    model: torch.nn.parallel.DataParallel,
    epochs: int,
    model_name: str,
    batch_size: int,
    kfold: int = 0,
) -> None:
    """
    Trains and validates a model across epochs, phases, and batches

    Args:
        dataloaders (Dict[str, torch.utils.data.DataLoader]): training and validation dict that loads images with sampling procedure
        optimizer (torch.optim.SGD): an algorithm that modifies the attributes of the neural network
        model (torch.nn.parallel.data_parallel.DataParallel): saved and loaded model
        epochs (int): max number of epochs to train and validate
        model_name (str): name of model architecture
        batch_size (int): number of images read into memory at a time
        kfold (int): number of k-folds used in resampling procedure
    """
    since_total = time.time()
    val_best_acc = 0.0
    for epoch in range(epochs):
        since_epoch = time.time()
        t = cocpit.timing.EpochTime(since_total, since_epoch)
        for phase in determine_phases():
            # put model in correct mode based on if training or validating
            model = model.train() if phase == "train" else model.eval()
            if phase == "train":
                train = cocpit.train.Train(
                    dataloaders,
                    optimizer,
                    model,
                    model_name,
                    epoch,
                    epochs,
                    kfold,
                    batch_size,
                )
                train.run()
            else:
                val = cocpit.validate.Validation(
                    dataloaders,
                    optimizer,
                    model,
                    model_name,
                    epoch,
                    epochs,
                    kfold,
                    batch_size,
                    val_best_acc,
                )
                val_best_acc = val.run()
            t.print_time_one_epoch()
    try:
        t.print_time_all_epochs()
        t.write_times(model_name, kfold)
    except NameError:
        print("Number of epochs needs to increase")
