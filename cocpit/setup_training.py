"""
train model with k folds for cross validation across samples
called in __main__.py
"""
import os
import random

import numpy as np
import torch
import torch.utils.data.sampler as samp

import cocpit
import cocpit.data_loaders as data_loaders

import cocpit.config as config  # isort: split
from collections import Counter

from sklearn.model_selection import StratifiedKFold, train_test_split

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Runner(cocpit.train_model.Train):
    kfold: int
    batch_size: int
    model_name: str
    model: Any  # torch.nn.parallel.data_parallel.DataParallel
    epochs: int
    train_data: Any  # torch.utils.data.Subset
    val_data: Any  # torch.utils.data.Subset
    train_labels: Any  # List[int]

    dataloaders: Any = field(
        default_factory=dict, init=False
    )  # torch.utils.data.DataLoader

    def update_save_names(self):
        """update save names for model and dataloader so that each fold gets saved"""
        config.VAL_LOADER_SAVENAME = (
            f"{config.VAL_LOADER_SAVE_DIR}e{config.MAX_EPOCHS}"
            f"_val_loader20_bs{config.BATCH_SIZE}"
            f"_k{str(self.kfold)}"
            f"_{len(config.MODEL_NAMES)}model(s).pt"
        )

        config.MODEL_SAVENAME = (
            f"{config.MODEL_SAVE_DIR}e{config.MAX_EPOCHS}"
            f"_bs{config.BATCH_SIZE}"
            f"_k{str(self.kfold)}"
            f"_{len(config.MODEL_NAMES)}model(s).pt"
        )

    def create_dataloaders(self, balance_weights: bool = True):
        """create dataloaders based on split from StratifiedKFold"""

        sampler = (
            data_loaders.balanced_sampler(self.train_labels)
            if balance_weights
            else None
        )
        train_loader = data_loaders.create_loader(
            self.train_data, batch_size=self.batch_size, sampler=sampler
        )

        if config.VALID_SIZE < 0.01:
            # use all data for training - no val loader
            val_loader = None
        else:
            val_sampler = samp.RandomSampler(self.val_data)
            val_loader = data_loaders.create_loader(
                self.val_data, batch_size=100, sampler=val_sampler
            )
            if config.SAVE_MODEL:
                data_loaders.save_valloader(self.val_data)

        self.dataloaders = {"train": train_loader, "val": val_loader}

    def run(self):
        self.update_save_names()
        self.create_dataloaders()
        self.model_config()
        self.train_model()


#############
def print_composition(train_labels, val_labels):
    """prints length of train and test data based on validation %"""
    print(len(train_labels), len(val_labels), len(train_labels) + len(val_labels))
    print("train counts")
    print(Counter(train_labels))
    print("val counts")
    print(Counter(val_labels))


def data_setup(train_indices, val_indices, composition: bool = True):
    """apply different transforms to train and
    validation datasets from ImageFolder"""
    data = data_loaders.get_data("train")
    train_labels = list(map(data.targets.__getitem__, train_indices))
    train_data = torch.utils.data.Subset(data, train_indices)

    data = data_loaders.get_data("val")
    val_data = torch.utils.data.Subset(data, val_indices)
    val_labels = list(map(data.targets.__getitem__, val_indices))

    if composition:
        print_composition(train_labels, val_labels)

    return train_data, val_data, train_labels


def kfold_training(batch_size, model_name, model, epochs):
    """
    1. split dataset into folds
        preserve the percentage of samples for each class with stratified
    2. create dataloaders
    3. initialize and train model
    """
    skf = StratifiedKFold(n_splits=config.KFOLD, shuffle=True, random_state=42)
    # datasets based on phase get called again in data_setup
    # needed here to initialize for skf.split
    data = data_loaders.get_data("val")
    for kfold, (train_indices, val_indices) in enumerate(
        skf.split(data.imgs, data.targets)
    ):
        print("KFOLD iteration: ", kfold)

        # apply appropriate transformations for training and validation sets
        train_data, val_data, train_labels = data_setup(train_indices, val_indices)

        r = Runner(
            kfold,
            batch_size,
            model_name,
            model,
            epochs,
            train_data,
            val_data,
            train_labels,
        )

        r.run()


def nofold_indices():
    """
    if not applying cross-fold validation, split training dataset
    based on config.VALID_SIZE
    shuffle first and then split dataset
    """

    total_files = sum(len(files) for r, d, files in os.walk(config.DATA_DIR))
    print(f"len files {total_files}")

    # randomly split indices for training and validation indices according to valid_size
    if config.VALID_SIZE < 0.01:
        # use all of the data
        train_indices = np.arange(0, total_files)
        random.shuffle(train_indices)
        val_indices = None
    else:
        train_indices, val_indices = train_test_split(
            list(range(total_files)), test_size=config.VALID_SIZE
        )
    return train_indices, val_indices


def nofold_training(batch_size, model_name, model, epochs, kfold=0):
    """
    execute training once through - no folds
    composition (bool): whether to print the length of train and test data based on validation %
    """
    train_indices, val_indices = nofold_indices()
    train_data, val_data, train_labels = data_setup(train_indices, val_indices)
    r = Runner(
        kfold, batch_size, model_name, model, epochs, train_data, val_data, train_labels
    )
    r.run()


def main(batch_size, model_name, epochs):

    model = cocpit.models.initialize_model(model_name)
    model = cocpit.model_config.to_device(model)
    if config.KFOLD != 0:
        kfold_training(batch_size, model_name, model, epochs)
    else:
        nofold_training(batch_size, model_name, model, epochs)
