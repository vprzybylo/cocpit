"""
setup training and validation indices for labels
and data based on k-fold cross validation
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
class FoldSetup:
    """
    setup training and validation indices for labels
    and data based on k-fold cross validation
    """

    batch_size: int
    epochs: int
    kfold: int = 0

    train_data: Any = field(default=None, init=False)  # torch.utils.data.Subset
    val_data: Any = field(default=None, init=False)  # torch.utils.data.Subset
    train_labels: Any = field(default=None, init=False)  # List[int]
    val_labels: Any = field(default=None, init=False)
    train_indices: Any = field(default=None, init=False)
    val_indices: Any = field(default=None, init=False)

    def nofold_indices(self):
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
            self.train_indices = np.arange(0, total_files)
            random.shuffle(self.train_indices)
        else:
            self.train_indices, self.val_indices = train_test_split(
                list(range(total_files)), test_size=config.VALID_SIZE
            )

    def print_composition(self):
        """prints length of train and test data based on validation %"""
        print(
            len(self.train_labels),
            len(self.val_labels),
            len(self.train_labels) + len(self.val_labels),
        )
        print("train counts")
        print(Counter(self.train_labels))
        print("val counts")
        print(Counter(self.val_labels))

    def split_data(self, composition: bool = True):
        """
        Create subset of data and labels for train and val based on indices
        """
        data = data_loaders.get_data("train")
        self.train_labels = list(map(data.targets.__getitem__, self.train_indices))
        self.train_data = torch.utils.data.Subset(data, self.train_indices)

        data = data_loaders.get_data("val")
        self.val_data = torch.utils.data.Subset(data, self.val_indices)
        self.val_labels = list(map(data.targets.__getitem__, self.val_indices))

        if composition:
            self.print_composition()

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

    def kfold_training(self):
        """
        Split dataset into folds
        Preserve the percentage of samples for each class with stratified
        Create dataloaders for each fold
        """
        skf = StratifiedKFold(n_splits=config.KFOLD, shuffle=True, random_state=42)
        # datasets based on phase get called again in split_data
        # needed here to initialize for skf.split
        data = data_loaders.get_data("val")
        for self.kfold, (self.train_indices, self.val_indices) in enumerate(
            skf.split(data.imgs, data.targets)
        ):
            print("KFOLD iteration: ", self.kfold)

            # apply appropriate transformations for training and validation sets
            self.split_data()
            self.update_save_names()
            self.create_dataloaders()

    def train_loader(self, balance_weights: bool = True):
        """
        Create train loader that iterates images in batches
        Option to balance the distribution of sampled images
            if there is class imbalance
        """
        sampler = (
            data_loaders.balanced_sampler(self.train_labels)
            if balance_weights
            else None
        )
        return data_loaders.create_loader(
            self.train_data, batch_size=self.batch_size, sampler=sampler
        )

    def val_loader(self):
        """
        Create validation loader to be iterated in batches
        Option to use the entire labeled dataset if VALID_SIZE small
        """
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
        return val_loader

    def create_dataloaders(self):
        """create dict of train/val dataloaders based on split and sampler from StratifiedKFold"""
        self.dataloaders = {"train": self.train_loader(), "val": self.val_loader()}


def main(batch_size, epochs):
    f = FoldSetup(batch_size, epochs)
    if config.KFOLD != 0:
        f.kfold_training()
    else:
        f.nofold_indices()
        f.split_data()
        f.create_dataloaders()
