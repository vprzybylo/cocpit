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
class FoldSetup:
    batch_size: int
    model_name: str
    model: Any
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

    def data_setup(self, composition: bool = True):
        """apply different transforms to train and
        validation datasets from ImageFolder"""
        data = data_loaders.get_data("train")
        self.train_labels = list(map(data.targets.__getitem__, self.train_indices))
        self.train_data = torch.utils.data.Subset(data, self.train_indices)

        data = data_loaders.get_data("val")
        self.val_data = torch.utils.data.Subset(data, self.val_indices)
        self.val_labels = list(map(data.targets.__getitem__, self.val_indices))

        if composition:
            self.print_composition()

    def kfold_training(self):
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
        for self.kfold, (self.train_indices, self.val_indices) in enumerate(
            skf.split(data.imgs, data.targets)
        ):
            print("KFOLD iteration: ", self.kfold)

            # apply appropriate transformations for training and validation sets
            self.data_setup()

            r = Runner(self.batch_size, self.model_name, self.model, self.epochs)
            r.run()


@dataclass
class Runner(FoldSetup, cocpit.train_model.Train):

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


def main(batch_size, model_name, epochs):

    model = cocpit.models.initialize_model(model_name)
    model = cocpit.model_config.to_device(model)
    f = FoldSetup(batch_size, model_name, model, epochs)
    if config.KFOLD != 0:
        f.kfold_training()
    else:
        f.nofold_indices()
        f.data_setup()
        r = Runner(f.batch_size, f.model_name, f.model, f.epochs)
        r.run()
