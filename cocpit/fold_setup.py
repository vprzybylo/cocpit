import os
import random

import numpy as np
import pandas as pd
import torch
import torch.utils.data.sampler as samp

import cocpit
import cocpit.data_loaders as data_loaders

import cocpit.config as config  # isort: split
from collections import Counter

from sklearn.model_selection import train_test_split

from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class FoldSetup:
    """
    Setup training and validation dataloaders based on k-fold cross validation. Called in __main__.py

    Args:
        batch_size (int): number of images read into memory at a time
        kfold (int): fold index in k-fold cross validation loop
        train_indices (List[int]): list of indices of data for training
        val_indices (List[int]): list of indices of data for validation
    """

    batch_size: int
    kfold: int
    train_indices: List[int] = field(default_factory=list)
    val_indices: List[int] = field(default_factory=list)

    dataloaders: Dict[str, torch.utils.data.DataLoader] = field(init=False)
    train_data: torch.utils.data.Subset = field(init=False)
    val_data: torch.utils.data.Subset = field(init=False)
    train_labels: List[int] = field(init=False)
    val_labels: List[int] = field(init=False)

    def print_composition(self) -> None:
        """
        Prints length of train and test data based on validation %
        defined in config.py
        """
        print(
            len(self.train_labels),
            len(self.val_labels),
            len(self.train_labels) + len(self.val_labels),
        )
        print("train counts")
        print(Counter(self.train_labels))
        print("val counts")
        print(Counter(self.val_labels))

    def split_data(self, composition: bool = False) -> None:
        """
        Create a subset of data and labels for training
        and validation based on indices

        Args:
            composition (bool): whether to print the class totals for each dataset
        """
        self.train_data = data_loaders.get_data("train")
        self.val_data = data_loaders.get_data("val")

        if composition:
            self.print_composition()

    def update_save_names(self) -> None:
        """
        Update config save names for model and validation dataloader
        so that each fold gets saved"""
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

    def train_loader(
        self, balance_weights: bool = True
    ) -> torch.utils.data.DataLoader:
        """
        - Create train loader that iterates images in batches
        - Balance the distribution of sampled images given imbalance

        Args:
            balance_weights (bool): pull from training dataset evenly among classes
        Returns:
            torch.utils.data.DataLoader: an iterable dataloader for training
        """
        sampler = (
            data_loaders.balanced_sampler(self.train_data.labels)
            if balance_weights
            else None
        )
        return data_loaders.create_loader(
            self.train_data, batch_size=self.batch_size, sampler=sampler
        )

    def val_loader(self) -> None:
        """
        - Create validation loader to be iterated in batches
        - Option to use the entire labeled dataset if config.VALID_SIZE small
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

    def create_dataloaders(self) -> None:
        """Create dict of train/val dataloaders based on split and sampler from StratifiedKFold"""
        self.dataloaders = {
            "train": self.train_loader(),
            "val": self.val_loader(),
        }
