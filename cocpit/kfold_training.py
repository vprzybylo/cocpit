"""
train model with k folds for cross validation across samples
called in __main__.py
"""

import torch
import cocpit
import cocpit.data_loaders as data_loaders

# import cocpit.auto_str as auto_str
import cocpit.config as config  # isort: split
from cocpit.auto_str import auto_str

from collections import Counter

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


@auto_str
class Runner:
    def __init__(
        self, model_name, epochs, kfold, data, train_indices, val_indices, batch_size
    ):
        self.model_name = model_name
        self.epochs = epochs
        self.kfold = kfold
        self.data = data
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.batch_size = batch_size

    def train_val_composition(self):
        """
        prints length of train and test data based on validation %
        """
        train_y = list(map(self.data.targets.__getitem__, self.train_indices))
        test_y = list(map(self.data.targets.__getitem__, self.val_indices))
        print(len(train_y), len(test_y), len(train_y) + len(test_y))
        print("train counts")
        print(Counter(train_y))
        print("val counts")
        print(Counter(test_y))

    def update_save_names(self):
        '''update save names for model and dataloader so that each fold gets saved'''
        config.VAL_LOADER_SAVENAME = (
            f"{config.VAL_LOADER_SAVE_DIR}e{config.MAX_EPOCHS}"
            f"_val_loader20_bs{config.BATCH_SIZE}"
            f"_k{str(self.kfold)}_vgg16.pt"
        )
        config.MODEL_SAVENAME = (
            f"{config.MODEL_SAVE_DIR}e{config.MAX_EPOCHS}"
            f"_bs{config.BATCH_SIZE}"
            f"_k{str(self.kfold)}_vgg16.pt"
        )

    def initialize_model(self):
        self.model = cocpit.models.initialize_model(self.model_name)

    def create_dataloaders(self, balance_weights: bool =True) -> dict[str, torch.utils.data.dataloader.DataLoader]:
        '''create dataloaders based on split from StratifiedKFold'''
        loaders = data_loaders.Loader(
            self.data,
            self.train_indices,
            self.val_indices,
            self.batch_size,
        )
        train_sampler = loaders.balance_weights() if balance_weights else None
        train_loader = loaders.create_trainloader(train_sampler)

        if config.VALID_SIZE < 0.01:
            # use all data for training - no val loader
            val_loader = None
        else:
            val_loader = loaders.create_valloader()
            if config.SAVE_MODEL:
                loaders.save_valloader()

        dataloaders_dict = {"train": train_loader, "val": val_loader}
        return dataloaders_dict

    def train_model(self, dataloaders_dict: dict) -> None:
        '''train model'''
        cocpit.train_model.train_model(
            self.kfold,
            self.model,
            self.batch_size,
            self.model_name,
            self.epochs,
            dataloaders_dict,
        )


def main(data, batch_size, model_name, epochs, composition=False):
    """
    split dataset into folds
    create dataloaders
    initialize and train model
    save classification report
    """
    # preserve the percentage of samples for each class with stratified
    skf = StratifiedKFold(n_splits=config.KFOLD, shuffle=True, random_state=42)
    for kfold, (train_indices, val_indices) in enumerate(
        skf.split(data.imgs, data.targets)
    ):
        print("KFOLD iteration: ", kfold)

        execute = Runner(
            model_name, epochs, kfold, data, train_indices, val_indices, batch_size
        )
        if composition:
            execute.train_val_composition()
        execute.initialize_model()
        execute.update_save_names()
        dataloaders_dict = execute.create_dataloaders()
        execute.train_model(dataloaders_dict)
