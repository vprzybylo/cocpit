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

from cocpit.auto_str import auto_str


@auto_str
class Runner:
    def __init__(self, model_name, epochs, kfold, train_data, val_data, batch_size):
        self.model_name = model_name
        self.epochs = epochs
        self.kfold = kfold
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = batch_size

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

    def create_dataloaders(self, train_labels, balance_weights: bool = True):
        '''create dataloaders based on split from StratifiedKFold'''

        sampler = (
            data_loaders.balanced_sampler(train_labels) if balance_weights else None
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

        dataloaders_dict = {"train": train_loader, "val": val_loader}
        return dataloaders_dict

    def train_model(self, dataloaders_dict):
        '''train model'''
        t = cocpit.train_model.Train(
            self.kfold,
            self.model,
            self.batch_size,
            self.model_name,
            self.epochs,
            dataloaders_dict,
        )
        t.model_config()
        t.train_model()


#############
def print_composition(train_labels, val_labels):
    '''prints length of train and test data based on validation %'''
    print(len(train_labels), len(val_labels), len(train_labels) + len(val_labels))
    print("train counts")
    print(Counter(train_labels))
    print("val counts")
    print(Counter(val_labels))


def data_setup(train_indices, val_indices, composition: bool = False):
    '''apply different transforms to train and
    validation datasets from ImageFolder'''
    data = data_loaders.get_data('train')
    print(len(data))
    train_labels = list(map(data.targets.__getitem__, train_indices))
    train_data = torch.utils.data.Subset(data, train_indices)

    data = data_loaders.get_data('val')
    val_data = torch.utils.data.Subset(data, val_indices)
    val_labels = list(map(data.targets.__getitem__, val_indices))

    if composition:
        print_composition(train_labels, val_labels)

    return train_data, val_data, train_labels


def kfold_training(batch_size, model_name, epochs):
    '''
    1. split dataset into folds
        preserve the percentage of samples for each class with stratified
    2. create dataloaders
    3. initialize and train model
    '''
    skf = StratifiedKFold(n_splits=config.KFOLD, shuffle=True, random_state=42)
    # datasets based on phase get called again in data_setup
    # needed here to initialize for skf.split
    data = data_loaders.get_data('val')
    for kfold, (train_indices, val_indices) in enumerate(
        skf.split(data.imgs, data.targets)
    ):
        print("KFOLD iteration: ", kfold)

        # apply appropriate transformations for training and validation sets
        train_data, val_data, train_labels = data_setup(train_indices, val_indices)

        execute = Runner(model_name, epochs, kfold, train_data, val_data, batch_size)

        execute.initialize_model()
        execute.update_save_names()
        dataloaders_dict = execute.create_dataloaders(train_labels)
        execute.train_model(dataloaders_dict)


def nofold_indices():
    '''
    if not applying cross-fold validation, split training dataset
    based on config.VALID_SIZE
    shuffle first and then split dataset'''

    total_files = sum([len(files) for r, d, files in os.walk(config.DATA_DIR)])
    print(f'len files {total_files}')

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


def nofold_training(batch_size, model_name, epochs, kfold=0):
    '''
    execute training once through - no folds
    composition (bool): whether to print the length of train and test data based on validation %
    '''
    train_indices, val_indices = nofold_indices()
    train_data, val_data, train_labels = data_setup(train_indices, val_indices)

    execute = Runner(model_name, epochs, kfold, train_data, val_data, batch_size)
    execute.initialize_model()
    execute.update_save_names()
    dataloaders_dict = execute.create_dataloaders(train_labels)
    execute.train_model(dataloaders_dict)


def main(batch_size, model_name, epochs):

    if config.KFOLD != 0:
        kfold_training(batch_size, model_name, epochs)
    else:
        nofold_training(batch_size, model_name, epochs)
