"""
Retrives data loaders from Pytorch for training and validation data
"""

from cocpit.kfold_training import Runner
import cocpit.config as config  # isort: split
from cocpit.auto_str import auto_str
import os

import numpy as np
import torch
import torch.utils.data.sampler as sampler
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import datasets, transforms

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_data():
    """
    Use the Pytorch ImageFolder class to read in root directory
    that holds subfolders of each class for training data
    Applies transforms
    Params
    ------
    data_dir (str): root dir for training data
    Returns
    -------
    data (tuple): (image, label, path)
    """
    all_transforms = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    return ImageFolderWithPaths(root=config.DATA_DIR, transform=all_transforms)


class ImageFolderWithPaths(datasets.ImageFolder):
    """
    Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = original_tuple + (path,)
        return (tuple_with_path, index)


class TestDataSet(Dataset):
    """
    dataloader for new unseen data
    """

    def __init__(self, open_dir, file_list):

        self.open_dir = open_dir
        self.file_list = list(file_list)
        self.transform = transforms.Compose(
            [
                transforms.Resize((224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        self.path = os.path.join(self.open_dir, self.file_list[idx])
        image = Image.open(self.path)
        tensor_image = self.transform(image)
        return (tensor_image, self.path)


@auto_str
class Loader(Runner):
    """
    creates training and validation Pytorch dataloaders
    option to weight based on class count
    """
    def __init__(
        self,
        data,
        train_indices,
        val_indices,
        batch_size,
        shuffle=True,
    ):
        self.data=data
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.train_labels = list(map(self.data.targets.__getitem__, train_indices))
        self.train_data = torch.utils.data.Subset(self.data, train_indices)
        self.val_data = torch.utils.data.Subset(self.data, val_indices)

    def make_weights_for_balanced_classes(self):
        """
        creates weights for each class for sampler such that lower count classes
        are sampled more frequently and higher count classes are sampled less
        Returns
        -------
        - class_sample_counts (list): # of samples per class
        - train_samples_weights (list): weights for each class for sampling
        """

        # only weight the training dataset
        class_sample_counts = [0] * len(config.CLASS_NAMES)
        for target in self.train_labels:
            class_sample_counts[target] += 1
        print("counts per class in training data: ", class_sample_counts)

        class_weights = 1.0 / torch.Tensor(class_sample_counts)
        train_samples_weights = [
            class_weights[class_id] for class_id in self.train_labels
        ]

        return class_sample_counts, torch.DoubleTensor(train_samples_weights)

    def balance_weights(self):
        '''create a training dataloader for unbalanced class counts'''

        # For an unbalanced dataset create a weighted sampler
        class_counts, train_samples_weights = self.make_weights_for_balanced_classes()

        # Make a sampler to undersample classes with the highest counts
        train_sampler = sampler.WeightedRandomSampler(
            train_samples_weights, len(train_samples_weights), replacement=True
        )
        return train_sampler

    def create_trainloader(self, train_sampler):
        '''Make an iterable of batches across the training dataset'''
        train_loader = torch.utils.data.DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            sampler=train_sampler,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )
        return train_loader

    def create_valloader(self):
        '''Make an iterable of batches across the validation dataset'''
        val_loader = torch.utils.data.DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )
        return val_loader

    def save_valloader(self):
        '''save validation dataloader based on paths in config.py'''
        if not os.path.exists(config.VAL_LOADER_SAVE_DIR):
            os.makedirs(config.VAL_LOADER_SAVE_DIR)
        torch.save(self.val_data, config.VAL_LOADER_SAVENAME)

