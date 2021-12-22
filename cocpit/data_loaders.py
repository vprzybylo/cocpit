"""
Retrives data loaders from Pytorch for training and validation data
"""

import cocpit.config as config  # isort: split
import os

import numpy as np
import torch
import torch.utils.data.sampler as sampler
from PIL import ImageFile
from torchvision import datasets, transforms

from cocpit.auto_str import auto_str

ImageFile.LOAD_TRUNCATED_IMAGES = True


def get_data(phase):
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

    transform_dict = {
        'train': transforms.Compose(
            [
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        'val': transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    }

    return ImageFolderWithPaths(root=config.DATA_DIR, transform=transform_dict[phase])


@auto_str
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


@auto_str
class Loader:
    """
    creates training and validation Pytorch dataloaders
    option to weight based on class count
    """

    def __init__(self, train_labels, batch_size):
        self.train_labels = train_labels
        self.batch_size = batch_size

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
        print(
            "counts per class in training data (before sampler): ", class_sample_counts
        )

        class_weights = 1.0 / torch.Tensor(class_sample_counts)
        train_samples_weights = [
            class_weights[class_id] for class_id in self.train_labels
        ]

        return class_sample_counts, torch.DoubleTensor(train_samples_weights)

    def balanced_sampler(self):
        '''create a training dataloader for unbalanced class counts'''

        # For an unbalanced dataset create a weighted sampler
        class_counts, train_samples_weights = self.make_weights_for_balanced_classes()

        # Make a sampler to undersample classes with the highest counts
        return sampler.WeightedRandomSampler(
            train_samples_weights, len(train_samples_weights), replacement=True
        )

    def create_loader(self, data, sampler, pin_memory=True):
        '''Make an iterable of batches across either
        the training or validation dataset'''
        return torch.utils.data.DataLoader(
            data,
            batch_size=self.batch_size,
            sampler=sampler,
            num_workers=config.NUM_WORKERS,
            pin_memory=pin_memory,
        )

    def save_valloader(self, val_data):
        '''save validation dataloader based on paths in config.py'''
        if not os.path.exists(config.VAL_LOADER_SAVE_DIR):
            os.makedirs(config.VAL_LOADER_SAVE_DIR)
        torch.save(val_data, config.VAL_LOADER_SAVENAME)
