"""
Retrives data loaders from Pytorch
"""

import cocpit.config as config  # isort: split
import os
import torch
import torch.utils.data
import torch.utils.data.sampler as sampler
from PIL import Image, ImageFile
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from typing import List, Union, Optional
from cocpit.auto_str import auto_str
import numpy as np
import pandas as pd
import random

ImageFile.LOAD_TRUNCATED_IMAGES = True


class DatasetCSV(torch.utils.data.Dataset):
    def __init__(self, root, transform, labels, imgs):
        self.root = root
        self.transform = transform
        self.labels = labels
        self.imgs = imgs

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        image = Image.open(os.path.join(self.root, self.imgs.iloc[index]))
        self.tensor_image = self.transform(image)
        self.label = self.labels[index]
        return self.tensor_image, self.label


def get_data(phase: str) -> DatasetCSV:
    """
    - Use the Pytorch ImageFolder class to read in training data
    - Training data needs to be organized all in one folder with subfolders for each class
    - Applies transforms and data augmentation

    Args:
        phase (str): 'train' or 'val'
    Returns:
        data (tuple): (image, label, path)
    """

    transform_dict = {
        "train": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                # transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    }
    if phase == "train":
        df_from_csv = pd.read_csv(config.DATA_DIR_PREDEFINED_TRAIN)
    elif phase == "val":
        df_from_csv = pd.read_csv(config.DATA_DIR_PREDEFINED_VAL)
    else:
        df_from_csv = pd.read_csv(config.DATA_DIR_PREDEFINED_TEST)

    labels = df_from_csv["cat"]
    imgs = df_from_csv["path"]

    return DatasetCSV(
        root=config.DATA_DIR,
        transform=transform_dict[phase],
        labels=labels,
        imgs=imgs,
    )


def balanced_sampler(train_labels: List[int]) -> sampler.WeightedRandomSampler:
    """
    - Creates weights for each class for use in the dataloader sampler argument
    - Lower count classes are sampled more frequently and higher count classes are sampled less frequently
    - Only used in the training dataloader

    Args:
        train_labels (List[int]): numerically labeled classes for training dataset

    Returns:
        class_sample_counts (List): number of samples per class
        train_samples_weights (torch.DoubleTensor): weights for each class for sampling
    """
    # class_sample_counts = [0] * len(config.CLASS_NAMES)

    # for target in train_labels.values:
    #     print(target)
    #     class_sample_counts[target] += 1

    class_sample_counts = train_labels.value_counts()
    print(
        "counts per class in training data (before sampler): ",
        class_sample_counts,
    )

    class_weights = 1.0 / class_sample_counts

    train_samples_weights = [
        float(class_weights[class_id]) for class_id in train_labels.values
    ]

    return sampler.WeightedRandomSampler(
        train_samples_weights, len(train_samples_weights), replacement=True
    )


def seed_worker(worker_id) -> None:
    torch_seed = torch.initial_seed()
    random.seed(torch_seed + worker_id)
    if torch_seed >= 2 ** 30:  # make sure torch_seed + workder_id < 2**32
        torch_seed = torch_seed % 2 ** 30
    np.random.seed(torch_seed + worker_id)


def create_loader(
    data: torch.utils.data.Subset,
    batch_size: int,
    sampler: Optional[torch.utils.data.Sampler],
    pin_memory: bool = True,
) -> torch.utils.data.DataLoader:
    """
    Make an iterable of batches across a dataset

    Args:
        data (torch.utils.data.Subset): the dataset to load
        batch_size (int): number of images to be read into memory at a time
        sampler (torch.utils.data.Sampler): the method used to iterate over indices of dataset (e.g., random shuffle)
        pin_memory (bool): For data loading, passing pin_memory=True to a DataLoader will automatically
                           put the fetched data Tensors in pinned memory, and thus enables faster data
                           transfer to CUDA-enabled GPUs.
    Returns:
        torch.utils.data.DataLoader: a dataset to be iterated over using sampling strategy

    """
    g = torch.Generator()
    g.manual_seed(0)

    return torch.utils.data.DataLoader(
        data,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
    )


def save_valloader(val_data: torch.utils.data.Subset) -> None:
    """
    Save validation dataloader based on paths in config.py

    Args:
        val_data (torch.utils.data.Subset): the validation dataset
    """
    if not os.path.exists(config.VAL_LOADER_SAVE_DIR):
        os.makedirs(config.VAL_LOADER_SAVE_DIR)
    torch.save(val_data, config.VAL_LOADER_SAVENAME)
