"""
Retrives data loaders from Pytorch for training and validation data
"""

import cocpit.config as config  # isort: split

import itertools
import os
from collections import Counter

import numpy as np
import torch
import torch.utils.data.sampler as sampler
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets, transforms


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


def make_weights_for_balanced_classes(train_labels):
    """
    creates weights for each class for sampler such that lower count classes
    are sampled more frequently and higher count classes are sampled less
    Params
    ------
    - train_labels (list): labels of training dataset
    Returns
    -------
    - class_sample_counts (list): # of samples per class
    - train_samples_weights (list): weights for each class for sampling
    """

    # only weight the training dataset
    class_sample_counts = [0] * len(config.CLASS_NAMES)
    for target in train_labels:
        class_sample_counts[target] += 1
    print("counts per class in training data: ", class_sample_counts)

    class_weights = 1.0 / torch.Tensor(class_sample_counts)
    train_samples_weights = [class_weights[class_id] for class_id in train_labels]

    return class_sample_counts, torch.DoubleTensor(train_samples_weights)


def create_dataloaders(
    data,
    train_indices,
    val_indices,
    batch_size,
    shuffle=True,
    balance_weights=True,
):

    """
    get dataloaders
    Params
    -----
    - data (tuple): (sample, target) where target
        is class_index of the target class
    - train_indices (list): training dataset indices
    - val_indices (list): validation dataset indices
    - batch_size (int): batch size for dataloader
    - shuffle (bool): whether to shuffle the data per epoch
    - balance_weights (bool): True creates a weighted sampler for class imbalance
    Returns
    -------
    - train_loader (obj): dataloader iterable for training dataset
    - val_loader (obj): dataloader iterable for validation dataset
    """

    # Get a list of labels according to train_indices to obtain weighting for sampling
    train_labels = list(map(data.targets.__getitem__, train_indices))

    # Make a training and validation dataset of images and labels according to indices
    train_data = torch.utils.data.Subset(data, train_indices)
    val_data = torch.utils.data.Subset(data, val_indices)

    if balance_weights:
        # For an unbalanced dataset create a weighted sampler
        class_counts, train_samples_weights = make_weights_for_balanced_classes(
            train_labels
        )

        # Make a sampler to undersample classes with the highest counts
        train_sampler = sampler.WeightedRandomSampler(
            train_samples_weights, len(train_samples_weights), replacement=True
        )

        # Make an iterable of batches across the training dataset
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            sampler=train_sampler,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )
    else:
        # Make an iterable of batches across the training dataset
        train_loader = torch.utils.data.DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=config.NUM_WORKERS,
            pin_memory=True,
        )

    if config.VALID_SIZE < 0.01:
        # use all data for training - no val loader
        return train_loader, None
    # Make an iterable of batches across the validation dataset
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )
    if config.SAVE_MODEL:
        torch.save(val_data, config.VAL_LOADER_SAVENAME)

    return train_loader, val_loader


def get_test_loader_df(
    open_dir, file_list, batch_size=100, shuffle=False, pin_memory=True
):
    """
    Utility function for loading and returning a multi-process test iterator
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir (str): path directory to the dataset.
    - batch_size (int): how many samples per batch to load.
    - num_workers (int): number of subprocesses to use when loading the dataset.
    - shuffle (bool): whether to shuffle the dataset after every epoch.
    - pin_memory (bool): whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator (image, path)
    """

    test_data = TestDataSet(open_dir, file_list)

    return torch.utils.data.DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=pin_memory,
    )


def get_val_loader_predictions(model, val_data, batch_size, shuffle=True):
    """
    get a list of hand labels and predictions from a saved dataloader/model
    Params
    ------
    - model (obj): torch.nn.parallel.data_parallel.DataParallel loaded from saved file
    - val_data (obj): Loads an object saved with torch.save() from a file
    - batch_size (int): how many samples per batch to load
    - shuffle (bool): whether to shuffle the dataset after every epoch.

    Returns
    -------
    - all_preds (list): predictions from a model
    - all_labels (list): correct/hand labels
    """
    # transforms already applied in get_data() before saving
    val_loader = torch.utils.data.DataLoader(
        val_data,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=config.NUM_WORKERS,
        pin_memory=True,
    )

    all_preds = []
    all_labels = []
    with torch.no_grad():

        for batch_idx, ((imgs, labels, img_paths), index) in enumerate(val_loader):
            # get the inputs
            inputs = imgs.to(config.DEVICE)
            labels = labels.to(config.DEVICE)

            output = model(inputs)
            pred = torch.argmax(output, 1)

            all_preds.append(pred.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.asarray(list(itertools.chain(*all_preds)))
    all_labels = np.asarray(list(itertools.chain(*all_labels)))

    return all_preds, all_labels
