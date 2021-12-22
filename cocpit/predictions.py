import itertools
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

import cocpit.config as config
import cocpit.data_loaders as data_loaders
from cocpit.auto_str import auto_str


@auto_str
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


def test_loader(open_dir, file_list, batch_size=100, shuffle=False, pin_memory=True):
    """
    Loads and returns a multi-process test iterator
    """

    test_data = TestDataSet(open_dir, file_list)
    loaders = data_loaders.Loader(train_labels=None, batch_size=batch_size)
    return loaders.create_loader(test_data, sampler=None)


def val_loader_predictions(model, val_data, batch_size, shuffle=True):
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

    loaders = data_loaders.Loader(train_labels=None, batch_size=batch_size)
    val_loader = loaders.create_loader(val_data, sampler=None)

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
