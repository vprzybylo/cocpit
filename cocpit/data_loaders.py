'''
Retrives data loaders from Pytorch for training and validation data
'''
import itertools
import numpy as np
import torch
from torchvision import datasets, transforms
import torch.utils.data.sampler as sampler

class ImageFolderWithPaths(datasets.ImageFolder):
    """Custom dataset that includes image file paths. Extends
    torchvision.datasets.ImageFolder
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns 
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return (tuple_with_path, index)
    
def get_data(data_dir):
    '''
    Use the Pytorch ImageFolder class to read in root directory
    that holds subfolders of each class for training data
    Applies transforms
    Params
    ------
    data_dir (str): root dir for training data
    Returns
    -------
    data (tuple): (sample, target) where target is class_index of the target class 
    '''
    all_transforms = transforms.Compose([transforms.Resize(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225])])

    data = ImageFolderWithPaths(root=data_dir, transform=all_transforms)
    return data


def make_weights_for_balanced_classes(train_labels, nclasses):
    '''
    creates weights for each class for sampler such that lower count classes 
    are sampled more frequently and higher count classes are sampled less
    Params
    ------
    - train_target (): labels of training dataset
    - nclasses (int): number of classes
    Returns
    -------
    - class_sample_counts (list): # of samples per class
    - train_samples_weights (list): weights for each class for sampling
    '''
    # only weight the training dataset 

    class_sample_counts = [0] * nclasses
    for target in train_labels:  
        class_sample_counts[target] += 1
    print('counts per class: ', class_sample_counts)

    class_weights = 1./torch.Tensor(class_sample_counts)
    train_samples_weights = [class_weights[class_id] for class_id in train_labels]

    return class_sample_counts, torch.DoubleTensor(train_samples_weights)


def create_dataloaders(data, train_indices, val_indices, batch_size,
                       save_model, val_loader_savename, 
                       class_names, data_dir, valid_size=0.2, num_workers=20,
                       shuffle=True):

    '''
    get dataloaders
    Params
    -----
    - data (tuple): (sample, target) where target is class_index of the target class 
    - train_indices (list): training dataset indices
    - val_indices (list): validation dataset indices
    - class_names (list): list of strings of classes
    - data_dir (str): directory for training dataset
    - batch_size (int): batch size for dataloader
    - num_workers (int): # of cpus to be used during data loading
    - shuffle (bool): whether to shuffle the data per epoch
    - valid_size (float): % of data used for validation dataset (0.0-1.0 = 0%-100%)
    Returns
    -------
    - train_loader (obj): dataloader iterable for training dataset
    - val_loader (obj): dataloader iterable for validation dataset
    '''

    # Get a list of labels according to train_indices to obtain weighting for sampling
    train_labels = list(map(data.targets.__getitem__, train_indices))

    # Make a training and validation dataset of images and labels according to indices
    train_data = torch.utils.data.Subset(data, train_indices)
    val_data = torch.utils.data.Subset(data, val_indices)

    # For an unbalanced dataset create a weighted sampler 
    class_counts, train_samples_weights = make_weights_for_balanced_classes(train_labels,
                                                                            len(class_names))
    # Make a sampler to undersample classes with the highest counts
    train_sampler = sampler.WeightedRandomSampler(train_samples_weights,
                                                  len(train_samples_weights),
                                                  replacement=True)
    
    # Make an iterable of batches across the training dataset
    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=batch_size,        
                                               sampler=train_sampler,
                                               num_workers=num_workers,
                                               pin_memory=True)
    
    if valid_size < 0.01:
        # use all data for training - no val loader
        return train_loader, None
    else:

        # Make an iterable of batches across the validation dataset
        val_loader = torch.utils.data.DataLoader(val_data,
                                                 batch_size=batch_size,
                                                 shuffle=shuffle,
                                                 num_workers=num_workers,
                                                 pin_memory=True)
        if save_model:
            torch.save(val_data, val_loader_savename)

        return train_loader, val_loader


def get_test_loader(datadir,
                    batch_size,
                    num_workers,
                    shuffle=True,
                    pin_memory=True):
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
    - data_loader: test set iterator
    """
    # resizing helps memory usage
    transforms_ = transforms.Compose([transforms.Resize((224,224)), 
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

    all_data_wpath = ImageFolderWithPaths(datadir, transform=transforms_)

    testloader = torch.utils.data.DataLoader(all_data_wpath, pin_memory=True,
                                             shuffle=shuffle, batch_size=batch_size,
                                             num_workers=num_workers)
    return testloader

def get_val_loader_predictions(model, val_data, device, batch_size, shuffle=True, num_workers=20):
    '''
    get a list of hand labels and predictions from a saved dataloader/model
    Params
    ------
    - model (obj): torch.nn.parallel.data_parallel.DataParallel loaded from saved file
    - val_data (obj): Loads an object saved with torch.save() from a file
    - device (obj): use cuda if available
    - batch_size (int): how many samples per batch to load
    - shuffle (bool): whether to shuffle the dataset after every epoch.
    - num_workers (int): number of subprocesses to use when loading the dataset
    
    Returns
    -------
    - all_preds (list): predictions from a model
    - all_labels (list): correct/hand labels 
    '''
    #transforms already applied in get_data() before saving
    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=batch_size,
                                             shuffle=shuffle,
                                             num_workers=num_workers,
                                             pin_memory=True)
    all_preds= []
    all_labels = []
    with torch.no_grad():

        for batch_idx, ((imgs, labels, img_paths), index) in enumerate(val_loader):
            # get the inputs
            inputs = imgs.to(device)
            labels = labels.to(device)

            output = model(inputs)
            pred = torch.argmax(output, 1)

            all_preds.append(pred.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
    
    all_preds = np.asarray(list(itertools.chain(*all_preds)))
    all_labels = np.asarray(list(itertools.chain(*all_labels)))
    
    return all_preds, all_labels

