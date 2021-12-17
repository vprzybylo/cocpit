import torch
import numpy as np

def get_test_loader_df(
    open_dir, file_list, batch_size=100, shuffle=False, pin_memory=True
):
    """
    Utility function for loading and returning a multi-process test iterator
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