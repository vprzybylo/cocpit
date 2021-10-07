"""
train model without folds for cross validation
called in __main__.py under build_model

isort:skip_file
"""
import cocpit  # isort: split

import cocpit.config as config

from sklearn.model_selection import StratifiedKFold, train_test_split


def main(data, batch_size, model_name, epochs, kfold=0):
    """
    create dataloaders
    initialize and train model
    """
    total_size = len(data)

    # randomly split indices for training and validation indices according to valid_size
    if config.VALID_SIZE < 0.01:
        # use all of the data
        train_indices = np.arange(0, total_size)
        random.shuffle(train_indices)
        val_indices = None
    else:
        train_indices, val_indices = train_test_split(
            list(range(total_size)), test_size=config.VALID_SIZE
        )

    # DATALOADERS
    train_loader, val_loader = cocpit.data_loaders.create_dataloaders(
        data, train_indices, val_indices, batch_size
    )

    dataloaders_dict = {"train": train_loader, "val": val_loader}

    # INITIALIZE MODEL
    model = cocpit.models.initialize_model(model_name)

    # TRAIN MODEL
    cocpit.train_model.train_model(
        kfold,
        model,
        batch_size,
        model_name,
        epochs,
        dataloaders_dict,
    )
