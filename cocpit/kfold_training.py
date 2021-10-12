"""
train model with k folds for cross validation across samples
called in __main__.py
"""

import cocpit

import cocpit.config as config  # isort: split

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split


def train_val_composition(data, train_indices, val_indices):
    """
    confirms length of train and test data based on validation %
    """
    train_y = list(map(data.targets.__getitem__, train_indices))
    test_y = list(map(data.targets.__getitem__, val_indices))
    print(len(train_y), len(test_y), len(train_y) + len(test_y))
    print("train counts")
    print(Counter(train_y))
    print("val counts")
    print(Counter(test_y))


def main(data, batch_size, model_name, epochs):
    """
    split dataset into folds
    create dataloaders
    initialize and train model
    save classification report
    """

    fold_report = []  # holds classification metric performances per kfold

    # preserve the percentage of samples for each class with stratified
    skf = StratifiedKFold(n_splits=config.KFOLD, shuffle=True, random_state=42)
    for kfold, (train_indices, val_indices) in enumerate(
        skf.split(data.imgs, data.targets)
    ):
        print("KFOLD iteration: ", kfold)
        # train_val_composition(data, train_indices, val_indices)

        config.VAL_LOADER_SAVENAME = (
            f"{config.MODEL_SAVE_DIR}e{config.MAX_EPOCHS}"
            f"_val_loader20_bs{config.BATCH_SIZE}"
            f"_k{str(kfold)}_vgg16.pt"
        )

        config.MODEL_SAVENAME = (
            f"{config.MODEL_SAVE_DIR}e{config.MAX_EPOCHS}"
            f"_bs{config.BATCH_SIZE}"
            f"_k{str(kfold)}_vgg16.pt"
        )

        # DATALOADERS based on split from StratifiedKFold
        (train_loader, val_loader,) = cocpit.data_loaders.create_dataloaders(
            data,
            train_indices,
            val_indices,
            batch_size,
        )

        dataloaders_dict = {"train": train_loader, "val": val_loader}

        # INITIALIZE MODEL
        model = cocpit.models.initialize_model(model_name)

        # TRAIN MODEL
        clf_report = cocpit.train_model.train_model(
            kfold,
            model,
            batch_size,
            model_name,
            epochs,
            dataloaders_dict,
        )
        fold_report.append(clf_report)

    # concatenate all metric reports from each fold and model and write
    concat_df = pd.concat(fold_report)
    if config.SAVE_ACC:
        concat_df.to_csv(config.METRICS_SAVENAME, mode="a")
