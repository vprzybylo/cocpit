#!/usr/bin/env python
# coding: utf-8

"""
Build and train the ML model for different CNNs to predict
ice crystal type
"""
import os
import random
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn

import cocpit


def set_random_seed(random_seed):
    if random_seed is not None:
        print("Set random seed as {}".format(random_seed))
        os.environ["PYTHONHASHSEED"] = str(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.set_num_threads(1)
        cudnn.benchmark = False
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )


def train_val_composition(data, train_indices, val_indices):
    '''
    confirms length of train and test data based on validation %
    '''
    train_y = list(map(data.targets.__getitem__, train_indices))
    test_y = list(map(data.targets.__getitem__, val_indices))
    print(len(train_y), len(test_y), len(train_y) + len(test_y))
    print("train counts")
    print(Counter(train_y))
    print("val counts")
    print(Counter(test_y))


########## MAIN ###########


def main(
    params,
    experiment,
    acc_savename_train,
    acc_savename_val,
    save_acc,
    save_model,
    metrics_savename,
    valid_size,
    num_workers,
):
    if experiment is not None:  # comet logging
        log_exp = True
    else:
        log_exp = False

    num_classes = len(params["class_names"])

    data = cocpit.data_loaders.get_data(params["data_dir"])
    fold_report = []  # holds classification metric performances per kfold

    for batch_size in params["batch_size"]:
        print("BATCH SIZE: ", batch_size)
        for model_name in params["model_names"]:
            print("MODEL: ", model_name)
            for epochs in params["max_epochs"]:
                print("MAX EPOCH: ", epochs)

                # K-FOLD
                if params["kfold"] != 0:
                    # preserve the percentage of samples for each class with stratified
                    skf = StratifiedKFold(
                        n_splits=params["kfold"], shuffle=True, random_state=42
                    )
                    for i, (train_indices, val_indices) in enumerate(
                        skf.split(data.imgs, data.targets)
                    ):
                        print("KFOLD iteration: ", i)
                        # train_val_composition(data, train_indices, val_indices)
                        model_savename = (
                            "/data/data/saved_models/no_mask/"
                            + "e"
                            + str(params["max_epochs"][0])
                            + "_bs"
                            + str(params["batch_size"][0])
                            + "_k"
                            + str(i)
                            + "_"
                            + str(len(params["model_names"]))
                            + "models_no_blank_vgg16_v1.3.0"
                        )
                        val_loader_savename = (
                            "/data/data/saved_val_loaders/no_mask/"
                            + "val_loader"
                            + str(params["max_epochs"][0])
                            + "_bs"
                            + str(params["batch_size"][0])
                            + "_k"
                            + str(i)
                            + "_"
                            + str(len(params["model_names"]))
                            + "models_vgg16_v1.3.0.pt"
                        )
                        # DATALOADERS
                        (
                            train_loader,
                            val_loader,
                        ) = cocpit.data_loaders.create_dataloaders(
                            data,
                            train_indices,
                            val_indices,
                            batch_size,
                            save_model,
                            val_loader_savename,
                            class_names=params["class_names"],
                            data_dir=params["data_dir"],
                            valid_size=valid_size,
                            num_workers=num_workers,
                        )

                        dataloaders_dict = {"train": train_loader, "val": val_loader}

                        # INITIALIZE MODEL
                        model = cocpit.models.initialize_model(model_name, num_classes)

                        # TRAIN MODEL
                        clf_report = cocpit.train_model.train_model(
                            experiment,
                            log_exp,
                            model,
                            i,
                            batch_size,
                            params["class_names"],
                            model_name,
                            model_savename,
                            acc_savename_train,
                            acc_savename_val,
                            save_acc,
                            save_model,
                            dataloaders_dict,
                            epochs,
                            num_classes,
                            valid_size=valid_size,
                        )
                        fold_report.append(clf_report)
                    # concatenate all metric reports from each fold and model and write
                    concat_df = pd.concat(fold_report)
                    if save_acc:
                        concat_df.to_csv(metrics_savename, mode="a")

                else:  # no kfold
                    i = 0  # kfold false for savename
                    total_size = len(data)
                    # randomly split indices for training and validation indices according to valid_size
                    if valid_size < 0.01:
                        # use all of the data
                        train_indices = np.arange(0, total_size)
                        random.shuffle(train_indices)
                        val_indices = None
                    else:
                        train_indices, val_indices = train_test_split(
                            list(range(total_size)), test_size=valid_size
                        )
                    # train_val_composition(data, train_indices, val_indices)

                    model_savename = (
                        "/data/data/saved_models/no_mask/"
                        + "e"
                        + str(params["max_epochs"][0])
                        + "_bs"
                        + str(params["batch_size"][0])
                        + "_"
                        + str(len(params["model_names"]))
                        + "models_vgg_16_v1.3.0"
                    )

                    val_loader_savename = (
                        "/data/data/saved_val_loaders/no_mask/"
                        + "val_loader"
                        + str(params["max_epochs"][0])
                        + "_bs"
                        + str(params["batch_size"][0])
                        + "_"
                        + str(len(params["model_names"]))
                        + "models_vgg_16_v1.3.0.pt"
                    )

                    # DATALOADERS
                    train_loader, val_loader = cocpit.data_loaders.create_dataloaders(
                        data,
                        train_indices,
                        val_indices,
                        batch_size,
                        save_model,
                        val_loader_savename,
                        class_names=params["class_names"],
                        data_dir=params["data_dir"],
                        valid_size=valid_size,
                        num_workers=num_workers,
                    )

                    dataloaders_dict = {"train": train_loader, "val": val_loader}

                    # INITIALIZE MODEL
                    model = cocpit.models.initialize_model(model_name, num_classes)

                    # TRAIN MODEL
                    clf_report = cocpit.train_model.train_model(
                        experiment,
                        log_exp,
                        model,
                        i,
                        batch_size,
                        params["class_names"],
                        model_name,
                        model_savename,
                        acc_savename_train,
                        acc_savename_val,
                        save_acc,
                        save_model,
                        dataloaders_dict,
                        epochs,
                        num_classes,
                        valid_size=valid_size,
                    )

                    if save_acc:
                        clf_report.to_csv(metrics_savename, mode="a")


if __name__ == "__main__":

    main(
        params,
        log_exp,
        acc_savename_train,
        acc_savename_val,
        metrics_savename,
        save_acc,
        save_model,
        valid_size,
        num_workers,
        num_classes,
    )
