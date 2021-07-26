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
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torchvision import models

import cocpit.data_loaders as data_loaders
import cocpit.train_ML_model as train_ML_model


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


def set_parameter_requires_grad(model, feature_extract):
    """
    Flag for feature extracting
        when False, finetune the whole model,
        when True, only update the reshaped layer params
    """
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(
    model_name, num_classes, feature_extract=False, use_pretrained=False
):
    # all input size of 224
    if model_name == "resnet18":
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet34":
        model_ft = models.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "resnet152":
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)

    elif model_name == "alexnet":
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "vgg16":
        model_ft = models.vgg16_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "vgg19":
        model_ft = models.vgg19_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)

    elif model_name == "squeezenet":
        model_ft = models.squeezenet1_1(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(
            512, num_classes, kernel_size=(7, 7), stride=(2, 2)
        )
        # model_ft.num_classes = num_classes

    elif model_name == "densenet169":
        model_ft = models.densenet169(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    elif model_name == "densenet201":
        model_ft = models.densenet201(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)

    elif model_name == "efficient":
        model_ft = EfficientNet.from_name("efficientnet-b0")
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft


def train_val_composition(data, train_indices, val_indices):
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

    num_classes = len(params["class_names"])

    data = data_loaders.get_data(params["data_dir"])
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
                            + "models_no_blank"
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
                            + "models_vgg16_no_blank.pt"
                        )
                        # DATALOADERS
                        train_loader, val_loader = data_loaders.create_dataloaders(
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
                        model = initialize_model(model_name, num_classes)

                        # TRAIN MODEL
                        clf_report = train_ML_model.train_model(
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
                        + "models_no_blank_alltraindata"
                    )

                    val_loader_savename = (
                        "/data/data/saved_val_loaders/no_mask/"
                        + "val_loader"
                        + str(params["max_epochs"][0])
                        + "_bs"
                        + str(params["batch_size"][0])
                        + "_"
                        + str(len(params["model_names"]))
                        + "models_v1.0.0_no_blank.pt"
                    )

                    # DATALOADERS
                    train_loader, val_loader = data_loaders.create_dataloaders(
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
                    model = initialize_model(model_name, num_classes)

                    # TRAIN MODEL
                    clf_report = train_ML_model.train_model(
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
