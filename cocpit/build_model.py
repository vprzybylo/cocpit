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

########## MAIN ###########


def main(
    params,
    experiment,
    acc_savename_train,
    acc_savename_val,
    metrics_savename,
    save_acc,
    save_model,
    valid_size,
    num_workers,
    num_classes,
):

    data = cocpit.data_loaders.get_data(params["data_dir"])

    # loop through batch sizes, models, epochs, and/or folds
    for batch_size in params["batch_size"]:
        print("BATCH SIZE: ", batch_size)
        for model_name in params["model_names"]:
            print("MODEL: ", model_name)
            for epochs in params["max_epochs"]:
                print("MAX EPOCH: ", epochs)

                # K-FOLD
                if params["kfold"] != 0:
                    cocpit.kfold_training.main(
                        data,
                        params,
                        experiment,
                        acc_savename_train,
                        acc_savename_val,
                        metrics_savename,
                        save_acc,
                        save_model,
                        valid_size,
                        num_workers,
                        num_classes,
                    )
                else:  # no kfold
                    cocpit.no_fold_training.main(
                        data,
                        params,
                        experiment,
                        acc_savename_train,
                        acc_savename_val,
                        metrics_savename,
                        save_acc,
                        save_model,
                        valid_size,
                        num_workers,
                        num_classes,
                    )


if __name__ == "__main__":

    main(
        params,
        experiment,
        acc_savename_train,
        acc_savename_val,
        metrics_savename,
        save_acc,
        save_model,
        valid_size,
        num_workers,
        num_classes,
    )
