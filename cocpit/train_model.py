"""
train the CNN model(s)
"""

import copy
import csv
import itertools
import operator
import time

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import cocpit.classification_metrics as classification_metrics
import cocpit.train_metrics as metrics


def set_dropout(model, drop_rate=0.1):
    """
    technique to fight overfitting and improve neural network generalization
    """
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Dropout):
            child.p = drop_rate
        set_dropout(child, drop_rate=drop_rate)


def to_device(device, model):
    """
    push model to gpu(s) if available
    """
    # Send the model to GPU
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)
    return model


def count_parameters(model):
    '''
    number of model parameters to update
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def update_params(model, feature_extract=False):
    """
    When feature extracting, we only want to update the parameters
    of the last layer, or in other words, we only want to update the
    parameters for the layer(s) we are reshaping. Therefore, we do
    not need to compute the gradients of the parameters that we are
    not changing, so for efficiency we set the .requires_grad attribute
    to False. This is important because by default, this attribute is
    set to True. Then, when we initialize the new layer and by default
    the new parameters have .requires_grad=True so only the new layer’s
    parameters will be updated. When we are finetuning we can leave all
    of the .required_grad’s set to the default of True.
    """
    params_to_update = model.parameters()
    if feature_extract:
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params_to_update.append(param)
                # print("\t",name)

    return params_to_update


def label_counts(i, labels, num_classes):
    """
    Calculate the # of labels per batch to ensure
    weighted random sampler is correct
    """
    label_cnts = [0] * len(range(num_classes))
    for n in range(len(range(num_classes))):
        label_cnts[n] += len(np.where(labels.numpy() == n)[0])

    for n in range(len(range(num_classes))):
        # print("batch index {}, {} counts: {}".format(
        i, n, (labels == n).sum()
    print('LABEL COUNT = ', label_cnts)

    return label_cnts


def get_normalization_values(dataloaders_dict, phase):
    '''
    Get mean and standard deviation of pixel values
    across all batches
    '''
    mean = 0.0
    std = 0.0
    nb_samples = 0.0
    for ((inputs, labels, paths), index) in dataloaders_dict[phase]:
        batch_samples = inputs.size(0)
        data = inputs.view(batch_samples, inputs.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples
    print(mean, std)
    return mean, std


def train_model(
    experiment,
    log_exp,
    model,
    kfold,
    batch_size,
    class_names,
    model_name,
    model_savename,
    acc_savename_train,
    acc_savename_val,
    save_acc,
    save_model,
    dataloaders_dict,
    epochs,
    valid_size=0.2,
    clf_report=None,
):

    set_dropout(model, drop_rate=0.0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = to_device(device, model)
    params_to_update = update_params(model)
    print("model params: ", count_parameters(model))

    optimizer = optim.SGD(params_to_update, lr=0.01, momentum=0.9, nesterov=True)

    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=0, verbose=True, eps=1e-04
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    copy.deepcopy(model.state_dict())
    best_acc_val = 0.0
    best_acc_train = 0.0
    since_total = time.time()

    for epoch in range(epochs):
        since_epoch = time.time()
        # print('Epoch {}/{}'.format(epoch+1,num_epochs))
        print("-" * 20)

        # Each epoch has a training and validation phase
        if valid_size < 0.1:
            phases = ["train"]
        else:
            phases = ["train", "val"]

        for phase in phases:
            print("Phase: {}".format(phase))
            totals_train = 0
            totals_val = 0
            running_loss_train = 0.0
            running_loss_val = 0.0
            running_corrects_train = 0
            running_corrects_val = 0
            all_preds = []
            all_labels = []
            # label_cnts_total = np.zeros(len(class_names))

            if phase == "train":
                model.train()
            else:
                model.eval()

            # get pytorch transform normalization values per channel
            # iterates over training
            # mean, std = get_normalization_values(dataloaders_dict, phase)

            for i, ((inputs, labels, paths), index) in enumerate(
                dataloaders_dict[phase]
            ):
                # uncomment to print cumulative sum of images per class, per batch
                # ensures weighted sampler is working properly
                # if phase == 'train':
                # label_cnts = label_counts(i, labels, num_classes)
                # label_cnts_total = list(map(operator.add, label_cnts, label_cnts_total))
                # print(label_cnts_total)

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()  # a clean up step for PyTorch

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "val":
                        all_preds.append(preds.cpu().numpy())
                        all_labels.append(labels.cpu().numpy())

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()  # compute updates for each parameter
                        optimizer.step()  # make the updates for each parameter

                # calculate batch metrics
                if phase == "train":
                    (
                        running_loss_train,
                        running_corrects_train,
                        totals_train,
                    ) = metrics.batch_train_metrics(
                        i,
                        loss,
                        inputs,
                        preds,
                        labels,
                        running_loss_train,
                        running_corrects_train,
                        totals_train,
                        dataloaders_dict,
                        phase,
                    )
                else:
                    (
                        running_loss_val,
                        running_corrects_val,
                        totals_val,
                    ) = metrics.batch_val_metrics(
                        i,
                        loss,
                        inputs,
                        preds,
                        labels,
                        running_loss_val,
                        running_corrects_val,
                        totals_val,
                        dataloaders_dict,
                        phase,
                    )

            # calculate epoch metrics
            if phase == "train":
                epoch_loss_train, epoch_acc_train = metrics.epoch_train_metrics(
                    experiment,
                    running_loss_train,
                    totals_train,
                    running_corrects_train,
                    scheduler,
                    log_exp,
                    save_acc,
                    acc_savename_train,
                    model_name,
                    epoch,
                    epochs,
                    kfold,
                    batch_size,
                )

                if (
                    valid_size < 0.01
                    and epoch_acc_train > best_acc_train
                    and save_model
                ):
                    best_acc_train = epoch_acc_train
                    # save/load best model weights
                    torch.save(model, model_savename)

            else:
                epoch_loss_val, epoch_acc_val = metrics.epoch_val_metrics(
                    experiment,
                    running_loss_val,
                    totals_val,
                    running_corrects_val,
                    scheduler,
                    log_exp,
                    save_acc,
                    acc_savename_val,
                    model_name,
                    epoch,
                    epochs,
                    kfold,
                    batch_size,
                )
                # deep copy the model
                if epoch_acc_val > best_acc_val and save_model:
                    best_acc_val = epoch_acc_val
                    # save/load best model weights
                    torch.save(model, model_savename)

                if epoch == epochs - 1:
                    # flatten from appending in batches
                    all_preds = np.asarray(list(itertools.chain(*all_preds)))
                    all_labels = np.asarray(list(itertools.chain(*all_labels)))
                    clf_report = classification_metrics.metrics_report(
                        all_labels, all_preds, class_names
                    )
                    clf_report = classification_metrics.add_model_fold_to_clf_report(
                        clf_report, kfold, model_name
                    )

        time_elapsed = time.time() - since_epoch
        print(
            "Epoch complete in {:.0f}m {:.0f}s".format(
                time_elapsed // 60, time_elapsed % 60
            )
        )

    time_elapsed = time.time() - since_total
    print(
        "All epochs comlete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )

    with open(
        '/data/data/saved_timings/model_timing_only_cpu.csv', 'a', newline=''
    ) as file:
        writer = csv.writer(file)
        writer.writerow([model_name, epoch, kfold, time_elapsed])

    return clf_report
