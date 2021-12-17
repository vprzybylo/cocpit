"""
train the CNN model(s)
"""
import csv
import operator
import time

import numpy as np
import torch
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau

import cocpit
import cocpit.config as config  # isort:split


def train_model(
    kfold,
    model,
    batch_size,
    model_name,
    epochs,
    dataloaders_dict,
    clf_report=None,
):

    ## model configurations ##
    cocpit.model_config.set_dropout(model, drop_rate=0.0)
    model = cocpit.model_config.to_device(model)
    params_to_update = cocpit.model_config.update_params(model)
    optimizer = optim.SGD(params_to_update, lr=0.01, momentum=0.9, nesterov=True)
    criterion = nn.CrossEntropyLoss()  # Loss function
    scheduler = ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=0, verbose=True, eps=1e-04
    )
    #########################

    train_metrics = cocpit.metrics.Metrics()
    val_metrics = cocpit.metrics.Metrics()

    since_total = time.time()

    for epoch in range(epochs):
        since_epoch = time.time()
        # print('Epoch {}/{}'.format(epoch+1,num_epochs))
        print("-" * 20)

        # Each epoch has a training and validation phase
        if config.VALID_SIZE < 0.1:
            phases = ["train"]
        else:
            phases = ["train", "val"]

        for phase in phases:
            print("Phase: {}".format(phase))

            # reset totals for each epoch and each phase
            train_metrics.reset_totals()
            val_metrics.reset_totals()

            # label_cnts_total = np.zeros(len(class_names))

            if phase == "train":
                model.train()
            else:
                model.eval()

            # get pytorch transform normalization values per channel
            # iterates over training
            # mean, std = cocpit.model_config.normalization_values(dataloaders_dict, phase)

            for batch, ((inputs, labels, paths), index) in enumerate(
                dataloaders_dict[phase]
            ):
                # uncomment to print cumulative sum of images per class, per batch
                # ensures weighted sampler is working properly
                # if phase == 'train':
                # label_cnts = cocpit.model_config.label_counts(i, labels)
                # label_cnts_total = list(map(operator.add, label_cnts, label_cnts_total))
                # print(label_cnts_total)

                inputs = inputs.to(config.DEVICE)
                labels = labels.to(config.DEVICE)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()  # compute updates for each parameter
                        optimizer.step()  # make the updates for each parameter

                        # CALCULATE BATCH METRICS
                        train_metrics.update_batch_metrics(loss, inputs, preds, labels)
                        if (batch + 1) % 5 == 0:
                            train_metrics.print_batch_metrics(
                                labels, batch, phase, dataloaders_dict
                            )

                if phase == 'val':
                    val_metrics.update_batch_metrics(loss, inputs, preds, labels)
                    if (batch + 1) % 5 == 0:
                        val_metrics.print_batch_metrics(
                            labels, batch, phase, dataloaders_dict
                        )
                    # append batch prediction and labels for plots
                    val_metrics.all_preds.append(preds.cpu().numpy())
                    val_metrics.all_labels.append(labels.cpu().numpy())

            # CALCULATE EPOCH METRICS
            if phase == "train":
                cocpit.metrics.log_metrics(
                    train_metrics,
                    kfold,
                    model,
                    batch_size,
                    model_name,
                    epoch,
                    epochs,
                    scheduler,
                    phase,
                    acc_savename=config.ACC_SAVENAME_TRAIN,
                )

                if config.VALID_SIZE < 0.01:
                    # when using all of the training data
                    # and no validation, save model based
                    # on best training iteration
                    if (
                        train_metrics.epoch_acc > train_metrics.best_acc
                        and config.SAVE_MODEL
                    ):
                        train_metrics.best_acc = train_metrics.epoch_acc
                        # save/load best model weights
                        if not os.path.exists(config.MODEL_SAVE_DIR):
                            os.makedirs(config.MODEL_SAVE_DIR)
                        torch.save(model, config.MODEL_SAVENAME)

            else:
                cocpit.metrics.log_metrics(
                    val_metrics,
                    kfold,
                    model,
                    batch_size,
                    model_name,
                    epoch,
                    epochs,
                    scheduler,
                    phase,
                    acc_savename=config.ACC_SAVENAME_VAL,
                )

                if val_metrics.epoch_acc > val_metrics.best_acc and config.SAVE_MODEL:
                    val_metrics.best_acc = val_metrics.epoch_acc
                    # save/load best model weights
                    torch.save(model, config.MODEL_SAVENAME)

                if (
                    epoch == epochs - 1
                    and (config.KFOLD != 0 and kfold == config.KFOLD - 1)
                    or (config.KFOLD == 0)
                ):
                    cocpit.metrics.log_confusion_matrix(val_metrics)
                if epoch == epochs - 1:
                    # save classification report
                    cocpit.metrics.sklearn_report(
                        val_metrics,
                        kfold,
                        model_name,
                    )

                # reduce learning rate upon plateau in epoch validation accuracy
                scheduler.step(val_metrics.epoch_acc)

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
        "/data/data/saved_timings/model_timing_only_cpu.csv", "a", newline=""
    ) as file:
        writer = csv.writer(file)
        writer.writerow([model_name, epoch, kfold, time_elapsed])
