#!/usr/bin/env python
# coding: utf-8

'''
Build and train the ML model for different CNNs to predict
ice crystal type
'''

import cocpit.data_loaders as data_loaders
import cocpit.calculate_metrics as metrics

from comet_ml import Experiment
import copy
import numpy as np
import time
import pandas as pd
import csv
from operator import add
from collections import Counter

import torch
from torch import nn
from torchvision import models
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from efficientnet_pytorch import EfficientNet
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

import PIL
from PIL import Image
from PIL import ImageFile
from pathlib import Path
ImageFile.LOAD_TRUNCATED_IMAGES = True

def set_random_seed(random_seed):
    if random_seed is not None:
        print("Set random seed as {}".format(random_seed))
        os.environ['PYTHONHASHSEED'] = str(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.set_num_threads(1)
        cudnn.benchmark = False
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')


def set_parameter_requires_grad(model, feature_extract):
    """
    Flag for feature extracting
        when False, finetune the whole model,
        when True, only update the reshaped layer params
    """
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes,
                     feature_extract=False, use_pretrained=False):
    #all input size of 224
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
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    elif model_name == "vgg16":
        model_ft = models.vgg16_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    elif model_name == "vgg19":
        model_ft = models.vgg19_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)

    elif model_name == "squeezenet":
        model_ft = models.squeezenet1_1(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes,\
                                           kernel_size=(7,7), stride=(2,2))
        #model_ft.num_classes = num_classes

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
        model_ft = EfficientNet.from_name('efficientnet-b0')
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft


def set_dropout(model, drop_rate=0.1):
    """
    technique to fight overfitting and improve neural network generalization
    """
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Dropout):
            child.p = drop_rate
        set_dropout(child, drop_rate=drop_rate)


def to_device(device, model):
    '''
    push model to gpu(s) if available
    '''
    # Send the model to GPU
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)
    return model


def update_params(model, feature_extract=False):
    '''
    If finetuning, update all parameters. If using 
    feature extract method, only update the parameter initialized
    i.e. the parameters with requires_grad is True
    '''
    params_to_update = model.parameters()
    if feature_extract:
        params_to_update = []
        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                #print("\t",name)
#     else:
#         for name,param in model.named_parameters():
#             if param.requires_grad == True:
#                 print("\t",name)


def label_counts(i, labels, num_classes):
    '''
    Calculate the # of labels per batch to ensure 
    weighted random sampler is correct
    '''    
    label_cnts = [0]*len(range(num_classes))
    for n in range(len(range(num_classes))):
        label_cnts[n] += len(np.where(labels.numpy() == n)[0])

    for n in range(len(range(num_classes))):
        #print("batch index {}, {} counts: {}".format(
        i, n, (labels == n).sum()
    #print('LABEL COUNT = ', label_cnts)
    
    return label_cnts


def train_model(experiment, log_exp, model, kfold, model_name, model_savename,
                acc_savename_train, acc_savename_val, save_acc, save_model,
                dataloaders_dict, epochs, num_classes):

    set_dropout(model, drop_rate=0.0)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = to_device(device, model)
    update_params(model)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)

    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5,
                                  patience=0, verbose=True, eps=1e-04)

    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc_val = 0.0
    since_total = time.time()
    for epoch in range(epochs):
        since_epoch = time.time()
        #print('Epoch {}/{}'.format(epoch+1,num_epochs))
        print('-' * 20)
    
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            print('Phase: {}'.format(phase))
            totals_train = 0
            totals_val = 0
            running_loss_train = 0.0
            running_loss_val = 0.0
            running_corrects_train = 0
            running_corrects_val = 0

            if phase == 'train':
                model.train() 
            else:
                model.eval()
            
            # Iterate over data in batches
            label_cnts_total = [0]*len(range(num_classes))
            for i, (inputs, labels, paths) in enumerate(dataloaders_dict[phase]):
                # uncomment to print cumulative sum of images per class, per batch
                # ensures weighted sampler is working properly
                #if phase == 'train':
#                     label_cnts = label_counts(i, labels, num_classes)
#                     label_cnts_total = list(map(add, label_cnts, label_cnts_total))
#                     print(label_cnts_total)

                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad() # a clean up step for PyTorch

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()  # compute updates for each parameter
                        optimizer.step()  # make the updates for each parameter 
                        
                # calculate batch metrics
                if phase == 'train':
                    running_loss_train, running_corrects_train, totals_train = \
                        metrics.batch_train_metrics(i, loss, inputs, preds, labels,
                                                    running_loss_train, running_corrects_train,
                                                    totals_train, dataloaders_dict, phase)
                else:
                    running_loss_val, running_corrects_val, totals_val = \
                        metrics.batch_val_metrics(i, loss, inputs, preds, labels,
                                                  running_loss_val, running_corrects_val,
                                                  totals_val, dataloaders_dict, phase)
            # calculate epoch metrics
            if phase == 'train':
                epoch_loss_train, epoch_acc_train = \
                    metrics.epoch_train_metrics(experiment, running_loss_train,
                                                totals_train, running_corrects_train,
                                                scheduler, log_exp, save_acc,
                                                acc_savename_train, model_name,
                                                epoch, epochs, kfold)

            else: 
                epoch_loss_val, epoch_acc_val = \
                    metrics.epoch_val_metrics(experiment, running_loss_val, totals_val,
                                              running_corrects_val, scheduler, log_exp,
                                              save_acc, acc_savename_val, model_name,
                                              epoch, epochs, kfold)
                #deep copy the model
                if epoch_acc_val > best_acc_val:
                    best_acc_val = epoch_acc_val
                    best_model_wts = copy.deepcopy(model.state_dict())
                    #save/load best model weights
                    if save_model:
                        torch.save(model, model_savename+'_'+model_name)

        time_elapsed = time.time() - since_epoch
        print('Epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    time_elapsed = time.time() - since_total
    print('All epochs comlete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    #with open('/data/data/saved_models/model_timing.csv', 'a', newline='') as file:
    #    writer = csv.writer(file)
    #    writer.writerow([model_name, epoch, kfold, time_elapsed])


def train_val_composition(data, train_indices, val_indices):
    train_y = list(map(data.targets.__getitem__, train_indices))
    test_y = list(map(data.targets.__getitem__, val_indices))
    print('train counts')
    print(Counter(train_y))
    print('val counts')
    print(Counter(test_y))

    
########## MAIN ###########

def main(params, log_exp, model_savename, acc_savename_train,
         acc_savename_val, save_acc, save_model, masked_dir,
         valid_size, num_workers, num_classes):
    
    if log_exp:
        experiment = Experiment(api_key="6tGmiuOfY08czs2b4SHaHI2hw",
                        project_name="multi-campaigns", workspace="vprzybylo")
        experiment.log_code('/data/data/notebooks/Pytorch_mult_campaigns.ipynb')
    else:
        experiment = None
    
    data = data_loaders.get_data(params['data_dir'])

    for batch_size in params['batch_size']:
        print('BATCH SIZE: ', batch_size) 
        for model_name in params['model_names']: 
            print('MODEL: ', model_name)
            for epochs in params['max_epochs']:
                print('MAX EPOCH: ', epochs)

                #K-FOLD 
                if params['kfold']!=0:
                    kfold=True
                    # preserve the percentage of samples for each class with stratified
                    skf = StratifiedKFold(n_splits=params['kfold'])
                    for i, (train_indices, val_indices) in enumerate(skf.split(data.imgs, data.targets)):
                        print('KFOLD: ', i)
                        # train_val_composition(data, train_indices, val_indices)

                        # DATALOADERS
                        train_loader, val_loader = \
                            data_loaders.create_dataloaders(data,
                                                            train_indices,
                                                            val_indices,
                                                            class_names=params['class_names'],
                                                            data_dir=params['data_dir'],
                                                            batch_size=batch_size,
                                                            save_model=save_model,
                                                            masked_dir=masked_dir,
                                                            num_workers=num_workers,
                                                            valid_size=valid_size)

                        dataloaders_dict = {'train': train_loader, 'val': val_loader}

                        # INITIALIZE MODEL
                        model = initialize_model(model_name, num_classes)

                        # TRAIN MODEL
                        train_model(experiment, log_exp,
                                    model, i, model_name,
                                    model_savename,
                                    acc_savename_train,
                                    acc_savename_val,
                                    save_acc,
                                    save_model,
                                    dataloaders_dict,
                                    epochs,
                                    num_classes)
                else:  # no kfold
                    kfold=False
                    i=0  # kfold false for savename
                    total_size = len(data)
                    # randomly split indices for training and validation indices according to valid_size
                    train_indices, val_indices = train_test_split(list(range(total_size)),
                                                          test_size=valid_size)
        
                    #DATALOADERS
                    train_loader, val_loader = \
                        data_loaders.create_dataloaders(data,
                                                        train_indices,
                                                        val_indices,
                                                        class_names=params['class_names'],
                                                        data_dir=params['data_dir'],
                                                        batch_size=batch_size,
                                                        save_model=save_model,
                                                        masked_dir=masked_dir,
                                                        num_workers=num_workers,
                                                        valid_size=valid_size)

                    dataloaders_dict = {'train': train_loader, 'val': val_loader}

                    # INITIALIZE MODEL
                    model = initialize_model(model_name, num_classes)

                    # TRAIN MODEL
                    train_model(experiment,log_exp,
                                model, i, model_name,
                                model_savename,
                                acc_savename_train,
                                acc_savename_val,
                                save_acc,
                                save_model,
                                dataloaders_dict,
                                epochs,
                                num_classes)
if __name__ == '__main__':

    main(params, log_exp, model_savename,
         acc_savename_train, acc_savename_val,
         save_acc, save_model, masked_dir, 
         valid_size, num_workers, num_classes)
    