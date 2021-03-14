#!/usr/bin/env python
# coding: utf-8

from comet_ml import Experiment

experiment = Experiment(api_key="6tGmiuOfY08czs2b4SHaHI2hw",
                        project_name="multi-campaigns", workspace="vprzybylo")
import copy
import numpy as np
import time
import pandas as pd
import csv

import torch
from torch import nn
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from efficientnet_pytorch import EfficientNet

import PIL
from PIL import Image
from PIL import ImageFile
from pathlib import Path
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib.pyplot as plt

plt_params = {'axes.labelsize': 'xx-large',
         'axes.titlesize':'xx-large',
         'xtick.labelsize':'x-large',
         'ytick.labelsize':'xx-large'}
plt.rcParams.update(plt_params)


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
        return tuple_with_path

#### equal pull from classes
def make_weights_for_balanced_classes(train_imgs, nclasses):
    #only weight the training dataset 

    class_sample_counts = [0] * nclasses
    for item in train_imgs:  
        class_sample_counts[item[1]] += 1
    print('counts per class: ', class_sample_counts)

#     weight_per_class = [0.] * nclasses
#     N = float(sum(class_sample_counts))
#     for i in range(nclasses): 
#         weight_per_class[i] = N/float(class_sample_counts[i])
#     weight = [0] * len(images)
#     for idx, val in enumerate(images):
#         weight[idx] = weight_per_class[val[1]]

    class_weights = 1./torch.Tensor(class_sample_counts)
    train_targets = [sample[1] for sample in train_imgs]
    train_samples_weights = [class_weights[class_id] for class_id in train_targets]

    return class_sample_counts, torch.DoubleTensor(train_samples_weights)

def make_histogram_classcounts(class_names, class_counts):
    fig, ax = plt.subplots(figsize=(9,5))

    width = 0.75 # the width of the bars 
    ind = np.arange(len(class_counts))  # the x locations for the groups
    ax.barh(class_names, class_counts, width, color="blue", align='center', tick_label=class_names)
    #ax.set_yticks(ind+width/2)
    #plt.xticks(rotation=-90, ha='center')

    for i, v in enumerate(class_counts):
        ax.text(v, i-.1, str(v), color='blue')
    ax.set_xlabel("Count")
    #ax.set_xlim(0,2500)
    #plt.savefig('../plots/class_counts.png', dpi=300, format='png', bbox_inches='tight')
    plt.show()

def load_split_train_val(train_data, val_data, class_names,\
                         datadir, batch_size, show_sample=True,\
                         num_workers=32, valid_size = .8):
    
    # For an unbalanced dataset we create a weighted sampler              
    class_counts, train_samples_weights =\
                        make_weights_for_balanced_classes(train_data.dataset.imgs, len(class_names))
    make_histogram_classcounts(class_names, class_counts)
    
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_samples_weights, 
                                                                   len(train_samples_weights),
                                                                   replacement=True)                     
    trainloader = torch.utils.data.DataLoader(train_data.dataset,\
                                              batch_size=batch_size,                         
                                              sampler = train_sampler,\
                                              num_workers=num_workers,\
                                              pin_memory=True)    
    
    val_sampler = SubsetRandomSampler(val_data.indices)                 
    valloader = torch.utils.data.DataLoader(val_data.dataset,\
                                            batch_size=batch_size,\
                                            sampler = val_sampler,\
                                            num_workers=num_workers,\
                                            pin_memory=True)  

#     val_samples_weights = make_weights_for_balanced_classes(val_data.dataset.imgs,\
#                                                            len(range(len(class_names)))
    
#     val_sampler = torch.utils.data.sampler.WeightedRandomSampler(val_samples_weights, 
#                                                                    len(val_samples_weights),
#                                                                    replacement=True)                   
#     valloader = torch.utils.data.DataLoader(val_data.dataset, batch_size=batch_size,                 
#                                             sampler = val_sampler, num_workers=num_workers, pin_memory=True)    

    if show_sample:
        show_sample(train_data, train_sampler)

    return trainloader, valloader

def show_sample(train_data, train_sampler):

    batch_size_sampler=20
    sample_loader = torch.utils.data.DataLoader(train_data.dataset, batch_size=batch_size_sampler,                                                 sampler = train_sampler, num_workers=1, drop_last=True)
    data_iter = iter(sample_loader)

    images, labels, paths = data_iter.next()
    fig, ax = plt.subplots(batch_size_sampler//5, 5, figsize=(10, 8))

    for j in range(images.size()[0]):

        # Undo preprocessing
        image = images[j].permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        image = std * image + mean

        # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
        image = np.clip(image, 0, 1)
        ax = ax.flatten()
        ax[j].set_title(str(class_names[labels[j]]))
        ax[j].axis('off')
        ax[j].imshow(image)
    plt.show()

def get_test_loader(datadir,
                    batch_size,
                    num_workers,
                    shuffle=True,
                    pin_memory=True):
    """
    Utility function for loading and returning a multi-process
    test iterator 
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - shuffle: whether to shuffle the dataset after every epoch.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - data_loader: test set iterator.
    """
    transforms_ = transforms.Compose([transforms.Resize((224,224)),  #resizing helps memory usage
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    all_data_wpath = ImageFolderWithPaths(datadir,transform=transforms_)

    testloader = torch.utils.data.DataLoader(all_data_wpath,pin_memory=True,shuffle=shuffle,
                    batch_size=batch_size, num_workers=num_workers)  

    return testloader


# Flag for feature extracting. When False, we finetune the whole model,
#   when True we only update the reshaped layer params
def set_parameter_requires_grad(model, feature_extract):
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False

            
def initialize_model(model_name, num_classes, feature_extract=False, use_pretrained=False):
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
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(7,7), stride=(2,2))
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
        #torch.hub.list('rwightman/gen-efficientnet-pytorch')
        #model_ft = torch.hub.load('rwightman/gen-efficientnet-pytorch', 'efficientnet_b0', pretrained=False)
        model_ft = EfficientNet.from_name('efficientnet-b0')
    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft


def set_dropout(model, drop_rate=0.1):
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Dropout):
            child.p = drop_rate
        set_dropout(child, drop_rate=drop_rate)


def train_model(experiment, model, kfold, model_name, model_savename,\
                acc_savename_train, acc_savename_val,save_acc, save_model,\
                dataloaders_dict, epochs, num_classes, feature_extract=False):

    set_dropout(model, drop_rate=0.0)

    #feature extract False for all layers to be updated

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Send the model to GPU
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(device)

    # Gather the parameters to be optimized/updated in this run. If we are
    #  finetuning we will be updating all parameters. However, if we are
    #  doing feature extract method, we will only update the parameters
    #  that we have just initialized, i.e. the parameters with requires_grad
    #  is True.
    params_to_update = model.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []

        for name,param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                #print("\t",name)
    #else:
        #for name,param in model.named_parameters():
            #if param.requires_grad == True:
                #print("\t",name)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
    # step_size: at how many multiples of epoch you decay
    # step_size = 1, after every 1 epoch, new_lr = lr*gamma 
    # step_size = 2, after every 2 epoch, new_lr = lr*gamma 
    # gamma = decaying factor
    #scheduler = StepLR(optimizer, step_size=1, gamma=0.1)
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5,\
                                  patience=0, verbose=True, eps=1e-04)

    # Setup the loss fxn
    criterion = nn.CrossEntropyLoss()

    val_acc_history = []
    train_acc_history = []
    val_loss_history = []
    train_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc_val = 0.0
    since_total = time.time()

    step = 0
    label_counts = [0]*len(range(num_classes))
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
                #logger = logger_train

            else:
                model.eval()   
                #logger = logger_val

            # Iterate over data.
            for i, (inputs, labels, paths) in enumerate(dataloaders_dict[phase]):
                for n in range(len(range(num_classes))):
                    label_counts[n] += len(np.where(labels.numpy() == n)[0])

#                 for n in range(len(range(num_classes))):
#                     print("batch index {}, {} counts: {}".format(
#                         i, n, (labels == n).sum()))


#                print('LABEL COUNT = ', label_counts)

                inputs = inputs.to(device)
                labels = labels.to(device)
                #print(inputs.device)

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
                        loss.backward() # compute updates for each parameter
                        optimizer.step() # make the updates for each parameter 

                if phase == 'train':
                    #Batch accuracy and loss statistics   
                    batch_loss_train = loss.item() * inputs.size(0)     
                    batch_corrects_train = torch.sum(preds == labels.data) 

                    #for accuracy and loss statistics overall 
                    running_loss_train += loss.item() * inputs.size(0)
                    running_corrects_train += torch.sum(preds == labels.data)
                    totals_train += labels.size(0)

                    if (i+1) % 5 == 0:
                        print("Training, Batch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(i+1,                                                                      len(dataloaders_dict[phase]),                                                                       batch_loss_train/labels.size(0),                                                                       float(batch_corrects_train)/labels.size(0)))
                    step += 1

                else:
                    #Batch accuracy and loss statistics  
                    batch_loss_val = loss.item() * inputs.size(0)     
                    batch_corrects_val = torch.sum(preds == labels.data) 

                    #for accuracy and loss statistics overall
                    running_loss_val += loss.item() * inputs.size(0)
                    running_corrects_val += torch.sum(preds == labels.data)
                    totals_val += labels.size(0)

                    if (i+1) % 3 == 0:
                        print("Validation, Batch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(i+1,                                                                      len(dataloaders_dict[phase]),                                                                       batch_loss_val/labels.size(0),                                                                       float(batch_corrects_val)/labels.size(0)))
            if phase == 'train':
                #epoch loss and accuracy stats    
                epoch_loss_train = running_loss_train / totals_train
                epoch_acc_train = running_corrects_train.double() / totals_train
                scheduler.step(epoch_acc_train) #reduce learning rate if not improving acc
                experiment.log_metric('train scheduler', scheduler)

                #write acc and loss to file within epoch iteration
                if save_acc:
                    with open(acc_savename_train, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([model_name, epoch, kfold,\
                                         epoch_acc_train.cpu().numpy(), epoch_loss_train])
                    file.close()

                print("Training Epoch {}/{}, Loss: {:.3f}, Accuracy: \033[1m {:.3f} \033[0m".format(epoch+1,\
                                                                                                    epochs,\
                                                                                            epoch_loss_train,\
                                                                                            epoch_acc_train))
                train_acc_history.append(epoch_acc_train)
                train_loss_history.append(epoch_loss_train)
                experiment.log_metric('epoch_acc_train', epoch_acc_train*100)
                experiment.log_metric('epoch_loss_train', epoch_loss_train)

            else: 
                epoch_loss_val = running_loss_val / totals_val
                epoch_acc_val = running_corrects_val.double() / totals_val
                scheduler.step(epoch_acc_val) #reduce learning rate if not improving acc
                experiment.log_metric('val scheduler', scheduler)

                #write acc and loss to file within epoch iteration
                if save_acc:
                    with open(acc_savename_val, 'a', newline='') as file:
                        writer = csv.writer(file)
                        writer.writerow([model_name, epoch, kfold, epoch_acc_val.cpu().numpy(), epoch_loss_val])
                    file.close()

                print("Validation Epoch {}/{}, Loss: {:.3f}, Accuracy: \033[1m {:.3f} \033[0m".format(epoch+1,epochs, epoch_loss_val, epoch_acc_val))
                val_acc_history.append(epoch_acc_val)
                val_loss_history.append(epoch_loss_val)
                experiment.log_metric('epoch_acc_val', epoch_acc_val*100)
                experiment.log_metric('epoch_loss_val', epoch_loss_val)

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

    return


# # MAIN
def main(params, model_savename, acc_savename_train, acc_savename_val,\
                                   save_acc, save_model, valid_size, num_workers, num_classes):
    #experiment.log_code('/data/data/notebooks/Pytorch_mult_campaigns.ipynb')

    all_transforms = transforms.Compose([transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    #custom dataset that includes entire path
    all_data_wpath = ImageFolderWithPaths(params['data_dir'],transform=all_transforms) 

    for batch_size in params['batch_size']:
        print('BATCH SIZE: ', batch_size) 
        for model_name in params['model_names']: 
            print('MODEL: ', model_name)
            for epochs in params['max_epochs']:
                print('MAX EPOCH: ', epochs)

                #K-FOLD 
                if params['kfold']!=0:
                    train_score = pd.Series(dtype=np.float64)
                    val_score = pd.Series(dtype=np.float64)

                    total_size = len(all_data_wpath)
                    fraction = 1/params['kfold']
                    seg = int(total_size * fraction)
                    # tr:train,val:valid; r:right,l:left;  eg: trrr: right index of right side train subset 
                    # index: [trll,trlr],[vall,valr],[trrl,trrr]

                    for i in range(params['kfold']):
                        print('KFOLD: ', i)
                        trll = 0
                        trlr = i * seg
                        vall = trlr
                        valr = i * seg + seg
                        trrl = valr
                        trrr = total_size

                        print("train indices: [%d,%d),[%d,%d), test indices: [%d,%d)" 
                          % (trll,trlr,trrl,trrr,vall,valr))

                        train_left_indices = list(range(trll,trlr))
                        train_right_indices = list(range(trrl,trrr))

                        train_indices = train_left_indices + train_right_indices
                        val_indices = list(range(vall,valr))

                        train_data = torch.utils.data.dataset.Subset(all_data_wpath, train_indices)
                        val_data = torch.utils.data.dataset.Subset(all_data_wpath, val_indices)                        

                        train_loader, val_loader = load_split_train_val(
                                train_data,
                                val_data,
                                class_names=params['class_names'], 
                                datadir=params['data_dir'],
                                batch_size=batch_size,
                                show_sample=False,
                                num_workers=num_workers)

                        dataloaders_dict = {'train': train_loader, 'val': val_loader}

                        #INITIALIZE MODEL
                        model = initialize_model(model_name, num_classes)

                        #TRAIN MODEL
                        train_model(experiment,
                                    model, i, model_name,
                                    model_savename,
                                    acc_savename_train,
                                    acc_savename_val,
                                    save_acc,
                                    save_model,
                                    dataloaders_dict,
                                    epochs, 
                                    num_classes)
                else: #no kfold
                    i=0
                    train_length = int(valid_size*len(all_data_wpath))
                    val_length = len(all_data_wpath)-train_length
                    train_data, val_data = torch.utils.data.random_split(all_data_wpath,(train_length,val_length))                

                    train_loader, val_loader = load_split_train_val(
                            train_data,
                            val_data,
                            class_names=params['class_names'], 
                            datadir=params['data_dir'],
                            batch_size=batch_size,
                            show_sample=False,
                            num_workers=num_workers)

                    dataloaders_dict = {'train': train_loader, 'val': val_loader}

                    #INITIALIZE MODEL
                    model = initialize_model(model_name, num_classes)

                    #TRAIN MODEL
                    train_model(experiment,
                                model, i, model_name,
                                model_savename,
                                acc_savename_train,
                                acc_savename_val,
                                save_acc,
                                save_model,
                                dataloaders_dict,
                                epochs, 
                                num_classes)
                    
    return

if __name__ == '__main__':

    
    main(params, model_savename, acc_savename_train, acc_savename_val,\
                                   save_acc, save_model, valid_size, num_workers, num_classes)