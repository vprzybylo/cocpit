"""
-trains pytorch models 
-balances class counts for uniform distribution in training
-sets up dataloaders
-returns model accuracies
-alterable hyperparameters under 'params' in the ./__main__.py executable
-saves the model (if not None)
"""

from torchvision import datasets, transforms, models
import torch 
from torch.utils.data.sampler import SubsetRandomSampler
from torch import nn
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy 
import time
import numpy as np

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
    
def load_split_train_val(class_names, datadir, batch_size, num_workers=32, valid_size = .8):
    
    all_transforms = transforms.Compose([transforms.Resize(224),
                        transforms.ToTensor(),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    
    all_data_wpath = ImageFolderWithPaths(datadir,transform=all_transforms) #custom dataset that includes entire path
    
#     num_train = len(all_data_wpath)
#     indices = list(range(num_train))
#     split = int(np.floor(valid_size * num_train))
#     np.random.shuffle(indices)
#     train_idx, val_idx = indices[split:], indices[:split-1]
    
#     train_data = torch.utils.data.Subset(all_data_wpath, train_idx)
#     val_data = torch.utils.data.Subset(all_data_wpath, val_idx)
    
    train_length = int(valid_size*len(all_data_wpath))
    val_length = len(all_data_wpath)-train_length
    train_data, val_data = torch.utils.data.random_split(all_data_wpath,(train_length,val_length))
    #print(len(train_data), len(val_data))
    
    # For an unbalanced dataset we create a weighted sampler              
    class_counts, train_samples_weights = make_weights_for_balanced_classes(train_data.dataset.imgs, len(class_names))                                                                 
    #make_histogram_classcounts(class_names, class_counts)
    
    train_sampler = torch.utils.data.sampler.WeightedRandomSampler(train_samples_weights, 
                                                                   len(train_samples_weights),
                                                                   replacement=True)                     
    trainloader = torch.utils.data.DataLoader(train_data.dataset, batch_size=batch_size,                         
                                            sampler = train_sampler, num_workers=num_workers, pin_memory=True)    
    
    val_sampler = SubsetRandomSampler(val_data.indices)                 
    valloader = torch.utils.data.DataLoader(val_data.dataset, batch_size=batch_size,                             
                                            sampler = val_sampler, num_workers=num_workers, pin_memory=True)  

#     val_samples_weights = make_weights_for_balanced_classes(val_data.dataset.imgs, len(range(num_classes)))                                                                   
    
#     val_sampler = torch.utils.data.sampler.WeightedRandomSampler(val_samples_weights, 
#                                                                    len(val_samples_weights),
#                                                                    replacement=True)                     
#     valloader = torch.utils.data.DataLoader(val_data.dataset, batch_size=batch_size,                              
#                                             sampler = val_sampler, num_workers=num_workers, pin_memory=True)    

            
    return trainloader, valloader


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
    transforms_ = transforms.Compose([transforms.Resize(224),  #resizing helps memory usage
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


def initialize_model(model_name, num_classes, feature_extract, use_pretrained=False):
    # Initialize these variables which will be set in this if statement. Each of these
    # variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet18":
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    elif model_name == "resnet34":
        model_ft = models.resnet34(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "resnet152":
        model_ft = models.resnet152(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224
        
    elif model_name == "vgg16":
        """ VGG
        """
        model_ft = models.vgg16_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg19":
        """ VGG
        """
        model_ft = models.vgg19_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_1(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(7,7), stride=(2,2))
        #model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet169":
        """ Densenet
        """ 
        model_ft = models.densenet169(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    elif model_name == "densenet201":
        """ Densenet
        """ 
        model_ft = models.densenet201(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size
            
def train_model(model_name, savename, dataloaders_dict, epochs, num_classes, is_inception, feature_extract=False):
    #current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    #logger_train = Logger('./logs/'+current_time+'/train/')
    #logger_val = Logger('./logs/'+current_time+'/val/')
    model, input_size = initialize_model(model_name=model_name, num_classes=num_classes, feature_extract=feature_extract, use_pretrained=False)
    
    def set_dropout(model, drop_rate=0.1):
        for name, child in model.named_children():
            
            if isinstance(child, torch.nn.Dropout):
                child.p = drop_rate
            set_dropout(child, drop_rate=drop_rate)
    set_dropout(model, drop_rate=0.0)
    
#     model.classifier = nn.Sequential(*[model.classifier()[i] for i in range(7) if i != 2 and i !=5])
#     print(model.classifier())

    #feature extract False for all layers to be updated
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Send the model to GPU
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "out of", torch.cuda.device_count(), "GPUs!")
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
    
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=0, verbose=True, eps=1e-04)
    print(scheduler)
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
                    # makes sure to clear the intermediate values for evaluation
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
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
                    #tensorboard_logging(logger, batch_loss_train, labels, batch_corrects_train, step, model)
                    
                    #for accuracy and loss statistics overall 
                    running_loss_train += loss.item() * inputs.size(0)
                    running_corrects_train += torch.sum(preds == labels.data)
                    totals_train += labels.size(0)
                    
                    if (i+1) % 5 == 0:
                        print("Training, Batch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(i+1,\
                                                                      len(dataloaders_dict[phase]), \
                                                                      batch_loss_train/labels.size(0), \
                                                                      float(batch_corrects_train)/labels.size(0)))

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
                        print("Validation, Batch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(i+1,\
                                                                      len(dataloaders_dict[phase]), \
                                                                      batch_loss_val/labels.size(0), \
                                                                      float(batch_corrects_val)/labels.size(0)))

            if phase == 'train':
                #epoch loss and accuracy stats    
                epoch_loss_train = running_loss_train / totals_train
                epoch_acc_train = running_corrects_train.double() / totals_train
                scheduler.step(epoch_acc_train) #reduce learning rate if not improving acc
                print("Training Epoch {}/{}, Loss: {:.3f}, Accuracy: \033[1m {:.3f} \033[0m".format(epoch+1,epochs, epoch_loss_train, epoch_acc_train))
                #tensorboard_logging(logger, epoch_loss_train, epoch_acc_train, epoch, model)
                train_acc_history.append(epoch_acc_train)
                train_loss_history.append(epoch_loss_train)

            else: 
                epoch_loss_val = running_loss_val / totals_val
                epoch_acc_val = running_corrects_val.double() / totals_val
                scheduler.step(epoch_acc_val) #reduce learning rate if not improving acc

                print("Validation Epoch {}/{}, Loss: {:.3f}, Accuracy: \033[1m {:.3f} \033[0m".format(epoch+1,epochs, epoch_loss_val, epoch_acc_val))
                #tensorboard_logging(logger, epoch_loss_val, epoch_acc_val, epoch, model)
                val_acc_history.append(epoch_acc_val)
                val_loss_history.append(epoch_loss_val)
                
                #deep copy the model
                if epoch_acc_val > best_acc_val:
                    best_acc_val = epoch_acc_val
                    best_model_wts = copy.deepcopy(model.state_dict())
                    # save/load best model weights
                    if savename is not 'None':
                        torch.save(model, savename)

        time_elapsed = time.time() - since_epoch
        print('Epoch complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    time_elapsed = time.time() - since_total
    print('All epochs comlete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return model, train_acc_history, val_acc_history, train_loss_history, val_loss_history

def main(params, num_workers, num_classes):
    
    for batch_size in params['batch_size']:
        print('NEW BATCH SIZE: ', batch_size) 
        train_loader, val_loader = load_split_train_val(
            class_names=params['class_names'], 
            datadir=params['data_dir'],
            batch_size=batch_size,
            num_workers=num_workers)

        dataloaders_dict = {'train': train_loader, 'val': val_loader}

        model_train_accs = []
        model_val_accs = []  
        model_train_loss = []
        model_val_loss = []
        for model_name in params['model_names']: 
            for epochs in params['max_epochs']:
                model_ft, train_acc_history, val_acc_history, train_loss_history, val_loss_history= train_model(
                    model_name,
                    params['savename'],
                    dataloaders_dict,
                    epochs, 
                    num_classes,
                    is_inception=False
                )
                
                model_val_accs.append(val_acc_history)
                model_train_accs.append(train_acc_history)
                model_train_loss.append(train_loss_history)
                model_val_loss.append(val_loss_history)
                
                
    return model_name, model_train_accs, model_val_accs, model_train_loss, model_val_loss

