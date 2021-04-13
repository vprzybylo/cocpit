import torch
import csv

def batch_train_metrics(i, loss, inputs, preds, labels, running_loss_train,\
                  running_corrects_train, totals_train, dataloaders_dict, phase):
    """
    Calculate loss and accuracy on training iterations
    """
    #Batch accuracy and loss statistics
    batch_loss_train = loss.item() * inputs.size(0)
    batch_corrects_train = torch.sum(preds == labels.data)

    #for accuracy and loss statistics overall 
    running_loss_train += loss.item() * inputs.size(0)
    running_corrects_train += torch.sum(preds == labels.data)
    totals_train += labels.size(0)

    if (i+1) % 5 == 0:
        print("Training, Batch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(i+1,len(dataloaders_dict[phase]),\
                                     batch_loss_train/labels.size(0),\
                                     float(batch_corrects_train)/labels.size(0)))
    return running_loss_train, running_corrects_train, totals_train
        
def batch_val_metrics(i, loss, inputs, preds, labels, running_loss_val,\
               running_corrects_val, totals_val, dataloaders_dict, phase):
    """
    Calculate loss and accuracy on validation iterations
    """
    #Batch accuracy and loss statistics  
    batch_loss_val = loss.item() * inputs.size(0)
    batch_corrects_val = torch.sum(preds == labels.data)

    #for accuracy and loss statistics overall
    running_loss_val += loss.item() * inputs.size(0)
    running_corrects_val += torch.sum(preds == labels.data)
    totals_val += labels.size(0)
    if (i+1) % 5 == 0:
        print("Validation, Batch {}/{}, Loss: {:.3f}, Accuracy: {:.3f}".format(i+1,len(dataloaders_dict[phase]),\
                                        batch_loss_val/labels.size(0),\
                                        float(batch_corrects_val)/labels.size(0)))
    return running_loss_val, running_corrects_val, totals_val

def epoch_train_metrics(experiment, running_loss_train, totals_train,\
                       running_corrects_train, scheduler, log_exp,\
                       save_acc, acc_savename_train, model_name,\
                       epoch, epochs, kfold, batch_size):
    '''
    Calculate epoch loss and accuracy during training
    '''

    epoch_loss_train = running_loss_train / totals_train
    epoch_acc_train = running_corrects_train.double() / totals_train
    if log_exp:
        experiment.log_metric('train scheduler', scheduler)

    #write acc and loss to file within epoch iteration
    if save_acc:
        with open(acc_savename_train, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([model_name, epoch, kfold, batch_size,\
                             epoch_acc_train.cpu().numpy(), epoch_loss_train])
        file.close()

    print("Training Epoch {}/{}, Loss: {:.3f}, Accuracy: \033[1m {:.3f} \033[0m".format(epoch+1,epochs,\
                                                     epoch_loss_train,\
                                                     epoch_acc_train))
    if log_exp:
        experiment.log_metric('epoch_acc_train', epoch_acc_train*100)
        experiment.log_metric('epoch_loss_train', epoch_loss_train)
        
    return epoch_loss_train, epoch_acc_train

def epoch_val_metrics(experiment, running_loss_val, totals_val,\
                       running_corrects_val, scheduler, log_exp,\
                       save_acc, acc_savename_val, model_name,\
                       epoch, epochs, kfold, batch_size):
    '''
    Calculate epoch loss and accuracy during validation
    '''
    epoch_loss_val = running_loss_val / totals_val
    epoch_acc_val = running_corrects_val.double() / totals_val
    scheduler.step(epoch_acc_val) #reduce learning rate if not improving acc
    if log_exp:
        experiment.log_metric('val scheduler', scheduler)

    #write acc and loss to file within epoch iteration
    if save_acc:
        with open(acc_savename_val, 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([model_name, epoch, kfold, batch_size,\
                             epoch_acc_val.cpu().numpy(), epoch_loss_val])
        file.close()

    print("Validation Epoch {}/{}, Loss: {:.3f}, Accuracy: \033[1m {:.3f} \033[0m".format(epoch+1,epochs,\
                                                     epoch_loss_val, epoch_acc_val))
    if log_exp:
        experiment.log_metric('epoch_acc_val', epoch_acc_val*100)
        experiment.log_metric('epoch_loss_val', epoch_loss_val)
    
    return epoch_loss_val, epoch_acc_val