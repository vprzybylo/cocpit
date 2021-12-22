'''
model configurations for:
    - dropout
    - device settings
    - parameters to update
    - checking label counts within a batch
    - normalization values for transformations
'''

import torch
from torch import nn

import cocpit.config as config


def set_dropout(model, drop_rate=0.1):
    """
    technique to fight overfitting and improve neural network generalization
    """
    for name, child in model.named_children():
        if isinstance(child, torch.nn.Dropout):
            child.p = drop_rate
        set_dropout(child, drop_rate=drop_rate)


def to_device(model):
    """
    push model to gpu(s) if available
    """

    # Send the model to GPU
    if torch.cuda.device_count() > 1:
        print("Using", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    model = model.to(config.DEVICE)
    return model


def count_parameters(model):
    """
    number of model parameters to update
    """
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

    print("model params: ", count_parameters(model))

    return params_to_update


def label_counts(i, labels):
    """
    Calculate the # of labels per batch to ensure
    weighted random sampler is correct
    """

    num_classes = len(config.CLASS_NAMES)
    label_cnts = [0] * len(range(num_classes))
    for n in range(len(range(num_classes))):
        label_cnts[n] += len(np.where(labels.numpy() == n)[0])

    for n in range(len(range(num_classes))):
        # print("batch index {}, {} counts: {}".format(
        i, n, (labels == n).sum()
    print("LABEL COUNT = ", label_cnts)

    return label_cnts


def normalization_values(dataloaders_dict, phase):
    """
    Get mean and standard deviation of pixel values
    across all batches
    """
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
