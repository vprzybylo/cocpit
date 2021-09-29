'''
torchvision models
all pretrained on the 1000-class Imagenet dataset
we update all weights but use this model architecture
'''
from efficientnet_pytorch import EfficientNet
from torch import nn
from torchvision import models

import cocpit.config as config  # isort:split


def set_parameter_requires_grad(model, feature_extract):
    """
    Flag for feature extracting
        when False, finetune the whole model,
        when True, only update the reshaped layer params
    """
    if feature_extract:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, feature_extract=False, use_pretrained=False):

    '''
    params:
    -------
    model_name: name of the model to train
    num_classes: number of classes
    feature_extract: default = False; start with a pretrained model and only
                    update the final layer weights from which we derive predictions
    use_pretrained: default = False; update all of the model’s
                    parameters for our new task (retrain)
    '''

    num_classes = len(config.CLASS_NAMES)

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
