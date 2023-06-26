import torchvision
from efficientnet_pytorch import EfficientNet
from torch import nn

import cocpit.config as config  # isort:split


class Model:
    """
    Model initializations from torchvision. We always train from scratch.

    Args:
        num_classes (int): number of classes
        model (torchvision.models): a torchvision.models instance
    """

    def __init__(
        self,
    ):
        self.num_classes = len(config.CLASS_NAMES)
        self.model: torchvision.models = None

    def set_parameter_requires_grad(self) -> None:
        """
        Finetune the whole model when feature_extract is True, else only update the reshaped layer params
        """
        for param in self.model.parameters():
            param.requires_grad = False

    def resnet_classifier(self) -> None:
        """
        create linear output layer equal to number of classes
        """
        self.set_parameter_requires_grad() if config.FEATURE_EXTRACT else None
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)

    def vgg_classifier(self) -> None:
        """
        create linear output layer equal to number of classes
        """
        self.set_parameter_requires_grad() if config.FEATURE_EXTRACT else None
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, self.num_classes)

    def densenet_classifier(self) -> None:
        """
        create linear output layer equal to number of classes
        """
        self.set_parameter_requires_grad() if config.FEATURE_EXTRACT else None
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, self.num_classes)

    def resnet18(self) -> None:
        """resnet 18 architecture"""
        self.model = torchvision.models.resnet18(
            pretrained=config.USE_PRETRAINED
        )
        self.resnet_classifier()

    def resnet34(self) -> None:
        """resnet 34 architecture"""
        self.model = torchvision.models.resnet34(
            pretrained=config.USE_PRETRAINED
        )
        self.resnet_classifier()

    def resnet152(self) -> None:
        """resnet 152 architecture"""
        self.model = torchvision.models.resnet152(
            pretrained=config.USE_PRETRAINED
        )
        self.resnet_classifier()

    def alexnet(self) -> None:
        """alexnet architecture"""
        self.model = torchvision.models.alexnet(
            pretrained=config.USE_PRETRAINED
        )
        self.vgg_classifier()

    def vgg16(self) -> None:
        """VGG 16 architecture"""
        self.model = torchvision.models.vgg16_bn(
            pretrained=config.USE_PRETRAINED
        )
        self.vgg_classifier()

    def vgg19(self) -> None:
        """VGG 19 architecture"""
        self.model = torchvision.models.vgg19_bn(
            pretrained=config.USE_PRETRAINED
        )
        self.vgg_classifier()

    def densenet169(self) -> None:
        """Densenet 169 architecture"""
        self.model = torchvision.models.densenet169(
            pretrained=config.USE_PRETRAINED
        )
        self.densenet_classifier()

    def densenet201(self) -> None:
        """Densenet 201 architecture"""
        self.model = torchvision.models.densenet201(
            pretrained=config.USE_PRETRAINED
        )
        self.densenet_classifier()

    def efficient(self) -> None:
        """EfficientNet-b0 architecture"""
        self.model = EfficientNet.from_name("efficientnet-b0")
