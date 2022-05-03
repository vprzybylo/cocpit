import torchvision
from efficientnet_pytorch import EfficientNet
from torch import nn

import cocpit.config as config  # isort:split


class Model:
    """
    Model initializations from torchvision. We always train from scratch.

    Args:
        feature_extract (bool): update only the last layer parameters. Default False
        use_pretrained (bool): use a pre-trained model to extract meaningful features from new samples. Default False.
        num_classes (int): number of classes
        model (torchvision.models): a torchvision.models instance
    """

    def __init__(
        self,
        feature_extract: bool = False,
        use_pretrained: bool = False,
    ):
        self.feature_extract = feature_extract
        self.use_pretrained = use_pretrained
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
        self.set_parameter_requires_grad() if self.feature_extract else None
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, self.num_classes)

    def vgg_classifier(self) -> None:
        """
        create linear output layer equal to number of classes
        """
        self.set_parameter_requires_grad() if self.feature_extract else None
        num_ftrs = self.model.classifier[6].in_features
        self.model.classifier[6] = nn.Linear(num_ftrs, self.num_classes)

    def densenet_classifier(self) -> None:
        """
        create linear output layer equal to number of classes
        """
        self.set_parameter_requires_grad() if self.feature_extract else None
        num_ftrs = self.model.classifier.in_features
        self.model.classifier = nn.Linear(num_ftrs, self.num_classes)

    def resnet18(self) -> None:
        """resnet 18 architecture"""
        self.model = torchvision.models.resnet18(pretrained=self.use_pretrained)
        self.resnet_classifier()

    def resnet34(self) -> None:
        """resnet 34 architecture"""
        self.model = torchvision.models.resnet34(pretrained=self.use_pretrained)
        self.resnet_classifier()

    def resnet152(self) -> None:
        """resnet 152 architecture"""
        self.model = torchvision.models.resnet152(pretrained=self.use_pretrained)
        self.resnet_classifier()

    def alexnet(self) -> None:
        """alexnet architecture"""
        self.model = torchvision.models.alexnet(pretrained=self.use_pretrained)
        self.vgg_classifier()

    def vgg16(self) -> None:
        """VGG 16 architecture"""
        self.model = torchvision.models.vgg16_bn(pretrained=self.use_pretrained)
        self.vgg_classifier()
        print(self.model)

    def vgg19(self) -> None:
        """VGG 19 architecture"""
        self.model = torchvision.models.vgg19_bn(pretrained=self.use_pretrained)
        self.vgg_classifier()

    def densenet169(self) -> None:
        """Densenet 169 architecture"""
        self.model = torchvision.models.densenet169(pretrained=self.use_pretrained)
        self.densenet_classifier()

    def densenet201(self) -> None:
        """Densenet 201 architecture"""
        self.model = torchvision.models.densenet201(pretrained=self.use_pretrained)
        self.densenet_classifier()

    def efficient(self) -> None:
        """EfficientNet-b0 architecture"""
        self.model = EfficientNet.from_name("efficientnet-b0")


def initialize_model(
    model_name: str, feature_extract: bool = False, use_pretrained: bool = False
) -> torchvision.models:
    """
    Set up model architectures. All input size of 224

    Args:
        feature_extract (bool): Start with a pretrained model and only
                                update the final layer weights from which we derive predictions
        use_pretrained (bool): Update all of the model’s parameters (retrain). Default = False
    """

    m = Model(feature_extract, use_pretrained)
    # call method based on str model name
    method = getattr(Model, model_name)
    method(m)
    return m.model
