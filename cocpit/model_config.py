import cocpit.config as config
import torch
from torch import nn, optim
import torchvision


class ModelConfig:
    """
    Model configurations for:
        - dropout
        - optimizer
        - device settings (cpu/gpu)
        - parameters to update
    Args:
        model (torchvision.models): loaded pytorch model object
        optimizer (torch.optim.sgd.SGD): an algorithm that modifies the attributes of the neural network
    """

    def __init__(self, model: torchvision.models):
        self.model = model
        self.optimizer: torchvision.optimizer = None

    def update_params(self, feature_extract: bool = False):
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

        Args:
            feature_extract (bool): only update the weights of the last layer
        Returns:
            params_to_update (generator): model parameters that get updated
        """
        params_to_update = self.model.parameters()

        if feature_extract:
            params_to_update = [
                param
                for _, param in self.model.named_parameters()
                if param.requires_grad
            ]

        print("model params: ", self.count_parameters())
        return params_to_update

    def count_parameters(self) -> int:
        """
        Number of model parameters to update
        """
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def set_optimizer(self, lr=0.01) -> None:
        """
        Model optimizer for stochastic gradient decent

        Args:
            lr (float): the learning rate for SGD
        """
        self.optimizer = optim.SGD(
            self.update_params(), lr=lr, momentum=0.9, nesterov=True
        )

    def set_criterion(self) -> None:
        self.criterion = nn.CrossEntropyLoss()

    def set_dropout(self, drop_rate=0.1) -> None:
        """
        Apply dropout rate: a technique to fight overfitting and improve neural network generalization

        Args:
            drop_rate (float): dropout rate for neurons - 1 = 100%
        """
        for m in self.model.modules():
            if isinstance(m, nn.Dropout):
                m.p = drop_rate

    def to_device(self) -> None:
        """
        Push model to gpu(s) if available
        """
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(config.DEVICE)
