"""
Created on Thu Oct 26 11:23:47 2017

@author: Utku Ozbulak - github.com/utkuozbulak
Modified by Vanessa Przybylo
"""
import torch
from torch.nn import ReLU
from cocpit import config as config
import numpy as np
import cv2
from typing import Tuple, Optional


class GuidedBackprop:
    """
    Produces gradients generated with guided back propagation from the given image
    """

    def __init__(self):
        self.model: torch.nn.parallel.data_parallel.DataParallel = torch.load(
            "/ai2es/saved_models/v0.0.0/e[30]_bs[64]_k0_1model(s).pt"
        ).to(config.DEVICE)
        self.gradients: torch.Tensor = None
        self.forward_relu_outputs = []
        self.model_output = None
        self.one_hot_output: torch.Tensor = None
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self) -> None:
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model.module.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def update_relus(self):
        """
        Updates relu activation functions so that
            1- stores output in forward pass
            2- imputes zero for gradient values that are less than zero
        """

        def relu_backward_hook_function(module, grad_in, grad_out):
            """
            If there is a negative gradient, change it to zero
            """
            # Get last forward output
            corresponding_forward_output = self.forward_relu_outputs[-1]
            corresponding_forward_output[corresponding_forward_output > 0] = 1
            modified_grad_out = corresponding_forward_output * torch.clamp(
                grad_in[0], min=0.0
            )
            del self.forward_relu_outputs[-1]  # Remove last forward output
            return (modified_grad_out,)

        def relu_forward_hook_function(module, ten_in, ten_out) -> None:
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model.module.features._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def backward_pass(self) -> None:
        """Backward pass"""
        self.model_output.backward(gradient=self.one_hot_output)

    def zero_grads(self) -> None:
        """Zero gradients"""
        self.model.zero_grad()

    def encode_target_class(self, target_class: int) -> None:
        """Target for backprop"""
        self.one_hot_output = (
            torch.FloatTensor(1, self.model_output.size()[-1]).zero_().to(config.DEVICE)
        )
        self.one_hot_output[0][target_class] = 1

    def generate_gradients(
        self,
        input_image: torch.Tensor,
        target_size: Tuple[int, int],
        target_class: Optional[int] = None,
    ) -> None:
        self.model_output = self.model(input_image)

        if target_class is None:
            target_class = np.argmax(self.model_output.data.cpu().numpy())
        self.encode_target_class(target_class)
        self.zero_grads()
        self.backward_pass()
        # Convert Pytorch variable to numpy array
        # [0] to get rid of the first channel (1,3,224,224)
        return cv2.resize(
            self.gradients.cpu().numpy()[0, 0, :, :],
            target_size,
        )
