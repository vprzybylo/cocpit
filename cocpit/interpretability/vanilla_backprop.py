"""
Created on Thu Oct 26 11:19:58 2017
@author: Utku Ozbulak - github.com/utkuozbulak
Modified by Vanessa Przybylo
"""
import torch
from cocpit import config as config
import numpy as np
import cv2
from typing import Tuple, Optional
from torch.nn import ReLU


class Gradient:
    """
    Produces gradients generated with vanilla back propagation from the image
    """

    def __init__(self, model, guided_backprop=True):
        self.model: torch.nn.parallel.data_parallel.DataParallel = model
        self.gradients = None
        self.forward_relu_outputs = []
        self.model.eval()
        self.one_hot_output: torch.Tensor = None
        self.model_output = None
        self.first_layer_hook()
        if guided_backprop:
            self.update_relus()

    def first_layer_hook(self):
        """Register hook to the first layer"""

        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

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

    def target_class(self, target_class):
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
    ):
        # Forward
        self.model_output = self.model(input_image)
        if target_class is None:
            target_class = np.argmax(self.model_output.data.cpu().numpy())
        # Zero grads
        self.model.zero_grad()
        self.target_class(target_class)
        # Backward pass
        self.model_output.backward(gradient=self.one_hot_output)
        return cv2.resize(self.gradients.cpu().numpy()[0, 0, :, :], target_size)
