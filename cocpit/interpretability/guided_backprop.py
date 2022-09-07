"""
Created on Thu Oct 26 11:23:47 2017
@author: Utku Ozbulak - github.com/utkuozbulak
"""
import torch
from torch.nn import ReLU
import numpy as np
from cocpit import config as config
from cocpit.interpretability.misc_funcs import (
    convert_to_grayscale,
    save_gradient_images,
    get_positive_negative_saliency,
)


class GuidedBackprop:
    """
    Produces gradients generated with guided back propagation from the given image
    """

    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.guided_grads = None
        self.pos_saliency = None
        self.neg_saliency = None
        self.forward_relu_outputs = []
        # Put model in evaluation mode
        self.model.eval()
        self.update_relus()
        self.hook_layers()

    def hook_layers(self):
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

        def relu_forward_hook_function(module, ten_in, ten_out):
            """
            Store results of forward pass
            """
            self.forward_relu_outputs.append(ten_out)

        # Loop through layers, hook up ReLUs
        for pos, module in self.model.module.features._modules.items():
            if isinstance(module, ReLU):
                module.register_backward_hook(relu_backward_hook_function)
                module.register_forward_hook(relu_forward_hook_function)

    def target_class(self, target_class: int) -> None:
        """Target for backprop"""
        self.one_hot_output = (
            torch.FloatTensor(1, self.model_output.size()[-1]).zero_().to(config.DEVICE)
        )
        self.one_hot_output[0][target_class] = 1

    def generate_gradients(self, input_image, target_class=None):
        self.model_output = self.model(input_image)
        self.model.zero_grad()
        if target_class is None:
            target_class = np.argmax(self.model_output.data.cpu().numpy())
        self.target_class(target_class)
        self.model_output.backward(gradient=self.one_hot_output)
        return self.gradients.data.cpu().numpy()[0]
