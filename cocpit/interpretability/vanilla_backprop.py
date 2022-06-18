"""
Created on Thu Oct 26 11:19:58 2017
@author: Utku Ozbulak - github.com/utkuozbulak
Modified by Vanessa Przybylo
"""
import torch
from cocpit import config as config
import numpy as np
import cv2


class VanillaBackprop:
    """
    Produces gradients generated with vanilla back propagation from the image
    """

    def __init__(self):
        self.model = torch.load(config.MODEL_SAVENAME).to(config.DEVICE)
        self.gradients = None
        # Put model in evaluation mode
        self.model.eval()
        # Hook the first layer to get the gradient
        self.hook_layers()
        self.model_output = None

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]

        # Register hook to the first layer
        first_layer = list(self.model.module.features._modules.items())[0][1]
        first_layer.register_backward_hook(hook_function)

    def target_class(self, target_class):
        """Target for backprop"""
        self.one_hot_output = torch.FloatTensor(1, self.model_output.size()[-1]).zero_()
        self.one_hot_output = self.one_hot_output.to(config.DEVICE)
        self.one_hot_output[0][target_class] = 1

    def generate_gradients(self, input_image, target_size, target_class=None):
        # Forward
        input_image = input_image.to(config.DEVICE)
        self.model_output = self.model(input_image)
        if target_class is None:
            target_class = np.argmax(self.model_output.data.cpu().numpy())
        # Zero grads
        self.model.zero_grad()
        self.target_class(target_class)
        # Backward pass
        self.model_output.backward(gradient=self.one_hot_output)
        self.gradients = cv2.resize(
            np.transpose(self.gradients.cpu().numpy()[0], (1, 2, 0)), target_size
        )
        return self.gradients
