"""
Created on Thu Oct 26 11:06:51 2017

@author: Utku Ozbulak - github.com/utkuozbulak
"""
from PIL import Image
import numpy as np
import torch
from cocpit import config as config
import cv2


class CamExtractor:
    """
    Extracts cam features from the model
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None

    def save_gradient(self, grad):
        self.gradients = grad

    def forward_pass_on_convolutions(self, x):
        """
        Does a forward pass on convolutions, hooks the function at given layer
        """
        conv_output = None
        for module_pos, module in self.model.module.features._modules.items():
            x.to(config.DEVICE)
            x = module(x)  # Forward
            if int(module_pos) == self.target_layer:
                x.register_hook(self.save_gradient)
                conv_output = x  # Save the convolution output on that layer
        return conv_output, x

    def forward_pass(self, x):
        """
        Does a full forward pass on the model
        """
        # Forward pass on the convolutions
        conv_output, x = self.forward_pass_on_convolutions(x)
        x = x.view(x.size(0), -1)  # Flatten
        # Forward pass on the classifier

        x = self.model.module.classifier(x)
        return conv_output, x


class GradCam:
    """
    Produces class activation map
    """

    def __init__(self, target_layer):
        self.model = torch.load(config.MODEL_SAVENAME).to(config.DEVICE)
        self.model = self.model.to(config.DEVICE)
        self.model.eval()
        # Define extractor
        self.extractor = CamExtractor(self.model, target_layer)

    def generate_cam(self, input_image, target_class=None):
        # Full forward pass
        # conv_output is the output of convolutions at specified layer
        input_image = input_image.to(config.DEVICE)
        conv_output, model_output = self.extractor.forward_pass(input_image)
        if target_class is None:
            target_class = np.argmax(model_output.data.cpu().numpy())
        # Target for backprop
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_()
        one_hot_output = one_hot_output.to(config.DEVICE)
        one_hot_output[0][target_class] = 1

        # Zero grads
        self.model.module.features.zero_grad()
        self.model.module.classifier.zero_grad()
        # Backward pass with specified target
        model_output.backward(gradient=one_hot_output, retain_graph=True)
        # Get hooked gradients
        guided_gradients = self.extractor.gradients.data.cpu().numpy()[0]
        # Get convolution outputs
        target = conv_output.data.cpu().numpy()[0]
        # Get weights from gradients
        weights = np.mean(
            guided_gradients, axis=(1, 2)
        )  # Take averages for each gradient

        # Create empty numpy array for cam
        cam = np.zeros(target.shape[1:], dtype=np.float32)
        # Multiply each weight with its conv output and then, sum
        for i, w in enumerate(weights):
            cam += w * target[i, :, :]
        cam = np.maximum(cam, 0)
        cam = (cam - np.min(cam)) / (np.max(cam) - np.min(cam))  # Normalize between 0-1
        # cam = np.uint8(cam * 255)  # Scale between 0-255 to visualize
        # cam = np.uint8(Image.fromarray(cam).resize(720,1280))
        cam = cv2.resize(np.asarray(Image.fromarray(cam)), (1280, 720)) / 255
        return cam, config.CLASS_NAMES[target_class]
