"""
Gathers interpretability output for 6 panel plot.
Called in /ai2es/notebooks/classify_real_preds.ipynb
"""

from cocpit.interpretability.misc_funcs import (
    preprocess_image,
    normalize,
)
from cocpit.interpretability import gradcam, vanilla_backprop, guided_backprop
import matplotlib
import PIL
from typing import Tuple
import numpy as np


class Interp:
    """
    Holds interpretability methods
    Args:
        gradients (np.ndarray):  a vector which gives us the direction in which the loss function has the steepest ascent.
        pos_saliency (np.ndarray): Positive values in the gradients in which a small change to that pixel will increase the output value
        neg_saliency (np.ndarray): Negative values in the gradients

    """

    def __init__(
        self,
        image: PIL.Image,
    ):
        self.image = image
        self.target_size: Tuple[int, int] = (
            np.shape(self.image)[1],
            np.shape(self.image)[0],
        )
        self.pos_saliency: np.ndarray(
            np.shape(self.image)[0], np.shape(self.image)[1], float
        ) = None
        self.neg_saliency: np.ndarray(
            np.shape(self.image)[0], np.shape(self.image)[1], float
        ) = None
        self.cam: np.ndarray(
            np.shape(self.image)[0], np.shape(self.image)[1], float
        ) = None
        self.vanilla_grads: np.ndarray(
            np.shape(self.image)[0], np.shape(self.image)[1], float
        ) = None
        self.gradients: np.ndarray(
            np.shape(self.image)[0], np.shape(self.image)[1], float
        ) = None

    def generate_guided_grads(self):
        """
        Guided backpropagation and saliency maps.
        Positive and negative gradients indicate the direction in which we
        would have to change this feature to increase the conditional
        probability of the attended class given this input example and
        their magnitude shows the size of this effect.
        """
        GBP = guided_backprop.GuidedBackprop()
        self.gradients = GBP.generate_gradients(self.prep_img, self.target_size)
        self.pos_saliency = np.maximum(0, self.gradients) / self.gradients.max()
        self.neg_saliency = np.maximum(0, -self.gradients) / -self.gradients.min()

    def generate_vanilla_grads(self) -> None:
        """gradients for vanilla backpropagation"""
        VBP = vanilla_backprop.VanillaBackprop()
        vanilla_grads = VBP.generate_gradients(self.prep_img, self.target_size)
        self.vanilla_grads = normalize(vanilla_grads)

    def generate_cam(self):
        """generate gradient class activation map"""
        grad_cam = gradcam.GradCam(target_layer=42)
        self.cam = grad_cam.generate_cam(self.prep_img, self.target_size)

    def interp_runner(self) -> None:
        """
        Calculate gradients used in interpretability figures
        """
        self.prep_img = preprocess_image(self.image).cuda()
        self.generate_cam()
        self.generate_guided_grads()
        self.generate_vanilla_grads()
