from cocpit.interpretability.misc_funcs import (
    preprocess_image,
    normalize,
    apply_colormap_on_image,
)
from cocpit.interpretability import gradcam, vanilla_backprop, guided_backprop
import matplotlib.pyplot as plt
import PIL
from typing import Tuple
import numpy as np


class Interp:
    """
    Holds interpretability methods
    Args:
        gradients (np.ndarray):  a vector which gives us the direction in which the loss function has the steepest ascent.
        pos_saliency (np.ndarray): Positive values in the gradients in which a small change to that pixel will increase the output value
        neg_saliency (np.ndarray): Negative values in the gradients in which a small change to that pixel will decrease the output value

    """

    def __init__(
        self,
        image: PIL.Image,
        label: str,
        prob: float,
        station: str = None,
        date: str = None,
        precip: bool = None,
    ):

        self.image = image
        self.target_size = (np.shape(self.image)[1], np.shape(self.image)[0])
        self.label = label
        self.prob = prob
        self.precip = precip
        self.station = station
        self.date = date
        self.gradients = None
        self.pos_saliency = None
        self.neg_saliency = None
        self.cam = None
        self.vanilla_grads = None
        self.gradients = None

    def plot_saliency_pos(self, ax: plt.Axes):
        """
        plot positive saliency - where gradients are positive after RELU
        """
        ax.imshow(self.pos_saliency)
        ax.axes.set_title("Positive Saliency")
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])

    def plot_saliency_neg(self, ax: plt.Axes):
        """
        plot negative saliency - where gradients are positive after RELU
        """
        ax.imshow(self.neg_saliency)
        ax.axes.set_title("Negative Saliency")
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])

    def get_guided_grads(self):
        """
        Guided backpropagation and saliency maps.
        Positive and negative gradients indicate the direction in which we
        would have to change this feature to increase the conditional
        probability of the attended class given this input example and
        their magnitude shows the size of this effect.
        """
        GBP = guided_backprop.GuidedBackprop()
        self.gradients = GBP.generate_gradients(self.prep_img, self.target_size)
        self.pos_saliency = (
            np.maximum(0, self.gradients[:, :, 0]) / self.gradients[:, :, 0].max()
        )
        self.neg_saliency = (
            np.maximum(0, -self.gradients[:, :, 0]) / -self.gradients[:, :, 0].min()
        )

    def get_vanilla_grads(self) -> None:
        """gradients for vanilla backpropagation"""
        VBP = vanilla_backprop.VanillaBackprop()
        vanilla_grads = VBP.generate_gradients(self.prep_img, self.target_size)
        self.vanilla_grads = normalize(vanilla_grads)

    def plot_vanilla_bp(self, ax: plt.Axes) -> None:
        """plot vanilla backpropagation gradients"""
        ax.imshow(self.vanilla_grads)
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        ax.axes.set_title("Vanilla Backpropagation")

    def generate_cam(self):
        """generate gradient class activation map"""
        grad_cam = gradcam.GradCam(target_layer=26)
        self.cam = grad_cam.generate_cam(self.prep_img)

    def plot_gradcam(self, ax: plt.Axes) -> None:
        """plot gradient class activation map"""
        heatmap = apply_colormap_on_image(self.cam, self.image, alpha=0.5)
        ax.imshow(heatmap)
        ax.axes.set_title("GRAD-CAM")
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])

    def plot_guided_gradcam(self, ax: plt.Axes) -> None:
        """
        Guided Grad CAM combines the best of Grad CAM,
        which is class-discriminative and localizes relevant image regions,
        and Guided Backpropagation, which visualizes gradients with respect
        to the image where negative gradients set to zero to highlight
        import pixel in the image when backpropagating through ReLU layers.
        """
        cam_gb = np.multiply(self.cam, self.gradients[:, :, 0])
        ax.imshow(cam_gb)
        ax.axes.set_title("Guided GRAD-CAM")
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])

    def show_original(self, ax1: plt.Axes) -> None:
        """
        display the raw image
        Args:
            ax1 (plt.Axes): subplot axis
        """
        ax1.imshow(self.image)
        # if self.precip:
        ax1.set_title(
            f"Prediction: {self.label}\n"
            f"Station: {self.station}\n"
            f"Probability: {self.prob}\n"
            f"Date: {self.date}\n"
            f"5 min precip accumulation {self.precip}"
        )
        ax1.axis("off")

    def call_plots(self, figsize: Tuple[int, int] = (12, 6), ncols=3, nrows=2):
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
            constrained_layout=True, figsize=figsize, ncols=ncols, nrows=nrows
        )
        self.show_original(
            ax1,
        )
        self.plot_vanilla_bp(ax4)
        self.plot_gradcam(ax5)
        self.plot_guided_gradcam(ax6)
        self.plot_saliency_pos(ax2)
        self.plot_saliency_neg(ax3)
        # self.save(fig)

    def interp(self) -> None:
        """
        Calculate gradients used in interpretability figures
        """
        self.prep_img = preprocess_image(self.image).cuda()
        self.generate_cam()
        self.get_guided_grads()
        self.get_vanilla_grads()
        self.call_plots()
        plt.show()
