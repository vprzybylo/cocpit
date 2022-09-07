"""
Make interpretability 6 panel plot.
Called in /ai2es/notebooks/classify_real_preds.ipynb
"""

from cocpit.interpretability import gradcam, vanilla_backprop, interp
from cocpit.interpretability.misc_funcs import apply_colormap_on_image
import matplotlib.pyplot as plt
import matplotlib
import PIL
from typing import Tuple
import numpy as np
import torch


class Plot(interp.Interp):
    """Plot interpretability output in 6 panels"""

    def __init__(self, model, image):
        self.model: torch.nn.parallel.data_parallel.DataParallel = model
        self.image: PIL.Image = image
        super().__init__(model, image)

    def plot_saliency_pos(self, ax: plt.Axes) -> None:
        """
        plot positive saliency - where gradients are positive after RELU
        """
        ax.imshow(self.pos_saliency)
        ax.axes.set_title("Positive Saliency")
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])

    def plot_saliency_neg(self, ax: plt.Axes) -> None:
        """
        plot negative saliency - where gradients are positive after RELU
        """
        ax.imshow(self.cam_gb)
        ax.axes.set_title("Negative Saliency")
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])

    def plot_cam_all_layers(self, ax: plt.Axes) -> None:
        ax.imshow(self.cam_gb)
        ax.axes.set_title("Negative Saliency")
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])

    def plot_vanilla_bp(self, ax: plt.Axes) -> None:
        """plot vanilla backpropagation gradients"""
        ax.imshow(self.vanilla_grads)
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])
        ax.axes.set_title("Vanilla Backpropagation")

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
        cam_gb = np.multiply(self.cam, self.gradients)
        ax.imshow(cam_gb)
        ax.axes.set_title("Guided GRAD-CAM")
        ax.axes.xaxis.set_ticks([])
        ax.axes.yaxis.set_ticks([])

    def make_plots(
        self, figsize: Tuple[int, int] = (12, 6), ncols=3, nrows=2
    ) -> matplotlib.figure.Figure:
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
            constrained_layout=True, figsize=figsize, ncols=ncols, nrows=nrows
        )
        ax1.imshow(self.image)
        ax1.axis("off")
        self.plot_vanilla_bp(ax4)
        self.plot_gradcam(ax5)
        self.plot_guided_gradcam(ax6)
        self.plot_saliency_pos(ax2)
        self.plot_saliency_neg(ax3)
        return fig
