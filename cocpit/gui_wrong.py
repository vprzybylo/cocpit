import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from ipywidgets import Button
import PIL
import cocpit
from cocpit.interpretability.misc_funcs import (
    normalize,
    preprocess_image,
    apply_colormap_on_image,
)
import cocpit.config as config
from cocpit.auto_str import auto_str
from typing import Optional, Tuple
import cv2
import os
from cocpit.interpretability import gradcam, vanilla_backprop, guided_backprop

plt_params = {
    "axes.labelsize": "large",
    "axes.titlesize": "large",
    "xtick.labelsize": "large",
    "ytick.labelsize": "large",
    "legend.title_fontsize": 12,
}
plt.rcParams["font.family"] = "serif"
plt.rcParams.update(plt_params)


@auto_str
class GUI:
    """
    - ipywidget buttons to label incorrect predictions from a dataloader.
    - The dataloader, model, and all class variables are initialized in notebooks/move_wrong_predictions.ipynb

    Args:
        wrong_trunc (List[int]): indices where the model predictions are wrong
        labels (np.ndarray[int]): image labels
        paths (np.ndarray[str]): image paths
        topk_props (np.ndarray[float]): top predicted probabilites
        topk_classes (np.ndarray[int]): classes related to the top predicted probabilites
    """

    def __init__(
        self,
        paths,
        topk_probs,
        topk_classes,
        wrong_trunc=None,
        labels=None,
        precip=None,
    ):
        self.index = 0
        self.paths = paths
        self.topk_probs = topk_probs
        self.topk_classes = topk_classes
        self.labels = labels
        if wrong_trunc:
            self.paths = np.array(paths)[wrong_trunc]
            self.topk_probs = np.array(topk_probs)[wrong_trunc]
            self.topk_classes = np.array(topk_classes)[wrong_trunc]
            self.labels = np.array(labels)[wrong_trunc]
        self.output = ipywidgets.Output()  # main display
        self.next_btn = Button(description="Next")
        self.next_btn.on_click(self.on_button_next)
        self.count = 0  # number of moved images
        self.precip = precip
        self.prep_image = None
        self.cam = None
        self.vanilla_grads = None
        self.gradients = None
        self.target_size = None

    def open_image(self) -> Optional[PIL.Image.Image]:
        try:
            self.image = PIL.Image.open(self.paths[self.index])
            self.target_size = (np.shape(self.image)[1], np.shape(self.image)[0])
        except FileNotFoundError:
            print("The file cannot be found.")
            return

    def on_button_next(self, b) -> None:
        """
        when the next button is clicked, make a new image and bar chart appear
        by updating the index within the wrong predictions by 1

        Args:
            b: button instance
        """
        self.index = self.index + 1
        self.interp()

    def show_original(self, ax1: plt.Axes) -> None:
        """
        display the raw image

        Args:
            ax1 (plt.Axes): subplot axis
        """
        clear_output()  # so that the next fig doesnt display below
        ax1.imshow(self.image)
        station = self.paths[self.index].split("/")[-1].split("_")[-1].split(".")[0]
        if self.precip:
            ax1.set_title(
                f"Prediction: {[config.CLASS_NAMES[e] for e in self.topk_classes[self.index]][0]}\n"
                f"Station: {station}\n"
                f"1 min precip accumulation: {self.precip[self.index].values[0]}"
            )
        else:
            pred_list = [config.CLASS_NAMES[e] for e in self.topk_classes[self.index]]
            pred_mag = [np.round(i * 100, 2) for i in self.topk_probs[self.index]]

            if self.labels.any():
                ax1.set_title(
                    f"Human label: {config.CLASS_NAMES[self.labels[self.index]]}\n"
                    f"Model Prediction [%]: \n"
                    f"{', '.join(repr(e) for e in pred_list)}\n"
                    f"{', '.join(repr(e) for e in pred_mag)}"
                )
            else:
                ax1.set_title(
                    f"Prediction [%]: \n"
                    f"{', '.join(repr(e) for e in pred_list)}\n"
                    f"{', '.join(repr(e) for e in pred_mag)}"
                )
        ax1.axis("off")

    def bar_chart(self, ax3: plt.Axes) -> None:
        """create barchart that outputs top k predictions for a given image

        Args:
            ax3 (plt.Axes): subplot axis
        """
        y_pos = np.arange(len(self.topk_probs[self.index]))
        ax3.barh(y_pos, self.topk_probs[self.index], align="center")
        ax3.set_yticks(y_pos)
        ax3.set_yticklabels(
            [config.CLASS_NAMES[e] for e in self.topk_classes[self.index]]
        )
        ax3.yaxis.set_label_position("right")
        ax3.yaxis.tick_right()
        ax3.invert_yaxis()  # labels read top-to-bottom
        ax3.set_title("Class Probability")

    def plot_saliency(self, ax2: plt.Axes, size: int = 224) -> None:
        """create saliency map for image in test dataset

        Args:
            ax2 (plt.Axes): subplot axis
            size (int): image size for transformation
        """
        image = cocpit.plotting_scripts.saliency.preprocess(self.image, size)
        saliency, _, _ = cocpit.plotting_scripts.saliency.get_saliency(image)
        saliency = cv2.resize(
            np.array(np.transpose(saliency, (1, 2, 0))), self.target_size
        )
        ax2.imshow(saliency, cmap=plt.cm.hot)
        ax2.axes.xaxis.set_ticks([])
        ax2.axes.yaxis.set_ticks([])
        ax2.set_title("Saliency Map")

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
        grad_cam = gradcam.GradCam(target_layer=42)
        self.cam = grad_cam.generate_cam(self.prep_img)

    def plot_gradcam(self, ax: plt.Axes) -> None:
        """plot gradient class activation map"""
        heatmap = apply_colormap_on_image(self.cam, self.image, alpha=0.5)
        ax.imshow(heatmap)
        ax.axes.set_title("GRAD-CAM")
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

    def save(
        self,
        fig: plt.Axes,
        directory: str = "/ai2es/codebook_dataset/carly/interpretability",
        class_="obstructed",
    ):
        if not os.path.exists(os.path.join(directory, class_)):
            os.makedirs(os.path.join(directory, class_))
        fig.savefig(
            os.path.join(directory, class_, self.paths[self.index].split("/")[-1])
        )

    def call_plots(self, figsize: Tuple[int, int] = (12, 6), ncols=3, nrows=2):
        fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(
            constrained_layout=True, figsize=figsize, ncols=ncols, nrows=nrows
        )
        self.show_original(ax1)
        # self.plot_saliency(ax2)
        # self.bar_chart(ax3)
        self.plot_vanilla_bp(ax4)
        self.plot_gradcam(ax5)
        self.plot_guided_gradcam(ax6)
        self.plot_saliency_pos(ax2)
        self.plot_saliency_neg(ax3)
        # self.save(fig)

    def interp(self) -> None:
        """
        Calculate gradients used in interpretability
        """
        with self.output:
            # add chart to ipywidgets.Output()
            if self.index == len(self.topk_probs):
                print("You have completed looking at all predictions!")
                return
            else:
                self.open_image()
                self.prep_img = preprocess_image(self.image).cuda()
                self.generate_cam()
                self.get_guided_grads()
                self.get_vanilla_grads()
                self.call_plots()
                plt.show()
