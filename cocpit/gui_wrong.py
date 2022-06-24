import shutil

import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from ipywidgets import Button
import PIL
from typing import List, Optional
from cocpit.auto_str import auto_str
import cocpit
from cocpit.interpretability import guided_backprop
from cocpit.interpretability.misc_funcs import preprocess_image

plt_params = {
    "axes.labelsize": "xx-large",
    "axes.titlesize": "xx-large",
    "xtick.labelsize": "xx-large",
    "ytick.labelsize": "xx-large",
    "legend.title_fontsize": 12,
}
plt.rcParams["font.family"] = "serif"
plt.rcParams.update(plt_params)


class Interp:
    """
    Holds interpretability methods
    Args:
        gradients (np.ndarray):  a vector which gives us the direction in which the loss function has the steepest ascent.
        pos_saliency (np.ndarray): Positive values in the gradients in which a small change to that pixel will increase the output value
        neg_saliency (np.ndarray): Negative values in the gradients in which a small change to that pixel will decrease the output value

    """

    def __init__(self):
        self.gradients = None
        self.pos_saliency = None
        self.neg_saliency = None

    def plot_saliency_pos(self, ax: plt.Axes):
        """
        plot positive saliency - where gradients are positive after RELU
        """
        ax.imshow(self.pos_saliency)
        ax.axes.set_title("Positive Saliency")
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


@auto_str
class GUI(Interp):
    """
    - ipywidget buttons to label incorrect predictions from a dataloader.
    - The dataloader, model, and all class variables are initialized in notebooks/move_wrong_predictions.ipynb

    Args:
        wrong_trunc (List[int]): indices where the model predictions are wrong
        labels (np.ndarray[int]): image labels
        paths (np.ndarray[str]): image paths
        topk_probs (np.ndarray[float]): top predicted probabilites
        topk_classes (np.ndarray[int]): classes related to the top predicted probabilites

        buttons (List[widgets.Button]): list of buttons for class names
        next_btn (widgets.Button): next button to move index by one
        count (int): number of moved images

        image (torch.Tensor): preprocessed image. Default None, defined in interp()
        prep_image (torch.Tensor): preprocessed image. Default None, defined in interp()
        target_size (Tuple[int, int]): original image size for resizing interpretability plots
        center (ipywidgets.Output()): main display
    """

    def __init__(
        self,
        wrong_trunc: List[int],
        labels: np.ndarray,
        paths: np.ndarray,
        topk_probs: np.ndarray,
        topk_classes: np.ndarray,
    ):
        self.index = 0
        self.labels = np.array(labels)[wrong_trunc]
        self.paths = np.array(paths)[wrong_trunc]
        self.topk_probs = np.array(topk_probs)[wrong_trunc]
        self.topk_classes = np.array(topk_classes)[wrong_trunc]

        self.buttons = []
        self.next_btn = Button(
            description="Next",
            style=dict(
                font_style="italic",
                font_weight="bold",
                font_variant="small-caps",
            ),
        )
        self.count = 0

        self.image = None
        self.prep_image = None
        self.target_size = None
        self.center = ipywidgets.Output()

    def open_image(self):
        """
        Open an image from a path at a given index

        Returns:
            Union[Image.Image, None]: opened PIL image or None if no image is opened

        Raises:
            FileNotFoundError: File already moved and cannot be opened
        """

        try:
            self.image = PIL.Image.open(self.paths[self.index])
            self.target_size = (np.shape(self.image)[1], np.shape(self.image)[0])

        except FileNotFoundError:
            print("This file cannot be found.")

    def make_buttons(self) -> None:
        """Make buttons for each category"""

        for idx, label in enumerate(cocpit.config.CLASS_NAMES):
            self.buttons.append(
                Button(
                    description=label,
                )
            )
            self.buttons[idx].on_click(self.save_image)
        self.next_btn.on_click(self.on_button_next)

    def on_button_next(self, b) -> None:
        """
        When the next button is clicked, make a new image and bar chart appear
        by updating the index within the wrong predictions by 1
        """

        self.index = self.index + 1
        self.visualizations()

    def align_buttons(self):
        """
        Alter layout based on # of classes
        """
        with self.center:
            if len(cocpit.config.CLASS_NAMES) > 5:
                # align buttons vertically
                self.label_btns = ipywidgets.VBox(
                    [self.buttons[i] for i in range(len(cocpit.config.CLASS_NAMES))]
                )
            else:
                # align buttons horizontally
                self.label_btns = ipywidgets.HBox(
                    [self.buttons[i] for i in range(len(cocpit.config.CLASS_NAMES))],
                )

    def init_fig(self, ax1: plt.Axes) -> None:
        """
        Display the raw image

        Args:
            image (Image.Image): opened image
            ax1 (plt.Axes): subplot axis
        """
        clear_output()  # so that the next fig doesnt display below
        ax1.imshow(self.image, aspect="auto")
        ax1.set_title(
            f"Human Labeled as: {cocpit.config.CLASS_NAMES[self.labels[self.index]]}\n"
            f"Model Labeled as: {[cocpit.config.CLASS_NAMES[e] for e in self.topk_classes[self.index]][0]}\n"
        )
        ax1.axis("off")

    def bar_chart(self, ax2) -> None:
        """
        Create barchart that outputs top k predictions for a given image

        Args:
            ax2 (plt.Axes): subplot axis
        """

        y_pos = np.arange(len(self.topk_probs[self.index]))
        ax2.bar(y_pos, self.topk_probs[self.index])
        ax2.set_ylim(0.0, 1.0)
        ax2.set_xticks(y_pos)
        ax2.set_xticklabels(
            [cocpit.config.CLASS_NAMES[e] for e in self.topk_classes[self.index]]
        )
        ax2.tick_params(axis="x", rotation=45)
        # ax2.invert_yaxis()  # labels read top-to-bottom
        ax2.set_title("Class Probability")

    def save_image(self, b) -> None:
        """
        Move the image based on dropdown selection

        Args:
            b: button instance
        """

        filename = self.paths[self.index].split("/")[-1]

        try:
            shutil.move(
                f"{cocpit.config.DATA_DIR}{cocpit.config.CLASS_NAME_MAP[cocpit.config.CLASS_NAMES[self.labels[self.index]]]}/{filename}",
                f"{cocpit.config.DATA_DIR}{cocpit.config.CLASS_NAME_MAP[b.description]}/{filename}",
            )
            self.count += 1
            print(f"moved {self.count} images")

        except FileNotFoundError:
            print(self.paths[self.index])
            print("File not found or directory does not exist. Not moving.")

    def visualizations(self) -> None:
        """
        Use the human and model labels and classes to
        create a bar chart with the top k predictions
        from the image at the current index
        """
        # add chart to ipywidgets.Output()
        with self.center:
            if self.index == len(self.topk_probs):
                print("You have completed looking at all incorrect predictions!")
                return
            else:
                _, (ax1, ax2) = plt.subplots(
                    constrained_layout=True, figsize=(14, 7), ncols=2, nrows=1
                )
                self.open_image()
                self.init_fig(ax1)
                self.prep_img = preprocess_image(self.image).cuda()
                # self.get_guided_grads()
                # self.plot_saliency_pos(ax2)
                self.bar_chart(ax2)
                plt.show()
                # fig.savefig(f"/ai2es/plots/wrong_preds{self.index}.pdf")
