import shutil

import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from ipywidgets import Button
from PIL import Image
from typing import List, Optional
from cocpit.auto_str import auto_str
import cocpit

plt_params = {
    "axes.labelsize": "xx-large",
    "axes.titlesize": "xx-large",
    "xtick.labelsize": "xx-large",
    "ytick.labelsize": "xx-large",
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
        all_labels (np.ndarray[int]): image labels
        all_paths (np.ndarray[str]): image paths
        all_topk_props (np.ndarray[float]): top predicted probabilites
        all_topk_classes (np.ndarray[int]): classes related to the top predicted probabilites
    """

    def __init__(
        self,
        wrong_trunc: List[int],
        all_labels: np.ndarray,
        all_paths: np.ndarray,
        all_topk_probs: np.ndarray,
        all_topk_classes: np.ndarray,
    ):

        self.index = 0
        self.all_labels = all_labels[wrong_trunc]
        self.all_paths = all_paths[wrong_trunc]
        self.all_topk_probs = all_topk_probs[wrong_trunc]
        self.all_topk_classes = all_topk_classes[wrong_trunc]

        self.label = self.all_labels[self.index]
        self.next_btn = Button(
            description="Next",
            style=dict(
                font_style="italic",
                font_weight="bold",
                font_variant="small-caps",
            ),
        )
        self.buttons = []
        self.count = 0  # number of moved images
        self.center = ipywidgets.Output()  # center image with predictions

    def open_image(self) -> Optional[Image.Image]:
        """
        Open an image from a path at a given index

        Returns:
            Union[Image.Image, None]: opened PIL image or None if no image is opened

        Raises:
            FileNotFoundError: File already moved and cannot be opened
        """

        try:
            return Image.open(self.all_paths[self.index])
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

    def init_fig(self, image: Image.Image, ax1: plt.Axes) -> None:
        """
        Display the raw image

        Args:
            image (Image.Image): opened image
            ax1 (plt.Axes): subplot axis
        """
        clear_output()  # so that the next fig doesnt display below
        ax1.imshow(image, aspect="auto")
        ax1.set_title(
            f"Human Labeled as: {cocpit.config.CLASS_NAMES[self.all_labels[self.index]]}\n"
            f"Model Labeled as: {[cocpit.config.CLASS_NAMES[e] for e in self.all_topk_classes[self.index]][0]}\n"
        )
        ax1.axis("off")

    def bar_chart(self, ax2) -> None:
        """
        Create barchart that outputs top k predictions for a given image

        Args:
            ax2 (plt.Axes): subplot axis
        """

        y_pos = np.arange(len(self.all_topk_probs[self.index]))
        ax2.barh(y_pos, self.all_topk_probs[self.index])
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(
            [cocpit.config.CLASS_NAMES[e] for e in self.all_topk_classes[self.index]]
        )
        ax2.tick_params(axis="y", rotation=45)
        ax2.invert_yaxis()  # labels read top-to-bottom
        ax2.set_title("Class Probability")

    def plot_saliency(self, image: Image.Image, ax2: plt.Axes, size: int = 224) -> None:
        """Create saliency map for image in test dataset

        Args:
            image (PIL.Image.Image): opened image
            ax2 (plt.Axes): subplot axis
            size (int): image size for transformation
        """
        image = cocpit.plotting_scripts.saliency.preprocess(image.convert("RGB"), size)
        saliency, _, _ = cocpit.plotting_scripts.saliency.get_saliency(image)
        ax2.imshow(saliency[0], cmap=plt.cm.hot, aspect="auto")
        ax2.axes.xaxis.set_ticks([])
        ax2.axes.yaxis.set_ticks([])

    def save_image(self, b) -> None:
        """
        Move the image based on dropdown selection
        """

        filename = self.all_paths[self.index].split("/")[-1]

        try:
            shutil.move(
                f"{cocpit.config.DATA_DIR}{cocpit.config.CLASS_NAME_MAP[cocpit.config.CLASS_NAMES[self.all_labels[self.index]]]}/{filename}",
                f"{cocpit.config.DATA_DIR}{cocpit.config.CLASS_NAME_MAP[b.description]}/{filename}",
            )
            self.count += 1
            print(f"moved {self.count} images")

        except FileNotFoundError:
            print(self.all_paths[self.index])
            print("File not found or directory does not exist. Not moving.")

    def visualizations(self) -> None:
        """
        Use the human and model labels and classes to
        create a bar chart with the top k predictions
        from the image at the current index
        """
        # add chart to ipywidgets.Output()
        with self.center:
            if self.index == len(self.all_topk_probs):
                print("You have completed looking at all incorrect predictions!")
                return
            else:
                image = self.open_image()
                _, (ax1, ax2, ax3) = plt.subplots(
                    constrained_layout=True, figsize=(19, 5), ncols=3, nrows=1
                )
                if image:
                    self.init_fig(image, ax1)
                    self.plot_saliency(image, ax2)
                    self.bar_chart(ax3)
                    plt.show()
                    # fig.savefig(f"/ai2es/plots/wrong_preds{self.index}.pdf")
