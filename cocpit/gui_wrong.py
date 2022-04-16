"""
Holds the class for ipywidget buttons to
label incorrect predictions from a dataloader.
The dataloader, model, and all class variables
are initialized in notebooks/move_wrong_predictions.ipynb
"""
import shutil

import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from ipywidgets import Button
import PIL
from typing import Optional

import cocpit.config as config
from cocpit.auto_str import auto_str

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
    """create widgets"""

    def __init__(
        self, wrong_trunc, all_labels, all_paths, all_topk_probs, all_topk_classes
    ):

        self.index = 0
        self.all_labels = all_labels[wrong_trunc]
        self.all_paths = all_paths[wrong_trunc]
        self.all_topk_probs = all_topk_probs[wrong_trunc]
        self.all_topk_classes = all_topk_classes[wrong_trunc]
        self.label = self.all_labels[self.index]
        self.next_btn = Button(description="Next")
        self.menu = ipywidgets.Dropdown(
            options=config.CLASS_NAMES,
            description="Category:",
            value=config.CLASS_NAMES[self.label],
        )
        self.count = 0  # number of moved images
        self.center = ipywidgets.Output()  # center image with predictions

    def open_image(self) -> Optional[PIL.Image.Image]:
        try:
            return PIL.Image.open(self.all_paths[self.index])
        except FileNotFoundError:
            print("The file cannot be found.")
            return

    def button_functions(self) -> None:
        """
        create next button that progresses through incorrect predictions
        define save image on change for dropdown
        """
        self.visualizations()
        self.menu.observe(self.on_change, names="value")
        self.next_btn.on_click(self.on_button_next)

    def on_change(self, change) -> None:
        """
        when a class in the dropdown is selected, move the image and save
        it to the specified class
        """
        self.save_image(change)

    def on_button_next(self, b) -> None:
        """
        when the next button is clicked, make a new image and bar chart appear
        by updating the index within the wrong predictions by 1

        Args:
            b: button instance
        """

        self.index = self.index + 1
        self.visualizations()

        # keep the default dropdown value to agg
        # don't want it to change based on previous selection
        self.menu.value = config.CLASS_NAMES[self.label]

    def show_image(self, image: PIL.Image.Image, ax1: plt.Axes) -> None:
        """
        display the raw image

        Args:
            image (PIL.Image.Image): opened image
            ax1 (plt.Axes): subplot axis
        """
        ax1.imshow(image)
        ax1.set_title(
            f"Human Labeled as: {config.CLASS_NAMES[self.all_labels[self.index]]}\n"
            f"Model Labeled as: {[config.CLASS_NAMES[e] for e in self.all_topk_classes[self.index]][0]}\n"
            f"Index number: {self.index+1}"
        )
        ax1.axis("off")

    def bar_chart(self, ax2) -> None:
        """
        create barchart that outputs top k predictions for a given image

        Args
            ax2 (plt.Axes): subplot axis
        """

        y_pos = np.arange(len(self.all_topk_probs[self.index]))
        ax2.barh(y_pos, self.all_topk_probs[self.index], align="center")
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(self.all_topk_classes[self.index])
        ax2.tick_params(axis="y", rotation=45)
        ax2.invert_yaxis()  # labels read top-to-bottom
        ax2.set_title("Class Probability")

    def save_image(self, change) -> None:
        """
        move the image based on dropdown selection

        Args:
            change (button dropdown instance): new class label
        """
        filename = self.all_paths[self.index].split("/")[-1]

        print(
            f"{config.DATA_DIR}{config.CLASS_NAME_MAP[config.CLASS_NAMES[self.all_labels[self.index]]]}/{filename}"
        )
        print(f"{config.DATA_DIR}{config.CLASS_NAME_MAP[change.new]}/{filename}")
        try:
            shutil.move(
                f"{config.DATA_DIR}{config.CLASS_NAME_MAP[config.CLASS_NAMES[self.all_labels[self.index]]]}/{filename}",
                f"{config.DATA_DIR}{config.CLASS_NAME_MAP[change.new]}/{filename}",
            )
            self.count += 1
            print(f"moved {self.count} images")

        except FileNotFoundError:
            print(self.all_paths[self.index])
            print("File not found or directory does not exist. Not moving.")

    def visualizations(self) -> None:
        """
        use the human and model labels and classes to
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
                clear_output()  # so that the next fig doesnt display below
                _, (ax1, ax2) = plt.subplots(
                    constrained_layout=True, figsize=(9, 9), ncols=1, nrows=2
                )
                self.show_image(image, ax1)
                self.bar_chart(ax2)
                plt.show()
                # fig.savefig(f"/ai2es/plots/wrong_preds{self.index}.pdf")
