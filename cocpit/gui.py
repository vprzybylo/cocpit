import shutil

import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output
from ipywidgets import Button
from PIL import Image
from typing import List, Union
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
    """
    Creates buttons in notebooks/gui_check_dataset_one_class.ipynb to label images

    Args:
        all_labels (List[int]): List of labels on training data from one specified class
        all_paths (List[str]): List of paths of labeleld data from one specified class
        index (int): index of the image in the list of paths
        count (int): number of moved images
        center (ipywidgets.Output()): main display
        menu (ipywidgets.Dropdown): class name dropdown to move images
        foward (ipywidgets.Button): next button
    """

    def __init__(self, all_labels: List[int], all_paths: List[str]):
        self.all_labels = all_labels
        self.all_paths = all_paths
        self.index = 0
        self.count = 0  # number of moved images
        self.center = None
        self.menu = None
        self.forward = None

    def open_image(self) -> Union[Image.Image, None]:
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
            print(
                "This file was already moved and cannot be found. Please hit"
                " Next."
            )
            return

    def make_buttons(self) -> None:
        """
        Use ipywidgets to create a box for the image, bar chart, dropdown,
        and next button
        """
        self.center = ipywidgets.Output()  # center image with predictions
        self.menu = ipywidgets.Dropdown(
            options=config.CLASS_NAMES,
            description="Category:",
            value=config.CLASS_NAMES[self.all_labels[self.index]],
        )
        with self.center:
            self.view_classifications()

        self.menu.observe(self.save_image, names="value")

        # create button that progresses through incorrect predictions
        self.forward = Button(description="Next")
        self.forward.on_click(self.on_button_next)

    def on_button_next(self, b) -> None:
        """
        - When the next button is clicked, make a new image and bar chart appear by updating the index within the wrong predictions by 1
        - b is the button instance but not actually passed.
        """
        self.index = self.index + 1
        with self.center:
            self.view_classifications()

        # Keep the default dropdown value
        # Don't want it to change based on previous selection
        self.menu.value = config.CLASS_NAMES[self.all_labels[self.index]]

    def view_classifications(self) -> None:
        """
        Show image
        """
        clear_output()  # so that the next fig doesnt display below
        image = self.open_image()
        _, ax1 = plt.subplots(
            constrained_layout=True, figsize=(5, 7), ncols=1, nrows=1
        )
        ax1.imshow(image)
        ax1.set_title(
            "Human Labeled as:"
            f" {config.CLASS_NAMES[self.all_labels[self.index]]}\n"
        )
        ax1.axis("off")
        plt.show()

    def save_image(self, change) -> None:
        """
        Move the image based on dropdown selection

        Args:
            change (traitlets.utils.bunch.Bunch): dropdown selection

        """
        filename = self.all_paths[self.index].split("/")[-1]

        data_dir = "/ai2es/night_precip_hand_labeled/2017/"

        print(
            f"{data_dir}{config.CLASS_NAMES[self.all_labels[self.index]]}/{filename}"
        )
        print(f"{data_dir}{change.new}/{filename}")
        try:
            shutil.move(
                f"{data_dir}{config.CLASS_NAMES[self.all_labels[self.index]]}/{filename}",
                f"{data_dir}{change.new}/{filename}",
            )
            self.count += 1
            print(f"moved {self.count} images")

        except FileNotFoundError:
            print(self.all_paths[self.index])
            print("File Not Found. Not moving.")
