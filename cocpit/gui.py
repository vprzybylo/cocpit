"""
Holds the class for ipywidget buttons to
label incorrect predictions from a dataloader.
The dataloader, model, and all class variables
are initialized in notebooks/move_wrong_predictions.ipynb
"""

import functools
import itertools
import shutil

import ipywidgets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from IPython.display import clear_output
from ipywidgets import Button
from PIL import Image

import cocpit.config as config
from cocpit.auto_str import auto_str
from cocpit.predictions import LoaderPredictions


@auto_str
class GUI:
    """
    creates buttons in notebooks/move_wrong_predictions.ipynb
    and notebooks/gui_check_dataset_one_class.ipynb
    to label predictions
    """

    def __init__(self, all_labels=None, all_paths=None, b=None, wrong=True):
        # b is an instance of TestBatchPredictions which inherits LoaderPredictions
        # passsing b in here so that LoaderPredictions __init__
        # is not called again/variables reinitialized.  Using in gui_move_wrong_predictions.ipynb
        self.wrong = wrong
        self.index = 0
        self.count = 0  # number of moved images
        self.center = None
        self.menu = None
        self.forward = None
        if self.wrong:
            # variables at wrong indices based on user input of specific categories to sift through
            # used in gui_move_wrong_prediction.ipynb
            self.all_labels = b.all_labels[b.wrong_trunc]
            self.all_paths = b.all_paths[b.wrong_trunc]
            self.all_topk_probs = b.all_topk_probs[b.wrong_trunc]
            self.all_topk_classes = b.all_topk_classes[b.wrong_trunc]

        else:
            # used in gui_check_dataset_one_class.ipynb
            # all labels and paths of training dataset
            self.all_labels = all_labels
            self.all_paths = all_paths

    def open_image(self):
        try:
            image = Image.open(self.all_paths[self.index])
        except FileNotFoundError:
            print("This file was already moved and cannot be found. Please hit Next.")
        return image

    def make_buttons(self):
        """
        use ipywidgets to create a box for the image, bar chart, dropdown,
        and next button
        """
        self.label = self.all_labels[self.index]
        self.center = ipywidgets.Output()  # center image with predictions
        self.menu = ipywidgets.Dropdown(
            options=config.CLASS_NAMES,
            description="Category:",
            value=config.CLASS_NAMES[self.label],
        )
        self.bar_chart()

        self.menu.observe(self.on_change, names="value")

        # create button that progresses through incorrect predictions
        self.forward = Button(description="Next")
        self.forward.on_click(self.on_button_next)

    def on_change(self, change):
        """
        when a class in the dropdown is selected, move the image and save
        it to the specified class
        """
        self.save_image(change)

    def on_button_next(self, b):
        """
        when the next button is clicked, make a new image and bar chart appear
        by updating the index within the wrong predictions by 1
        """
        self.index = self.index + 1
        self.bar_chart()

        # keep the default dropdown value to agg
        # don't want it to change based on previous selection
        self.menu.value = config.CLASS_NAMES[self.label]

    def bar_chart(self):
        """
        use the human and model labels and classes to
        create a bar chart with the top k predictions
        from the image at the current index
        """

        # add chart to ipywidgets.Output()
        with self.center:
            if self.wrong:
                self.topk_probs = self.all_topk_probs[self.index]
                self.topk_classes = self.all_topk_classes[self.index]

                # puts class names in order based on probabilty of prediction
                crystal_names = [config.CLASS_NAMES[e] for e in self.topk_classes]
                self.view_classifications_wrong(self.topk_probs, crystal_names)
            else:
                self.view_classifications()

    def view_classifications_wrong(self, probs, crystal_names):
        """
        create barchart that outputs top k predictions for a given image
        """
        clear_output()  # so that the next fig doesnt display below
        fig, (ax1, ax2) = plt.subplots(
            constrained_layout=True, figsize=(5, 7), ncols=1, nrows=2
        )
        image = self.open_image()
        ax1.imshow(image)
        ax1.set_title(
            f"Human Labeled as: {config.CLASS_NAMES[self.all_labels[self.index]]}\n"
            f"Model Labeled as: {crystal_names[0]}"
        )
        ax1.axis("off")

        y_pos = np.arange(len(self.topk_probs))
        ax2.barh(y_pos, probs, align="center")
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(crystal_names)
        ax2.tick_params(axis="y", rotation=45)
        ax2.invert_yaxis()  # labels read top-to-bottom
        ax2.set_title("Class Probability")
        plt.show()

    def view_classifications(self):
        """
        show image
        """
        clear_output()  # so that the next fig doesnt display below
        image = self.open_image()
        fig, ax1 = plt.subplots(
            constrained_layout=True, figsize=(5, 7), ncols=1, nrows=1
        )
        ax1.imshow(image)
        ax1.set_title(
            f"Human Labeled as: {config.CLASS_NAMES[self.all_labels[self.index]]}\n"
        )
        ax1.axis("off")
        plt.show()

    def save_image(self, change):
        """
        move the image based on dropdown selection
        """
        filename = self.all_paths[self.index].split("/")[-1]

        data_dir = f"/data/data/cpi_data/training_datasets/hand_labeled_resized_{config.TAG}_sideplanes_copy/"
        # print(f"{data_dir}{config.CLASS_NAMES[self.all_labels[self.index]]}/{filename}")
        # print(f"{data_dir}{change.new}/{filename}")
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
            pass
