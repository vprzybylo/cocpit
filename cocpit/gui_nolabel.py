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
    views predictions on test data in notebooks/check_classifications.ipynb
    """

    def __init__(self, b=None):
        self.b = b  # instance from cocpit.predictions.BatchPredictions
        self.index = 0
        self.count = 0  # number of moved images
        self.center = None
        self.menu = None
        self.forward = None

    def open_image(self):
        try:
            image = Image.open(self.b.all_paths[self.index])
        except FileNotFoundError:
            print("This file was already moved and cannot be found. Please hit Next.")
        return image

    def make_buttons(self):
        """
        use ipywidgets to create a box for the image, bar chart, dropdown,
        and next button
        """
        self.center = ipywidgets.Output()  # center image with predictions
        self.bar_chart()

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

    def bar_chart(self):
        """
        use the human and model labels and classes to
        create a bar chart with the top k predictions
        from the image at the current index
        """
        # add chart to ipywidgets.Output()
        with self.center:
            self.view_classifications()

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
            f"Model Labeled as: {config.CLASS_NAMES[self.b.all_max_preds[self.index]]}\n"
        )
        ax1.axis("off")
        plt.show()
