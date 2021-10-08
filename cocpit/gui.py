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
from IPython.display import clear_output, display
from ipywidgets import AppLayout, Button
from PIL import Image

import cocpit.config as config


class GUI:
    def __init__(
        self, all_labels, all_paths, all_topk_probs, all_topk_classes, all_max_preds
    ):
        self.index = 0
        self.all_labels = all_labels
        self.all_paths = all_paths
        self.all_topk_probs = all_topk_probs
        self.all_topk_classes = all_topk_classes
        self.all_max_preds = all_max_preds

    def make_buttons(self):
        self.center = ipywidgets.Output()  # center image with predictions

        self.menu = ipywidgets.Dropdown(
            options=[
                "agg",
                "budding",
                "bullets",
                "columns",
                "compact_irregs",
                "fragments",
                "planar_polycrsytals",
                "rimed",
                "spheres",
            ],
            description="Category:",
            value="agg",
        )
        self.menu.observe(self.on_change)

        # create button that progresses through incorrect predictions
        self.forward = Button(description="Next")
        self.forward.on_click(self.on_button_next)

        if self.index == 0:
            self.bar_chart()

    def on_change(self, change):
        self.save_image()

    def on_button_next(self, b):
        self.bar_chart()
        self.index += 1

    def bar_chart(self):

        self.label = self.all_labels[self.index]
        self.path = self.all_paths[self.index]
        self.topk_probs = self.all_topk_probs[self.index]
        self.topk_classes = self.all_topk_classes[self.index]
        self.max_pred = self.all_max_preds[self.index]

        # puts class names in order based on probabilty of prediction
        crystal_names = [config.CLASS_NAMES[e] for e in self.topk_classes]

        # add chart to ipywidgets.Output()
        with self.center:
            self.view_classifications(crystal_names)

    def view_classifications(self, crystal_names):

        clear_output()  # so that the next fig doesnt display below
        fig, (ax1, ax2) = plt.subplots(
            constrained_layout=True, figsize=(5, 7), ncols=1, nrows=2
        )

        try:
            image = Image.open(self.path)
            ax1.imshow(image)
            ax1.set_title(
                f"Human Labeled as: {config.CLASS_NAMES[self.label]}\n"
                f"Model Labeled as: {crystal_names[0]}"
            )
            ax1.axis("off")

            y_pos = np.arange(len(self.topk_probs))
            ax2.barh(y_pos, self.topk_probs, align="center")
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(crystal_names)
            ax2.tick_params(axis="y", rotation=45)
            ax2.invert_yaxis()  # labels read top-to-bottom
            ax2.set_title("Class Probability")
            plt.show()

        except FileNotFoundError:
            print("This file was already moved and cannot be found. Please hit Next.")
            pass

    def save_image(self):
        filename = self.path.split("/")[-1]
        # print(f'move {path} to {config.DATA_DIR}{crystal_names[0]}/{filename}')

        try:
            # print('path', self.path)
            # print(f"dropdown: {config.DATA_DIR}{self.menu.value}/{filename}")

            shutil.move(self.path, f"{config.DATA_DIR}{self.menu.value}/{filename}")
        except FileNotFoundError:
            print("file not found")
            pass
