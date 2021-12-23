"""
Holds the class for ipywidget buttons to
label images from a dataloader.
"""

import shutil

import ipywidgets
import matplotlib.pyplot as plt
from IPython.display import clear_output
from ipywidgets import Button
from PIL import Image

import cocpit.config as config
from cocpit.auto_str import auto_str


@auto_str
class GUI:
    '''
    creates buttons in notebooks/move_wrong_predictions.ipynb
    to label wrong predictions from validation dataloader
    '''

    def __init__(self, all_labels, all_paths):
        '''
        requires human labels, image absolute paths, the top k probabilities output
        by the model for a bar chart of predictions, the top k class names that
        correspond to the top k probabilities, and the class index with
        the highest probability
        '''
        self.index = 0
        self.count = 0  # number of moved images
        self.all_labels = all_labels
        self.all_paths = all_paths

    def make_buttons(self):
        '''
        use ipywidgets to create a box for the image, bar chart, dropdown,
        and next button
        '''
        self.center = ipywidgets.Output()  # center image with predictions

        self.label = self.all_labels[self.index]
        self.menu = ipywidgets.Dropdown(
            options=config.CLASS_NAMES,
            description="Category:",
            value=config.CLASS_NAMES[self.label],
        )
        self.bar_chart()

        self.menu.observe(self.on_change, names='value')

        # create button that progresses through incorrect predictions
        self.forward = Button(description="Next")
        self.forward.on_click(self.on_button_next)

    def on_change(self, change):
        '''
        when a class in the dropdown is selected, move the image and save
        it to the specified class
        '''
        self.save_image(change)

    def on_button_next(self, b):
        '''
        when the next button is clicked, make a new image and bar chart appear
        by updating the index within the wrong predictions by 1
        '''
        self.index = self.index + 1
        self.bar_chart()

        # keep the default dropdown value to agg
        # don't want it to change based on previous selection
        self.menu.value = config.CLASS_NAMES[self.label]

    def bar_chart(self):
        '''
        use the human and model labels and classes to
        create a bar chart with the top k predictions
        from the image at the current index
        '''
        self.label = self.all_labels[self.index]
        self.path = self.all_paths[self.index]

        # add chart to ipywidgets.Output()
        with self.center:
            self.view_classifications()

    def view_classifications(self):
        '''
        show image
        '''
        clear_output()  # so that the next fig doesnt display below
        try:
            fig, ax1 = plt.subplots(
                constrained_layout=True, figsize=(5, 7), ncols=1, nrows=1
            )
            image = Image.open(self.path)
            ax1.imshow(image)
            ax1.set_title(f"Human Labeled as: {config.CLASS_NAMES[self.label]}\n")
            ax1.axis("off")
            plt.show()

        except FileNotFoundError:
            print("This file was already moved and cannot be found. Please hit Next.")
            pass

    def save_image(self, change):
        '''
        move the image based on dropdown selection
        '''
        filename = self.path.split("/")[-1]

        data_dir = config.DATA_DIR[:-1] + '_copy/'

        try:
            print(f"{data_dir}{config.CLASS_NAMES[self.label]}/{filename}")
            print(f"{data_dir}{change.new}/{filename}")

            shutil.move(
                f"{data_dir}{config.CLASS_NAMES[self.label]}/{filename}",
                f"{data_dir}{change.new}/{filename}",
            )
            self.count += 1
            print(f'moved {self.count} images')

        except FileNotFoundError:
            print(self.path)
            print("File Not Found. Not moving.")
            pass
