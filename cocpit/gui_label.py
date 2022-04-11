"""
Holds the class for ipywidget buttons to
label from a folder of images.
Run in notebooks/label.ipynb
"""
import ipywidgets
import matplotlib.pyplot as plt
from IPython.display import clear_output
from ipywidgets import Button
import PIL
from shutil import copyfile
import os
from typing import Union, List
import cocpit.config as config


class GUI:
    """
    view and label images in notebooks/label.ipynb
    """

    def __init__(self, all_paths: List[str], folder_dest: str):
        self.all_paths = all_paths
        self.n_paths: int = len(self.all_paths)
        self.folder_dest = folder_dest
        self.index: int = 0
        self.center = ipywidgets.Output()
        self.undo_btn = Button(description="Undo")
        # split path from filename if paths coming from ai2es project
        self.filename = self.all_paths[self.index].split("/")[-1]
        self.buttons = []

    def open_image(self) -> Union[PIL.Image.Image, None]:
        try:
            return PIL.Image.open(self.all_paths[self.index])

        except FileNotFoundError:
            print("This file was already moved and cannot be found. Please hit Next.")

    def make_buttons(self) -> None:
        """buttons for each category and undo button"""
        self.undo_btn.on_click(self.undo)

        for idx, label in enumerate(config.CLASS_NAMES):
            self.buttons.append(Button(description=label))
            self.buttons[idx].on_click(self.cp_to_dir)

    def cp_to_dir(self, b) -> None:
        """
        copy from original dir to new directory with class label
        Args:
            b: button instance
        """

        output_path = os.path.join(
            self.folder_dest, config.CLASS_NAME_MAP[b.description], self.filename
        )
        copyfile(self.all_paths[self.index], output_path)
        self.index = self.index + 1
        self.display_image()

    def undo(self, b) -> None:
        """
        undo moving image into folder
        """
        self.index = self.index - 1
        self.display_image()

        # undo the move and remove file
        for label in config.CLASS_NAMES:
            if self.filename in os.listdir(os.path.join(self.folder_dest, label)):
                os.remove(
                    os.path.join(self.folder_dest, label, self.filename),
                )

    def display_image(self) -> None:
        """
        show image
        """
        with self.center:
            clear_output()  # so that the next fig doesnt display below
            image = self.open_image()
            fig, ax = plt.subplots(
                constrained_layout=True, figsize=(6, 6), ncols=1, nrows=1
            )
            ax.set_title(f"{self.index}/{self.n_paths}")
            ax.imshow(image)
            ax.axis("off")
            plt.show()
