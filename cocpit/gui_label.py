import ipywidgets
import matplotlib.pyplot as plt
from IPython.display import clear_output
from ipywidgets import Button
from PIL import Image
from shutil import copyfile
import os
from typing import Union, List
import cocpit.config as config
import pandas as pd


class GUI:
    """
    View and label images in notebooks/label.ipynb

    Args:
        all_paths (List[str]): paths to images
        folder_dest (str): where to save images to
        precip (pd.core.series.Series): corresponding precipitation measurements to images
    """

    def __init__(
        self,
        all_paths: List[str],
        folder_dest: str,
        precip: pd.core.series.Series = None,
    ):
        self.all_paths = all_paths
        self.n_paths: int = len(self.all_paths)
        self.folder_dest: str = folder_dest
        self.precip: pd.core.series.Series = precip
        self.index: int = 0
        self.center = ipywidgets.Output()
        self.undo_btn = Button(description="Undo")
        self.buttons = []

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
        """Make buttons for each class and an undo button"""
        self.undo_btn.on_click(self.undo)

        for idx, label in enumerate(config.CLASS_NAMES):
            self.buttons.append(Button(description=label))
            self.buttons[idx].on_click(self.cp_to_dir)

    def align_buttons(self) -> None:
        """Alter layout based on number of classes"""
        if len(config.CLASS_NAMES) > 5:
            # align buttons vertically
            self.label_btns = ipywidgets.VBox(
                [self.buttons[i] for i in range(len(config.CLASS_NAMES))]
            )
        else:
            # align buttons horizontally
            self.label_btns = ipywidgets.HBox(
                [self.buttons[i] for i in range(len(config.CLASS_NAMES))]
            )
        self.undo_btn = ipywidgets.HBox([self.undo_btn])

    def cp_to_dir(self, b) -> None:
        """
        Copy from original dir to new directory based on selected class label
        """
        # split path from filename if paths coming from ai2es project
        self.filename = self.all_paths[self.index].split("/")[-1]
        output_path = os.path.join(
            self.folder_dest,
            config.CLASS_NAME_MAP[b.description],
            self.filename,
        )

        copyfile(self.all_paths[self.index], output_path)
        # print(f"copying to {output_path}")
        self.index = self.index + 1
        self.display_image()

    def undo(self, b) -> None:
        """
        Undo moving image into folder

        Raises:
            FileNotFoundError: file already moved or not found
        """

        self.index = self.index - 1
        self.display_image()

        # split path from filename if paths coming from ai2es project
        self.filename = self.all_paths[self.index].split("/")[-1]

        # undo the move and remove file
        for label in config.CLASS_NAMES:
            if self.filename in os.listdir(
                os.path.join(self.folder_dest, config.CLASS_NAME_MAP[label])
            ):
                try:
                    os.remove(
                        os.path.join(
                            self.folder_dest,
                            config.CLASS_NAME_MAP[label],
                            self.filename,
                        ),
                    )
                except FileNotFoundError:
                    print(
                        f"{os.path.join(self.folder_dest, config.CLASS_NAME_MAP[label], self.filename)} not"
                        " found."
                    )

    def display_image(self) -> None:
        """
        Show image
        """
        with self.center:
            clear_output()  # so that the next fig doesnt display below
            image = self.open_image()
            fig, ax = plt.subplots(
                constrained_layout=True, figsize=(6, 6), ncols=1, nrows=1
            )
            if self.precip is not None:
                ax.set_title(
                    f"{self.index}/{self.n_paths} \n"
                    f" {self.all_paths[self.index].split('/')[-1]} \n 1 minute"
                    f" accumulated precip [mm]: {self.precip[self.index]}"
                )
            else:
                ax.set_title(
                    f"{self.index}/{self.n_paths} \n"
                    f" {self.all_paths[self.index].split('/')[-1]}"
                )
            ax.imshow(image)
            ax.axis("off")
            plt.show()
