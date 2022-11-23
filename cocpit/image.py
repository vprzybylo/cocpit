"""
Particle Image Classifier (PIC)

Opens image and calculates geormetric parameters using opencv
"""

import copy as cp
import os

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np


class Image:
    """
    Opens an image file (png) and processes the image

    Args:
        open_dir (str): directory to open image
        filename (str): file to be opened in the directory
    """

    def __init__(self, open_dir, filename, path=None):

        if path is not None:
            self.path = path
            self.im = cv2.cvtColor(
                cv2.imread(self.path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB
            )
        else:
            self.open_dir = open_dir
            self.filename = filename

            self.im = cv2.cvtColor(
                cv2.imread(self.open_dir + self.filename, cv2.IMREAD_UNCHANGED),
                cv2.COLOR_BGR2RGB,
            )
        self.image_og = cp.deepcopy(self.im)
        self.find_contours()
        # original image height and width before resizing
        self.height_og, self.width_og, _ = self.image_og.shape

    def resize_stretch(self, desired_size=1000) -> None:
        """
        Resized the image to the desired size while stretching the image instead of
        adding paddding or borders

        Args:
        desired_size (int): the images are resized as a square according to desired_size
        """
        self.im = cv2.resize(
            self.image_og,
            (desired_size, desired_size),
            interpolation=cv2.INTER_AREA,
        )

    def find_contours(self) -> None:
        """
        Finds contours in a binary image
        """
        self.gray = cv2.cvtColor(self.im, cv2.COLOR_BGR2GRAY)
        self.thresh = cv2.threshold(self.gray, 50, 255, cv2.THRESH_BINARY_INV)[1]

        self.contours, _ = cv2.findContours(
            self.thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        self.contours = sorted(self.contours, key=cv2.contourArea, reverse=True)

    #         plt.imshow(self.thresh)
    #         plt.show()

    def find_largest_contour(self) -> None:
        """define largest contour"""
        self.largest_contour = sorted(self.contours, key=cv2.contourArea, reverse=True)[
            0
        ]

    def find_largest_area(self) -> None:
        """define largest contour area"""
        self.area = cv2.contourArea(self.largest_contour)

    def morph_contours(self) -> None:
        """
        - Perform advanced morphological transformations using erosion and dilation.
        - Combines nearby gaps in contours
        """
        kernel = np.ones((5, 5), dtype="uint8")
        image_close = cv2.morphologyEx(self.thresh, cv2.MORPH_CLOSE, kernel)

        self.contours, _ = cv2.findContours(
            image_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        draw = cv2.drawContours(self.thresh, self.contours, -1, (0, 0, 255), 2)
        draw = cv2.fillPoly(self.thresh, self.contours, color=(255, 255, 255))
        # plt.imshow(draw)
        # plt.show()

        self.contours, _ = cv2.findContours(
            draw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # self.contours = sorted(contours, key=cv2.contourArea, reverse = True)

    def mask_background(self) -> None:
        """
        - Keeps the background all white surrounding the largest conotour.
        - Places the largest contour on an array of all the same color
        """
        mask = np.zeros(self.im.shape[:2], dtype="uint8")
        self.im = cv2.bitwise_and(self.im, self.im, mask=mask)

    #         plt.imshow(self.im)
    #         plt.show()

    def flip_imgs(self, save_dir) -> None:
        """Flip and save each flipped iteration"""
        plt.imsave(save_dir + self.filename, np.array(self.image_og))
        plt.imsave(save_dir + self.filename[:-4] + "_ud.png", np.flipud(self.image_og))
        plt.imsave(save_dir + self.filename[:-4] + "_lr.png", np.fliplr(self.image_og))
        plt.imsave(
            save_dir + self.filename[:-4] + "_ud_lr.png",
            np.flipud(np.fliplr(self.image_og)),
        )

    def save_image(self, save_dir: str, flip: bool = False) -> None:
        """
        Saves image to directory with option to flip image

        Args:
            save_dir (str): directory to be saved
            flip (boolean): whether or not to duplicate the image by
                            flipping it and saving it in multiple orientations
                            (up, down, left, right). Default False.
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        if flip:
            self.flip_imgs(save_dir)
        else:
            # save single image, no flipping:
            self.im = cv2.cvtColor(self.im, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(save_dir, str(self.filename)), np.array(self.im))

    def show_image(self) -> None:
        """Plot the image and the threshold"""
        plt.imshow(self.im)
        plt.show()
        plt.imshow(self.thresh)
        plt.show()
