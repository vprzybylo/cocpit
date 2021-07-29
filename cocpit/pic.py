"""
Particle Image Classifier (PIC)

Opens image and calculates geormetric parameters using opencv
"""

import os

import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np


class Image:

    """Opens an image file (png) and processes the image

    attributes:
    ----------
        open_dir (str):
            directory to be opened
        filename (str):
            file to be opened in the directory
    """

    def __init__(self, open_dir=None, filename=None, path=None):

        if path is not None:
            self.path = path
            self.image_og = cv2.cvtColor(
                cv2.imread(self.path, cv2.IMREAD_UNCHANGED), cv2.COLOR_BGR2RGB
            )
        else:
            self.open_dir = open_dir
            self.filename = filename
            self.image_og = cv2.cvtColor(
                cv2.imread(self.open_dir + self.filename, cv2.IMREAD_UNCHANGED),
                cv2.COLOR_BGR2RGB,
            )
        # original image height and width before resizing
        self.height_og, self.width_og, _ = self.image_og.shape

    def resize_stretch(self, desired_size=1000):
        """
        resized the image to the desired size while stretching the image instead of
        adding paddding or borders

        attributes:
        ----------
        desired_size (int):
            the images are resized as a square according to desired_size
        """

        self.im = cv2.resize(
            self.image_og, (desired_size, desired_size), interpolation=cv2.INTER_AREA
        )
        self.height, self.width, _ = self.im.shape

    def find_contours(self):
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

    def morph_contours(self):
        """
        Perform advanced morphological transformations using erosion and dilation.
        Combines nearby gaps in contours
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

    def mask_background(self):
        """
        Keeps the background all white surrounding the largest conotour.
        Places the largest contour on an array of all the same color
        """
        mask = np.zeros(self.im.shape[:2], dtype="uint8")
        # draw = cv2.drawContours(mask, [self.largest_contour], 0, (255,255,255), -1)
        self.im = cv2.bitwise_and(self.im, self.im, mask=mask)

    #         plt.imshow(self.im)
    #         plt.show()

    def contrast(self):
        """
        Returns the standard deviation of the image pixel values -
        higher std has more contrast
        """
        return self.im.std()

    def laplacian(self):
        """
        computes the variance (i.e. standard deviation squared)
        of a convolved 3 x 3 kernel.  The Laplacian highlights regions
        of an image containing rapid intensity changes. If the variance falls
        below a pre-defined threshold, then the image is considered blurry;
        otherwise, the image is not blurry.
        """
        return cv2.Laplacian(self.gray, cv2.CV_64F).var()

    def edges(self):
        """
        finds edges in the input image image and marks
        them in the output map edges using the Canny algorithm.
        The smallest value between threshold1 and threshold2 is used
        for edge linking. The largest value is used to find initial
        segments of strong edges.
        """
        min_threshold = 0.66 * np.mean(self.im)
        max_threshold = 1.33 * np.mean(self.im)
        edges = cv2.Canny(self.im, min_threshold, max_threshold)
        return edges

    def calculate_largest_contour(self):
        self.largest_contour = sorted(self.contours, key=cv2.contourArea, reverse=True)[
            0
        ]

    def calculate_area(self):
        self.area = cv2.contourArea(self.largest_contour)

    def calculate_perim(self):
        self.perim = cv2.arcLength(self.largest_contour, False)

    def phi(self):
        """
        calculate aspect ratio from rectangle
        """
        # box ONLY around the largest contour
        rect = cv2.minAreaRect(self.largest_contour)
        # get length and width of contour
        x = rect[1][0]
        y = rect[1][1]
        self.rect_length = max(x, y)
        self.rect_width = min(x, y)
        return self.rect_width / self.rect_length

    def create_ellipse(self):
        self.calculate_largest_contour()

        if len(self.largest_contour) >= 5:
            ellipse = cv2.fitEllipse(self.largest_contour)
            self.ellipse = cv2.ellipse(self.im, ellipse, (255, 255, 255), 5)

            # Gets width and height of rotated ellipse
            widthE = ellipse[1][0]
            heightE = ellipse[1][1]
            if widthE > heightE:
                return heightE / widthE
            else:
                return widthE / heightE

        return np.nan

    def extreme_points(self):
        """
        computes how separated the outer most points are on the largest contour
        higher std. deviation = more spread out
        """
        leftmost = tuple(
            self.largest_contour[self.largest_contour[:, :, 0].argmin()][0]
        )
        rightmost = tuple(
            self.largest_contour[self.largest_contour[:, :, 0].argmax()][0]
        )
        topmost = tuple(self.largest_contour[self.largest_contour[:, :, 1].argmin()][0])
        bottommost = tuple(
            self.largest_contour[self.largest_contour[:, :, 1].argmax()][0]
        )
        return np.std([leftmost, rightmost, topmost, bottommost])

    def filled_circular_area_ratio(self):
        """
        returns the area of the largest contour divided by the area of
        an encompassing circle - useful for spheres that have reflection
        spots that are not captured by the largest contour and leave a horseshoe pattern
        """
        _, radius = cv2.minEnclosingCircle(self.largest_contour)

        return self.area / (np.pi * radius ** 2)

    def circularity(self):
        return (4.0 * np.pi * self.area) / self.perim ** 2

    def roundness(self):
        """
        similar to circularity but divided by the perimeter
        that surrounds the largest contour squared instead of the
        actual convoluted perimeter
        """
        return (4.0 * np.pi * self.area) / (self.convex_perim(True) ** 2)

    def perim_area_ratio(self):
        return self.perim / self.area

    def convex_perim(self, closed_cnt=True):
        """
        returns the perimeter of the convex hull of the
        largest contour

        closed_cnt (boolean):
            perimeter must be of a closed contour
        """
        hull = cv2.convexHull(self.largest_contour)
        return cv2.arcLength(hull, closed_cnt)

    def convexity(self):
        return self.convex_perim() / self.perim()

    def complexity(self):
        """
        how intricate & complicated the particle is

        see:
            Schmitt, C. G., and A. J. Heymsfield (2014),
            Observational quantification of the separation of
            simple and complex atmospheric ice particles,
            Geophys. Res. Lett., 41, 1301–1307, doi:10.1002/ 2013GL058781.
        """

        _, radius = cv2.minEnclosingCircle(self.largest_contour)
        Ac = np.pi * radius ** 2

        return 10 * (0.1 - (self.area / (np.sqrt(self.area / Ac) * self.perim ** 2)))

    def solidity(self):
        return self.area / self.hull_area

    def equiv_d(self):
        """
        equivalent diameter of a circle with the
        same area as the largest contour
        """
        return np.sqrt(4 * self.area / np.pi)

    def calculate_hull_area(self):
        """
        area of a convex hull surrounding the largest contour
        """
        hull = cv2.convexHull(self.largest_contour)
        self.hull_area = cv2.contourArea(hull)

    def flip_imgs(self, save_dir):
        plt.imsave(save_dir + self.filename, np.array(self.image_og))
        plt.imsave(save_dir + self.filename[:-4] + "_ud.png", np.flipud(self.image_og))
        plt.imsave(save_dir + self.filename[:-4] + "_lr.png", np.fliplr(self.image_og))
        plt.imsave(
            save_dir + self.filename[:-4] + "_ud_lr.png",
            np.flipud(np.fliplr(self.image_og)),
        )

    def rotate(self):
        # loop over the rotation angles
        for angle in np.arange(0, 360, 100):
            self.saveimg = imutils.rotate_bound(self.saveimg, angle)

    #             plt.imshow(self.saveimg)
    #             plt.show()

    def save_image(self, save_dir, flip=False):
        """
        saves image to directory

        attributes:
        ----------
            save_dir (str):
                directory to be saved
            flip (boolean): False
                whether or not to duplicate the image by
                flipping it and saving it in multiple orientations
                (up, down, left, right)
        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        if flip:
            self.flip_imgs(save_dir)
        else:
            # save single image, no flipping:
            self.im = cv2.cvtColor(self.im, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join(save_dir, str(self.filename)), np.array(self.im))

    def show_image(self):
        plt.imshow(self.im)
        plt.show()
        plt.imshow(self.thresh)
        plt.show()
