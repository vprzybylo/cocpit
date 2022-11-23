"""Calculates geometric attributes of a particle in an image"""
import cv2
import numpy as np


class Geometry:
    """
    Args:
        phi (float): aspect ratio from rectangle
        extreme_points (float): computes how separated the outer most points are on the largest contour. Higher std. deviation = more spread out
        area (float): area of largest contour
        perim (float): perimeter of largest contour
        complexity (float): how intricate & complicated the particle is
        hull (np.ndarray): a convex hull around the largest contour
        convex_perim (float): perimeter of the convex hull of the largest contour
        hull_area (float): area of a convex hull surrounding the largest contour
        filled_circular_area_ratio (float):  area of the largest contour divided by the area of
                                     an encompassing circle - useful for spheres that have reflection
                                     spots that are not captured by the largest contour and leave a horseshoe pattern

        laplacian (float): Computes the variance (i.e. standard deviation squared)
                           of a convolved 3 x 3 kernel.  The Laplacian highlights regions
                           of an image containing rapid intensity changes. If the variance falls
                           below a pre-defined threshold, then the image is considered blurry;
                           otherwise, the image is not blurry.
        circularity (float): 4*pi*area/perimeter**2
        roundness (float): similar to circularity but divided by the perimeter
                   that surrounds the largest contour squared instead of the
                   actual convoluted perimeter
        perim_area_ratio (float): perimeter/area
        solidity (float): area/hull_area
        equiv_d (float): equivalent diameter of a circle with the same area as the largest contour
    """

    def __init__(self, gray, largest_contour):
        self.gray = gray
        self.largest_contour = largest_contour

    def calc_phi(self) -> None:
        """Calculate aspect ratio from rectangle"""
        # box ONLY around the largest contour
        rect = cv2.minAreaRect(self.largest_contour)
        # get length and width of contour
        x = rect[1][0]
        y = rect[1][1]
        rect_length = max(x, y)
        rect_width = min(x, y)
        self.phi = rect_width / rect_length

    def calc_extreme_points(self) -> None:
        """
        Computes how separated the outer most points are on the largest contour.
        Higher std. deviation = more spread out
        """
        left = tuple(self.largest_contour[self.largest_contour[:, :, 0].argmin()][0])
        right = tuple(self.largest_contour[self.largest_contour[:, :, 0].argmax()][0])
        top = tuple(self.largest_contour[self.largest_contour[:, :, 1].argmin()][0])
        bottom = tuple(self.largest_contour[self.largest_contour[:, :, 1].argmax()][0])
        self.extreme_points = np.std([left, right, top, bottom])

    def calc_filled_circular_area_ratio(self) -> None:
        """
        Returns the area of the largest contour divided by the area of
        an encompassing circle - useful for spheres that have reflection
        spots that are not captured by the largest contour and leave a horseshoe pattern
        """
        _, radius = cv2.minEnclosingCircle(self.largest_contour)
        self.filled_circular_area_ratio = self.area / (np.pi * radius**2)

    def calc_complexity(self) -> None:
        """
        How intricate & complicated the particle is

        see:
            Schmitt, C. G., and A. J. Heymsfield (2014),
            Observational quantification of the separation of
            simple and complex atmospheric ice particles,
            Geophys. Res. Lett., 41, 1301–1307, doi:10.1002/ 2013GL058781.
        """

        _, radius = cv2.minEnclosingCircle(self.largest_contour)
        Ac = np.pi * radius**2

        self.complexity = 10 * (
            0.1 - (self.area / (np.sqrt(self.area / Ac) * self.perim**2))
        )

    def runner(self):
        """perform all calculations"""
        self.calc_phi()
        self.calc_extreme_points()
        self.area = cv2.contourArea(self.largest_contour)
        self.perim = cv2.arcLength(self.largest_contour, False)
        self.calc_complexity()
        self.hull = cv2.convexHull(self.largest_contour)
        self.convex_perim = cv2.arcLength(self.hull, True)
        self.hull_area = cv2.contourArea(self.hull)
        self.calc_filled_circular_area_ratio()
        self.laplacian = cv2.Laplacian(self.gray, cv2.CV_64F).var()
        self.circularity = (4.0 * np.pi * self.area) / self.perim**2
        self.roundness = (4.0 * np.pi * self.area) / (self.convex_perim**2)
        self.perim_area_ratio = self.perim / self.area
        self.solidity = self.area / self.hull_area
        self.equiv_d = np.sqrt(4 * self.area / np.pi)
