import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as pil_img


class Image:
    def __init__(self, filename):
        self.filename = filename

    def open_image(self, open_dir):
        self.image_og = cv2.imread(open_dir + self.filename, cv2.IMREAD_UNCHANGED)
        # if self.image_og.shape[2] == 4 or  self.image_og.shape[2] == 3:
        self.image_og = cv2.cvtColor(self.image_og, cv2.COLOR_BGR2RGB)
        self.height, self.width, channel = self.image_og.shape

    def resize_stretch(self, desired_size):
        self.saveimg = cv2.resize(
            self.image_og, (desired_size, desired_size), interpolation=cv2.INTER_AREA
        )
        self.height, self.width, channel = self.saveimg.shape

    def find_contours(self):
        self.gray = cv2.cvtColor(self.image_og, cv2.COLOR_BGR2GRAY)
        self.thresh = cv2.threshold(self.gray, 50, 255, cv2.THRESH_BINARY_INV)[1]

        self.contours, hierarchy = cv2.findContours(
            self.thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        self.contours = sorted(self.contours, key=cv2.contourArea, reverse=True)
        # plt.imshow(self.thresh)
        # plt.show()

    def morph_contours(self):
        kernel = np.ones((5, 5), dtype="uint8")
        image_close = cv2.morphologyEx(self.thresh, cv2.MORPH_CLOSE, kernel)

        self.contours, hierarchy = cv2.findContours(
            image_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        draw = cv2.drawContours(self.thresh, self.contours, -1, (0, 0, 255), 2)
        draw = cv2.fillPoly(self.thresh, self.contours, color=(255, 255, 255))
        # plt.imshow(draw)
        # plt.show()

        self.contours, hierarchy = cv2.findContours(
            draw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        # self.contours = sorted(contours, key=cv2.contourArea, reverse = True)

    def largest_contour(self):
        return sorted(self.contours, key=cv2.contourArea, reverse=True)[0]

    def largest_contour_area(self):
        return cv2.contourArea(self.largest_contour())

    def convex_perim(self, closed_cnt):
        hull = cv2.convexHull(self.largest_contour())
        return cv2.arcLength(hull, closed_cnt)

    def perim_ratio(self):
        return self.perim() / self.convex_perim(self)

    def area(self):
        area = 0
        for c in self.contours:
            area += cv2.contourArea(c)
            return area

    def perim(self):
        if len(self.contours) == 1:
            return cv2.arcLength(self.largest_contour(), False)
        else:
            return 0

    def circularity(self):
        perim = self.perim()
        hull_area = self.hull_area()
        if (
            perim != 0
            and hull_area != 0
            and self.area() != 0
            and len(self.contours) == 1
        ):
            return (4.0 * np.pi * self.area()) / (self.convex_perim(True) ** 2)
            # return (4*np.pi*self.area)/(perim**2)
        else:
            return 0

    def roundness(self):
        return (4.0 * np.pi * self.area()) / (self.hull_area())

    def perim_area(self):
        return self.perim() / self.area()

    def solidity(self):
        if self.hull_area() != 0.0 and len(self.contours) == 1:
            return float(self.area()) / self.hull_area()
        else:
            return 0

    def equiv_d(self):
        return np.sqrt(4 * self.area() / np.pi)

    def hull_area(self):
        hull = cv2.convexHull(self.largest_contour())
        return cv2.contourArea(hull)

    def save_image(self, save_dir, flip=False):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        if flip:
            self.flip_imgs(save_dir)
        else:
            # save single image, no flipping:
            # if self.saveimg.shape[2] == 4 or  self.saveimg.shape[2] == 3:
            self.saveimg = cv2.cvtColor(self.saveimg, cv2.COLOR_BGR2RGB)
            cv2.imwrite(
                os.path.join(save_dir, str(self.filename)), np.array(self.saveimg)
            )


def main():
    open_dir = "../cpi_data/training_datasets/SIFT/bad_train_masked/"
    save_dir_nonspheres = "../cpi_data/training_datasets/SIFT/nonspheres/"
    desired_size = 1000
    count = 0
    for filename in os.listdir(open_dir):
        image = Image(filename)
        image.open_image(open_dir)
        image.find_contours()
        image.resize_stretch(desired_size)
        # image.save_image(save_dir, flip=True)
        if len(image.contours) > 1:
            image.morph_contours()
        if image.circularity() > 0.8 and image.solidity() > 0.8:
            count += 1
            # image.save_image(save_dir_spheres)
        else:
            image.save_image(save_dir_nonspheres)
    print("removed %d spheres" % count)


if __name__ == "__main__":
    main()
# 1. mask all backgrounds
# 2. remove all spheres and blank and resize
# 3. sift model: separate ice for training (good vs bad)
# 4. remove duplicates
# 5. ML model


# More detailed:
# 1. mask all backgrounds
# 2. remove all spheres and blank and resize
# 3. create dataset of good and bad images and train regression model
# 4. run all masked campaign images through model to output good ice
# 5. remove duplicates from good
# 6. create ML dataset from good ice for categories
# 7. run good campaigns through ML model
