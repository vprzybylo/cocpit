# -*- coding: utf-8 -*-
"""
cite: https://www.pyimagesearch.com/2020/04/20/detect-and-remove-duplicate-images-from-a-dataset-for-deep-learning/

The function accepts an image and hashSize and proceeds to:
    Detect and remove duplicate images from a dataset for deep learning.
    Convert the image to a single-channel grayscale image.
    Resize the image according to the hashSize. 
The algorithm requires that the width of the image have exactly 1 more column than the height as is evident by the dimension tuple.
Compute the relative horizontal gradient between adjacent column pixels. This is now known as the 'difference image.'
Apply our hashing calculation and return the result.
dataset: The path to your input dataset, which contains duplicates that you’d like to prune out of the dataset
remove: Indicates whether duplicates should be removed (deleted permanently) or whether you want to conduct a “dry run” so you can visualize the duplicates on your screen and see the hashes in your terminal
"""

from imutils import paths
import numpy as np
import cv2
import os

#os.environ['DISPLAY'] = ':0'

def dhash(image, hashSize=8):
    # convert the image to grayscale and resize the grayscale image,
    # adding a single column (width) so we can compute the horizontal
    # gradient
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (hashSize + 1, hashSize))
    # compute the (relative) horizontal gradient between adjacent
    # column pixels
    diff = resized[:, 1:] > resized[:, :-1]
    # convert the difference image to a hash and return it
    return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])
    
    
def remove_duplicates(open_dir, remove, df):
    
    # grab the paths to all images in our input dataset directory and
    # then initialize our hashes dictionary
    imagePaths = list(paths.list_images(open_dir))
    hashes = {}
    # loop over our image paths
    for imagePath in imagePaths:
        # load the input image and compute the hash
        image = cv2.imread(imagePath)
        h = dhash(image)
        # grab all image paths with that hash, add the current image
        # path to it, and store the list back in the hashes dictionary
        p = hashes.get(h, [])
        p.append(imagePath)
        hashes[h] = p

        # loop over the image hashes
        for (h, hashedPaths) in hashes.items():
            # check to see if there is more than one image with the same hash
            if len(hashedPaths) > 1:
                # check to see if this is a dry run
                if remove is False:
                    # initialize a montage to store all images with the same
                    # hash
                    montage = None
                    # loop over all image paths with the same hash
                    for p in hashedPaths:
                        # load the input image and resize it to a fixed width
                        # and heightG
                        image = cv2.imread(p)
                        image = cv2.resize(image, (150, 150))
                        # if our montage is None, initialize it
                        if montage is None:
                            montage = image
                        # otherwise, horizontally stack the images
                        else:
                            montage = np.hstack([montage, image])
                    # show the montage for the hash
                    #removed for execution

                # otherwise, we'll be removing the duplicate images
                else:
                    # loop over all image paths with the same hash *except*
                    # for the first image in the list (since we want to keep
                    # one, and only one, of the duplicate images)
                    for p in hashedPaths[1:]:
                        os.remove(p)
