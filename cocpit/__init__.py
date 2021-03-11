"""
COCPIT package:

Classification of Cloud Particle Imagery and Thermodynamics

modules:
-------

preprocess_sheets:
    extracts single images from the muliple images per frame or sheet
    saves single images to a directory for later creation of the database

pic: 'particle image classification'
    holds the main Image class for image manipulation using opencv 
    and calculates particle geometrical attributes 

build_spheres_sift:
    Trains and saves a logistic regression model on prelabeled data
    to predict if an image is a sphere 

    If the image is not a sphere, another logistic regression model
    is used to predict if an image represents quality ice or a 
    blurry/broken/blank/fragmented image. This model is called 
    SIFT or separate ice for training.

    In the case of quality ice, the amount of pixels touching the 
    image border are taken into account (i.e., an alterable cutoff
    measurement)

spheres_sift_prediction: 
    using the prebuilt logistic regression models, new predictions are 
    made to the single image directiory from preprocess_sheets and 
    a dataframe is created for quality ice images holding image/particle
    attributes determined in pic.py

build_ML_model:
    -loads the prebuilt pytorch models
    see: https://pytorch.org/docs/stable/torchvision/models.html
    -sets up the pytorch dataloaders, batch size, image directories, and includes gpu support
    -returns training, validation, and testing dataloaders
    -houses the execution of training the model for all epochs and batches 
    -returns accuracy and loss logs for each dataset (training and validation)

run_ML_model:
    -classifies good ice images through a convolutional neural network that
    was presaved or built in build_ML_model
    -transforms, makes predictions, and appends classification to dataframe 
"""
import cocpit.build_ML_model
import cocpit.build_spheres_sift
import cocpit.pic
import cocpit.process_png_sheets_with_text
import cocpit.remove_duplicates
import cocpit.run_ML_model
import cocpit.spheres_sift_prediction


# from os.path import dirname, basename, isfile, join
# import glob
# modules = glob.glob(join(dirname(__file__), "*.py"))
# __all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]
# from cocpit import *