"""
COCPIT package:

Classification of Cloud Particle Imagery and Thermodynamics

modules:
-------

add_date:
    - add a column to the dataframes for the date from the filename

auto_str:
    - automatically implements a string representation for classes
    instead of memory id
    - finds all attributes of the class
    - called with str(instance)

config-sample:
    - holds all user-defined variables
    - treated as global and used across modules
    - paths are to be modified for each user
    - rename config-sample to config.py when running code

data_loaders:
    - retrives data loaders from Pytorch for training and validation data
    - includes weighted and unweighted sampling
    - modified to append path to image (including class folder)

fold_setup:
    - setup training and validation indices for labels and data based on k-fold cross validation

geometry:
    - calculates particle geometrical attributes

geometry_runner:
    - holds the main Image class for image manipulation using opencv

gui_label:
    - an ipywidget interface for creating a training dataset
    - buttons to choose a label and move an image into that dir
    - called in notebooks/label.ipynb

gui_wrong:
    - an ipywidget interface for labeling incorrect predictions from a validation dataloader
    - should the model be right, images are moved within the training dataset
    - called in notebooks/move_wrong_predictions.ipynb

gui:
    - an ipywidget interface for nested classification that separates images within a class
    - used on the entire training dataset
    - called in notebooks/gui_check_dataset_one_class.ipynb

image_stats:
    - find the #/% of cutoff particles after removing blurry, fragmented, and spherical drops
    - used as a separate script (external - not being called in __main__.py)

interpretability:
    - gradcam, guided backprop, vanilla backprop

model_config:
    - model configurations for:
        - dropout
        - device settings
        - parameters to update
        - checking label counts within a batch
        - normalization values for transformations

models:
    - defines torchvision models

performance_metrics:
    - holds epoch and batch metrics for both the training and validation datasets
    - inherited by train and validation classes
    - logs metrics to console and/or comet-ml interface (see config.py to turn on)
    - writes metrics to csv's defined in config.py

predictions:
    - holds methods regarding making model predictions
    - for confusion matrices and running the model on new data

process_sheets:
    - extracts single images from the muliple images per frame or sheet
    - saves single images to a directory for later creation of the database

run_model:
    -classifies good ice images through a convolutional neural network that
    was presaved or built in build_ML_model
    -transforms, makes predictions, and appends classification to dataframe

runner:
    - iterates over epochs, batch sizes, and phases and calls training methods
    and validation methods

timing:
    - time one epoch, all epochs, and write to csv

setup_training:
    - train model with k folds for cross validation across samples called in __main__.py
    - runner class to get dataloaders and set up cross validation

train:
    - execution of running a model across batches for training set only
    - outputs performance metrics during training

validate:
    - execution of running a model across batches for validation set only
    - outputs performance metrics and saves confusion matrix and classification report

"""
import glob
from os.path import basename, dirname, isfile, join

from comet_ml import Experiment  # isort:split

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules if isfile(f) and not f.endswith("__init__.py")
]
from . import *  # noqa: F403 E402
