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

check_classifications:
    - check predictions from a saved CNN
    - called in check_classifications.ipynb for bar chart plot

config-sample:
    - holds all user-defined variables
    - treated as global and used across modules
    - paths are to be modified for each user
    - rename config-sample to config.py when running code

data_loaders:
    - retrives data loaders from Pytorch for training and validation data
    - includes weighted and unweighted sampling
    - modified to append path to image (including class folder)

geometric_attributes:
    - calculates particle geometric properties from pic.py
    - e.g., area ratio, roundness, aspect ratio
    - length and width of particle calculated in process_sheets.py before resizing

gui.py
    - An ipywidget interface for ensuring training dataset labels are correct
    - incorrect predictions on a validation dataloader is iterated over
    - images are displayed with model predictions and human label
    - if a label is incorrect a user can move the image within the labeled dataset using a dropdown menu
    - called in notebooks/gui_move_wrong_predictions.ipynb

image_stats:
    - find the #/% of cutoff particles after removing blurry, fragmented, and spherical drops
    - used as a separate script (external - not being called in __main__.py)

metrics:
    - holds epoch and batch metrics for both the training and validation datasets
    - called in train_model.py
    - updates and resets loss and acc totals within training loop
    - logs metrics to console and/or comet-ml interface (see config.py to turn on)
    - writes metrics to csv's defined in config.py
    - creates a sklearn classification report using the metrics

model_config:
    - model configurations for:
        - dropout
        - device settings
        - parameters to update
        - checking label counts within a batch
        - normalization values for transformations

models:
    - defines torchvision models

no_fold_training:
    - train model without folds for cross validation

pic: 'particle image classification'
    - holds the main Image class for image manipulation using opencv
    - calculates particle geometrical attributes

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

setup_training:
    - train model with k folds for cross validation across samples called in __main__.py
    - runner class to get dataloaders and set up cross validation

train_model:
    - houses the execution of training the model for all epochs and batches
    - iterates through training and validation phases for specified CNN
    - called in run_ML model
    - writes accuracy and loss logs for each dataset (training and validation)
    - returns classification report

"""
import glob
from os.path import basename, dirname, isfile, join

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')
]
from . import *  # noqa: F403 E402
