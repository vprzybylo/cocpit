"""
- THIS FILE SHOUlD BE ALTERED AND RENAMED config.py FOR EACH USER
- config.py in .gitignore to avoid version changes upon specifications
- holds all user-defined variables
- treated as global variables that do not change in any module
- used in each module through 'import cocpit.config as config'
- call using config.VARIABLE_NAME
- flags for what module of cocpit to run is found in the main directory in __main__.py (e.g., preprocess_sheets, build_model, ice_classification, geometric_attributes, add_date..)

isort:skip_file
"""

from comet_ml import Experiment  # isort:split

import os
from dotenv import load_dotenv
import torch
import sys

# cocpit version used in docker and git
TAG = "v1.4.0"

# extract each image from sheet of images
PREPROCESS_SHEETS = True

# create and save CNN
BUILD_MODEL = False

# run the category classification on quality images of ice particles
ICE_CLASSIFICATION = True

# calculates geometric particle properties and appends to databases
GEOMETRIC_ATTRIBUTES = True

# adds a column for the date from the filename
ADD_DATE = True

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Absolute path to to folder where the data and models live
# BASE_DIR = '/Volumes/TOSHIBA EXT/raid/data/cpi_data'
BASE_DIR = "/data/data"

# model to load
MODEL_PATH = f"{BASE_DIR}/saved_models/no_mask/{TAG}/e[15]_bs[64]_k0_vgg16.pt"

# workers for parallelization
NUM_CPUS = 10

# number of cpus used to load data in pytorch dataloaders
NUM_WORKERS = 10

# whether to save the individual extracted images
# used in process_png_sheets_with_text.py
SAVE_IMAGES = True

# percent of image that can intersect the border
CUTOFF = 10

# how many folds used in training (cross-validation)
# kold = 0 turns this off and splits the data according to valid_size
KFOLD = 0

# percent of the training dataset to use as validation
VALID_SIZE = 0.20

# images read into memory at a time during training
BATCH_SIZE = [64]

# number of epochs to train model
MAX_EPOCHS = [15]

# names of each ice crystal class
CLASS_NAMES = [
    "aggregate",
    "budding rosette",
    "bullet rosette",
    "column",
    "compact irregular",
    "fragment",
    "planar polycrystal",
    "rimed",
    "sphere",
]

# any abbreviations in folder names where the data lives for each class
CLASS_NAME_MAP = {
    "aggregate": "agg",
    "budding rosette": "budding",
    "bullet rosette": "bullet",
    "column": "column",
    "compact irregular": "compact_irreg",
    "fragment": "fragment",
    "planar polycrystal": "planar_polycrystal",
    "rimed": "rimed",
    "sphere": "sphere",
}

# models to train
MODEL_NAMES = [
    #     "efficient",
    #     "resnet18",
    #     "resnet34",
    #     "resnet152",
    #     "alexnet",
    "vgg16",
    #      "vgg19",
    #     "densenet169",
    #     "densenet201",
]

# model to load
MODEL_PATH = f"/data/data/saved_models/no_mask/{TAG}/e[15]_bs[64]_k1_vgg16.pt"

# directory that holds the training data
DATA_DIR = f"{BASE_DIR}/cpi_data/training_datasets/hand_labeled_{TAG}_noaug/"

# whether to save the model
SAVE_MODEL = True
# directory to save the trained model to

MODEL_SAVE_DIR = f"{BASE_DIR}/saved_models/no_mask/{TAG}/"

# directory to save validation data to
# for later inspection of predictions
VAL_LOADER_SAVE_DIR = f"{BASE_DIR}/saved_val_loaders/no_mask/{TAG}/"

MODEL_SAVENAME = (
    f"{MODEL_SAVE_DIR}e{MAX_EPOCHS}_" f"bs{BATCH_SIZE}_k{KFOLD}_" f"vgg16.pt"
)

VAL_LOADER_SAVENAME = (
    f"{VAL_LOADER_SAVE_DIR}e{MAX_EPOCHS}_"
    f"val_loader20_"
    f"bs{BATCH_SIZE}_k{KFOLD}_"
    f"vgg16.pt"
)

# write training loss and accuracy to csv
SAVE_ACC = True

# directory for saving training accuracy and loss csv's
ACC_SAVE_DIR = f"{BASE_DIR}/saved_accuracies/{TAG}/"

#  filename for saving training accuracy and loss
ACC_SAVENAME_TRAIN = (
    f"{ACC_SAVE_DIR}train_acc_loss_e{max(MAX_EPOCHS)}_"
    f"bs{max(BATCH_SIZE)}_k{KFOLD}_"
    f"{len(MODEL_NAMES)}model(s).csv"
)
#  output filename for validation accuracy and loss
ACC_SAVENAME_VAL = (
    f"{ACC_SAVE_DIR}val_acc_loss_e{max(MAX_EPOCHS)}_"
    f"bs{max(BATCH_SIZE)}_k{KFOLD}_"
    f"{len(MODEL_NAMES)}model(s).csv"
)
# output filename for precision, recall, F1 file
METRICS_SAVENAME = (
    f"{ACC_SAVE_DIR}val_metrics_e{max(MAX_EPOCHS)}_"
    f"bs{max(BATCH_SIZE)}_k{KFOLD}_"
    f"{len(MODEL_NAMES)}model(s).csv"
)

CONF_MATRIX_SAVENAME = "{BASE_DIR}/plots/conf_matrix.png"

# where to save final databases to
FINAL_DIR = f"{BASE_DIR}/final_databases/vgg16/{TAG}/"

# log experiment to comet for tracking?
LOG_EXP = False
if os.path.basename(sys.argv[0]) == "__main__.py":
    NOTEBOOK = False
else:
    NOTEBOOK = True

load_dotenv()  # loading sensitive keys from .env file
if LOG_EXP and NOTEBOOK is False and BUILD_MODEL:
    print("logging to comet ml...")
    API_KEY = os.getenv("API_KEY")
    WORKSPACE = os.getenv("WORKSPACE")
    PROJECT_NAME = os.getenv("PROJECT_NAME")
    experiment = Experiment(
        api_key=API_KEY,
        project_name=PROJECT_NAME,
        workspace=WORKSPACE,
    )

    params = {}
    for variable in [
        "TAG",
        "KFOLD",
        "BATCH_SIZE",
        "MAX_EPOCHS",
        "CLASS_NAMES",
        "VALID_SIZE",
        "MODEL_NAMES",
        "DATA_DIR",
        "MODEL_SAVE_DIR",
        "VAL_LOADER_SAVE_DIR",
        "SAVE_ACC",
        "NUM_WORKERS",
        "ACC_SAVENAME_TRAIN",
        "ACC_SAVENAME_VAL",
        "METRICS_SAVENAME",
        "MODEL_SAVENAME",
        "VAL_LOADER_SAVENAME",
    ]:
        params[variable] = eval(variable)

    experiment.log_parameters(params)
    experiment.add_tag(TAG)
else:
    experiment = None
