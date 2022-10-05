"""
- THIS FILE SHOUlD BE ALTERED AND RENAMED config.py FOR EACH USER
- config.py in .gitignore to avoid version changes upon specifications
- holds all user-defined variables
- treated as global variables that do not change in any module
- used in each module through 'import config as config'
- call using config.VARIABLE_NAME
isort:skip_file
"""
from comet_ml import Experiment  # isort:split
from ray import tune
import os
from dotenv import load_dotenv
import torch
import sys

# Absolute path to to folder where the data and models live
BASE_DIR = "/DRIVE_PYTORCH"

# version used in docker and git
TAG = "v0.0.0"

# create and save CNN
BUILD_MODEL = True

# classify images on new data?
CLASSIFICATION = False

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# workers for parallelization
NUM_CPUS = 5

# number of cpus used to load data in pytorch dataloaders
NUM_WORKERS = 10

# how many folds used in training (cross-validation)
# kold = 0 turns this off and splits the data according to valid_size
# cannot = 1
KFOLD = 0

# percent of the training dataset to use as validation
VALID_SIZE = 0.20

# ray tune hyperoptimization
TUNE = False

# images read into memory at a time during training
BATCH_SIZE = [64]
BATCH_SIZE_TUNE = [32, 64, 128, 256]

# number of epochs to train model
MAX_EPOCHS = [25]
MAX_EPOCHS_TUNE = [20, 30, 40]

# dropout rate (in model_config)
DROP_RATE_TUNE = [0.0, 0.3, 0.5]

# learning rate (in model_config)
LR_TUNE = [0.001, 0.01, 0.1]

# names of each ice crystal class
CLASS_NAMES = ["wet", "dry", "snow", "snow_severe"]
CLASS_NAMES_ALPHABETICAL = ["dry", "snow", "snow_severe", "wet"]

# any abbreviations in folder names where the data lives for each class
CLASS_NAME_MAP = {
    "wet": "wet",
    "dry": "dry",
    "snow": "snow",
    "snow_severe": "severe snow",
}

# models to train
MODEL_NAMES_TUNE = [
    "resnet18",
    "efficient",
    "resnet34",
    "resnet152",
    "alexnet",
    "vgg16",
    "vgg19",
    "densenet169",
    "densenet201",
]

MODEL_NAMES = ["vgg16"]
MODEL_NAMES_SAVE = (
    MODEL_NAMES if len(MODEL_NAMES) == 1 else f"{len(MODEL_NAMES)}model(s)"
)

config_ray = {
    "BATCH_SIZE": tune.choice(BATCH_SIZE_TUNE),
    "MODEL_NAMES": tune.choice(MODEL_NAMES_TUNE),
    "LR": tune.choice(LR_TUNE),
    "DROP_RATE": tune.choice(DROP_RATE_TUNE),
    "MAX_EPOCHS": tune.choice(MAX_EPOCHS_TUNE),
}


# directory that holds the training data
DATA_DIR = (
    "/DRIVE/site_analysis/corridor_LIE_east/training_data_py/training_data/"
)
# DATA_DIR = f"{BASE_DIR}/training_small/"

# whether to save the model
SAVE_MODEL = True

# directory to save the trained model to
MODEL_SAVE_DIR = f"{BASE_DIR}/saved_models/{TAG}/"

# model description
MODEL_DESC = f"LIE_6class"

# directory to save validation data to
# for later inspection of predictions
VAL_LOADER_SAVE_DIR = f"{BASE_DIR}/saved_val_loaders/{TAG}/"

# {len(MODEL_NAMES)}model(s)
MODEL_SAVENAME = f"{MODEL_SAVE_DIR}{MODEL_DESC}_e{MAX_EPOCHS}_bs{BATCH_SIZE}_k{KFOLD}_{MODEL_NAMES_SAVE}.pt"
# {len(MODEL_NAMES)}model(s)

VAL_LOADER_SAVENAME = f"{VAL_LOADER_SAVE_DIR}{MODEL_DESC}_e{MAX_EPOCHS}_val_loader20_bs{BATCH_SIZE}_k{KFOLD}_{MODEL_NAMES_SAVE}"
# REMOVE vgg16 at end

# Start with a pretrained model and only update the final layer weights
# from which we derive predictions
FEATURE_EXTRACT = False

# Update all of the model’s parameters (retrain). Default = False
USE_PRETRAINED = False

# write training loss and accuracy to csv
SAVE_ACC = True

# directory for saving training accuracy and loss csv's
ACC_SAVE_DIR = f"{BASE_DIR}/saved_accuracies/{TAG}/"

#  filename for saving training accuracy and loss
ACC_SAVENAME_TRAIN = (
    f"{ACC_SAVE_DIR}{MODEL_DESC}_train_acc_loss_e{max(MAX_EPOCHS)}_"
    f"bs{max(BATCH_SIZE)}_k{KFOLD}_"
    f"{MODEL_NAMES_SAVE}.csv"
)
#  output filename for validation accuracy and loss
ACC_SAVENAME_VAL = (
    f"{ACC_SAVE_DIR}{MODEL_DESC}_val_acc_loss_e{max(MAX_EPOCHS)}_"
    f"bs{max(BATCH_SIZE)}_k{KFOLD}_"
    f"{MODEL_NAMES_SAVE}.csv"
)
# output filename for precision, recall, F1 file
METRICS_SAVENAME = (
    f"{ACC_SAVE_DIR}{MODEL_DESC}_val_metrics_e{max(MAX_EPOCHS)}_"
    f"bs{max(BATCH_SIZE)}_k{KFOLD}_"
    f"{MODEL_NAMES_SAVE}.csv"
)

CONF_MATRIX_SAVENAME = f"{BASE_DIR}/plots/{MODEL_DESC}_e{MAX_EPOCHS}_bs{BATCH_SIZE}_k{KFOLD}_{MODEL_NAMES_SAVE}_conf_matrix.png"

# where to save final databases to
FINAL_DIR = f"{BASE_DIR}/final_databases/{MODEL_NAMES_SAVE}/{TAG}/"
# vgg16 in between

# log experiment to comet for tracking?
LOG_EXP = False
NOTEBOOK = os.path.basename(sys.argv[0]) != "__main__.py"
load_dotenv()  # loading sensitive keys from .env file
if LOG_EXP and not NOTEBOOK and BUILD_MODEL:
    print("logging to comet ml...")
    API_KEY = os.getenv("API_KEY")
    WORKSPACE = os.getenv("WORKSPACE")
    PROJECT_NAME = os.getenv("PROJECT_NAME")
    experiment = Experiment(
        api_key=API_KEY,
        project_name=PROJECT_NAME,
        workspace=WORKSPACE,
    )

    params = {
        variable: eval(variable)
        for variable in [
            "TAG",
            "KFOLD",
            "BATCH_SIZE",
            "MAX_EPOCHS",
            "CLASS_NAMES",
            "VALID_SIZE",
            "MODEL_NAMES",
            "DATA_DIR",
            "SAVE_MODEL",
            "MODEL_SAVE_DIR",
            "VAL_LOADER_SAVE_DIR",
            "SAVE_ACC",
            "NUM_WORKERS",
            "ACC_SAVENAME_TRAIN",
            "ACC_SAVENAME_VAL",
            "METRICS_SAVENAME",
            "MODEL_SAVENAME",
            "VAL_LOADER_SAVENAME",
        ]
    }

    experiment.log_parameters(params)
    experiment.add_tag(TAG)
else:
    experiment = None
