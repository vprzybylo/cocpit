'''
- holds all user-defined variables
- treated as global variables that do not change in any module
- used in each module through 'import cocpit.config as config'
- call using config.VARIABLE_NAME
- flags for what module of cocpit to run is found in the main directory in __main__.py (e.g., preprocess_sheets, build_model, ice_classification, geometric_attributes, add_date..)

isort:skip_file
'''

from comet_ml import Experiment  # isort:split

import os
from dotenv import load_dotenv
import torch
import sys

# cocpit version used in docker and git
TAG = 'v1.4.0'

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model to load
MODEL_PATH = f"/data/data/saved_models/no_mask/{TAG}/e15_bs64_1model(s).pt"

# workers for parallelization
NUM_CPUS = 5

# number of cpus used to load data in pytorch dataloaders
NUM_WORKERS = 20

# whether to save the individual extracted images
# used in process_png_sheets_with_text.py
SAVE_IMAGES = False

# percent of image that can intersect the border
CUTOFF = 10

# how many folds used in training (cross-validation)
# kold = 0 turns this off and splits the data according to valid_size
KFOLD = 5

# percent of the training dataset to use as validation
VALID_SIZE = 0.20

# images read into memory at a time during training
BATCH_SIZE = [64]

# number of epochs to train model
MAX_EPOCHS = [20]

# names of each ice crystal class
CLASS_NAMES = [
    "agg",
    "budding",
    "bullets",
    "columns",
    "compact_irregs",
    "fragments",
    "planar_polycrystals",
    "rimed",
    "spheres",
]

# models to train
MODEL_NAMES = [
    #     "efficient",
    #     "resnet18",
    #     "resnet34",
    #     "resnet152",
    #     "alexnet",
    "vgg16",
    #     "vgg19",
    #     "densenet169",
    #     "densenet201",
]


# directory that holds the training data
DATA_DIR = (
    f"/data/data/cpi_data/training_datasets/hand_labeled_resized_{TAG}_sideplanes/"
)

# whether to save the model
SAVE_MODEL = False
# directory to save the trained model to

MODEL_SAVE_DIR = f"/data/data/saved_models/no_mask/{TAG}/"

# directory to save validation data to
# for later inspection of predictions
VAL_LOADER_SAVE_DIR = f"/data/data/saved_val_loaders/no_mask/{TAG}/"

MODEL_SAVENAME = (
    f"{MODEL_SAVE_DIR}e{max(MAX_EPOCHS)}_"
    f"bs{max(BATCH_SIZE)}_"
    f"{len(MODEL_NAMES)}model(s).pt"
)

VAL_LOADER_SAVENAME = (
    f"{VAL_LOADER_SAVE_DIR}e{max(MAX_EPOCHS)}_"
    f"bs{max(BATCH_SIZE)}_"
    f"{len(MODEL_NAMES)}model(s).pt"
)

# write training loss and accuracy to csv
SAVE_ACC = False

# directory for saving training accuracy and loss csv's
ACC_SAVE_DIR = f"/data/data/saved_accuracies/{TAG}/"

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

CONF_MATRIX_SAVENAME = "/data/data/plots/conf_matrix.png"

# where to save final databases to
FINAL_DIR = "/data/data/final_databases/vgg16/"

# log experiment to comet for tracking?
LOG_EXP = True
if os.path.basename(sys.argv[0]) == "__main__.py":
    NOTEBOOK = False
else:
    NOTEBOOK = True

load_dotenv()  # loading sensitive keys from .env file
if LOG_EXP and NOTEBOOK is False:
    print('logging to comet ml...')
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
