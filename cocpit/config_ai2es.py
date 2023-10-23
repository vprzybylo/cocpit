"""
- holds all user-defined global variables that do not change in any module
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
BASE_DIR = "/home/vanessa/hulk/ai2es"

# /raid/NYSM/archive/nysm/netcdf/proc/ on hulk mounted
NC_FILE_DIR = f"{BASE_DIR}/5_min_obs"

# /raid/lgaudet/precip/Precip/NYSM_1min_data on hulk mounted
CSV_FILE_DIR = f"{BASE_DIR}/1_min_obs"

# where to write time-matched images and NYSM data
WRITE_PATH = f"{BASE_DIR}/matched_parquet/"

# root dir to raw camera images (before each year subdir) - mounted
PHOTO_DIR = f"{BASE_DIR}/cam_photos/"

# where the mesonet obs live in parquet format
# output from nysm_obs_to_parquet
PARQUET_DIR_5M = f"{BASE_DIR}/mesonet_parquet_5M"
PARQUET_DIR_1M = f"{BASE_DIR}/mesonet_parquet_1M"

# ai2es version used in docker and git
TAG = "v1.0.0"

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
MAX_EPOCHS = [2]
MAX_EPOCHS_TUNE = [20, 30, 40]

# dropout rate (in model_config)
DROP_RATE_TUNE = [0.0, 0.3, 0.5]

# dropout rate (in model_config)
WEIGHT_DECAY_TUNE = [1e-5, 1e-3, 1e-2, 1e-1]

# learning rate (in model_config)
LR_TUNE = [0.001, 0.01, 0.1]

# If evidential deep learning is True, the model outputs prediction uncertainty and minimizes evidence for out of distribution samples
EVIDENTIAL = True

# effect of the KL divergence in the loss for evidential deep learning
# (e.g., if >= epoch 10, prediction error term and evidence adjustment term equally weighted)
ANNEALING_STEP = 10 if EVIDENTIAL else 0

# names of each ice crystal class
CLASS_NAMES = [
    "no precipitation",
    "obstructed",
    "precipitation",
]

# any abbreviations in folder names where the data lives for each class
CLASS_NAME_MAP = {
    "no precipitation": "no_precip",
    "obstructed": "obstructed",
    "precipitation": "precip",
}

# models to train
MODEL_NAMES_TUNE = [
    "resnet18",
    "resnet34",
    "resnet152",
    "efficient",
    "alexnet",
    "vgg16",
    "vgg19",
    "densenet169",
    "densenet201",
]

MODEL_NAMES = [
    "vgg16",
]

CONFIG_RAY = {
    "BATCH_SIZE": tune.choice(BATCH_SIZE_TUNE),
    "MODEL_NAMES": tune.choice(MODEL_NAMES_TUNE),
    "LR": tune.choice(LR_TUNE),
    "WEIGHT_DECAY": tune.choice(WEIGHT_DECAY_TUNE),
    "DROP_RATE": tune.choice(DROP_RATE_TUNE),
    "MAX_EPOCHS": tune.choice(MAX_EPOCHS_TUNE),
}


# directory that holds the training data
DATA_DIR = f"{BASE_DIR}/codebook_dataset/combined_extra/"
# DATA_DIR = f"{BASE_DIR}/training_small/"

# whether to save the model
SAVE_MODEL = True

# directory to save the trained model to
MODEL_SAVE_DIR = f"{BASE_DIR}/saved_models/{TAG}/"
if EVIDENTIAL:
    MODEL_SAVE_DIR = f"{BASE_DIR}/saved_models/evidential/{TAG}/"

# If the validation dataset is coming from a csv
VAL_PREDEFINED = False

# directory to save validation data to
# for later inspection of predictions
VAL_LOADER_SAVE_DIR = f"{BASE_DIR}/saved_val_loaders/{TAG}/"
if EVIDENTIAL:
    VAL_LOADER_SAVE_DIR = f"{BASE_DIR}/saved_val_loaders/evidential/{TAG}/"

# model to load
MODEL_PATH = f"{BASE_DIR}/saved_models/{TAG}/e[30]_bs[64]_k0_1model(s).pt"
if EVIDENTIAL:
    MODEL_PATH = f"{BASE_DIR}/saved_models/evidential/{TAG}/e[30]_bs[64]_k0_1model(s).pt"


MODEL_SAVENAME = (
    f"{MODEL_SAVE_DIR}e{MAX_EPOCHS}_"
    f"bs{BATCH_SIZE}_"
    f"k{KFOLD}_"
    f"{len(MODEL_NAMES)}model(s).pt"
)

VAL_LOADER_SAVENAME = (
    f"{VAL_LOADER_SAVE_DIR}e{MAX_EPOCHS}_val_loader20_"
    f"bs{BATCH_SIZE}_"
    f"k{KFOLD}_"
    f"{len(MODEL_NAMES)}model(s).pt"
) 

# Start with a pretrained model and only update the final layer weights
# from which we derive predictions
FEATURE_EXTRACT = False

# write training loss and accuracy to csv
SAVE_ACC = True

# directory for saving training accuracy and loss csv's
ACC_SAVE_DIR = f"{BASE_DIR}/saved_accuracies/{TAG}/"
if EVIDENTIAL:
    ACC_SAVE_DIR = f"{BASE_DIR}/saved_accuracies/evidential/{TAG}/"

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

CONF_MATRIX_SAVENAME = f"{BASE_DIR}/plots/conf_matrix.png"
CLASSIFICATION_REPORT_SAVENAME = f"{BASE_DIR}/plots/classification_report.png"

# where to save final databases to
FINAL_DIR = f"{BASE_DIR}/final_databases/vgg16/{TAG}/"

# log experiment to comet for tracking?
LOG_EXP = False
NOTEBOOK = os.path.basename(sys.argv[0]) != "__main__.py"
load_dotenv()  # loading sensitive keys from .env file
if LOG_EXP and not NOTEBOOK:
    print("logging to comet ml...")
    API_KEY = os.getenv("API_KEY")
    WORKSPACE = os.getenv("WORKSPACE")
    PROJECT_NAME = os.getenv("PROJECT_NAME")
    experiment = Experiment(
        api_key=API_KEY,
        project_name=PROJECT_NAME,
        workspace=WORKSPACE,
    )

    PARAMS = {
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

    experiment.log_parameters(PARAMS)
    experiment.add_tag(TAG)
    experiment.add_tag("evidential")
else:
    experiment = None

STNID = [
    "ADDI",
    "ANDE",
    "BATA",
    "BEAC",
    "BELD",
    "BELL",
    "BELM",
    "BERK",
    "BING",
    "BKLN",
    "BRAN",
    "BREW",
    "BROC",
    "BRON",
    "BROO",
    "BSPA",
    "BUFF",
    "BURD",
    "BURT",
    "CAMD",
    "CAPE",
    "CHAZ",
    "CHES",
    "CINC",
    "CLAR",
    "CLIF",
    "CLYM",
    "COBL",
    "COHO",
    "COLD",
    "COPA",
    "COPE",
    "CROG",
    "CSQR",
    "DELE",
    "DEPO",
    "DOVE",
    "DUAN",
    "EAUR",
    "EDIN",
    "EDWA",
    "ELDR",
    "ELLE",
    "ELMI",
    "ESSX",
    "FAYE",
    "FRED",
    "GABR",
    "GFAL",
    "GFLD",
    "GROT",
    "GROV",
    "HAMM",
    "HARP",
    "HARR",
    "HART",
    "HERK",
    "HFAL",
    "ILAK",
    "JOHN",
    "JORD",
    "KIND",
    "LAUR",
    "LOUI",
    "MALO",
    "MANH",
    "MEDI",
    "MEDU",
    "MORR",
    "NBRA",
    "NEWC",
    "NHUD",
    "OLDF",
    "OLEA",
    "ONTA",
    "OPPE",
    "OSCE",
    "OSWE",
    "OTIS",
    "OWEG",
    "PENN",
    "PHIL",
    "PISE",
    "POTS",
    "QUEE",
    "RAND",
    "RAQU",
    "REDF",
    "REDH",
    "ROXB",
    "RUSH",
    "SARA",
    "SBRI",
    "SCHA",
    "SCHO",
    "SCHU",
    "SCIP",
    "SHER",
    "SOME",
    "SOUT",
    "SPRA",
    "SPRI",
    "STAT",
    "STEP",
    "SUFF",
    "TANN",
    "TICO",
    "TULL",
    "TUPP",
    "TYRO",
    "VOOR",
    "WALL",
    "WALT",
    "WANT",
    "WARS",
    "WARW",
    "WATE",
    "WBOU",
    "WELL",
    "WEST",
    "WFMB",
    "WGAT",
    "WHIT",
    "WOLC",
    "YORK",
]
