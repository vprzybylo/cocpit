#!/opt/conda/bin/python
"""COCPIT package for classifying ice crystal images from the CPI probe
Usage:
------
    $ pipenv run python ./__main__.py
    OR
    $ python ./__main__.py

Contact:
--------
-Vanessa Przybylo
- vprzybylo@albany.edu
More information is available at:
- https://vprzybylo.github.io/COCPIT/
"""
import os
import time

import pandas as pd
import torch
from comet_ml import Experiment
from dotenv import load_dotenv
from sqlalchemy import create_engine

import cocpit


def _preprocess_sheets():
    """
    separate individual images from sheets of images to be saved
    text can be on the sheet
    """
    start_time = time.time()
    # change to ROI_PNGS if 'sheets' came from processing rois in IDL
    # sheets came from data archived pngs online
    open_dir = "/data/data/cpi_data/campaigns/" + campaign + "/sheets/"
    save_images = True
    print("save images: ", save_images)

    # resize images to desired_size

    print("cutoff percentage allowed: ", cutoff)

    save_dir = "/data/data/cpi_data/campaigns/" + campaign + "/single_imgs_all/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    outname = "df_" + campaign + ".csv"
    outdir = "/data/data/final_databases/no_mask/"
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    save_df = os.path.join(outdir, outname)

    cocpit.process_png_sheets_with_text.main(
        open_dir,
        cutoff,
        save_dir,
        num_cpus,
        save_images,
        save_df=save_df,
        show_original=False,
        show_dilate=False,
        show_cropped=False,
    )

    print("time to preprocess sheets: %.2f" % (time.time() - start_time))


def _build_ML():
    """
    train ML models
    """
    print("training...")

    load_dotenv()  # loading sensitive keys from .env file

    params = {
        "kfold": 0,  # set to 0 to turn off kfold cross validation
        "batch_size": [64],
        "max_epochs": [20],
        "class_names": [
            "aggs",
            "budding",
            "bullets",
            "columns",
            "compact_irregs",
            "fragments",
            "plates",
            "rimed",
            "spheres",
        ],
        "model_names": ["vgg16"],
        "data_dir": "/data/data/cpi_data/training_datasets/"
        + "hand_labeled_resized_multcampaigns_v1.0.0_no_blank/",
        "log_exp": True,  # log experiment to comet
        "API_KEY": os.getenv("API_KEY"),
        "WORKSPACE": os.getenv("WORKSPACE"),
        "PROJECT_NAME": os.getenv("PROJECT_NAME"),
    }
    # 'model_names': ['efficient', 'resnet18', 'resnet34',
    #               'resnet152', 'alexnet', 'vgg16',
    #               'vgg19', 'densenet169', 'densenet201']}

    acc_savename_train = (
        "/data/data/saved_accuracies/no_mask/"
        + "/save_train_acc_loss_e"
        + str(params["max_epochs"][0])
        + "_bs"
        + str(params["batch_size"][0])
        + "_k"
        + str(params["kfold"])
        + "_"
        + str(len(params["model_names"]))
        + "models_no_blank.csv"
    )
    acc_savename_val = (
        "/data/data/saved_accuracies/no_mask/"
        + "/save_val_acc_loss_e"
        + str(params["max_epochs"][0])
        + "_bs"
        + str(params["batch_size"][0])
        + "_k"
        + str(params["kfold"])
        + "_"
        + str(len(params["model_names"]))
        + "models_no_blank.csv"
    )
    # savename for precision, recall, F1 file
    metrics_savename = (
        "/data/data/saved_accuracies/no_mask/"
        + "/save_val_metrics_e"
        + str(params["max_epochs"][0])
        + "_bs"
        + str(params["batch_size"][0])
        + "_k"
        + str(params["kfold"])
        + "_"
        + str(len(params["model_names"]))
        + "_no_blank.csv"
    )

    if params["log_exp"]:
        experiment = Experiment(
            api_key=params["API_KEY"],
            project_name=params["PROJECT_NAME"],
            workspace=params["WORKSPACE"],
        )
        experiment.log_parameters(params)
        experiment.add_tag("v1.2.0")
    else:
        experiment = None

    save_acc = True
    save_model = False
    valid_size = 0.2  # if <0.01, use all data with kfold set to 0 also
    len(params["class_names"])

    cocpit.build_ML_model.main(
        params,
        experiment,
        acc_savename_train,
        acc_savename_val,
        save_acc,
        save_model,
        metrics_savename,
        valid_size,
        num_workers,
    )


def _ice_classification():
    """
    classify good quality ice particles using the ML model
    """
    print("running ML model to classify ice...")

    start_time = time.time()

    class_names = [
        "agg",
        "budding",
        "bullet",
        "column",
        "compact irregular",
        "fragment",
        "plate",
        "rimed",
        "sphere",
    ]

    open_dir = "cpi_data/campaigns/" + campaign + "/single_imgs_all/"

    # load ML model for predictions
    model = torch.load(
        "/data/data/saved_models/no_mask/e20_bs64_k5_9models_v1.0.0_no_blank_vgg19"
    )

    # load df of quality ice particles to make predictions on
    df = pd.read_csv("final_databases/no_mask/df_" + campaign + ".csv")

    df = cocpit.run_ML_model.main(df, open_dir, class_names, cutoff, model, num_workers)

    # write database to file that holds predictions
    engine = create_engine(
        "sqlite:///final_databases_vgg19/no_mask/" + campaign + ".db", echo=False
    )
    df.to_sql(name=campaign, con=engine, index=False, if_exists="replace")
    df.to_csv("final_databases_vgg19/no_mask/" + campaign + ".csv", index=False)

    print("done classifying all images!")
    print("time to classify ice = %.2f seconds" % (time.time() - start_time))


if __name__ == "__main__":
    # extract each image from sheet of images
    preprocess_sheets = False

    # create CNN
    build_ML = False

    # run the category classification on quality images of ice particles
    ice_classification = True

    cutoff = 10  # percent of image that can intersect the border
    num_cpus = 5  # workers for parallelization
    num_workers = 20  # workers for data loaders

    campaigns = [
        "MACPEX",
        "ATTREX",
        "CRYSTAL_FACE_UND",
        "AIRS_II",
        "ARM",
        "CRYSTAL_FACE_NASA",
        "ICE_L",
        "IPHEX",
        "MC3E",
        "MIDCIX",
        "MPACE",
        "OLYMPEX",
        "POSIDON",
    ]
    campaigns = ["OLYMPEX"]
    if build_ML:
        campaigns = ["N/A"]  # only run once

    for campaign in campaigns:
        print("campaign: ", campaign)
        if preprocess_sheets:
            _preprocess_sheets()

        if build_ML:
            _build_ML()

        if ice_classification:
            _ice_classification()
