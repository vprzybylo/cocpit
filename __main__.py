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
import cocpit

import cocpit.config as config  # isort: split

import os
import time
import warnings

import pandas as pd
import torch


def _preprocess_sheets():
    """
    separate individual images from sheets of images to be saved
    text can be on the sheet
    """
    start_time = time.time()

    print("save images: ", config.SAVE_IMAGES)
    print("cutoff percentage allowed: ", config.CUTOFF)

    # where the sheets of images for each campaign live
    # if sheets were processed using rois in IDL, change 'sheets' to 'ROI_PNGS'
    sheet_dir = f"/data/data/cpi_data/campaigns/{campaign}/sheets/"
    save_dir = f"/data/data/cpi_data/campaigns/{campaign}/single_imgs_{config.TAG}/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cocpit.process_sheets.main(
        sheet_dir,
        save_dir,
        save_df=df_path,
        show_original=False,  # all set to False due to lack of display on server
        show_dilate=False,
        show_cropped=False,
    )

    print("time to preprocess sheets: %.2f" % (time.time() - start_time))


def _build_model():
    """
    train ML models
    """

    data = cocpit.data_loaders.get_data()

    # loop through batch sizes, models, epochs, and/or folds
    for batch_size in config.BATCH_SIZE:
        print("BATCH SIZE: ", batch_size)
        for model_name in config.MODEL_NAMES:
            print("MODEL: ", model_name)
            for epochs in config.MAX_EPOCHS:
                print("MAX EPOCH: ", epochs)

                # K-FOLD
                if config.KFOLD != 0:
                    cocpit.kfold_training.main(
                        data,
                        batch_size,
                        model_name,
                        epochs,
                    )
                else:  # no kfold
                    cocpit.no_fold_training.main(
                        data,
                        batch_size,
                        model_name,
                        epochs,
                    )


def _ice_classification():
    """
    classify good quality ice particles using the ML model
    """
    print("running ML model to classify ice...")

    start_time = time.time()

    # load ML model for predictions
    model = torch.load(config.MODEL_PATH)

    # load df of quality ice particles to make predictions on
    df = pd.read_csv(df_path)
    df = cocpit.run_model.main(df, open_dir, model)
    df.to_csv(df_path, index=False)

    print("done classifying all images!")
    print("time to classify ice = %.2f seconds" % (time.time() - start_time))


def _geometric_attributes():
    """
    calculates geometric particle properties and appends to the databases
    e.g., roundness, aspect ratio, area ratio, etc.
    see cocpit/geometric_attributes.py, which calls pic.py for calculations
    """

    # load df of quality ice particles to append particle attributes
    df = pd.read_csv(df_path)
    df = cocpit.geometric_attributes.main(df, open_dir)
    df.to_csv(df_path, index=False)


def _add_date():
    """
    add a column for the date from the filename
    """
    df = pd.read_csv(df_path)
    df = cocpit.add_date.main(df)
    df.to_csv(df_path, index=False)


if __name__ == "__main__":

    print(
        "num workers in loader = {}".format(config.NUM_WORKERS)
    ) if config.ICE_CLASSIFICATION or config.BUILD_MODEL else print(
        "num cpus for parallelization = {}".format(config.NUM_WORKERS)
    )

    campaigns = (
        ["N/A"]
        if config.BUILD_MODEL
        else [
            # "MACPEX",
            # "ATTREX",
            # "ISDAC",
            # "CRYSTAL_FACE_UND",
            # "AIRS_II",
            # "ARM",
            # "CRYSTAL_FACE_NASA",
            # "ICE_L",
            # "IPHEX",
            # "MC3E",
            # "MIDCIX",
            # "MPACE",
            "OLYMPEX",
            # "POSIDON",
        ]
    )
    for campaign in campaigns:
        print("campaign: ", campaign)
        # directory where the individual images live for each campaign
        open_dir = f"cpi_data/campaigns/{campaign}/single_imgs_v1.3.0/"

        # create dir for final databases
        outname = campaign + ".csv"
        if not os.path.exists(config.FINAL_DIR):
            os.makedirs(config.FINAL_DIR)
        df_path = os.path.join(config.FINAL_DIR, outname)

        if config.PREPROCESS_SHEETS:
            _preprocess_sheets()

        if config.BUILD_MODEL:
            _build_model()

        if config.ICE_CLASSIFICATION:
            _ice_classification()

        if config.GEOMETRIC_ATTRIBUTES:
            _geometric_attributes()

        if config.ADD_DATE:
            _add_date()
