#! /usr/bin/python3.9
"""COCPIT package for classifying ice crystal images from the CPI probe
Usage:
------
    $ python ./__main__.py

Contact:
--------
-Vanessa Przybylo
- vprzybylo@albany.edu
More information is available at:
- https://github.com/vprzybylo/cocpit
"""
import cocpit

from cocpit import config as config
import os
import time

import pandas as pd
import torch


def _preprocess_sheets(df_path: str, campaign: str) -> None:
    """
    - Separate individual images from sheets of images to be saved
    - Text can be on the sheet
    """
    start_time = time.time()

    print("save images: ", config.SAVE_IMAGES)
    print("cutoff percentage allowed: ", config.CUTOFF)

    # where the sheets of images for each campaign live
    # if sheets were processed using rois in IDL, change 'sheets' to 'ROI_PNGS'
    # sheet_dir and save_dir can't go in config since using campaign var
    sheet_dir = f"{config.BASE_DIR}/cpi_data/campaigns/{campaign}/sheets/"
    save_dir = (
        f"{config.BASE_DIR}/cpi_data/campaigns/{campaign}/single_imgs_{config.TAG}/"
    )
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


def nofold_training(
    model_name: str,
    batch_size: int,
    epochs: int,
    feature_extract: bool = False,
    use_pretrained: bool = False,
) -> None:
    """
    Create training and validation indices when k-fold cross validation
    not initialized (config.KFOLD=0)

    Args:
        model_name (str): name of model architecture
        batch_size (int): number of images read into memory at a time
        epochs (int): number of iterations on dataset
        feature_extract (bool): Start with a pretrained model and only
                                update the final layer weights from which we derive predictions
        use_pretrained (bool): Update all of the model’s parameters (retrain). Default = False
    """
    f = cocpit.fold_setup.FoldSetup(batch_size)
    f.nofold_indices()
    f.split_data()
    f.create_dataloaders()

    m = cocpit.models.Model(feature_extract, use_pretrained)
    # call method based on str model name
    method = getattr(cocpit.models.Model, model_name)
    method(m)

    c = cocpit.model_config.ModelConfig(m.model)
    c.set_optimizer()
    c.set_criterion()
    c.to_device()
    cocpit.runner.main(f, c, model_name, epochs)


def fold_training(model_name: str, batch_size: int, epochs: int) -> None:
    """
    Setup k-fold cross validation to labeled dataset

    Args:
        model_name (str): name of model architecture
        batch_size (int): number of images read into memory at a time
        epochs (int): number of iterations on dataset
    """
    f = cocpit.fold_setup.FoldSetup(model_name, batch_size, epochs)
    f.kfold_training()  # model config and calls to training happen in here


def _train_models() -> None:
    """
    Train ML models by looping through all batch sizes, models, epochs, and/or folds
    """
    for batch_size in config.BATCH_SIZE:
        print("BATCH SIZE: ", batch_size)
        for model_name in config.MODEL_NAMES:
            print("MODEL: ", model_name)
            for epochs in config.MAX_EPOCHS:
                print("MAX EPOCH: ", epochs)

                if config.KFOLD != 0:
                    fold_training(model_name, batch_size, epochs)
                else:
                    nofold_training(model_name, batch_size, epochs)


def _ice_classification(df_path: str, open_dir: str) -> None:
    """
    Classify quality ice particles using a trained ML model
    """
    start_time = time.time()

    # load ML model for predictions
    model = torch.load(config.MODEL_PATH)

    # load df of quality ice particles to make predictions on
    df = pd.read_csv(df_path)
    df = cocpit.run_model.main(df, open_dir, model)
    # df.to_csv(df_path, index=False)

    print("done classifying all images!")
    print("time to classify ice = %.2f seconds" % (time.time() - start_time))


def _geometric_attributes(df_path: str, open_dir: str):
    """
    Calculates geometric particle properties and appends to the databases
     - e.g., roundness, aspect ratio, area ratio, etc.
     - see cocpit/geometric_attributes.py, which calls cocpit/pic.py for calculations
    """

    # load df of quality ice particles to append particle attributes
    df = pd.read_csv(df_path)
    df = cocpit.geometry_runner.main(df, open_dir)
    df.to_csv(df_path, index=False)


def _add_date(df_path: str):
    """
    Add a column for the date from the filename

    Args:
        df_path
    """
    df = pd.read_csv(df_path)
    d = cocpit.add_date.Date(df)
    d.date_column()
    d.convert_date_format()
    d.move_to_front()
    d.df.to_csv(df_path, index=False)


def main() -> None:
    """
    Main pipeline which invokes action based on user defined configs for:
        - preprocessing raw cpi sheets of images
        - training models
        - using a trained model to make new predictions
        - calculating geometric attributes on images
        - adding dates from filename timestamps
    """
    for campaign in config.CAMPAIGNS:
        print("campaign: ", campaign)
        # directory where the individual images live for each campaign
        open_dir = f"/data/data/cpi_data/campaigns/{campaign}/single_imgs_v1.4.0/"

        # create dir for final databases
        df_path = os.path.join(config.FINAL_DIR, f"{campaign}.csv")

        if config.PREPROCESS_SHEETS:
            _preprocess_sheets(df_path, campaign)

        if config.BUILD_MODEL:
            _train_models()

        if config.ICE_CLASSIFICATION:
            _ice_classification(df_path, open_dir)

        if config.GEOMETRIC_ATTRIBUTES:
            _geometric_attributes(df_path, open_dir)

        if config.ADD_DATE:
            _add_date(df_path)


if __name__ == "__main__":
    main()
