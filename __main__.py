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
from sklearn.model_selection import StratifiedKFold
import pandas as pd
import torch


def makedirs(path: str, isfile: bool = False) -> None:
    """
    Creates a directory given a path to either a directory or file.
    If a directory is provided, creates that directory. If a file is provided (i.e. isfiled == True),
    creates the parent directory for that file.

    Args:
        path: str: Path to a directory or file.
        isfile: bool: Whether the provided path is a directory or file.
    """
    if isfile:
        path = os.path.dirname(path)
    if path != "":
        os.makedirs(path, exist_ok=True)


def _preprocess_sheets(df_path: str, campaign: str) -> None:
    """
    - Separate individual images from sheets of images to be saved
    - Text can be on the sheet

    Args:
        df_path (str): path to save dataframe to
        campaign (str): which campaign to process
    """
    start_time = time.time()

    print("save images: ", config.SAVE_IMAGES)
    print("cutoff percentage allowed: ", config.CUTOFF)

    # where the sheets of images for each campaign live
    # if sheets were processed using rois in IDL, change 'sheets' to 'ROI_PNGS'
    # sheet_dir and save_dir can't go in config since using campaign var
    sheet_dir = f"{config.BASE_DIR}/cpi_data/campaigns/{campaign}/sheets/"
    save_dir = f"{config.BASE_DIR}/cpi_data/campaigns/{campaign}/single_imgs_{config.TAG}/"
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


def kfold_training(batch_size: int, model_name: str, epochs: int) -> None:
    """
    - Split dataset into folds
    - Preserve the percentage of samples for each class with stratified
    - Create dataloaders for each fold

    Args:
        batch_size (int): number of images read into memory at a time
        model_name (str): name of model architecture
        epochs (int): number of iterations on dataset
    """
    skf = StratifiedKFold(n_splits=config.KFOLD, shuffle=True, random_state=42)
    # datasets based on phase get called again in split_data
    # needed here to initialize for skf.split
    data = cocpit.data_loaders.get_data("val")
    for kfold, (train_indices, val_indices) in enumerate(
        skf.split(data.imgs, data.targets)
    ):
        print("KFOLD iteration: ", kfold)

        # apply appropriate transformations for training and validation sets
        f = cocpit.fold_setup.FoldSetup(
            batch_size, kfold, train_indices, val_indices
        )
        f.split_data()
        f.update_save_names()
        f.create_dataloaders()
        model_setup(f, model_name, epochs)


def model_setup(
    f: cocpit.fold_setup.FoldSetup, model_name: str, epochs: int
) -> None:
    """
    Create instances for model configurations and training/validation. Runs model.

    Args:
        f (cocpit.fold_setup.FoldSetup): instance of FoldSetup class
        model_name (str): name of model architecture
        epochs (int): number of iterations on dataset
    """
    m = cocpit.models.Model()
    # call method based on str model name
    method = getattr(cocpit.models.Model, model_name)
    method(m)

    c = cocpit.model_config.ModelConfig(m.model)
    c.set_optimizer()
    c.set_criterion()
    c.to_device()
    cocpit.runner.main(
        f,
        c,
        model_name,
        epochs,
        kfold=0,
    )


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
                    # Setup k-fold cross validation on labeled dataset
                    kfold_training(batch_size, model_name, epochs)
                else:
                    f = cocpit.fold_setup.FoldSetup(batch_size, 0, [], [])
                    f.nofold_indices()
                    f.split_data()
                    f.create_dataloaders()
                    model_setup(f, model_name, epochs)


def _ice_classification(df_path: str, open_dir: str) -> None:
    """
    Classify quality ice particles using a trained ML model

    Args:
        df_path (str): path to save df to
        open_dir (str): directory where the test images live
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


def _geometric_attributes(df_path: str, open_dir: str) -> None:
    """
    Calculates geometric particle properties and appends to the databases
     - e.g., roundness, aspect ratio, area ratio, etc.

    Args:
        df_path (str): path to save df to
        open_dir (str): directory where the images live to process
    """

    # load df of quality ice particles to append particle attributes
    df = pd.read_csv(df_path)
    df = cocpit.geometry_runner.main(df, open_dir)
    df.to_csv(df_path, index=False)


def _add_date(df_path: str) -> None:
    """
    Add a column for the date from the filename

    Args:
        df_path (str): path to save df to
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
        open_dir = (
            f"/cocpit/cpi_data/campaigns/{campaign}/single_imgs_{config.TAG}/"
        )

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
