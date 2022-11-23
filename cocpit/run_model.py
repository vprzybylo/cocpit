import os
from collections import defaultdict
from typing import DefaultDict, List, Tuple

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from twilio.rest import Client

import cocpit.data_loaders as data_loaders
import cocpit.predictions as predictions
from cocpit import config as config

torch.cuda.empty_cache()


def test_loader(open_dir, df):
    """
    Create test dataloader

    Args:
        open_dir (str): path to the image directory
        df (pd.DataFrame): df with file names

    Returns:
        (torch.utils.data.DataLoader): a dataset to be iterated over using sampling strategy
    """
    file_list = df["filename"]
    test_data = data_loaders.TestDataSet(open_dir, file_list)
    return data_loaders.create_loader(test_data, batch_size=100, sampler=None)


def test_predictions(loader, model) -> Tuple[DefaultDict[str, List[float]], List[str]]:
    """
    Make predictions from loader

    Args:
        loader (torch.loader.Loader): test loader
        model (torch.model.Model): model used to make predictions

    Returns:
        class_probs (DefaultDict[str, List[float]): list of class probabilities for all images
        top_class (List[str]): highest class probability for all images
    """
    # defaultdict will "default" to an empty list if that key has not been set yet
    class_probs = defaultdict(list)
    top_class = []
    for imgs, _ in loader:
        p = predictions.BatchPredictions(imgs, model)
        with torch.no_grad():
            batch_output = p.preds_softmax().cpu().numpy() * 100
            for pred in batch_output:
                [
                    class_probs[config.CLASS_NAMES[c]].append(pred[c])
                    for c, _ in enumerate(config.CLASS_NAMES)
                ]
                top_class.append(config.CLASS_NAMES[np.argmax(pred)])

    return (class_probs, top_class)


def percent_category(df, category) -> None:
    """
    Find # and % of a class out of all images

    Args:
        df (pdf.DataFrame): dataframe of predictions
        category (str): the class to be considered
    """

    len_category = len(df[df["classification"] == category])
    perc_category = (len_category / len(df)) * 100
    perc_category = np.round(perc_category, 2)

    print(f"#/% {category}: ", len_category, perc_category)


def send_message() -> None:
    """
    - Use twilio to receive a text when the model has finished running
    - Add ACCOUNT_SID, AUTH_TOKEN, and PHONE_NUMBER to a .env file
    """
    load_dotenv()
    account_sid = os.getenv("ACCOUNT_SID")
    auth_token = os.getenv("AUTH_TOKEN")
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body="ML predictions completed!",
        from_="+19285175160",  # Provided phone number
        to=os.getenv("PHONE_NUMBER"),  # Your phone number
    )
    message.sid


def main(df, open_dir, model) -> pd.DataFrame:
    """
    Classifies unseen images and appends classification to dataframe

    Args:
        df (pd.DataFrame): df of filenames
        open_dir (str): directory where the test images live
        model (torch.model.Model): model to be used for classification
    Returns:
        df (pd.DataFrame): df with predictions
    """
    pd.options.mode.chained_assignment = None  # default='warn'
    loader = test_loader(open_dir, df)
    class_probs, top_class = test_predictions(loader, model)
    for column in sorted(class_probs.keys()):
        df[f"{column} [%]"] = class_probs[column]

    # append predictions to dataframe
    df["classification"] = top_class
    percent_category(df, category="fragment")
    percent_category(df, category="sphere")

    # don't include fragments or sphere classifications in dataframes
    df[(df["classification"] != "fragment") & (df["classification"] != "sphere")]

    # send_message()

    return df
