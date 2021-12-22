"""
classifies unseen images:
transforms, makes predictions, and appends classification to dataframe
"""
import os
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import Dataset
from twilio.rest import Client

import cocpit.config as config
import cocpit.data_loaders as data_loaders
import cocpit.predictions as predictions

torch.cuda.empty_cache()


def test_loader(open_dir, file_list, batch_size=100, pin_memory=True):
    """
    Loads and returns a multi-process test iterator
    """

    test_data = data_loaders.TestDataSet(open_dir, file_list)
    return data_loaders.create_loader(test_data, batch_size=batch_size, sampler=None)


def percent_category(df, category):
    """
    find # and % of a class out of all images for a campaign
    """

    len_category = len(df[df["classification"] == category])
    perc_category = (len_category / len(df)) * 100
    perc_category = np.round(perc_category, 2)

    print(f"#/% {category}: ", len_category, perc_category)


def append_classifications(df, top_class):
    '''append the top class'''

    df["classification"] = top_class

    percent_category(df, category="fragment")
    percent_category(df, category="sphere")

    # don't include fragments or sphere classifications in dataframes
    return df[(df["classification"] != "fragment") & (df["classification"] != "sphere")]


def send_message():
    '''
    use twilio to receive a text when the model has finished running!
    register for an account and then:
    add ACCOUNT_SID, AUTH_TOKEN, and PHONE_NUMBER to a .env file
    '''
    load_dotenv()
    account_sid = os.getenv('ACCOUNT_SID')
    auth_token = os.getenv('AUTH_TOKEN')
    client = Client(account_sid, auth_token)
    message = client.messages.create(
        body="ML predictions completed!",
        from_="+19285175160",  # Provided phone number
        to=os.getenv('PHONE_NUMBER'),
    )  # Your phone number
    message.sid


def main(df, open_dir, model):

    pd.options.mode.chained_assignment = None  # default='warn'

    file_list = df["filename"]
    loader = test_loader(open_dir, file_list)

    # make predictions from test_loader
    p = predictions.Predict(model, loader)
    d, top_class = p.all_predictions()

    for column in sorted(d.keys()):
        df[column] = d[column]

    # append predictions to dataframe for a campaign
    df = append_classifications(df, top_class)

    send_message()

    return df
