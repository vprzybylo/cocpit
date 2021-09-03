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
import torch.nn.functional as F
from dotenv import load_dotenv
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from twilio.rest import Client

import cocpit.data_loaders as data_loaders

torch.cuda.empty_cache()


def predict(test_loader, class_names, model):

    """Predict the classes of an image
    using a trained CNN.
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    d = defaultdict(list)
    top_class = []
    all_outputs = []
    for batch_idx, (inputs, img_paths) in enumerate(test_loader):
        with torch.no_grad():
            inputs = inputs.to(device)
            logits = model.forward(inputs)
            ps = F.softmax(logits, dim=1)
            outputs = ps.cpu().numpy() * 100  # (batch size, # classes)

            all_outputs.append(outputs)

            for pred in outputs:  # batch
                for c in range(len(class_names)):  # class
                    d[class_names[c]].append(pred[c])
                top_class.append(class_names[np.argmax(pred)])

    return d, top_class


def percent_category(df, category="fragment"):
    """
    find # and % of a category
    """

    len_category = len(df[df["classification"] == category])
    perc_category = (len_category / len(df)) * 100
    perc_category = np.round(perc_category, 2)

    return len_category, perc_category


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


def main(df, open_dir, class_names, cutoff, model, num_workers):

    pd.options.mode.chained_assignment = None  # default='warn'

    file_list = df["filename"]
    test_loader = data_loaders.get_test_loader_df(
        open_dir, file_list, num_workers=num_workers
    )

    d, top_class = predict(test_loader, class_names, model)

    for column in sorted(d.keys()):
        df[column] = d[column]

    df["classification"] = top_class

    len_frag, perc_category = percent_category(df, category="fragment")
    print("#/% fragment: ", len_frag, perc_category)

    len_sphere, perc_category = percent_category(df, category="sphere")
    print("#/% sphere: ", len_sphere, perc_category)

    df = df[(df["classification"] != "fragment") & (df["classification"] != "sphere")]

    send_message()

    return df
