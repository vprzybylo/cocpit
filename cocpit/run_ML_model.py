"""
classifies unseen images:
transforms, makes predictions, and appends classification to dataframe 
"""
import cocpit.data_loaders as data_loaders
import numpy as np
import pandas as pd
import os
import cv2
from PIL import Image
from collections import defaultdict
from twilio.rest import Client

import torch
from torchvision import transforms
from torch.utils.data import Dataset
import torch.nn.functional as F


def predict(test_loader, class_names, model, device):
    
    ''' Predict the classes of an image
    using a trained CNN.
    '''
    
    d = defaultdict(list)
    top_class = []
    all_outputs = []
    for batch_idx, (inputs, img_paths) in enumerate(test_loader):
        with torch.no_grad():
            inputs = inputs.to(device)
            logits = model.forward(inputs)
            ps = F.softmax(logits,dim=1)
            outputs = ps.cpu().numpy()*100 #(batch size, # classes)
            
            all_outputs.append(outputs)
    
            for pred in outputs: # batch
                for c in range(len(class_names)): # class
                    d[class_names[c]].append(pred[c])
                top_class.append(class_names[np.argmax(pred)])
            
    return d, top_class

def send_message():    
    account_sid = "AC6034e88973d880bf2244f62eec6fe356"
    auth_token = 'f374de1a9245649ef5c8bc3f6e4faa97'
    client = Client(account_sid, auth_token)    
    message = client.messages .create(body =  "ML predictions completed!", 
                                      from_ = "+19285175160", #Provided phone number 
                                      to = "+15187969534") #Your phone number
    message.sid
        

def main(df, open_dir, class_names, model, num_workers):
    pd.options.mode.chained_assignment = None  # default='warn'
    
    file_list = df['filename']
    test_loader = data_loaders.get_test_loader_df(open_dir,
                                                  file_list,
                                                  num_workers=num_workers)
  
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d, top_class = predict(test_loader,
                           class_names,
                           model, device)

    for column in sorted(d.keys()):
        df[column] = d[column]
    
    df['classification'] = top_class
    df = df[(df['classification'] != 'fragment') & 
            (df['classification'] != 'sphere')]
    
    send_message();
    
    return df                   
