#!/usr/bin/env python
import sys
sys.path.insert(0,'/data/data/')
import cocpit.run_ML_model
import pandas as pd
import os
import csv
import torch
import time

def get_files(path, samples):
    return os.listdir(path+samples)

model_names = ['resnet34', 'resnet18', 'resnet152',
              'alexnet', 'vgg16', 'vgg19', 'densenet169',
              'densenet201', 'efficient']
class_names=['agg','blank','blurry','budding','bullet',
                 'column','compact irregular',
                 'plate','rimed','sphere']
num_workers=20 #cpus to load data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
path = '/data/data/cpi_data/campaigns/time_test/'

with open('/data/data/saved_timings/model_timing_samples3.csv', 'a', newline='') as file:
    writer = csv.writer(file)
    for model_name in model_names:
        model=torch.load('/data/data/saved_models/no_mask/e30_bs128_k0_9models_v1.0.0_removed_'
                         +model_name)
        model = model.to(device)
        model = model.cuda()
        model.eval()
        for samples in ['100_images/', '1000_images/', '10000_images/']:
            open_dir = path+samples
            file_list = get_files(path, samples)
            testdata = cocpit.run_ML_model.TestDataSet(open_dir, file_list)
            test_loader = torch.utils.data.DataLoader(testdata, batch_size=100, shuffle=False, 
                                   num_workers=num_workers, drop_last=False)
          
            start = time.time()
            d, top_class = cocpit.run_ML_model.predict(test_loader, class_names, model, device)
            end = time.time()
            time_elapsed = end-start
            print(model_name, samples[:-8], time_elapsed)
            writer.writerow([model_name, samples[:-8], time_elapsed])
