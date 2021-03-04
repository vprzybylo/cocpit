"""
makes predictions on new data if sphere/no sphere (liquid/no liquid)
makes predictions on good vs bad ice images
builds dataframe using build_spheres_sift module but with no training flag (filename included in df)
"""

import time
import pickle
import os
import cocpit
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Create the decorator function
# def logging_time(func):
#      def logged(*args, **kwargs):
#         start_time = time()
#         func(*args, **kwargs)
#         elapsed_time = time() - start_time
#         print(f"{func.__name__} time elapsed: {elapsed_time:.5f}")
#         return logged
#      return logging_time

def open_model(model_path):
    return pickle.load(open(model_path, 'rb'))

def load_scaler(load_scaler_spheres):
    return pickle.load(open(load_scaler_spheres, 'rb'))

def make_prediction_spheres(num_cpus, desired_size, cutoff, open_dir, model_path_spheres, load_scaler_spheres):
    
    spheres_lg = open_model(model_path_spheres)
    
    print('making new predictions spheres')
    start_time = time.time()
    df = cocpit.build_spheres_sift.make_dataframe(num_cpus, open_dir, desired_size)
    print('time to create spheres df %.2f' %(time.time()-start_time))
    
    #Regression model prediction for spheres
    scaler = load_scaler(load_scaler_spheres)
    df_pred = df.drop(columns=['filename'])
    pred = scaler.transform(df_pred)
    
    start_time = time.time()
    df['pred_spheres'] = spheres_lg.predict(pred)
    print('time to make predictions spheres %.2f' %(time.time()-start_time))
    
    idx = np.where((df['pred_spheres'] < 0.25) & (df['contours'] != 0) & (df['cutoff'] < cutoff))
    df_spheres = df.loc[idx]

    idx = np.where((df['pred_spheres'] > 0.25) & (df['contours'] != 0) & (df['cutoff'] < cutoff))
    df_nospheres = df.loc[idx]
    print('# of liquid drops: ', len(df_spheres))
    return df_nospheres


def make_prediction_sift(df_nospheres, model_path_sift, load_scaler_sift):

    sift_lg = open_model(model_path_sift)
    print('making new predictions sift')
    #Find ice that is not blurry or broken
    scaler = load_scaler(load_scaler_sift)
    df_pred = df_nospheres.drop(columns=['filename', 'pred_spheres'])
    
    pred = scaler.transform(df_pred)
    
    df_nospheres['pred_sift'] = sift_lg.predict(pred)
    
    df_good_ice = df_nospheres[df_nospheres['pred_sift'] < 0.25]

    print('# of good ice images: ', len(df_good_ice))

    return df_good_ice
        