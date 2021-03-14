"""
Trains and saves a logistic regression model on prelabeled data
to predict if an image is a sphere 

If the image is not a sphere, another logistic regression model
is used to predict if an image represents quality ice or a 
blurry/broken/blank/fragmented image

In the case of quality ice, the amount of pixels touching the 
image border are taken into account (i.e., an alterable cutoff
measurement)

"""

import os 
import cocpit
import pickle
import numpy as np
import time
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.linear_model import LogisticRegression
from joblib import Parallel, delayed
from dask_ml.wrappers import ParallelPostFit

# Create the decorator function
# def logging_time(func):
#      def logged(*args, **kwargs):
#         start_time = time.time()
#         func(*args, **kwargs)
#         elapsed_time = time.time() - start_time
#         print(f"{func.__name__} time elapsed: {elapsed_time:.5f}")
#         return logged
#      return logging_time

def get_attributes(open_dir, filename, desired_size, good, train):
    
    #want a good/bad index for every file
    if train:
        if good:
            good_bad=0
        else:
            good_bad=1
    else:
        filenames=filename

    image = cocpit.pic.Image(open_dir, filename)
    image.resize_stretch(desired_size)
    image.find_contours()

    if len(image.contours)!=0:
        
        image.calculate_largest_contour()
        image.calculate_area()  
        
        if image.area != 0.0:
            image.morph_contours() 
            #image.mask_background()
            count_edge_px = np.count_nonzero(image.edges())
            if count_edge_px > 0:
                std=np.std(np.nonzero(image.edges()))
            else:
                std=0
            image.calculate_perim()
            image.calculate_hull_area()
            lapl=image.laplacian()
            contours=len(image.contours)
            edges=count_edge_px
            contrast=image.contrast()
            height=image.height_og
            width=image.width_og
            cnt_area=image.area
            solidity=image.solidity()
            complexity=image.complexity()
            equiv_d=image.equiv_d()
            convex_perim=image.convex_perim(True)
            hull_area=image.hull_area
            perim=image.perim
            phi=image.phi()
            circularity=image.circularity()
            cutoff=image.cutoff_perim()
            perim_area_ratio=image.perim_area_ratio()
            roundness=image.roundness()
            filled_circular_area_ratio=image.filled_circular_area_ratio()
            extreme_points=image.extreme_points()
        else:
            lapl=0
            contours=0
            edges=0
            contrast=0
            height=0
            width=0
            cnt_area=0
            solidity=0
            complexity=0
            equiv_d=0
            convex_perim=0
            hull_area=0
            perim=0
            phi=0
            circularity=0
            cutoff=0
            perim_area_ratio=0
            roundness=0
            filled_circular_area_ratio=0
            extreme_points=0
            std=0


    else:
        lapl=0
        contours=0
        edges=0
        contrast=0
        height=0
        width=0
        cnt_area=0
        solidity=0
        complexity=0
        equiv_d=0
        convex_perim=0
        hull_area=0
        perim=0
        phi=0
        circularity=0
        cutoff=0
        perim_area_ratio=0
        roundness=0
        filled_circular_area_ratio=0
        extreme_points=0
        std=0

    dicts = {}
    if train:
        keys = ['good_bad', 'height', 'width', 'lapl', 'contours', 'edges', 'std', 'cnt_area', \
            'contrast', 'circularity', 'solidity','complexity','equiv_d','convex_perim',\
            'hull_area', 'perim', 'phi', 'cutoff', 'extreme_points' ,\
                'filled_circular_area_ratio','roundness','perim_area_ratio']
        values = [good_bad, height, width, lapl, contours, edges, std, cnt_area, \
            contrast, circularity, solidity, complexity, equiv_d, convex_perim,\
            hull_area, perim, phi, cutoff, extreme_points, filled_circular_area_ratio,\
                roundness, perim_area_ratio]
    else:
        keys = ['filename', 'height', 'width', 'lapl', 'contours', 'edges', 'std', 'cnt_area', \
            'contrast', 'circularity', 'solidity','complexity','equiv_d','convex_perim',\
            'hull_area', 'perim', 'phi', 'cutoff', 'extreme_points' ,\
                'filled_circular_area_ratio','roundness','perim_area_ratio']
        values = [filenames, height, width, lapl, contours, edges, std, cnt_area, \
            contrast, circularity, solidity, complexity, equiv_d, convex_perim,\
            hull_area, perim, phi, cutoff, extreme_points, filled_circular_area_ratio,\
                roundness, perim_area_ratio]
    for key, val in zip(keys, values):
        dicts[key] = val
    df = pd.DataFrame(dicts, index=[0])
        
    return(df)

def make_dataframe(num_cpus, open_dir, desired_size, good=True, train=False):
    """
    loops through images in a directory of prelabeled spheres 
    to build a logistic regression model

    returns
    -------
        df (pd.DataFrame): dataframe with image attributes from cocpit.pic module
    """
         
    files= os.listdir(open_dir)
    start = time.time()
    dfs = Parallel(n_jobs=num_cpus)(delayed(get_attributes)(open_dir, filename, desired_size, good, train) for filename in files)

    # Concat dataframes to one dataframe
    df = pd.concat(dfs, ignore_index=True)
    end = time.time()
    print('Completed in: %.2f sec'%(end - start))

    return df

def split_data(df):
    # split dataset into x,y
    x = df.drop(['good_bad'], axis=1)
    y = df['good_bad']
    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=13)
    return X_train, X_test, y_train, y_test

def transform_data_min_max(X_train, X_test):
    """
    Normalizes features by scaling each feature to a given range.

    This estimator scales and translates each feature individually 
    such that it is in the given range on the training set, 
    e.g. between zero and one.


    attributes:
    ----------
    X_train, X_test, y_train, y_test (arr): raw split data

    returns:
    --------
    X_norm_train, X_norm_test (arr): transformed data

    """
    scaler = MinMaxScaler() 
    X_norm_train = scaler.fit_transform(X_train)
    X_norm_test = scaler.transform(X_test)
    return X_norm_train, X_norm_test, scaler

def transform_data_stand(X_train, X_test):   
    scaler = StandardScaler() 
    X_stand_train = scaler.fit_transform(X_train)
    X_stand_test = scaler.transform(X_test) 
    return X_stand_train, X_stand_test, scaler

def transform_pca(X_train, X_test):
    pca = PCA()
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)   
    return  X_train_pca, X_test_pca

def build_model(X_train, y_train, X_test):
    lg = ParallelPostFit(LogisticRegression(random_state=13, class_weight='balanced'))
    lg.fit(X_train,y_train) 
     #don't normalize/standardize y since already between 0-1 for categorical 
    #y_pred = lg1.predict(X_test)
    return lg

def main(num_cpus, desired_size, open_dirs_spheres, open_dirs_sift, save_df_spheres, save_df_sift,\
                                      save_model_spheres, save_model_sift, save_scaler_spheres, save_scaler_sift):
    #saving the dataframes for testing transformations
    #that way not reading in files each test
    dfs = []
    for open_dir in open_dirs_spheres:
        if open_dir == open_dirs_spheres[0]:
            print('training good spheres')
            dfs.append(make_dataframe(num_cpus, open_dir, desired_size, good=True, train=True))
        else:
            print('training bad spheres')
            dfs.append(make_dataframe(num_cpus, open_dir, desired_size, good=False, train=True))
    df_spheres = pd.concat([dfs[0], dfs[1]])
    df_spheres.to_pickle(save_df_spheres)
    
    dfs = []
    for open_dir in open_dirs_sift:
        if open_dir == open_dirs_sift[0]:
            print('training good ice')
            dfs.append(make_dataframe(num_cpus, open_dir, desired_size,  good=True, train=True))
        else:
            print('training bad ice')
            dfs.append(make_dataframe(num_cpus, open_dir, desired_size, good=False, train=True))
    df_sift = pd.concat([dfs[0], dfs[1]])
    df_sift.to_pickle(save_df_sift)

    dfs = [save_df_spheres, save_df_sift]
    for c, df in enumerate(dfs):
        df = pd.read_pickle(df)
        X_train, X_test, y_train, y_test = split_data(df)
        #X_norm_train, X_norm_test, scaler = transform_data_min_max(X_train, X_test)
        X_stand_train, X_stand_test, scaler = transform_data_stand(X_train, X_test)
        
        #X_train_pca, X_test_pca = transform_pca(X_train, X_test)
        lg = build_model(X_stand_train, y_train, X_stand_test)

        #save final logistic regression models and transformations
        if c == 0:
            pickle.dump(scaler, open(save_scaler_spheres, 'wb'))
            pickle.dump(lg, open(save_model_spheres, 'wb'))  
        else:
            pickle.dump(scaler, open(save_scaler_sift, 'wb'))
            pickle.dump(lg, open(save_model_sift, 'wb'))   
