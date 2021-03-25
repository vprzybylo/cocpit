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
from comet_ml import Experiment

import os
import time
import pandas as pd
import torch
from sqlalchemy import create_engine
import cocpit


def _preprocess_sheets():
    """
    separate individual images from sheets of images to be saved
    text can be on the sheet
    """
    # change to ROI_PNGS if 'sheets' came from processing rois in IDL
    # sheets came from data archived pngs online
    open_dir = '/data/data/cpi_data/campaigns/'+campaign+'/sheets/'
    save_images=True
    print('save images', save_images)

    if mask:
        save_dir = '/data/data/cpi_data/campaigns/'+campaign+'/single_imgs_masked/'
    else:
        save_dir = '/data/data/cpi_data/campaigns/'+campaign+'/single_imgs/'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cocpit.process_png_sheets_with_text.main(open_dir,
                                             mask,
                                             save_dir,
                                             num_cpus,
                                             save_images,
                                             show_original=False,
                                             show_dilate=False,
                                             show_cropped=False,
                                             show_mask=False)


def _build_spheres_sift():
    """
    creates spheres and sift model from prelabeled datasets
    """

    print('building spheres and sift models...')
    # resize images to desired_size
    desired_size = 1000

    # paths for opening training datasets
    base_dir='/data/data/cpi_data/training_datasets/'
    open_dirs_spheres = [base_dir+'SPHERES/good/', base_dir+'SPHERES/bad/']
    open_dirs_sift = [base_dir+'SIFT/good/',base_dir+'SIFT/bad/']

    # paths for saving df's, logistic regression models, and transformations
    save_df_spheres = '/data/data/saved_models/'+masked_dir+'/spheres_df.pkl'
    save_df_sift = '/data/data/saved_models/'+masked_dir+'/sift_df.pkl'
    # save final logistic regression models
    save_model_spheres = '/data/data/saved_models/'+masked_dir+'/spheres_lg.pkl'
    save_model_sift = '/data/data/saved_models/'+masked_dir+'/sift_lg.pkl'
    # save transformations
    save_scaler_spheres = '/data/data/saved_models/'+masked_dir+'/stand_scaler_spheres.pkl'
    save_scaler_sift = '/data/data/saved_models/'+masked_dir+'/stand_scaler_sift.pkl'

    cocpit.build_spheres_sift.main(mask, num_cpus,
                                   desired_size,
                                   open_dirs_spheres,
                                   open_dirs_sift,
                                   save_df_spheres, save_df_sift,
                                   save_model_spheres,
                                   save_model_sift,
                                   save_scaler_spheres,
                                   save_scaler_sift)


def _make_predictions():
    '''
    makes new predictions on campaign data for liquid/ice and good/bad quality
    returns a dataframe of quality ice particles including:
    -filename
    -particle attributes
    -binary prediction
    '''
    print('counting liquid drops and returning a dataframe'+
          'with quality ice images...')

    start_time = time.time()
    # resize images to desired_size
    desired_size = 1000
    cutoff = 10

    if mask:
        open_dir = '/data/data/cpi_data/campaigns/'+campaign+'/single_imgs_masked/'
    else:
        open_dir = '/data/data/cpi_data/campaigns/'+campaign+'/single_imgs/'

    model_path_spheres = '/data/data/saved_models/'+masked_dir+'/spheres_lg.pkl'
    model_path_sift = '/data/data/saved_models/'+masked_dir+'/sift_lg.pkl'
    load_scaler_spheres = '/data/data/saved_models/'+masked_dir+'/stand_scaler_spheres.pkl'
    load_scaler_sift = '/data/data/saved_models/'+masked_dir+'/stand_scaler_sift.pkl'

    df_nospheres = cocpit.spheres_sift_prediction.make_prediction_spheres(mask, num_cpus,
                                                                          desired_size,
                                                                          cutoff,
                                                                          open_dir,
                                                                          model_path_spheres,
                                                                          load_scaler_spheres)

    df_good_ice = cocpit.spheres_sift_prediction.make_prediction_sift(df_nospheres,
                                                                      model_path_sift,
                                                                      load_scaler_sift)
    print('time to create quality ice df %.2f' %(time.time()-start_time))
    print('removing duplicates...')

    len_before = len(df_good_ice)
    df_good_ice = df_good_ice.drop_duplicates(subset=df_good_ice.columns.difference(['filename']),
                                              keep='first')
    print('removed %d duplicates' %(len_before - len(df_good_ice)))

    df_good_ice.to_pickle('final_databases_v2/'+
                          masked_dir+'/df_good_ice_'+campaign+'.pkl')


def _build_ML():
    '''
    train ML models. params holds some hyperparams
    '''
    print('training...')

    params = {'kfold': 0,  # set to 0 to turn off kfold cross validation
              'masked': mask,
              'batch_size': [128],
              'max_epochs': [20],
              'class_names': ['aggs','blank','blurry',
                              'budding','bullets','columns',
                              'compact_irregs','fragments','needles',
                              'plates','rimed_aggs','rimed_columns','spheres'],
              'model_names': ['vgg19']}
#               'model_names': ['efficient', 'resnet18', 'resnet34',
#                               'resnet152', 'alexnet', 'vgg16',
#                               'vgg19', 'densenet169', 'densenet201']}

    if mask:
        params['data_dir'] = '/data/data/cpi_data/training_datasets/' + \
                             'hand_labeled_resized_multcampaigns_masked/'
    else:
        params['data_dir'] = '/data/data/cpi_data/training_datasets/' + \
                             'hand_labeled_resized_multcampaigns_clean/'

    model_savename = '/data/data/saved_models/' + masked_dir + \
                     '/e' + str(params['max_epochs'][0]) + \
                     '_bs' + str(params['batch_size'][0]) + \
                     '_k' + str(params['kfold']) + '_' + \
                     str(len(params['model_names']))+'models'
    acc_savename_train = '/data/data/saved_accuracies/'+masked_dir + \
                         '/save_train_acc_loss_e' + \
                         str(params['max_epochs'][0]) + \
                         '_bs' + str(params['batch_size'][0]) + \
                         '_k' + str(params['kfold']) + '_' + \
                         str(len(params['model_names']))+'models_3.csv'
    acc_savename_val = '/data/data/saved_accuracies/' + masked_dir + \
                       '/save_val_acc_loss_e' + \
                       str(params['max_epochs'][0]) + \
                       '_bs' + str(params['batch_size'][0]) + \
                       '_k' + str(params['kfold']) + '_' + \
                       str(len(params['model_names'])) + 'models_3.csv'

    log_exp = True  # log experiment to comet
    save_acc = False
    save_model = True
    valid_size = 0.2  # 80-20 split training-val
    num_workers = 20  # change to # of cores available to load images
    num_classes = len(params['class_names'])

    cocpit.build_ML_model.main(params, log_exp, model_savename,
                               acc_savename_train, acc_savename_val,
                               save_acc, save_model, masked_dir, valid_size,
                               num_workers, num_classes)


def _ice_classification():
    '''
    classify good quality ice particles using the ML model
    '''
    print('running ML model to classify ice...')

    start_time = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    class_names=['agg','blank','blurry','budding','bullet',
                 'column','compact irregular','fragment','needle',
                 'plate','rimed agg','rimed column','sphere']

    if mask:
        open_dir = 'cpi_data/campaigns/'+campaign+'/single_imgs_masked/'
    else:
        open_dir = 'cpi_data/campaigns/'+campaign+'/single_imgs/'

    model=torch.load('/data/data/saved_models/'+
                     masked_dir+'/e50_bs128_k0_8models_vgg19')

    df_good_ice = pd.read_pickle('final_databases_v2/'+
                                 masked_dir+'df_good_ice_'+campaign+'.pkl')

    df = cocpit.run_ML_model.main(df_good_ice,
                                  open_dir,
                                  device,
                                  class_names,
                                  model)

    # write database to file that holds predictions
    engine = create_engine('sqlite:///final_databases_v3/' +
                           masked_dir + '/' + campaign+'.db', echo=False)
    df.to_sql(name=campaign, con=engine, index=False, if_exists='replace')
    df.to_csv('final_databases_v3/' + masked_dir + '/' + campaign+'.csv', index=False)

    print('done classifying all images!')
    print('time to classify ice = %.2f seconds' %(time.time() - start_time))


def main():

    if preprocess_sheets:
        _preprocess_sheets()

    if build_spheres_sift:
        _build_spheres_sift()

    if make_predictions:
        _make_predictions()

    if build_ML:
        _build_ML()

    if ice_classification:
        _ice_classification()

if __name__ == "__main__":
    # extract each image from sheet of images
    preprocess_sheets = False

    # creates and saves spheres and sift model from prelabeled datasets
    build_spheres_sift = False

    # makes predictions for spheres/sift on new
    # data and return quality ice dataframe
    make_predictions = False

    # create CNN
    build_ML = True

    # run the category classification on quality images of ice particles
    ice_classification = False
    campaigns=['ICE_L']
    # campaigns=['ARM','ATTREX','CRYSTAL_FACE_NASA','CRYSTAL_FACE_UND',\
    #           'ICE_L','IPHEX','MACPEX','MC3E','MIDCIX','MPACE','POSIDON']

    mask=False  # mask background?
    print('masked background = ', mask)
    if mask:
        masked_dir = 'masked/'
    else:
        masked_dir = 'no_mask/'

    num_cpus=28  # workers for parallelization
    for campaign in campaigns:
        print(campaign)
        main()
