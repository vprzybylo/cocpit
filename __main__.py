"""COCPIT package for classifying ice crystal images from the CPI probe
Usage:
------
    $ pipenv run python ./__main__.py 

Contact:
--------
-Vanessa Przybylo
- vprzybylo@albany.edu
More information is available at:
- https://vprzybylo.github.io/COCPIT/
"""
import cocpit 
import torch
import time
from sqlalchemy import create_engine
import pandas as pd

if __name__ == "__main__":
    #starts with sheets of images and extracts each image from sheet
    preprocess_sheets = False
    #creates spheres and sift model from prelabeled datasets
    build_spheres_sift = False
    #makes predictions on new data and return quality ice dataframe
    make_predictions = True
    #create CNN
    build_ML = False
    #remove duplicates after saving good ice
    remove_duplicates = True
    #run the category classification on quality images of ice particles
    ice_classification = True
    
    campaign = 'AIRS_II'
    print(campaign)
    
    #####################
    if preprocess_sheets:
        print('extracting single images from "sheets"')

        open_dir = 'cpi_data/campaigns/'+campaign+'/sheets/'
        save_dir = 'cpi_data/campaigns/'+campaign+'/single_imgs/'
        #saves the extracted images from the sheets of images for use in the following steps
        write_single_images = True
        
        cocpit.preprocess_sheets.main(open_dir, save_dir, write_single_images)
    
    ######################
    if build_spheres_sift:
        print('building spheres and sift models...')
        #resize images to desired_size
        desired_size = 1000
        
        #paths for opening training datasets
        open_dirs_spheres = ['cpi_data/training_datasets/SPHERES/good/',\
                            'cpi_data/training_datasets/SPHERES/bad/']
        open_dirs_sift = ['cpi_data/training_datasets/SIFT/good/',\
                          'cpi_data/training_datasets/SIFT/bad/'] 
        
        #paths for saving df's, logistic regression models, and transformations
        save_df_spheres = "saved_models/spheres_nomask_df.pkl"
        save_df_sift = "saved_models/sift_nomask_df.pkl"
        save_model_spheres = "saved_models/spheres_nomask_lg.pkl"
        save_model_sift = "saved_models/sift_nomask_lg.pkl"
        save_scaler_spheres = 'saved_models/stand_scaler_spheres_nomask.pkl'
        save_scaler_sift = 'saved_models/stand_scaler_sift_nomask.pkl'
        
        
        cocpit.build_spheres_sift.main(desired_size, open_dirs_spheres, open_dirs_sift, save_df_spheres, save_df_sift,\
                                      save_model_spheres, save_model_sift, save_scaler_spheres, save_scaler_sift)

    ######################
    if make_predictions:
        print('counting liquid drops and returning a dataframe with quality ice images...')
        start_time = time.time()
        #resize images to desired_size
        desired_size = 1000
        cutoff = 10
        #campaign = '2002_CRYSTAL-FACE-UND'
        #open_dir = 'cpi_data/campaigns/'+campaign+'/single_imgs/'
        open_dir = 'cpi_data/campaigns/'+campaign+'/single_imgs/'
        model_path_spheres = 'saved_models/spheres_nomask_lg.pkl'
        model_path_sift = 'saved_models/sift_nomask_lg.pkl'
        load_scaler_spheres = 'saved_models/stand_scaler_spheres_nomask.pkl'
        load_scaler_sift = 'saved_models/stand_scaler_sift_nomask.pkl'

        df_nospheres = cocpit.spheres_sift_prediction.make_prediction_spheres(desired_size,
                                cutoff, open_dir, model_path_spheres,load_scaler_spheres)

        df_good_ice = cocpit.spheres_sift_prediction.make_prediction_sift(df_nospheres, model_path_sift, load_scaler_sift)
        print('time to create quality ice df %.2f' %(time.time()-start_time))
    
    ######################
    if build_ML:
        print('training the ML model and saving best iteration...')
        
        params = {'lr': [0.01],
        'batch_size': [128],
        'max_epochs': [20],
        'data_dir':'cpi_data/training_datasets/hand_labeled_resized_multcampaigns_clean/',
        'optimizer':[torch.optim.Adam, torch.optim.Adagrad, torch.optim.Adadelta, torch.optim.Adamax],
        #'momentum': [0.9, 0.999], 
        'class_names':['aggs','blank','blurry','budding','bullets','columns','compact irregulars',\
                       'fragments','needles','plates','rimed aggs','rimed columns','spheres'],
        #'model_names':['vgg19'],
        'model_names':['resnet18', 'resnet34', 'resnet152', 'alexnet', 'vgg16', 'vgg19', 'densenet169', 'densenet201'],
        'savename': 'saved_models/vgg19_bs128_e20_13classes'}

        num_workers = 20  #change to # of cores available to load images
        num_classes = len(params['class_names'])
    
        model_name, model_train_accs, model_val_accs, model_train_loss, \
            model_val_loss = cocpit.build_ML_model.main(params, num_workers, num_classes)

    ######################
    if remove_duplicates:
        print('removing duplicates...')
        len_before = len(df_good_ice)
        df_good_ice = df_good_ice.drop_duplicates(subset=df_good_ice.columns.difference(['filename']), keep='first')
        
        print('removed %d duplicates' %(len_before - len(df_good_ice)))
        df_good_ice.to_pickle('final_databases/no_mask/df_good_ice_'+campaign+'.pkl')
    
    ######################
    if ice_classification:
        print('running ML model to classify ice...')
        start_time = time.time()
        open_dir = 'cpi_data/campaigns/'+campaign+'/single_imgs/'  #same as above
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        class_names=['agg','blank','blurry','budding','bullet','column','compact irregular',\
                       'fragment','needle','plate','rimed agg','rimed column','sphere']
        model = torch.load('saved_models/bs128_e50_13classes_clean_vgg19')
        
        df_good_ice = pd.read_pickle('final_databases/no_mask/df_good_ice_'+campaign+'.pkl')
        df = cocpit.run_ML_model.main(df_good_ice, open_dir, device, class_names, model)
        
        #df = pd.read_csv('saved_models/test_save_df.csv')  
        engine = create_engine('sqlite:///final_databases/no_mask/'+campaign+'.db', echo=False)
        df.to_sql(name=campaign, con=engine, index=False, if_exists='replace')
        df.to_csv('final_databases/no_mask/'+campaign+'.csv', index=False)
        print('done classifying all images!')
        print('time to classify ice = %.2f' %(time.time() - start_time))