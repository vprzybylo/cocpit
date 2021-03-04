import sys
sys.path.append("../")
import cocpit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cocpit.pic as pic
import multiprocessing
from joblib import Parallel, delayed

def create_ellipse(campaign, desired_size, file):
    if campaign == 'OLYMPEX':
        image = pic.Image('../cpi_data/campaigns/'+campaign+'/single_imgs2/', file)
    else:
        image = pic.Image('../cpi_data/campaigns/'+campaign+'/single_imgs/', file)
    image.resize_stretch(desired_size)
    image.find_contours()
    return image.create_ellipse()

df_all = pd.DataFrame()
campaigns=['ARM', 'CRYSTAL_FACE_NASA', 'CRYSTAL_FACE_UND', 'ICE_L', 'MIDCIX', 'MPACE', 'OLYMPEX']
desired_size = 1000
num_cores = multiprocessing.cpu_count()
phi_ellipses = []
campaign_names=[]
    
for campaign in campaigns:
    print(campaign)
    df = pd.read_csv('../final_databases/no_mask/'+campaign+'.csv')
    df = df[(df['classification'] != 'blurry') & (df['classification'] != 'sphere')]
    phi_ellipse = Parallel(n_jobs=10)(delayed(create_ellipse)(campaign, desired_size, file) for file in df['filename'])
    
    df.insert(16, 'phi_ellipse', phi_ellipse)
    df.insert(0, 'campaign', [campaign]*len(df))
    df_all = df_all.append(df)
print('done')

df_all.to_csv('../final_databases/no_mask/all_campaigns.csv', index=False)
