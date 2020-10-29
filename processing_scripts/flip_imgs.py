import matplotlib.pyplot as plt
import numpy as np
import os

columns = ['/data/cpi_data/OLYMPEX/new_labels2/columns/','/data/cpi_data/OLYMPEX/new_labels/2columns_flip/']
rimed_columns=['/data/cpi_data/OLYMPEX/new_labels2/rimed_columns/','/data/cpi_data/OLYMPEX/new_labels2/rimed_columns_flip/']
aggs=['/data/cpi_data/OLYMPEX/new_labels2/aggs/','/data/cpi_data/OLYMPEX/new_labels2/aggs_flip/']
rimed_aggs=['/data/cpi_data/OLYMPEX/new_labels2/rimed_aggs_resized/','/data/cpi_data/OLYMPEX/new_labels2/rimed_aggs_flip/']
spheres=['/data/cpi_data/OLYMPEX/new_labels2/spheres/','/data/cpi_data/OLYMPEX/new_labels2/spheres_flip/']
junk=['/data/cpi_data/OLYMPEX/new_labels2/junk/','/data/cpi_data/OLYMPEX/new_labels2/junk_flip/']

all_paths=[rimed_aggs]

for i in all_paths:
    p1 = i[0]
    p2 = i[1]
    print(i, p1, p2)
    if not os.path.exists(p2):
    	os.makedirs(p2)
    for img in os.listdir(p1):
        im = plt.imread(p1+img)
        plt.imsave(p2+img,im)
        plt.imsave(p2+img[:-4]+'_ud.png',np.flipud(im))
        plt.imsave(p2+img[:-4]+'_lr.png',np.fliplr(im))
        plt.imsave(p2+img[:-4]+'_ud_lr.png',np.flipud(np.fliplr(im)))
