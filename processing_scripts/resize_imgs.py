from PIL import Image, ImageOps
import os
import matplotlib.pyplot as plt
import cv2
import numpy as np

desired_size = 1000
dir_save = '../cpi_data/campaigns/'

if not os.path.exists(dir_save):
        os.makedirs(dir_save)

for filename in os.listdir(dir_open):
    #print(filename)
    im_pth = dir_open+filename

    im = cv2.imread(im_pth, cv2.COLOR_BGR2RGB)
    old_size = im.size
    height, width = im.shape[:2]
    old_size = [width, height]
    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size])
    
    print(old_size)
    print(new_size)
    im = cv2.resize(im,new_size)   

    delta_w = desired_size - new_size[0]
    delta_h = desired_size - new_size[1]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    cv2.imwrite(dir_save+filename[:-4]+'.png', new_im)
