import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

time_start = time.time()

rootdir = '../cpi_data/training_datasets/hand_labeled_resized_multcampaigns_clean/'
savedir = '../cpi_data/training_datasets/hand_labeled_resized_multcampaigns_masked/'
if not os.path.exists(savedir):
    os.makedirs(savedir, exist_ok=True)
print('masking backgrounds...')
for subdir in os.listdir(rootdir):
    print(subdir)
    if not os.path.exists(os.path.join(savedir, subdir)):
        os.makedirs(os.path.join(savedir,subdir))
    for file in os.listdir(os.path.join(rootdir, subdir)):
        image_og = cv2.imread(os.path.join(rootdir, subdir, file), cv2.IMREAD_UNCHANGED)
        image_og =cv2.cvtColor(image_og, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image_og, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)[1]

        # get largest contour
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #returns image, contours, hierarchy
        if len(contours) != 0:
            mask = np.zeros(image_og.shape[:2], dtype="uint8")
            big_contour = max(contours, key=cv2.contourArea)
            draw = cv2.drawContours(mask, [big_contour], 0, (255,255,255), -1)
            #image_og=255-image_og
            # apply mask to image
            masked = cv2.bitwise_and(image_og, image_og, mask=mask)
            masked =255-masked

            cv2.imwrite(os.path.join(savedir, subdir, file), masked)

time_end = time.time()-time_start
print('processed in %d seconds' %(time_end))