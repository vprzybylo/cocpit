import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

time_start = time.time()
#could use arg parser here 
rootdir = '../cpi_data/training_datasets/SIFT/good_train/'
savedir = '../cpi_data/training_datasets/SIFT/good_train_masked/'
print('masking backgrounds...')
for subdir, dirs, files in os.walk(rootdir):

    for file in files:

        image_og = cv2.imread(os.path.join(subdir, file), cv2.IMREAD_UNCHANGED)
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
          
            final = cv2.bitwise_and(image_og, image_og, mask=mask)

            if not os.path.exists(os.path.join(savedir)):
                os.makedirs(os.path.join(savedir), exist_ok=True)
            cv2.imwrite(os.path.join(savedir, file), final)


time_end = time.time()-time_start
print('processed %d files in %d seconds' %(len(files), time_end))