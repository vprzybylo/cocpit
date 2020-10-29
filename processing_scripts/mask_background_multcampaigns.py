import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import time

time_start = time.time()

campaigns = ['2000_ARM', '2002_CRYSTAL-FACE-NASA', '2002_CRYSTAL-FACE-UND',\
             '2003_AIRS_II', '2004_Midcix', '2007_ICE_L', 'MPACE']
for campaign in campaigns:
    print(campaign)
    rootdir = '../cpi_data/campaigns/'+campaign+'/good_lowcutoff/'
    savedir = '../cpi_data/campaigns/'+campaign+'/masked_background/'
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

                # save result 
                direct = subdir.split('/')[-1]
                if not os.path.exists(os.path.join(savedir,direct)):
                    os.makedirs(os.path.join(savedir,direct), exist_ok=True)
                cv2.imwrite(os.path.join(savedir,direct,file), final)

    time_end = time.time()-time_start
    print('processed %s campaign in %d seconds' %(campaign, time_end))
                # show images
    #             print('original')
    #             plt.imshow(image_og)
    #             plt.show()
        #         print('thresh')
        #         plt.imshow(thresh)
        #         plt.show()
        #         print('mask')
        #         plt.imshow(mask)
        #         plt.show()
    #             print('new image')
    #             plt.imshow(new_image)
    #             plt.show()
