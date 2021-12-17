import numpy as np
import cv2
import imutils
import os

def main():
    open_dir = 'cpi_data/campaigns/OLYMPEX/sheets/'
    save_dir = 'cpi_data/campaigns/OLYMPEX/single_images/'
    write_single_images = True

    for filename in os.listdir(open_dir):
        image = cv2.imread(open_dir + filename)

        #crop for header
        height, width, channels = image.shape
        image = image[25:height, 0:width]

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 225, 255, cv2.THRESH_BINARY_INV)[1]

        # check to see if using OpenCV 2.X
        if imutils.is_cv2() or imutils.is_cv4():
            (cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                             cv2.CHAIN_APPROX_SIMPLE)

            # check to see if using OpenCV 3
        elif imutils.is_cv3():
            (_, cnts, _) = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

            # draw the contours on the image
            #cv2.drawContours(image, cnts, -1, (240, 0, 159), 3)
            #cv2.imshow("Image", image)
            #cv2.waitKey(0)

        for i,c in enumerate(cnts):
                #bounding rectangles to crop
            rect = cv2.boundingRect(c)
            x,y,w,h = rect
                #box = cv2.rectangle(image, (x,y), (x+w,y+h), (0,0,255), 2)
            cropped = image[y: y+h, x: x+w]
            #final_img = np.zeros((1000, 1000, 3), np.uint8)+255
            #final_img[0:cropped.shape[0], 0:cropped.shapesheets

            #cv2.imshow("Show Boxes", cropped)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()
            #print(directory+'singleimages/'+filename[:-4]+'_'+str(i)+'.png')

            #cv2.imwrite(directory[:-7]+'/'+save_dir+'/'
            #            +filename[:-4]+'_'+str(i)+'.png', cropped)
            

