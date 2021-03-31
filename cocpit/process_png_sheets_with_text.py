
import copy
import cv2
from functools import partial
from multiprocessing import Pool
import numpy as np
import os
from time import time
from twilio.rest import Client


class Image:
    """
    Reads, extracts, and performs morphological processing on png sheets of CPI images

    Attributes:
        file (str): file [including path] to sheet with cpi images
    """
    def __init__(self, open_dir, file, save_dir):
        self.open_dir = open_dir
        self.file = file
        self.save_dir = save_dir

    def read_image(self, show_original):
        """
        Reads sheet into mem

        Parameters:
            show_original (bool): whether to show the original sheet in an opencv window
        """
        self.image = cv2.imread(self.open_dir+self.file)
        # make a copy so that image can be altered and processed
        # image_og holds the original sheet to extract contours from
        self.image_og = copy.deepcopy(self.image)
        # crop for header (consistent at 25 pixels from top)
        height, width, channels = self.image.shape
        self.image = self.image[25:height, 0:width]
        self.image_og = self.image_og[25:height, 0:width]
        #uncomment below to better fit image window to screen (shrinks)
        #self.image = cv2.resize(self.image, (0,0), fx=0.8, fy=0.8)

        if show_original:
            cv2.imshow('original', self.image_og)
            cv2.waitKey(0)

    def erode(self, show_erode):
        kernel = np.ones((5,5),np.uint8)
        self.image = cv2.erode(self.image, kernel, iterations = 1)
        if show_erode:
            cv2.imshow("Image", self.image)
            cv2.waitKey(0)

    def dilate(self, show_dilate):
        kernel = np.ones((5,5),np.uint8)
        self.image = cv2.dilate(self.image, kernel, iterations = 1)
        if show_dilate:
            cv2.imshow("Image", self.image)
            cv2.waitKey(0)

    def morphology(self, show_morph):
        kernel = np.ones((5,5),np.uint8)
        self.image = cv2.morphologyEx(self.image, cv2.MORPH_OPEN, kernel)
        if show_morph:
            cv2.imshow("Image", self.image)
            cv2.waitKey(0)

    def remove_text(self):
        """
        Removes text such as date/time stamp by drawing over small contours in white.
        If the text is on the particle itself and connected to the largest contour, it stays
        """
        # first find all contours on sheet, convert to b/w
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        #threshold the binary image
        # contours dependent on hardcoded threshold - some thin white boundaries are 253 si 252 picks these up
        self.thresh = cv2.threshold(gray, 252, 255, cv2.THRESH_BINARY_INV)[1]
        (cnts, _) = cv2.findContours(self.thresh,
                                    cv2.RETR_LIST,
                                    cv2.CHAIN_APPROX_SIMPLE)
        #overlay white contours on white background to cover or mask text
        #only mask contours with small area (aka the text)
        #should a very small particle be < 2000 px^2, it is also masked
        for i, c in enumerate(cnts):
            if cv2.contourArea(c)<2000:
                cv2.drawContours(self.thresh, [c], 0, (255,255,255), -1)  #-1 fills contour

    def connected_component_label(self, show_threshold, show_rois):
        """
        Finds all connected components by traversing the sheet matrix
        after masking text and removing header.
        The resulting connected components are particles (rectangular ROIs) to extract
        Labels each individual component in a different color for visual aid

        Parameters:
            show_threshold (bool): whether to show the thresholded black and white sheet
            show_rois (bool): whether to show each particle (rectangular ROI) in a different color
        """
        #if masking text on self.image, redefine self.thresh here
        # gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # self.thresh = cv2.threshold(gray, 252, 255, cv2.THRESH_BINARY_INV)[1]

        if show_threshold:
           cv2.imshow('thresh', self.thresh)
           cv2.waitKey(0)

        # Applying cv2.connectedComponents()
        num_labels, labels = cv2.connectedComponents(self.thresh)

        # Map component labels to hue val
        label_hue = np.uint8(252*labels/np.max(labels))
        blank_ch = 255*np.ones_like(label_hue)
        self.labeled_img = cv2.merge([label_hue, blank_ch, blank_ch])

        # Converting cvt to BGR
        self.labeled_img = cv2.cvtColor(self.labeled_img, cv2.COLOR_HSV2BGR)
        #cv2.imshow('labeled img', self.labeled_img)
        #cv2.waitKey(0)

        # set background label to white
        self.labeled_img[label_hue==0] = 255

        #Show image after component labeling
        if show_rois:
            self.labeled = cv2.cvtColor(self.labeled_img, cv2.COLOR_BGR2RGB)
            cv2.imshow('rois', self.labeled)
            cv2.waitKey(0)

    def largest_contour(self, cnts):
        '''
        Find largest contour out of list of contours on image

        Parameters:
            cnts (list): list of contours
        '''
        self.largest_cnt= sorted(cnts, key=cv2.contourArea, reverse = True)[0]

    def mask_background(self, cropped, show_mask):
        '''
        Masks out particle background once the ROIs have been cropped
        Places the largest contour on a white background

        Parameters:
            cropped (array): the ROI in which the background should be masked
            show_mask (bool): whether to show the particle after masking the background
        '''
        #draw a the largest contour in white on a black background
        # will be inverted later
        mask = np.zeros(cropped.shape[:2], dtype="uint8")
        cv2.drawContours(mask, [self.largest_cnt], 0, (255,255,255), cv2.FILLED)

        cropped=255-cropped
        # apply mask to image
        masked = cv2.bitwise_and(cropped, cropped, mask=mask)
        masked =255-masked

        if show_mask:
            cv2.imshow("Masked", masked)
            cv2.waitKey(0)
        return masked

    def extract_contours(self, mask, show_cropped, show_mask, save_images):
        """
        Finds, extracts, masks, and saves ROIs

        Parameters:
            show_cropped (bool): whether to show the ROI regions
            save_images (bool): whether to save the final masked images
        """
        (cnts, _) = cv2.findContours(self.thresh,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

        #draw = cv2.drawContours(self.image, cnts, -1, (0,0,255), -1)
        #cv2.imshow("Image", draw)
        #cv2.waitKey(0)
        for i, c in enumerate(cnts):
            #remove small particles that have numbers attached
            #to largest contour
            if cv2.contourArea(c)>2000:
                # crop the rectangles/contours from the sheet 
                rect = cv2.boundingRect(c)
                x, y, w, h = rect
                cropped = self.image_og[y: y+h, x: x+w]

                #resize the cropped images to be the same size for CNN
                cropped = cv2.resize(cropped, (1000,1000), interpolation = cv2.INTER_AREA)

                if show_cropped:
                    cv2.imshow('cropped', cropped)
                    cv2.waitKey(0)

                #converts ROI cropped regions to b/w
                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)[1]
                #find contours within cropped regions
                (cnts, _) = cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

                if cnts: #make sure the thresholding picks up a contour in the rectangle
                    self.largest_contour(cnts)
                    #mask the background so that text is removed if it was 
                    #left behind due to being overlaid

                    if mask:
                        masked = self.mask_background(cropped, show_mask)
                        if save_images:
                            if not os.path.exists(self.save_dir):
                                print('making dir ', self.save_dir)
                                os.makedirs(self.save_dir)
                            cv2.imwrite(self.save_dir+self.file[:-4]+'_'+str(i)+'.png', masked)
                    else:
                        if save_images:
                            if not os.path.exists(self.save_dir):
                                os.makedirs(self.save_dir)
                            cv2.imwrite(self.save_dir+self.file[:-4]+'_'+str(i)+'.png', cropped)

    def run(self, mask, show_original, show_dilate, show_cropped, show_mask, save_images):
        ''' main method calls '''

        self.read_image(show_original)
        #self.erode(show_erode)
        self.dilate(show_dilate)
        self.remove_text()
        #self.connected_component_label(show_threshold, show_rois)
        self.extract_contours(mask, show_cropped, show_mask, save_images)

def send_message():    
    account_sid = "AC6034e88973d880bf2244f62eec6fe356"
    auth_token = 'f374de1a9245649ef5c8bc3f6e4faa97'
    client = Client(account_sid, auth_token)    
    message = client.messages .create(body =  "preprocessing text completed!", 
                                      from_ = "+19285175160", #Provided phone number 
                                      to =    "+15187969534") #Your phone number
    message.sid

def main(open_dir,\
         mask, \
         save_dir, \
         num_cpus, \
         save_images,\
         show_original,\
         show_dilate,\
         show_cropped,\
         show_mask):

    start = time()
    p = Pool(num_cpus)
    instances=[]
    for file in os.listdir(open_dir):
        img = Image(open_dir, file, save_dir)
        #img.main()
        instances.append(img)

    p.map(partial(Image.run, \
                  mask=mask,\
                  show_original=show_original,\
                  show_dilate=show_dilate,\
                  show_cropped=show_cropped,\
                  show_mask=show_mask,\
                  save_images=save_images), instances)
    p.close()

    end = time()
    print('It took ', (end - start)/60,' mins')
    send_message()
