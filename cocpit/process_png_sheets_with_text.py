import copy
import cv2
from functools import partial
from multiprocessing import Pool
import numpy as np
import pandas as pd
import os
from twilio.rest import Client

class Image:
    """
    Reads, extracts, and performs morphological processing on png sheets of CPI images

    Attributes:
        file (str): file [including path] to sheet with cpi images
    """
    def __init__(self, open_dir, file, save_dir, save_df):
        self.open_dir = open_dir
        self.file = file
        self.save_dir = save_dir
        self.save_df = save_df

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
        #sheet without header
        self.image = self.image[25:height, 0:width]
        #make an unchanged copy 
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
        #should a very small particle be < 200 px^2, it is also masked
        for i, c in enumerate(cnts):
            if cv2.contourArea(c)<200:
                cv2.drawContours(self.thresh, [c], 0, (255,255,255), -1)  #-1 fills contour

    def connected_component_label(self, show_threshold, show_rois):
        """
        Finds all connected components by traversing the sheet matrix
        after removing text and header.
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
        
    def cutoff(self):
        '''
        Determines the % of pixels that intersect
        the border or perimeter of the image
        '''
        #checking the percentage of the contour that touches the edge/border
        locations = np.where(self.thresh != 0)
        count = 0 #pixels touching border
        for xl,yl in zip(locations[0], locations[1]):
            if xl == 5 or yl == 5 or xl == self.height-5 or yl == self.width-5:
                #cv2.circle(self.im, (yl, xl), 1, (255,0,0), 4)
                #cv2.circle(self.thresh, (yl, xl), 1, (255,0,0), 4)
                count+=1
        cutoff_perc = (count/(2*self.height+2*self.width))*100
        return cutoff_perc

        
    def extract_contours(self, cutoff, show_cropped, save_images):
        """
        Finds, extracts and saves ROIs from sheets
        Saves filename, width, height, and cutoff to lists for a df

        Parameters:
            mask (bool): whether to mask the background of the particles
            show_cropped (bool): whether to show the ROI regions
            save_images (bool): whether to save the final images
        """
        
        files = [] # only save the filenames to csv that pass the following criteria
        widths = []
        heights = []
        cutoffs = []
        
        (cnts, _) = cv2.findContours(self.thresh,
                                cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)

        # draw = cv2.drawContours(self.image, cnts, -1, (0,0,255), -1)
        # cv2.imshow("Image", draw)
        # cv2.waitKey(0)
        for i, c in enumerate(cnts):
            # remove small particles that have numbers attached
            # to largest contour
            if cv2.contourArea(c)>200:
                # crop the rectangles/contours from the sheet 
                # save width and height for cutoff calculation
                rect = cv2.boundingRect(c)
                x, y, self.width, self.height = rect
                
                cropped = self.image_og[y: y+self.height, x: x+self.width]
                
                if show_cropped:
                    cv2.imshow('cropped', cropped)
                    cv2.waitKey(0)

                # converts ROI cropped regions to b/w
                # overwrites self.thresh from whole sheet to particle rectangle
                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                self.thresh = cv2.threshold(gray, 40, 255, cv2.THRESH_BINARY_INV)[1]
                
                # find contours within cropped regions
                (cnts, _) = cv2.findContours(self.thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

                if cnts and self.cutoff() < cutoff:  # make sure the thresholding picks up a contour in the rectangle
                    
                    # get cutoff of each particle and append to list
                    cutoffs.append(self.cutoff())

                    # resize the cropped images to be the same size for CNN
                    cropped = cv2.resize(cropped, (1000,1000), interpolation = cv2.INTER_AREA)
                    
                    self.file_out=self.file[:-4]+'_'+str(i)+'.png'
                    
                    files.append(self.file_out)
                    widths.append(self.width)
                    heights.append(self.height)
                    
                    self.largest_contour(cnts)

                    if save_images:
                        if not os.path.exists(self.save_dir):
                            print('making dir ', self.save_dir)
                            os.makedirs(self.save_dir)
                        cv2.imwrite(self.save_dir+self.file_out, cropped)
                        
        return files, widths, heights, cutoffs

    def run(self, cutoff, show_original,
            show_dilate, show_cropped, save_images):
        
        ''' main method calls '''

        self.read_image(show_original)
        #self.erode(show_erode)
        self.dilate(show_dilate)
        self.remove_text()
        #self.connected_component_label(show_threshold, show_rois)
        files, widths, heights, cutoffs = self.extract_contours(cutoff,
                                                                 show_cropped,
                                                                 save_images)
        
        return files, widths, heights, cutoffs

def make_df(files, widths, heights, cutoffs):
        """
        write files, original image widths, heights, and % cutoff
        to csv file per campaign
        """
        cutoffs_formatted = [ '%.2f' % elem for elem in cutoffs]
        df_dict = {'filename': files, 'width': widths,
                  'height': heights, 'cutoff': cutoffs_formatted}
        df = pd.DataFrame(df_dict)
        len_before = len(df)
        df.drop_duplicates(subset=['width', 'height','cutoff'], keep='first', inplace=True)
        print('removed %d duplicates' %(len_before - len(df)))
    
        df.to_csv(self.save_df)

def send_message():    
    account_sid = "AC6034e88973d880bf2244f62eec6fe356"
    auth_token = 'f374de1a9245649ef5c8bc3f6e4faa97'
    client = Client(account_sid, auth_token)    
    message = client.messages .create(body =  "preprocessing text completed!", 
                                      from_ = "+19285175160", #Provided phone number 
                                      to =    "+15187969534") #Your phone number
    message.sid

def main(open_dir, cutoff, save_dir, num_cpus,
         save_images, save_df, show_original,
         show_dilate, show_cropped):

    p = Pool(num_cpus)
    instances=[]
    files = os.listdir(open_dir)
    for file in files:
        img = Image(open_dir, file, save_dir, save_df)
        #img.main()
        instances.append(img)

    files, widths, heights, cutoffs = p.map(partial(Image.run, 
                                          cutoff=cutoff,
                                          show_original=show_original,
                                          show_dilate=show_dilate,
                                          show_cropped=show_cropped,
                                          save_images=save_images), instances)
    p.close()
    p.join()
    make_df(files, widths, heights, cutoffs)
    send_message()
