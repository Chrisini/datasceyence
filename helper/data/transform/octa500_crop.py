# Import
import warnings
# from deprecated import deprecated


from skimage import io, color
import skimage.io

import copy #Permits deep copying objects

import scipy.io

import cv2
import numpy as np


def octa500_crop(img, msk):
    
    padding = 10
    
    gray = (img).astype(np.uint8)  

    gray = cv2.GaussianBlur(gray,(5,5),0)

    # threshold 
    thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    hh, ww = thresh.shape

    # make bottom 2 rows black where they are white the full width of the image
    thresh[hh-3:hh, 0:ww] = 0

    # get bounds of white pixels
    white = np.where(thresh==255)
    xmin, ymin, xmax, ymax = np.min(white[1]), np.min(white[0]), np.max(white[1]), np.max(white[0])
    #print("xmin, xmax, ymin, ymax:", xmin, xmax, ymin, ymax)

    # crop the image at the bounds adding rows at the top and bottom (y)
    img_cropped = gray[ymin-padding:ymax+padding, xmin:xmax+1]

    #print("gray shape:", gray.shape)
    #print("crop shape:", crop.shape)

    msk_cropped = msk.copy()
    for i in range(len(msk_cropped)):
        msk_cropped[i] = msk_cropped[i]-ymin+padding+1 # +1 looks better ...
        
        
    return img_cropped, msk_cropped, thresh