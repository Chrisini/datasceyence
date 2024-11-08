# Import
import warnings
# from deprecated import deprecated


from skimage import io, color
import skimage.io

import copy #Permits deep copying objects

import scipy.io

import cv2
import numpy as np


def octa500_resize(img, msk, size=28):

    new_size = (size, size)

    # image
    img_resized = cv2.resize(img, new_size)

    # mask

    try:
        # Calculate scaling factors
        scale_x = new_size[0] / img.shape[1]
        scale_y = new_size[1] / img.shape[0]

        # Scale the coordinates
        # scaled_coordinates = coordinates * [scale_x, scale_y]

        #print("crop shape from threshold:", img.shape)

        # mask_resized_layerid
        
        msk_resized = []
        
        for boundary in msk:
            # list(np.resize(boundary, (new_size)))
            msk_resized.append(np.array(list(zip( boundary, range(len(boundary)) ))) * [scale_y, scale_x])
        
        """
        mask_resized_0 = 
        mask_resized_1 = list(np.resize(msk[1], (new_size)))
        mask_resized_2 = list(np.resize(msk[2], (new_size)))
        mask_resized_3 = list(np.resize(msk[3], (new_size)))
        mask_resized_4 = list(np.resize(msk[4], (new_size)))
        mask_resized_5 = list(np.resize(msk[5], (new_size)))

        #print("length of layer 1:", len(mask_resized_0))

        mask_resized_0 = np.array(list(zip( msk[0], range(len(msk[0])) ))) * [scale_y, scale_x]
        mask_resized_1 = np.array(list(zip( msk[1], range(len(msk[0])) ))) * [scale_y, scale_x]
        mask_resized_2 = np.array(list(zip( msk[2], range(len(msk[0])) ))) * [scale_y, scale_x]
        mask_resized_3 = np.array(list(zip( msk[3], range(len(msk[0])) ))) * [scale_y, scale_x]
        mask_resized_4 = np.array(list(zip( msk[4], range(len(msk[0])) ))) * [scale_y, scale_x]
        mask_resized_5 = np.array(list(zip( msk[5], range(len(msk[0])) ))) * [scale_y, scale_x]
        """
        #print("length of layer 1:", len(mask_resized_0))

        #print("amount of layers:", len(msk))
    except:
        print("potentially no mask found")
        
        
    return img_resized, msk_resized