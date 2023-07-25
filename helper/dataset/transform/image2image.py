from dataset.transform.template import TemplateTransform

import random


import torchvision.transforms
import torchvision.transforms.functional

from PIL import Image, ImageDraw, ImageChops

import random
import skimage.io
import cv2
import numpy as np

import fda
from skimage.color import gray2rgb, rgb2gray

#class SeamlessCloneTransform(TemplateTransform):
    
    
    
    
    
    
#class ColourTransform(TemplateTransform):
    
    
    
    
    
    
class FourierDomainAdapTransform(TemplateTransform): 
    
    def __init__(self, p=1, tgt_paths=None, channels=1, image_size=512):
        TemplateTransform.__init__(self, p=p)
        self.p = p
        self.apply_to_mask = False
        self.tgt_paths = tgt_paths
        self.channels = channels
        self.image_size = image_size
        
        # print("transform: self.tgt_paths", self.tgt_paths)
        
    def _get_random_tgt_img(self):
        
        i_path = random.choice(list(self.tgt_paths))

        if self.channels == 1:
            # image = Image.open(i_path).convert('L')
            if ".tif" in i_path:
                tgt_img = skimage.io.imread(i_path, as_gray=True, plugin='tifffile')
            else:
                tgt_img = skimage.io.imread(i_path, as_gray=True)

        else:
            # image = Image.open(i_path).convert('RGB')
            if ".tif" in i_path:
                tgt_img = skimage.io.imread(i_path, as_gray=False, plugin='tifffile')
            else:
                tgt_img = skimage.io.imread(i_path, as_gray=False)

        #import matplotlib.pyplot as plt
        #plt.figure()
        #plt.imshow(tgt_img)
        #plt.title("the cirrus thing, orig")
        
        tgt_img = cv2.resize(tgt_img, (self.image_size, self.image_size), interpolation = cv2.INTER_AREA)
        
        #import matplotlib.pyplot as plt
        #plt.figure()
        #plt.imshow(tgt_img)
        #plt.title("the cirrus thing, 128 x 128")
        
        
        return tgt_img
    
    def _change_image(self, keyword="img"):
        # just use if source data aka has mask
        
        if self.item["has_mask"] and self.tgt_paths is not None:
            
            domain1_img = self.item[keyword]
            domain2_img = self._get_random_tgt_img()
        
            #print(len(domain1_img.shape))
            #print(len(domain2_img.shape))
            
            if len(domain1_img.shape) == 2:
                domain1_img = gray2rgb(domain1_img)
            if len(domain2_img.shape) == 2:
                domain2_img = gray2rgb(domain2_img)
                
            adapted_im1 = fda.fda(domain1_img, domain2_img, beta=0.5)
            
            """
            LB: beta in the paper, controls the size of the low frequency window to be replaced.
            entW: weight on the entropy term.
            ita: coefficient for the robust norm on entropy.
            switch2entropy: entropy minimization kicks in after this many steps.
            """
            
            #import matplotlib.pyplot as plt
            #plt.figure()
            #plt.imshow(adapted_im1)
            #plt.title("adapted")
            
            
            adapted_im1 = rgb2gray(adapted_im1)
            adapted_im1 = skimage.util.img_as_ubyte(adapted_im1, force_copy=False)
            
            self.item[keyword] = adapted_im1
            
        
    
#class MunitTransform(TemplateTransform):