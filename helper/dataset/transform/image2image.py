from dataset.transform.template import TemplateTransform

import random


import torchvision.transforms
import torchvision.transforms.functional

from PIL import Image, ImageDraw, ImageChops

import random
import skimage.io
import cv2
import numpy as np


#class SeamlessCloneTransform(TemplateTransform):
    
    
    
    
    
    
#class ColourTransform(TemplateTransform):
    
    
    
    
    
    
class FourierDomainAdapTransform(TemplateTransform): 
    
    def __init__(self, p=1):
        TemplateTransform.__init__(self, p=p)
        self.p = p
        self.apply_to_mask = False
        
    def _get_random_target_img(self):
        
        
        
        tgt_img = skimage.io.imread(i_path_random, as_gray=True)
        
        return tgt_img
        

    def _change_image(self, keyword="img"):
        # horizontally flip image
        
        
        
        if self.item["has_mask"]:
            
            domain1_img = self.item[keyword]
            domain2_img = self._get_random_target_img()

            adapted_im1 = fda.fda(domain1_img, domain2_img, beta=0.005)

            self.item[keyword] = adapted_im1
        
    
#class MunitTransform(TemplateTransform):