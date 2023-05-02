from transform.template import TemplateTransform

import random

class SeamlessCloneTransform(TemplateTransform):
    
    def __init__(self, p=1, image_size=512, zoom_factor=1):
        # default values, change in child class
        self.p = p
        self.apply_to_mask = True
        
        print(random.randint(0,9))
        self.dst["img"] = None
        self.dst["mask"] = None
    
    
    def _change_image(self, keyword):
        
        src_img = self.item[keyword]
                
        new_img = None
                
        self.item[keyword] = new_img
        
    
    def _change_mask(self, keyword="msk"):
        if self.apply_to_mask:
            for key, value in self.item.items():
                if keyword in key:
                    # copy original transform
                    self._change_image(key)