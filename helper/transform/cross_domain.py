import random

class TemplateTransform(object):
    
    def __init__(self, p=1, image_size=512, zoom_factor=1):
        # default values, change in child class
        self.p = p
        self.apply_to_mask = True
    
    
    def _change_image(self, keyword):
        
        src_img = self.item["img"]
        src_msk = self.item["msk"]
        
        print(random.randint(0,9))
        dst_img = None
        dst_mask = None
        
        new_img = None
        new_msk = None
        
        
        self.item["img"] = new_img
        self.item["msk"] = new_mask
        
    
    def _change_mask(self, keyword="msk"):
        if self.apply_to_mask:
            for key, value in self.item.items():
                if keyword in key:
                    # copy original transform
                    self._change_image(key)