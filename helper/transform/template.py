class TemplateTransform(object):
    
    def __init__(self, p=1, image_size=512, zoom_factor=1):
        # default values, change in child class
        self.p = p
        self.image_size = image_size
        self.zoom_factor = zoom_factor
        self.apply_to_mask = False
    
    def __call__(self, item):
        self.item = item
        if random.random() < self.p:
            self._change_image(keyword="img")
            # if mask, change mask
            self._change_mask(keyword="msk")
            # if lat, change lat
            self._change_label_lat(keyword="lat")
        return self.item
    
    def _change_image(self, keyword):
        pass
    
    def _change_label_lat(self, keyword="lat"):
        pass
    
    def _change_mask(self, keyword="msk"):
        if self.apply_to_mask:
            for key, value in self.item.items():
                if keyword in key:
                    # copy original transform
                    self._change_image(key)