class TwoCropTransform:
    # =============================================================================
    # This function is needed for the supervised contrastive loss
    # we take an image, and return two transformed versions of it
    # =============================================================================
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, item):
        
        tmp1 = {"img" : item["img"]}
        tmp2 = {"img" : item["img"]}
                
        item["img1"] = self.transform(tmp1)["img"]
        item["img2"] = self.transform(tmp2)["img"]
        
        return item