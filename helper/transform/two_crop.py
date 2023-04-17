class TwoCropTransform:
    # =============================================================================
    # This function is needed for the supervised contrastive loss
    # we take an image, and return two transformed versions of it
    # =============================================================================
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img):
        return [self.transform(img), self.transform(img)]