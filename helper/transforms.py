class TilesAndCrop(object):
    # =============================================================================
    # Resize and Crop (v1, use middle)
    # Resize all to same size, then crop to get a square
    # =============================================================================

    def __init__(self, image_size=128):
        # =============================================================================
        # x/y = col/row
        # A 0/0 D 1/0 G 2/0
        # B 0/1 E 1/1 H 2/1 
        # C 0/2 F 1/2 I 2/2
        # =============================================================================

        self.resize = transforms.Resize(size=round(image_size))
        self.centercrop = transforms.CenterCrop(size = image_size)

        one_third = math.ceil(image_size/3)
        two_thirds = math.ceil(image_size/3)*2
        three_thirds = math.ceil(image_size/3)*3

        

        self.regions = [(0, 0, one_third, one_third), # A
                        (0, one_third, one_third, two_thirds), # B
                        (0, two_thirds, one_third, three_thirds), # C
                        (one_third, 0, two_thirds, one_third), # D
                        (one_third, one_third, two_thirds, two_thirds), # E
                        (one_third, two_thirds, two_thirds, three_thirds), # F
                        (two_thirds, 0, three_thirds, one_third), # G
                        (two_thirds, one_third, three_thirds, two_thirds), # H
                        (two_thirds, two_thirds, three_thirds, three_thirds)  # I
                        ]

        self.random_region = list(range(0,len(self.regions),1))

    def __call__(self, item):
        # resize to same size
        item['image'] = self.resize(item['image'])
        # crop to get a square
        item['image'] = self.centercrop(item['image'])

        new_image = item['image'].copy()

        random.shuffle(self.random_region)

        for i, region in enumerate(self.regions):

            reg = item['image'].crop(region)

            # random flip of tiles - get rid of any positional context
            if random.random() > 0.6:
                reg = F.vflip(reg)
            if random.random() > 0.6:
                reg = F.hflip(reg)

            new_image.paste(reg, self.regions[self.random_region[i]])

        item['image'] = new_image
        
        return item


# =============================================================================
# testing the transform
# =============================================================================



class RandomHorizontalFlip(object):
    # =============================================================================
    # Random Horizontal Flip (use early)
    # Flips image horizontal and changes label left/right
    # when random value higher than threshold probability p
    # =============================================================================

    def __init__(self, p):
        self.p = p
    
    def __call__(self, item):
        
        if random.random() < self.p:
            # horizontal flip
            item['image'] = F.hflip(item['image'])
            
            for key, value in item.items():
                if "lat" in key: # e.g. meta_lat_number
                    lat_label = item[key]
                    # isApple = True if fruit == 'Apple' else False
                    # change left(0)/right(1) label
                    item[key] = 1 if lat_label == 0 else 0

        return item

# =============================================================================
# testing the transform
# =============================================================================



class RandomBlur(object):
    # =============================================================================
    # Random Blur (use early)
    # sigma: If it is tuple of float (min, max), sigma is chosen randomly 
    # in the given range.
    # =============================================================================

    def __init__(self, configs):
        self.gauss = transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))
        
    def __call__(self, item):
        item['image'] = self.gauss(item['image'])
        return item   

# =============================================================================
# testing the transform
# =============================================================================



class RandomAffine(object):
    # =============================================================================
    # Random Affine
    # todo, might be broken
    # =============================================================================
    
    def __init__(self, configs):
        self.resize = transforms.Resize(size = configs.image_size)
        self.randomaffine = transforms.RandomAffine(configs.affine_rotate,
                                                    translate   = (configs.affine_trans_x, configs.affine_trans_y),
                                                    scale       = (configs.affine_scale_x, configs.affine_scale_y),
                                                    shear       = configs.affine_shear)

    def __call__(self, item):
        image = item['image']
        # resize, so we get some standard size
        image = self.resize(image)
        # degrees, translate, scale, shear, resample, fillcolor
        image = self.randomaffine(image)
        
        item['image'] = image
        return item


# =============================================================================
# testing the transform
# =============================================================================



        
class ResizeCrop(object):
    # =============================================================================
    # Resize and Crop (v1, use middle)
    # Resize all to same size, then crop to get a square
    # =============================================================================

    def __init__(self, configs): # 400
     self.resize = transforms.Resize(size=round(configs.image_size))
     self.centercrop = transforms.CenterCrop(size = configs.image_size)

    def __call__(self, item):
        # resize to same size
        item['image'] = self.resize(item['image'])
        # crop to get a square
        item['image'] = self.centercrop(item['image'])
        return item

# =============================================================================
# testing the transform
# =============================================================================





class ZoomCrop(object):
    # =============================================================================
    # Resize and Crop (v2, use middle)
    # Resize all to same size (by zoom), then crop to get a square
    # =============================================================================

    def __init__(self, configs):
        self.resize = transforms.Resize(size=round(configs.image_size*1.4))
        self.centercrop = transforms.CenterCrop(size=configs.image_size)

    def __call__(self, item):
        # resize to same size
        item['image'] = self.resize(item['image'])
        # crop to get a square
        item['image'] = self.centercrop(item['image'])
        return item


# =============================================================================
# testing the transform
# =============================================================================



    
class ResizeRotateCrop(object):
# =============================================================================
# Resize, Rotate and Crop (v3, use middle)
# Resize all to same size, then crop to get a square
# =============================================================================

    def __init__(self, configs, angle):
        self.resize = transforms.Resize(size=round(configs.image_size))
        self.angle = angle
        self.centercrop = transforms.CenterCrop(size=configs.image_size)

    def __call__(self, item):
        image = item['image']
        # resize to same size
        item['image'] = self.resize(item['image'])
        # rotate image
        item['image'] = F.rotate(item['image'], self.angle)
        # crop to get a square
        item['image'] = self.centercrop(item['image'])
        return item

# =============================================================================
# testing the transform
# =============================================================================





class CircleMask(object):
    # =============================================================================
    # Circle Mask (use before to tensor)   
    # create a circle mask around item to get rid of bias
    # =============================================================================

    def __call__(self, item):
        image = item['image']
        circle_radius = int((image.size[0] - 2)/2)
        black = Image.new('RGB', (image.size), color = "black")
        draw = ImageDraw.Draw(black)
        img_w, img_h = image.size
        topleft = ((img_w / 2) - (circle_radius), (img_h / 2) - (circle_radius))
        downright = ((img_w / 2) + (circle_radius), (img_h / 2) + (circle_radius))
        draw.ellipse((topleft, downright), outline=(0,0,0), fill='white')        
        item['image'] = ImageChops.multiply(image, black)
        return item


# =============================================================================
# testing the transform
# =============================================================================





class ToTensor(object):
    # =============================================================================
    # ToTensor class (use last)
    # Creates a tensor from an image
    # =============================================================================
    
    def __call__(self, item):
        item['image'] = F.to_tensor(item['image'])        
        return item

# =============================================================================
# testing the transform
# =============================================================================





class RandomErasing(object):
    # =============================================================================
    # Random Erasing (use after to tensor)
    # =============================================================================
    
    def __init__(self, configs):
        self.random_erase = transforms.RandomErasing(p=configs.erase_prob, 
                                                     scale=(configs.erase_scale_x, configs.erase_scale_y), 
                                                     ratio=(configs.erase_ratio_x, configs.erase_ratio_y), 
                                                     value=configs.erase_value, 
                                                     inplace=configs.erase_inplace)
        
    def __call__(self, item):
        item['image'] = self.random_erase(item['image'])
        return item

# =============================================================================
# testing the transform
# =============================================================================





class Normalise(object):
    # =============================================================================
    # Min-Max Normalise (use middle)
    # used for both train and validation
    # =============================================================================

    def __init__(self, configs):

        if hasattr(configs, 'normalise'):
            self.normalise = configs.normalise
        else:
            self.normalise = True

    def __call__(self, item):

        if self.normalise == True:
            image = item['image']
            if image.max() == image.min() or image.max() < image.min():
                pass
            else:
                # min max normalisation
                image = (image - image.min()) / (image.max() - image.min())
            item['image'] = image
        return item

# =============================================================================
# testing the transform
# =============================================================================
