from dataset.transform.template import TemplateTransform

import torchvision.transforms
import torchvision.transforms.functional

from PIL import Image, ImageDraw, ImageChops

import random

import cv2

import numpy as np

class RandomHorizontalFlip(TemplateTransform):
    # =============================================================================
    # Random Horizontal Flip (use early)
    # Flips image horizontal and changes label left/right
    # when random value higher than threshold probability p
    # =============================================================================
    
    def __init__(self, p=0.5):
        TemplateTransform.__init__(self, p=p)
        self.p = p
        self.apply_to_mask = True

    def _change_image(self, keyword):
        # horizontally flip image
        self.item[keyword] = torchvision.transforms.functional.hflip(self.item[keyword])
        
    def _change_label_lat(self, keyword="lat"):
        # isApple = True if fruit == 'Apple' else False
        # change left(0)/right(1) label
        
        for key, value in self.item.items():
            if keyword in key: # e.g. meta_lat_number
                lat_label = self.item[key]
                self.item[key] = 1 if lat_label == 0 else 0
                
class RandomVerticalFlip(TemplateTransform):
    # =============================================================================
    # Random Vertical Flip (use early)
    # when random value higher than threshold probability p
    # do not use if laterality is important
    # =============================================================================
    
    def __init__(self, p=0.5):
        TemplateTransform.__init__(self, p=p)
        self.p = p
        self.apply_to_mask = True

    def _change_image(self, keyword):
        # horizontally flip image
        self.item[keyword] = torchvision.transforms.functional.vflip(self.item[keyword])


class RandomAugmentationsSoft(TemplateTransform):
    # =============================================================================
    # Random Augmentations that have no image transformation
    # brightness, contrast, gamma, sharpness
    # (use early)
    # =============================================================================

    def __init__(self, p=0.5):
        TemplateTransform.__init__(self, p=p)
        self.apply_to_mask=False
        
    def _change_image(self, keyword):
        image = self.item[keyword]
        
        # any non negative number. 0 gives a black image, 1 gives the original image while 2 increases the brightness by a factor of 2
        brightness_factor = random.uniform(0.8, 1.2)
        image = torchvision.transforms.functional.adjust_brightness(image, brightness_factor)

        # any non negative number. 0 gives a solid gray image, 1 gives the original image while 2 increases the contrast by a factor of 2.
        contrast_factor = random.uniform(0.8, 1.2)
        image = torchvision.transforms.functional.adjust_contrast(image, contrast_factor)

        # gamma larger than 1 make the shadows darker, while gamma smaller than 1 make dark regions lighter
        gamma_value = random.uniform(0.8, 1.2)
        image = torchvision.transforms.functional.adjust_gamma(image, gamma_value)

        #any non negative number. 0 gives a blurred image, 1 gives the original image while 2 increases the sharpness by a factor of 2.
        sharpness_factor = random.uniform(0.8, 1.2)
        image = torchvision.transforms.functional.adjust_sharpness(image, sharpness_factor)
        
        if random.random() < 1: # in 20 percent of the cases
            image = torchvision.transforms.functional.invert(image)
        
        self.item[keyword] = image
        
        
class RandomAugmentations(TemplateTransform):
    # =============================================================================
    # Random Augmentations that have no image transformation
    # brightness, contrast, gamma, sharpness
    # (use early)
    # =============================================================================

    def __init__(self, p=0.5):
        TemplateTransform.__init__(self, p=p)
        self.apply_to_mask=False
        
    def _change_image(self, keyword):
        image = self.item[keyword]
        
        if self.item["has_mask"]:
            # any non negative number. 0 gives a black image, 1 gives the original image while 2 increases the brightness by a factor of 2
            brightness_factor = random.uniform(0.5, 1.5)
            image = torchvision.transforms.functional.adjust_brightness(image, brightness_factor)

            # any non negative number. 0 gives a solid gray image, 1 gives the original image while 2 increases the contrast by a factor of 2.
            contrast_factor = random.uniform(0.5, 1.5)
            image = torchvision.transforms.functional.adjust_contrast(image, contrast_factor)

            # gamma larger than 1 make the shadows darker, while gamma smaller than 1 make dark regions lighter
            gamma_value = random.uniform(0.5, 1.5)
            image = torchvision.transforms.functional.adjust_gamma(image, gamma_value)

            #any non negative number. 0 gives a blurred image, 1 gives the original image while 2 increases the sharpness by a factor of 2.
            sharpness_factor = random.uniform(0.5, 1.5)
            image = torchvision.transforms.functional.adjust_sharpness(image, sharpness_factor)

            if random.random() < 1: # in 20 percent of the cases
                image = torchvision.transforms.functional.invert(image)

            self.item[keyword] = image
        
class RandomBlur(TemplateTransform):
    # =============================================================================
    # Random Blur (use early)
    # sigma: If it is tuple of float (min, max), sigma is chosen randomly 
    # in the given range.
    # also in previous random augmentations
    # =============================================================================

    def __init__(self, p=0.2):
        TemplateTransform.__init__(self, p=p)
        self.apply_to_mask=False
        self.gauss = torchvision.transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 5))
        
    def _change_image(self, keyword):
        if self.item["has_mask"]:
            self.item[keyword] = self.gauss(self.item[keyword])

        
class ResizeCrop(TemplateTransform):
    # =============================================================================
    # Resize and Crop (v1, use middle)
    # Resize all to same size, then crop to get a square
    # zoom factor possible
    # =============================================================================

    def __init__(self, image_size=512, zoom_factor=1):
        TemplateTransform.__init__(self, image_size=image_size, zoom_factor=zoom_factor)
        self.apply_to_mask=True
        self.resize = torchvision.transforms.Resize(size=round(image_size*zoom_factor))
        self.centercrop = torchvision.transforms.CenterCrop(size=image_size)

    def _change_image(self, keyword):
        # resize to same size
        self.item[keyword] = self.resize(self.item[keyword])
        # crop to get a square
        self.item[keyword] = self.centercrop(self.item[keyword])
        

class TileCrop(TemplateTransform):
    # =============================================================================
    # Resize and Crop (v1, use middle)
    # Crop tiles
    # =============================================================================

    def __init__(self, tile_size=224, zoom_factor=1, n_tiles=5):
        TemplateTransform.__init__(self)
        
        self.fivecrop = torchvision.transforms.FiveCrop(size=tile_size)

    def _change_image(self, keyword):
        # five crop
        self.item[keyword] = self.fivecrop(self.item[keyword])[1]
        

class MaskCrop(TemplateTransform):
    
    def __init__(self, image_size=512):
        TemplateTransform.__init__(self, image_size=image_size)
    
    def __call__(self, item):
        self.item = item
        if self.item["mask_crop"]: # only apply if mask crop is true
            self._change_image_and_mask()
        return self.item
    
    def _change_image_and_mask(self, crop_number=700):
        
        img = self.item["img"]
        msk = self.item["msk"]
        
        center_tmp = int(img.shape[1]/2)
        crop_tmp = 200
        
        if len(img.shape) == 2:
            gray = int(np.mean(img[center_tmp-crop_tmp:center_tmp+crop_tmp, center_tmp-crop_tmp:center_tmp+crop_tmp])) # img.mean(axis=0).mean(axis=0) 
            img = cv2.copyMakeBorder(img, 700, 700, 700, 700, cv2.BORDER_CONSTANT, None, value = gray)
            img[np.where((img==0))] = gray
        else:
            gray = img.mean(axis=0).mean(axis=0)
            # image, recolour black part
            img = cv2.copyMakeBorder(img, 700, 700, 700, 700, cv2.BORDER_CONSTANT, None, value = gray)
            img[np.where((img==[0,0,0]).all(axis=2))] = gray
            
        # print("gray",gray)

        # get center point of RoI
        msk = cv2.copyMakeBorder(msk, 700, 700, 700, 700, cv2.BORDER_REPLICATE)
        if len(msk.shape) != 2:
            msk = cv2.cvtColor(msk, cv2.COLOR_BGR2GRAY)
        num_labels, labels_im = cv2.findContours(msk, cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)    
        if len(num_labels) > 0:
            max_area = max(num_labels, key = cv2.contourArea)
            # cv2.drawContours(msk,[max_area],0,255,-1)
            # print(max_area)

            # calculate moments of binary image
            M = cv2.moments(max_area)
            # calculate x, y coordinate of center
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            try:
                # crop
                tmp_img = img[cY-crop_number:cY+crop_number, cX-crop_number:cX+crop_number]
                tmp_msk = msk[cY-crop_number:cY+crop_number, cX-crop_number:cX+crop_number]
                tmp_img = cv2.resize(tmp_img, (self.image_size, self.image_size), interpolation = cv2.INTER_AREA)
                tmp_msk = cv2.resize(tmp_msk, (self.image_size, self.image_size), interpolation = cv2.INTER_AREA) 
                
                self.item["img"] = tmp_img
                self.item["msk"] = tmp_msk
                
            except:
                pass

        else:
            print("no label available, transform file")
            
            tmp_img = img
            tmp_msk = msk
            

        

        
        return self.item
        

class CircleMask(TemplateTransform):
    # =============================================================================
    # Circle Mask (use before to tensor)   
    # create a circle mask around image
    # useful for classification
    # =============================================================================
    
    def __init__(self):
        TemplateTransform.__init__(self)
        self.apply_to_mask=False

    def _change_image(self, keyword):
        image = self.item[keyword]
        circle_radius = int((image.size[0] - 2)/2)
        black = Image.new('RGB', (image.size), color = "black")
        draw = ImageDraw.Draw(black)
        img_w, img_h = image.size
        topleft = ((img_w / 2) - (circle_radius), (img_h / 2) - (circle_radius))
        downright = ((img_w / 2) + (circle_radius), (img_h / 2) + (circle_radius))
        draw.ellipse((topleft, downright), outline=(0,0,0), fill='white')        
        self.item[keyword] = ImageChops.multiply(image, black)

class ToTensor(TemplateTransform):
    # =============================================================================
    # ToTensor class (use last)
    # Creates a tensor from an image
    # =============================================================================
    
    def __init__(self):
        TemplateTransform.__init__(self)
        self.apply_to_mask=True
    
    def _change_image(self, keyword):
        self.item[keyword] = torchvision.transforms.functional.to_tensor(self.item[keyword])        
        
class ToPillow(TemplateTransform):
    # =============================================================================
    # ToTensor class (use last)
    # Creates a tensor from an image
    # =============================================================================
    
    def __init__(self):
        TemplateTransform.__init__(self)
        self.apply_to_mask=True
    
    def _change_image(self, keyword):
        self.item[keyword] = torchvision.transforms.functional.to_pil_image(self.item[keyword])        

class RandomErasing(TemplateTransform):
    # =============================================================================
    # Random Erasing (use after to tensor)
    # useful for classification
    # =============================================================================
    
    def __init__(self, p=0.5):
        TemplateTransform.__init__(self, p=p)
        self.apply_to_mask=False
        self.random_erase = torchvision.transforms.RandomErasing(p)
        
    def _change_image(self, keyword):
        self.item[keyword] = self.random_erase(self.item[keyword])

class Normalise(TemplateTransform):
    # =============================================================================
    # Min-Max Normalise (use last)
    # used for both train and inference
    # =============================================================================
    
    def __init__(self):
        TemplateTransform.__init__(self)
        self.apply_to_mask=False

    def _change_image(self, keyword):
        
        image = self.item[keyword]
        if image.max() == image.min() or image.max() < image.min():
            pass
        else:
            # min max normalisation
            image = (image - image.min()) / (image.max() - image.min())
            self.item[keyword] = image
