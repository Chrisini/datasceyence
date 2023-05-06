from dataset.template import TemplateDataset
from dataset.transform.transform import *

import torch

import pandas as pd
import torchvision.transforms

import numpy as np

class MeanTeacherTrainDataset(TemplateDataset):
    # =============================================================================
    #
    # val needs different transforms!!!! todo
    #
    # =============================================================================
    
    def __init__(self, mode="train", channels=1, image_size=500, csv_filenames=["data_ichallenge_amd.csv", "data_ichallenge_non_amd.csv"]):
        super().__init__(mode, channels, image_size, csv_filenames)

    def __len__(self):
        return len(self.csv_data)
    
    def __getitem__(self, index):
        # =============================================================================
        # parameters:
        #   index of single image from dataloader
        # returns:
        #   dictionary "item" with:
        #       image (transformed)
        #       label
        # notes:
        # =============================================================================
        
        if torch.is_tensor(index):
            index=index.tolist()

        i_path = self.csv_data.iloc[index]['img_path']    
        if self.channels == 1:
            image = Image.open(i_path).convert('L')
        else:
            image = Image.open(i_path).convert('RGB')
           
        if 'msk_path' in self.csv_data.iloc[index].keys():
            if self.csv_data.iloc[index]['msk_path'] is not np.nan:
                m_path = self.csv_data.iloc[index]['msk_path']
                #print(i_path)
                #print(m_path)
                mask = Image.open(m_path).convert('L')
                has_mask = True
            else: 
                mask = None
                has_mask = False
        else: 
            mask = None
            has_mask = False
                
        weight = self.csv_data.iloc[index]['weight']
        
        dataset_type = self.csv_data.iloc[index]['dataset_type']
        
        # img, lbl_whatever, msk_whatever
        item = {
            'img' : image, # img_s
            # 'img_t' : image,
            'msk' : mask,
            "weight" : weight,
            "mbs_class" : dataset_type, # mixed batch sampler, class for data imbalance handling
            "has_mask" : has_mask
        } 
        
        if self.transforms:
            item = self.transforms(item)
            
        if item["msk"] is None:
            item["msk"] = torch.zeros(1, self.image_size, self.image_size)
            # item["msk"] = torch.empty((0), dtype=torch.float32)
        
        return item
    
    def get_mbs_labels(self):
        # =============================================================================
        # notes:
        #    mixed batch sampler based on the dataset: adam, plex, palm, refuge, ...  
        # =============================================================================
        return list(self.csv_data["dataset_type"])
    
    def get_transforms(self):
        # =============================================================================
        # notes:
        #   overwritten for training set
        #   when overwriting a transform, use own function ToTensor instead of transforms.ToTensor
        #   dream_c{label}_{patch_id}.jpg
        # =============================================================================
        
        transform_list = [
            ResizeCrop(self.image_size),
            RandomVerticalFlip(p=0.1),
            RandomHorizontalFlip(p=0.5),
            RandomAugmentations(p=0.3),
            RandomBlur(p=0.3),
            ToTensor(),
            Normalise()
        ]
        
        return transform_list
    
    
class MeanTeacherValDataset(TemplateDataset):
    # =============================================================================
    #
    # val needs different transforms!!!! todo
    #
    # =============================================================================

    def __init__(self, mode="val", channels=1, image_size=500, csv_filenames=["data_ichallenge_amd.csv", "data_ichallenge_non_amd.csv"]):
        super().__init__(mode, channels, image_size, csv_filenames)
        
         
    def __len__(self):
        return len(self.csv_data)
    
    def __getitem__(self, index):
        # =============================================================================
        # parameters:
        #   index of single image from dataloader
        # returns:
        #   dictionary "item" with:
        #       image (transformed)
        #       label
        # notes:
        # =============================================================================
        
        if torch.is_tensor(index):
            index=index.tolist()

        i_path = self.csv_data.iloc[index]['img_path']    
        if self.channels == 1:
            image = Image.open(i_path).convert('L')
        else:
            image = Image.open(i_path).convert('RGB')
           
        m_path = self.csv_data.iloc[index]['msk_path']
        mask = Image.open(m_path).convert('L')
  
        # img, lbl_whatever, msk_whatever
        item = {
            'img' : image,
            'msk' : mask
        } 
        
        if self.transforms:
            item = self.transforms(item)
        
        return item
    

    
    def get_transforms(self):
        # =============================================================================
        # notes:
        #   overwritten for training set
        #   when overwriting a transform, use own function ToTensor instead of transforms.ToTensor
        #   dream_c{label}_{patch_id}.jpg
        # =============================================================================
        
        transform_list = [
            ResizeCrop(self.image_size),
            ToTensor(),
            Normalise()
        ]
        
        return transform_list