from dataset.template import TemplateDataset
from dataset.transform.transform import *

import torch

import pandas as pd
import torchvision.transforms

class MeanTeacherDataset(TemplateDataset):
    # =============================================================================
    #
    # Parent Dataset
    # create objects based on child class
    #
    # =============================================================================

    def __init__(self, mode, channels=1, image_size=500, csv_filenames=["data_ichallenge_amd.csv", "data_ichallenge_non_amd.csv"]):
        super(TemplateDataset, self).__init__()
        
        self.mode = mode # train/val
        self.image_size = image_size
        
        self.channels=channels
        
        csv_list = []

        for i, filename in enumerate(csv_filenames):
            df = pd.read_csv(filename, delimiter=";")
            df["dataset_type"] = [i]*len(df.index)
            csv_list.append(df)

        self.csv_data = pd.concat(csv_list, axis=0, ignore_index=True)
                
        self.csv_data = self.csv_data[self.csv_data["mode"].str.contains(mode)]
        
        self.transforms = torchvision.transforms.Compose(self.get_transforms())
         
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

        path = self.csv_data.iloc[index]['img_path']    
        if self.channels == 1:
            image = Image.open(path).convert('L')
        else:
            image = Image.open(path).convert('RGB')
            
        mask = self.csv_data.iloc[index]['msk_path']
        mask = Image.open(path).convert('L')
        
        weight = self.csv_data.iloc[index]['weight']
        
        dataset_type = self.csv_data.iloc[index]['dataset_type']
        
        """
                
        # 's_inp', 't_inp' and 'gt' are tuples
            s_inp, t_inp, gt = self._batch_prehandle(inp, gt, True) # this should be in dataloader
        """
        
        # To change: you can add labels here
        item = {
            'img' : image, # img_s
            # 'img_t' : image,
            'msk' : mask,
            "weight" : weight,
            "mbs_value" : dataset_type 
        } 
        
        if self.transforms:
            item = self.transforms(item)
        
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
            RandomAugmentations(),
            ToTensor(),
            Normalise()
        ]
        
        return transform_list