from data.transform.transform import *
from data.transform.image2image import *

import os

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
import torchvision.transforms


class TemplateData():
    # =============================================================================
    #
    # Parent Dataset
    # create objects based on child class
    #
    # =============================================================================

    def __init__(self, train_kwargs, model_kwargs):
        super(TemplateData, self).__init__()
        
        self.transforms = torchvision.transforms.Compose(self.get_transforms(train_kwargs))
        
        model_kwargs['n_classes'] = 0
        
        # train dataloader
        self.train_dataloader = None

        # val dataloader
        self.val_dataloader = None

        # test dataloader with batch size
        self.test_dataloader = None 
        
        # same as test dataloader, but batch size = 1
        self.xai_dataloader = None
        
        self.set_data(None, None, train_kwargs)
        
    def set_data(self, train_indices, val_indices, test_indices, trainset, valset, testset, train_kwargs):
        
        # train subset
        train_subset = torch.utils.data.Subset(trainset, train_indices)
        # val subset
        val_subset = torch.utils.data.Subset(valset, val_indices)
        # test subset
        if train_kwargs["test_size"] > 0:
            testset = torch.utils.data.Subset(testset, test_indices)
        
        # train dataloader
        self.train_dataloader = torch.utils.data.DataLoader(train_subset, 
                                                       shuffle=True, 
                                                       batch_size=train_kwargs["batch_size"], 
                                                       num_workers=train_kwargs["num_workers"])

        # val dataloader
        self.val_dataloader = torch.utils.data.DataLoader(val_subset, 
                                                     shuffle=False, 
                                                     batch_size=train_kwargs["batch_size"], 
                                                     num_workers=train_kwargs["num_workers"]
                                                     # , persistent_workers=True
                                                     )
        
        # test dataloader with batch size
        self.test_dataloader = torch.utils.data.DataLoader(testset, 
                                                     shuffle=False, 
                                                     batch_size=train_kwargs["batch_size"], 
                                                     num_workers=train_kwargs["num_workers"]
                                                     # , persistent_workers=True
                                                    )    
        
        # same as test dataloader, but batch size = 1
        self.xai_dataloader = torch.utils.data.DataLoader(testset, 
                                                     shuffle=False, 
                                                     batch_size=1, 
                                                     num_workers=train_kwargs["num_workers"]
                                                     # , persistent_workers=True
                                                    )    
        
    
    def get_transforms(self, train_kwargs):
        # =============================================================================
        # notes:
        #   overwritten for training set
        #   when overwriting a transform, use own function ToTensor instead of transforms.ToTensor
        #   dream_c{label}_{patch_id}.jpg
        # =============================================================================
        
        # gray
        transform_list = [
            transforms.Resize(size=train_kwargs["img_size"]),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
        
        # rgb to gray
        transform_list = [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(size=train_kwargs["img_size"]),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ]
        
        # rgb
        transform_list = [
            transforms.Resize(size=train_kwargs["img_size"]),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ]
        
        # gray to rgb
        transform_list = [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize(size=train_kwargs["img_size"]),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))
        ]
        
        # multiple
        transform_list = [
            ResizeCrop(self.image_size),
            RandomVerticalFlip(p=0.1),
            RandomHorizontalFlip(p=0.5),
            RandomAugmentationsSoft(p=self.p_aug),
            RandomBlur(p=0.3),
            ToTensor(),
            Normalise()
        ]
        return transform_list


class TemplateDataset(Dataset):
    # =============================================================================
    #
    # Parent Dataset
    # create objects based on child class
    #
    # =============================================================================

    def __init__(self, mode="train", index_col=None, channels=1, image_size=500, csv_filenames=["data_ichallenge_amd.csv", "data_ichallenge_non_amd.csv"], p_aug=0.5, reduced_data=False):
        super(TemplateDataset, self).__init__()
        
        self.mode = mode # train/val
        self.image_size = image_size
        
        self.p_aug = p_aug
        
        self.channels=channels
        
        csv_list = []

        for i, filename in enumerate(csv_filenames):
            df = pd.read_csv(filename, delimiter=";", index_col=index_col)
            df["dataset_type"] = [i]*len(df.index)
            csv_list.append(df)

        self.csv_data = pd.concat(csv_list, axis=0, ignore_index=False)
        
        #print("here template")
        #print(self.csv_data)
                
        self.csv_data = self.csv_data[self.csv_data["mode"].str.contains(mode)]
        
        if reduced_data:
            self.csv_data = self.csv_data.sample(frac=1).reset_index(drop=True)
            self.csv_data = self.csv_data.head(200)
        
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

        path = self.csv_data.iloc[index]['image_path']       
        image = Image.open(path)
            
        if self.transforms:
            # apply transforms to both images
            tct = TwoCropTransform(self.transforms)
            image = tct(image)
            
        label = self.csv_data.iloc[index]['lbl']
        
        # To change: you can add labels here
        item = {
            'img' : image,
            'lbl' : label
        } 
        
        return item
    
    def get_mbs_labels(self):
        return list(self.csv_data["lbl"])
    
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
            RandomAugmentationsSoft(p=self.p_aug),
            RandomBlur(p=0.3),
            ToTensor(),
            Normalise()
        ]
        return transform_list

