import torch
import torchvision # from torchvision import datasets, transforms
import numpy as np
from sklearn.model_selection import train_test_split
from medmnist import OCTMNIST # , INFO
from data.template import TemplateDataLoaderWrapper

# we overwrite this, check on github whether the values still apply to the original dataset!
# https://github.com/MedMNIST/MedMNIST/blob/main/medmnist/info.py
INFO = {
    "octmnist": {
        "python_class": "OCTMNIST",
        "description": "The OCTMNIST is based on a prior dataset of 109,309 valid optical coherence tomography (OCT) images for retinal diseases. The dataset is comprised of 4 diagnosis categories, leading to a multi-class classification task. We split the source training set with a ratio of 9:1 into training and validation set, and use its source validation set as the test set. The source images are gray-scale, and their sizes are (384−1,536)×(277−512). We center-crop the images and resize them into 1×28×28.",
        "url": "https://zenodo.org/records/10519652/files/octmnist.npz?download=1", # -> 28
        "MD5": "c68d92d5b585d8d81f7112f81e2d0842", # -> 28
        #"url_64": "https://zenodo.org/records/10519652/files/octmnist_64.npz?download=1",
        #"MD5_64": "e229e9440236b774d9f0dfef9d07bdaf",
        #"url_128": "https://zenodo.org/records/10519652/files/octmnist_128.npz?download=1",
        #"MD5_128": "0a97e76651ace45c5d943ee3f65b63ae",
        #"url_224": "https://zenodo.org/records/10519652/files/octmnist_224.npz?download=1",
        #"MD5_224": "abc493b6d529d5de7569faaef2773ba3", # -> other sizes
        "task": "multi-class",
        "label": {
            "0": "cnv", # cnv + amd?
            "1": "dme", # dr
            "2": "drusen", # amd?
            "3": "normal", # normal
        },
        "n_channels": 1,
        "n_samples": {"train": 97477, "val": 10832, "test": 1000},
        "license": "CC BY 4.0",
    }
}

class DataLoaderOCTMNIST(TemplateDataLoaderWrapper): # TemplateDataLoaderWrapper
    def __init__(self, train_kwargs, model_kwargs):
        
        # transforms
        train_transforms, inference_transforms = self.get_transforms(train_kwargs)
        self.transforms = torchvision.transforms.Compose(train_transforms)
        self.transforms = torchvision.transforms.Compose(inference_transforms)
        
        # dataset
        trainset = OCTMNIST(split="train", transform=self.transforms, download=True)
        valset = OCTMNIST(split="val", transform=self.transforms, download=True)
        testset = OCTMNIST(split="test", transform=self.transforms, download=True) 
        
        # info
        self.info = INFO['octmnist']
        model_kwargs['n_classes'] = len(self.info['label'])

        # from parent
        train_indices, val_indices, test_indices = self.get_indices(train_kwargs)
        self.set_data(train_indices=train_indices, val_indices=val_indices, test_indices=test_indices, 
                      trainset=trainset, valset=valset, testset=testset, 
                      train_kwargs=train_kwargs) # TemplateData             
        self.log_info()

    
    def get_transforms(self, train_kwargs):
        
        # grayscale (1 channel)
        # all the fun stuff
        train_transforms = [
            torchvision.transforms.Resize(size=train_kwargs["img_size"]),
            torchvision.transforms.RandomRotation(degrees=2),
            torchvision.transforms.RandomHorizontalFlip(p=0.5),
            torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1),
            torchvision.transforms.RandomAdjustSharpness(sharpness_factor=1.2, p=train_kwargs["p_augment"]),
            torchvision.transforms.RandomApply([torchvision.transforms.GaussianBlur(kernel_size=3, sigma=(0.5, 1.2))], p=train_kwargs["p_augment"]),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ]
        
        # std
        inference_transforms = [
            torchvision.transforms.Resize(size=train_kwargs["img_size"]),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ]

        return train_transforms, inference_transforms
            
        
        
        
        
