import torch
import torchvision # from torchvision import datasets, transforms
import numpy as np
from sklearn.model_selection import train_test_split
from data.template import TemplateData

INFO = {
    "octmnist": {
        "python_class": "OCTA500",
        "description": "The OCTA500 is based on optical coherence tomography (OCT) images for retinal diseases. The dataset is comprised of n diagnosis categories, leading to a multi-class classification task. We split the dataset with a ratio of n:n:n into training, validation and testset. The source images are gray-scale, and their sizes are (n−n)×(n−n). We center-crop the images and resize them into c×w×h.",
        "task": "multi-class",
        "label": {
            "0": "choroidal neovascularization",
            "1": "diabetic retinopathy",
            "2": "drusen",
            "3": "normal",
        },
        "n_channels": 1,
        "n_samples": {"train": 0, "val": 0, "test": 0},
    },
}

class DataOCTA500(TemplateData):
    def __init__(self, train_kwargs, model_kwargs):
        
        # transforms
        self.transforms = torchvision.transforms.Compose(self.get_transforms(train_kwargs))
        
        trainset = OCTMNIST(split="train", transform=self.transforms, download=True)
        valset = OCTMNIST(split="val", transform=self.transforms, download=True)
        testset = OCTMNIST(split="test", transform=self.transforms, download=True) 
        
        info = INFO['octmnist']
        model_kwargs['n_classes'] = len(info['label'])

        # indices for splitting and/or reducing data
        train_indices = range(train_kwargs["train_size"])
        val_indices = range(train_kwargs["val_size"])
        test_indices = range(train_kwargs["test_size"])
        
        # from parent
        self.set_data(train_indices=train_indices, val_indices=val_indices, test_indices=test_indices, 
                      trainset=trainset, valset=valset, testset=testset, 
                      train_kwargs=train_kwargs) # TemplateData     
        
        self.log_info()
    
    def log_info(self):
        
        info = INFO['octmnist']
        for value, key in info.items():
            print(value, ":", key)
    
    def get_transforms(self, train_kwargs):
        
        # grayscale (1 channel)
        transform_list = [
                          torchvision.transforms.Resize(size=train_kwargs["img_size"]),
                          torchvision.transforms.ToTensor(),
                          torchvision.transforms.Normalize((0.1307,), (0.3081,))
                         ]

        return transform_list
            
        
        
        
        
