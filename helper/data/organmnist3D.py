import torch
import torchvision # from torchvision import datasets, transforms
import numpy as np
from sklearn.model_selection import train_test_split
from medmnist import OrganMNIST3D, INFO
from data.template import TemplateDataLoaderWrapper

class DataLoaderOrganMNIST3D(TemplateDataLoaderWrapper):
    def __init__(self, train_kwargs, model_kwargs):
        
        # transforms
        self.transforms = torchvision.transforms.Compose(self.get_transforms(train_kwargs))
        
        # dataset
        trainset = OrganMNIST3D(split="train", transform=self.transforms, download=True)
        valset = OrganMNIST3D(split="val", transform=self.transforms, download=True)
        testset = OrganMNIST3D(split="test", transform=self.transforms, download=True) 
        
        # info
        info = INFO['organmnist3d']
        model_kwargs['n_classes'] = len(info['label'])

        # from parent
        train_indices, val_indices, test_indices = self.get_indices(train_kwargs)
        self.set_data(train_indices=train_indices, val_indices=val_indices, test_indices=test_indices, 
                      trainset=trainset, valset=valset, testset=testset, 
                      train_kwargs=train_kwargs) # TemplateData             
        self.log_info()
    
    def log_info(self):
        
        info = INFO['organmnist3d']
        for value, key in info.items():
            print(value, ":", key)
    
    def get_transforms(self, train_kwargs):
        
        # grayscale (1 channel)
        # doesn't work for 3d rn
        transform_list = [
                          #torchvision.transforms.Resize(size=train_kwargs["img_size"]),
                          torchvision.transforms.ToTensor(),
                          torchvision.transforms.Normalize((0.1307,), (0.3081,))
                         ]

        return transform_list
            
        
        
        
        
