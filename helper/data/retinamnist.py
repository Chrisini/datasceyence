import torch
import torchvision # from torchvision import datasets, transforms
import numpy as np
from sklearn.model_selection import train_test_split
from medmnist import INFO, RetinaMNIST
from data.template import TemplateDataLoaderWrapper

class DataLoaderRetinaMNIST(TemplateDataLoaderWrapper):
    def __init__(self, train_kwargs, model_kwargs):
        
        # transforms
        self.transforms = torchvision.transforms.Compose(self.get_transforms(train_kwargs))       
        
        trainset = RetinaMNIST(split="train", transform=self.transforms, download=True)
        valset = RetinaMNIST(split="train", transform=self.transforms, download=True)
        testset = RetinaMNIST(split="test", transform=self.transforms, download=True) 
        
        self.info = INFO['retinamnist']
        model_kwargs['n_classes'] = len(self.info['label'])
        
        train_indices = range(train_kwargs["train_size"])
        val_indices = range(train_kwargs["val_size"])
        test_indices = range(train_kwargs["test_size"])
        
        self.set_data(train_indices=train_indices, val_indices=val_indices, test_indices=test_indices, 
                      trainset=trainset, valset=valset, testset=testset, 
                      train_kwargs=train_kwargs) # TemplateData      
        
        self.log_info()
    
    
    def get_transforms(self, train_kwargs):
        
        # rgb (3 channels) to grayscale (1 channel)
        transform_list = [torchvision.transforms.Grayscale(num_output_channels=1),
                          torchvision.transforms.Resize(size=train_kwargs["img_size"]),
                          torchvision.transforms.ToTensor(),
                          torchvision.transforms.Normalize((0.1307,), (0.3081,))
                         ]
        
        return transform_list
            
        
        
        
        
