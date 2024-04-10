import torch
import torchvision # from torchvision import datasets, transforms
import numpy as np
from sklearn.model_selection import train_test_split
from medmnist import OCTMNIST
from data.template import TemplateData

class DataOCTMNIST(TemplateData):
    def __init__(self, train_kwargs, model_kwargs):
        
        # transforms
        self.transforms = torchvision.transforms.Compose(self.get_transforms(train_kwargs))
        
        trainset = OCTMNIST(split="train", transform=self.transforms, download=True)
        valset = OCTMNIST(split="val", transform=self.transforms, download=True)
        testset = OCTMNIST(split="test", transform=self.transforms, download=True) 
        
        model_kwargs['n_classes'] = len(trainset.labels)

        # indices for splitting and/or reducing data
        train_indices = range(train_kwargs["train_size"])
        val_indices = range(train_kwargs["val_size"])
        test_indices = range(train_kwargs["test_size"])
                
        self.set_data(train_indices=train_indices, val_indices=val_indices, test_indices=test_indices, 
                      trainset=trainset, valset=valset, testset=testset, 
                      train_kwargs=train_kwargs) # TemplateData     
        
        self.log_info()
    
    def log_info(self):
        import medmnist
        from medmnist import INFO, Evaluator

        info = INFO['octmnist']
        task = info['task']
        n_channels = info['n_channels']
        n_classes = len(info['label'])

        DataClass = getattr(medmnist, info['python_class'])

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
            
        
        
        
        
