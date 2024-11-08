import torch
import torchvision # from torchvision import datasets, transforms
import numpy as np
from sklearn.model_selection import train_test_split
from data.template import TemplateDataLoaderWrapper


class DataLoaderMNIST(TemplateDataLoaderWrapper):
    def __init__(self, train_kwargs, model_kwargs):
        
        # transforms
        self.transforms = torchvision.transforms.Compose(self.get_transforms(train_kwargs))
        
        dataset = torchvision.datasets.MNIST('examples/example_data/mnist', train=True, download=True,
                                       transform=self.transforms)
        testset = torchvision.datasets.MNIST('examples/example_data/mnist', train=False, download=True,
                                      transform=self.transforms)
        
        model_kwargs['n_classes'] = len(torchvision.datasets.MNIST.classes)
        
        # indices for splitting and/or reducing data
        indices = np.arange(len(dataset))
        train_indices, val_indices = train_test_split(indices, 
                                                      train_size=train_kwargs["train_size"], 
                                                      test_size=train_kwargs["val_size"], 
                                                      stratify=dataset.targets)
        
        test_indices = range(train_kwargs["test_size"])
        
        self.set_data(train_indices=train_indices, val_indices=val_indices, test_indices=test_indices, 
                      trainset=dataset, valset=dataset, testset=testset, 
                      train_kwargs=train_kwargs) # TemplateData   
        
        self.log_info()
    
    def log_info(self):
        print("MNIST classes:",torchvision.datasets.MNIST.classes)
    
    def get_transforms(self, train_kwargs):
        
        transform_list = [torchvision.transforms.Resize(size=train_kwargs["img_size"]),
                          torchvision.transforms.ToTensor(),
                          torchvision.transforms.Normalize((0.1307,), (0.3081,))
                         ]
        
        return transform_list
            
        
        
        
        
