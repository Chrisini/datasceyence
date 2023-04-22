import os

import torch
import torch.nn
from torch.autograd import Variable


class DecentBlock_Head_Template(torch.nn.Module):
    
    def forward(self, img):
        return self.fc(img)


class DecentBlock_Head_Regression(DecentBlock_Head_Template):
    # =============================================================================
    # template for a fusion module
    # =============================================================================
    
    def __init__(self, in_channels, n_classes, dropout=0.5):
        
        self.out_classes = out_classes
        
        # get feature number (before identity)
        num_filters = in_channels
        num_filters2 = int(num_filters/2)
        
        # ReLU activation at the end of regression
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(num_filters, num_filters2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(num_filters2, num_filters2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(num_filters2, self.n_classes),
            torch.nn.ReLU()
        )
        
    def get_output_size(self):
        return self.out_classes
    
        
class DecentBlock_Head_Classification(DecentBlock_Head_Template):
    # =============================================================================
    # template for a fusion module
    # =============================================================================
    
    def __init__(self, in_channels, out_classes):
        
        self.out_classes = out_classes
        
        # get feature number (before identity)
        num_filters = in_channels
        num_filters2 = int(num_filters/2)
        
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(dropout),
            torch.nn.Linear(num_filters, num_filters2),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(num_filters2, num_filters2),
            torch.nn.LeakyReLU(),
            torch.nn.Linear(num_filters2, self.out_classes)
        )
        
    def get_output_size(self):
        return self.out_classes
    
    

          
    
    
        
        
        