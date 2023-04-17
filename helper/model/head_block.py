import os

import torch
import torch.nn as nn
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
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_filters, num_filters2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_filters2, num_filters2),
            nn.LeakyReLU(),
            nn.Linear(num_filters2, self.n_classes),
            nn.ReLU()
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
        
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(num_filters, num_filters2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(num_filters2, num_filters2),
            nn.LeakyReLU(),
            nn.Linear(num_filters2, self.out_classes)
        )
        
    def get_output_size(self):
        return self.out_classes
    
    

          
    
    
        
        
        