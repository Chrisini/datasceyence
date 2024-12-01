# =============================================================================
# alphabetic order misc
# =============================================================================
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
# =============================================================================
# torch
# =============================================================================
import torch
import torch.nn as nn
import torchvision # for model
import torch.nn # for torch.nn.Identity


class DecentBaseline(nn.Module):
    def __init__(self, model_kwargs, log_dir="", ckpt_path='') -> None:
        super(DecentBaseline, self).__init__()
        
        self.n_classes = model_kwargs["n_classes"]  
        self.in_channels = model_kwargs["in_channels"]  
        
        # backbone
        self.e_b0 = torchvision.models.efficientnet_b0()
        self.e_b0.features[0][0] = torch.nn.Conv2d(self.in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)        
        
        # extra conv
        output_size = torchvision.models.efficientnet_b0().features[-1][1].num_features 
        self.random_conv = torch.nn.Conv2d(output_size, self.n_classes, kernel_size=(1, 1), stride=(1, 1))

        
    def forward(self, x, mode=None): # normal image, not a real x
        
        x = self.e_b0.features(x) # only take the feature part!!
        x = self.random_conv(x)        
        x = torch.nn.functional.max_pool2d(x, kernel_size=x.size()[2:]) # global max pooling
        x = x.reshape(x.size(0), -1) # respahe or flatten
        
        return x
    
    
    def activations_hook(self, grad):
        # hook for the gradients of the activations
        self.gradients = None
        
    def get_activations_gradient(self):
        return None
    
    def get_activations(self, x):
        return None
    
    def plot_incoming_connections(self, current_epoch=0):
        pass
    
    def plot_outgoing_connections(self, current_epoch=0): # plot_layer_of_1_channel
        pass
        
    def get_cc_and_ci_loss_term(self):      
        return None

    def update(self, current_epoch):        
        pass
        
    def get_everything(self, counter):
        pass
        

