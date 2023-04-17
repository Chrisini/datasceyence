import os

import torch
import torch.nn
from torch.autograd import Variable


class DecentBlock_Fusion_Template(torch.nn.Module):
    # =============================================================================
    # template for a fusion module
    # =============================================================================
    
    def forward(self, img):
        return self.fusion_layer(img)
          
class DecentBlock_Fusion_Conv1x1_v1(DecentFusion_Template):
    # =============================================================================
    # conv 1x1
    # =============================================================================
    
    def __init__(self, amount_of_blocks, input_channels=116, output_channels=512):
        
        self.output_channels = output_channels
        
        # layer between decent blocks and combined layers
        # todo, figure out how to get the 116 automatically - 58 * 2
        # fusion_conv
        self.fusion_layer = torch.nn.Conv2d(input_channels*amount_of_blocks, self.output_channels, (1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1))
        
    def get_output_size(self):
        return self.output_channels
        
class DecentBlock_Fusion_Conv1x1_v2(DecentFusion_Template):
    # =============================================================================
    # conv 1x1
    # =============================================================================
    
    def __init__(self, amount_of_blocks, input_channels=116, output_channels=512):
        
        self.output_channels = output_channels
        
        # layer between decent blocks and combined layers
        # todo, figure out how to get the 116 automatically - 58 * 2
        # fusion_conv        
        self.fusion_layer = torch.nn.Sequential(
            torch.nn.Conv2d(input_channels*amount_of_blocks, self.output_channels, (1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1)),
            torch.nn.BatchNorm2d(output_channels),
            torch.nn.ReLU()
        )
        
    def get_output_size(self):
        return self.output_channels