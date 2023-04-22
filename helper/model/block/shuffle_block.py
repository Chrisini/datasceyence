import os

import torch
import torch.nn
from torch.autograd import Variable
from torchvision.models import resnet50, shufflenet_v2_x1_0, ShuffleNet_V2_X1_0_Weights

class DecentBlock_Shuffle_MLP(torch.nn.Module):
    # =============================================================================
    # use this
    # encoder backbone (shufflenet_v2_x1_0) + projection head (MLP)
    # output size of MLP: 128 
    # this feature vector is then used for the SupConLoss

    # Replace model head:
    # https://discuss.pytorch.org/t/how-to-replace-a-models-head/109002/2
    # shuffle net: (fc): Linear(in_features=1024, out_features=1000, bias=True)

    # this function is used to train a DecentBlock with the SupConLoss
    # =============================================================================
    
    def __init__(self, out_features=128):
        super(DecentBlock_Shuffle_MLP, self).__init__()
        
        self.out_features = out_features
        
        # encoder
        try:
            shufflenet = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        except: 
            shufflenet = shufflenet_v2_x1_0(pretrained=True)
                    
        # size of encoder output
        encoder_out = shufflenet.fc.in_features
            
        # placeholder identity operator 
        shufflenet.fc = torch.nn.Identity()
                
        # encoder without fully connected classification head (linear layer)
        self.encoder = shufflenet
        
        # MLP head
        self.mlp_head = torch.nn.Sequential(
            torch.nn.Linear(encoder_out, encoder_out),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(encoder_out, self.out_features)
        )
        
    def get_output_size(self):
        return self.out_features

    def forward(self, x):
        x = self.encoder(x)
        x = self.mlp_head(x)
        block_output = torch.nn.functional.normalize(x, dim=1)
        return block_output
    
class DecentBlock_Shuffle_116OutChannels(torch.nn.Module):
    # =============================================================================
    # use this
    # encoder backbone (shufflenet_v2_x1_0) + projection head (MLP)
    # output size of Net: 116 Channels
    # this feature vector is then used for the SupConLoss

    # Replace model head:
    # https://discuss.pytorch.org/t/how-to-replace-a-models-head/109002/2
    # shuffle net: (fc): Linear(in_features=1024, out_features=1000, bias=True)

    # this class is used as early block, needs an already trained ShuffleNet
    # =============================================================================
    
    def __init__(self, ckpt_early_blocks_path=None, ckpt_early_blocks=None, out_channels=5, device="cpu"):
        super(DecentBlock_Shuffle_116OutChannels, self).__init__()
        
        self.out_channels = out_channels
        
        # we use a shufflenet pretrained on data from a previous step
        # these layer should be frozen during training of this model
        decent_block = shufflenet_v2_x1_0()
        decent_block.fc = torch.nn.Identity()

        # load shuffle net weights here
        if ckpt_early_blocks_path is not None and ckpt_early_blocks is not None:
            checkpoint = torch.load(os.path.join(ckpt_early_blocks_path, ckpt_early_blocks))
            decent_block.load_state_dict(checkpoint['model'])

        # needs to be in torch.nn.Seq for some reason ...
        # frozen
        self.decent_block_116 = torch.nn.Sequential(*(list(decent_block.children())[:-4])).to(device)
        self.decent_block_116.requires_grad_(False) # todo check whether this works

        # trainable
        self.decent_block_reduction = torch.nn.Sequential(
                                              torch.nn.Conv2d(116, self.out_channels, (1, 1), stride=(1, 1), padding=(0, 0), dilation=(1, 1)),
                                              torch.nn.BatchNorm2d(self.out_channels),
                                              torch.nn.ReLU()
                                              ).to(device)
        
        if False:
            print("original")
            print("*"*50)
            print(decent_block)
            print("116")
            print("*"*50)
            print(self.decent_block_116)
            
    def get_output_size(self):
        return self.out_channels
        
    def forward(self, x):
        
        x = self.decent_block_116(x)
        block_output = self.decent_block_reduction(x)
        
        return block_output 


class DecentBlock_Shuffle_Linear(torch.nn.Module):
    # =============================================================================
    # encoder backbone (shufflenet_v2_x1_0) + classification head (Linear)
    # output size of classifier: n (amount of classes) 
    
    # this class is used to fine-tune a DecentBlock with e.g. CrossEntropyLoss
    # this is currently not in use
    # =============================================================================

    def __init__(self, out_classes=10):
        super(DecentBlock_Shuffle_Linear, self).__init__()
        
        self.num_classes
        
        # encoder
        try:
            shufflenet = shufflenet_v2_x1_0(weights=ShuffleNet_V2_X1_0_Weights.IMAGENET1K_V1)
        except: 
            shufflenet = shufflenet_v2_x1_0(pretrained=True)
            
        # size of encoder output
        encoder_out = shufflenet.fc.in_features
        
        # encoder without fully connected classification head (linear layer)
        self.encoder = shufflenet
        self.encoder.fc = torch.nn.Linear(encoder_out, self.out_classes)
                
        
    def get_output_size(self):
        return self.out_classes

    def forward(self, x):
        block_output = self.encoder(x)
        return block_output