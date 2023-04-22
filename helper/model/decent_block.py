import torch.nn
from model.block.shuffle_block import *

class DecentBlock(torch.nn.Module):
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
    
    def __init__(self, ckpt_early_blocks_path, ckpt_early_blocks, out_channels=5, device="cpu"):
        super(DecentBlock, self).__init__()
        
        self.decent_block = DecentBlock_Shuffle_116OutChannels(ckpt_early_blocks_path, ckpt_early_blocks, out_channels, device)

    def forward(self, x):
        
        return self.decent_block(x)