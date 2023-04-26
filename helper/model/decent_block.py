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
    
    def __init__(self, ckpt_early_blocks_path, ckpt_early_blocks, output=2, device="cpu", mode="use"):
        super(DecentBlock, self).__init__()
        
        if "train_mlp" in mode: 
            self.decent_block = DecentBlock_Shuffle_MLP(out_features=output) # 128 features
        elif "train_linear" in mode:
            self.decent_block = DecentBlock_Shuffle_Linear(out_classes=output) # 2 classes
        elif "use" in mode:
            self.decent_block = DecentBlock_Shuffle_116(ckpt_early_blocks_path, ckpt_early_blocks, out_channels=output, device=device) # 5 channels

    def forward(self, x):
        
        return self.decent_block(x)