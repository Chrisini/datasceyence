import torch.nn

class DecentBlock_Late_ResNet(torch.nn.Module):
    # =============================================================================
    # =============================================================================

    def __init__(self, num_classes=10):
        super(DecentBlock_ResNet_LateBlock, self).__init__()
        
        self.r50 = resnet50(pretrained=True)
        
        # remove early layers
        self.r50.conv1 = torch.nn.Identity()
        self.r50.bn1 = torch.nn.Identity()
        self.r50.relu = torch.nn.Identity()
        self.r50.maxpool = torch.nn.Identity()
        self.r50.layer1 = torch.nn.Identity()
        self.r50.layer2 = torch.nn.Identity()
        
        self.out_features = self.r50.fc.in_features
                
        # remove head
        self.r50.fc = torch.nn.Identity()
        
    def get_out_features(self):
        return self.out_features

    def forward(self, x):
        block_output = self.r50(x)
        return block_output