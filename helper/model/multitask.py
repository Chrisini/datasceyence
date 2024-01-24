class MultiTaskNet(nn.Module):
    # =============================================================================
    # Multi Task Classification ResNet    
    # =============================================================================

    def __init__(self, model, heads):
        # =============================================================================
        # Model, with ResNet backbone and multiple outputs:
        #    https://discuss.pytorch.org/t/modify-resnet50-to-give-multiple-outputs/46905
        # dropout with sequential:
        #    https://discuss.pytorch.org/t/resnet-last-layer-modification/33530
        # Layers for multi-task
        #    https://stackoverflow.com/questions/42480111/model-summary-in-pytorch
        #    (classifier): Sequential (
        #    (0): Dropout (p = 0.5)
        #    (1): Linear (25088 -> 4096)
        #    (2): ReLU (inplace)
        #    (3): Dropout (p = 0.5)
        #    (4): Linear (4096 -> 4096)
        #    (5): ReLU (inplace)
        #    (6): Linear (4096 -> 1000)
        #    )
        # =============================================================================
        
        super(EyeNet, self).__init__()
        
        if "resnet50" in model.lower():
            self.model = torchvision.models.resnet50(pretrained=True)
        elif "resnet101" in model.lower():
            self.model = torchvision.models.resnet101(pretrained=True)
        else:
            raise ValueError("Torchvision model not found")
            
        #self.head_name = type(self.heads).__name__
        self.heads = heads
        
        # get feature number (before identity)
        num_filters = self.model.fc.in_features
        num_filters2 = int(num_filters/2)
        
        # replace last layer
        self.model.fc = nn.Identity()
        
        # create new head for each task using module list
        # https://stackoverflow.com/questions/59763775/how-to-use-pytorch-to-construct-multi-task-dnn-e-g-for-more-than-100-tasks
        self.fc = nn.ModuleList([])
        
        for i, head in enumerate(self.heads):
            if "Reg" in head.head_name:
                # ReLU activation at the end of regression
                self.fc.append(nn.Sequential(
                    nn.Dropout(head.dropout),
                    nn.Linear(num_filters, num_filters2),
                    nn.ReLU(),
                    nn.Dropout(head.dropout),
                    nn.Linear(num_filters2, num_filters2),
                    nn.LeakyReLU(),
                    nn.Linear(num_filters2, head.class_amount),
                    nn.ReLU()
                ))
            else:
                # linear layer at the end of classification
                self.fc.append(nn.Sequential(
                    nn.Dropout(head.dropout),
                    nn.Linear(num_filters, num_filters2),
                    nn.ReLU(),
                    nn.Dropout(head.dropout),
                    nn.Linear(num_filters2, num_filters2),
                    nn.LeakyReLU(),
                    nn.Linear(num_filters2, head.class_amount)
                ))
    
    def forward(self, image):
        # =============================================================================
        # model_outputs: dictionary with all outputs: 
        #   {head name : prediction model output for head}
        # =============================================================================
        model_outputs = {}
        # pass image through model
        feature_vector = self.model(image)
        for i, fc_layer, head in enumerate(zip(self.fc, self.heads)):
            # model output of head = pass feature vector through head
            model_outputs[type(head).__name__] = fc_layer(feature_vector)
        return model_outputs