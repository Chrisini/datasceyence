from visualisation.hook import Hook

import matplotlib.pyplot as plt
import numpy as np

import torch

from PIL import Image

import random


class FeatureMap():
    # =============================================================================
    # ??? https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030
    # ??? https://towardsdatascience.com/convolutional-neural-network-feature-map-and-filter-visualization-f75012a5a49c
    # =============================================================================

    def __init__(self, model, layer, device="cpu", ckpt_net_path=None, iterations=200, lr=1):
        # =============================================================================
        # Initialise iter, lr, model, layer
        # =============================================================================

        # settings for dreams
        self.iterations=iterations
        self.lr=lr
        self.device = device

        # model
        if ckpt_net_path is not None:
            model.load_state_dict(torch.load(ckpt_net_path)["model"]) # 'dir/decentnet_epoch_19_0.3627.ckpt'
        self.model = model.eval()
        
        # the (conv) layer to be visualised
        self.layer = layer
        print("Layer:", self.layer)

    def run(self, img_tensor):
        # =============================================================================
        # Feature map visualisation using hooks       
        # A high activation means a certain feature was found. 
        # A feature map is called the activations of a layer after the convolutional operation.
        # =============================================================================
        
        self.img_tensor = img_tensor

        self.img_tensor = self.img_tensor.to(self.device).unsqueeze_(0)
        hook = Hook(self.layer)
        output = self.model(img_tensor)
        self.feature_maps = hook.output.squeeze()

    def plot(self):
        # =============================================================================
        # plot 15 random feature maps + original image
        # =============================================================================
        fig, axarr = plt.subplots(4, 4)
        plt.figure(figsize=(100,100))
        amount = self.feature_maps.shape[0]
        random_samples = random.sample(range(0, amount), 16)
        counter = 0      
        for idx in range(0, 4):
            for idx2 in range(0, 4):
                axarr[idx, idx2].axis('off')
                axarr[idx, idx2].imshow(self.feature_maps[random_samples[counter]].cpu().detach().numpy())
                counter += 1
                
        # overwrite first image with original image        
        axarr[0,0].imshow(self.img_tensor.squeeze().cpu().detach().numpy().transpose(1, 2, 0))

        plt.close()
    

    
    
    
    
    
    

