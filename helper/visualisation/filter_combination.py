from visualisation.hook import Hook

import matplotlib.pyplot as plt
import numpy as np

import torch

from PIL import Image

import random

import torch.nn

from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor


# DO NOT USE, MAKES NO SENSE

class FilterCombination():
    # =============================================================================
    # ??? https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030
    # ??? https://towardsdatascience.com/convolutional-neural-network-feature-map-and-filter-visualization-f75012a5a49c
    # =============================================================================

    def __init__(self, model, layer, return_nodes, device="cpu", ckpt_net_path=None, iterations=200, lr=1):
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
        # extracting the model features at the particular layer number
        self.layer = layer
        print("Layer:", self.layer)
        
        self.return_nodes = return_nodes
    

    def run(self):
        
        pass

        
        #print(get_graph_node_names(self.model))

        #a = create_feature_extractor(self.model, return_nodes=self.return_nodes)
        #print(a)
        
    def plot(self):

        #checking whether the layer is convolution layer or not 
        if isinstance(self.layer, torch.nn.Conv2d):
          #getting the weight tensor data
          weight_tensor = self.layer.weight.data #modelfeatures[layer_num].weight.data

          #print(self.layer.weight.data)

          t = weight_tensor
          #kernels depth * number of kernels
          nplots = t.shape[0]*t.shape[1]
          ncols = 12
          
          nrows = 1 + nplots//ncols
          #convert tensor to numpy image
          npimg = np.array(t.cpu().numpy(), np.float32)
          
          ncols = 3
          nrows = 3

          count = 0
          fig = plt.figure(figsize=(ncols, nrows))
          
          #looping through all the kernels in each channel

          print("shape", t.shape)
          for i in range(3): # t.shape[0]
              for j in range(3): # t.shape[1]
                  count += 1
                  ax1 = fig.add_subplot(nrows, ncols, count)
                  npimg = np.array(t[i, j].cpu().numpy(), np.float32)

                  print(np.unique(npimg))

                  # npimg = (npimg - np.mean(npimg)) / np.std(npimg)
                  # npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
                  ax1.imshow(npimg)
                  ax1.set_title(str(i) + ',' + str(j))
                  ax1.axis('off')
                  ax1.set_xticklabels([])
                  ax1.set_yticklabels([])
        
          plt.tight_layout()
          plt.show()
              