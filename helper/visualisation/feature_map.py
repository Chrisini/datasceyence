import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import torch
import os

class FeatureMap():
    # =============================================================================
    # unit test: 
    #    examples/utest_vis_feature_map.ipynb
    # sources:
    #    https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030
    #    https://towardsdatascience.com/convolutional-neural-network-feature-map-and-filter-visualization-f75012a5a49c
    #    https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/4
    # =============================================================================

    def __init__(self, model, layer, layer_str, log_dir, device="cpu"):
        # =============================================================================
        # Set model and layer
        # =============================================================================

        self.device = device
        self.log_dir = log_dir

        # model
        self.model = model.to(device).eval()
        
        # the (conv) layer to be visualised
        self.layer = layer
        self.layer_str = layer_str
        
        #print("")
        #print("Layer:", self.layer)
        #print("")

    def run(self, img_tensor, batch_idx):
        # =============================================================================
        # Feature map visualisation using hooks       
        # A high activation means a certain feature was found. 
        # A feature map is called the activations of a layer after the convolutional operation.
        # =============================================================================
        
        #print("i", img_tensor.data.shape)
        self.ii = img_tensor.data
        self.batch_idx = batch_idx
            
        # hook = Hook(module=self.layer)
        
        active = {}
        def get_active(name):
            def hook(model, input, output): # hi
                active[name] = output.data.detach()
            return hook


        model = self.model.eval()
        self.layer.register_forward_hook(get_active(self.layer_str))
        
        try:
            output = self.model(img_tensor, mode='explain')
        except:
            output = self.model(img_tensor)
        
        
        self.feature_maps = active[self.layer_str]

        print('self.feature_maps', self.feature_maps.data.shape)
        
        '''
        output = self.model(img_tensor, mode='explain')
        self.feature_maps = hook.output.data # .squeeze()
        
        print('o', output.shape)
        print('i', self.ii.shape)
        print('h', hook.output)
        print('f', self.feature_maps.shape)
        '''

    def log(self):
        # =============================================================================
        # plot and save 15 random feature maps + original image
        # =============================================================================
        
        # plt.figure(figsize=(100,100))
        amount = self.feature_maps.shape[1]
        print("amount of feature maps:", amount)
        if amount < 9:
            sample_amount = amount
            y_axis = 3
            x_axis = 3
            # if x_axis == 1: x_axis = 2
        elif amount < 16:
            sample_amount = amount
            y_axis = 4
            x_axis = 4
        else:
            sample_amount = 16
            x_axis = 4
            y_axis = 4
        
        fig, axarr = plt.subplots(x_axis, y_axis)
            
        # currently not random
        random_samples = range(0, amount) # random.sample(range(0, amount), sample_amount)
        print("random_samples", random_samples)
        counter = 0  
        idx, idx2 = [0, 0]
        for idx in range(0, x_axis):

            for idx2 in range(0, y_axis):
                
                axarr[idx, idx2].axis('off')
                try:
                    #print(self.feature_maps.squeeze().shape)
                    #print("try 1")
                    axarr[idx, idx2].imshow(self.feature_maps.squeeze()[random_samples[counter]].cpu().detach().numpy())
                    counter += 1
                    
                except:
                    try:
                        #print("try 2")
                        axarr[idx, idx2].imshow(self.feature_maps.cpu().detach().numpy())
                        counter += 1
                    except:
                        try:
                            #print("try 3")
                            axarr[idx, idx2].imshow( (self.feature_maps.squeeze()[random_samples[counter]]).cpu().detach().numpy().transpose(1, 2, 0))
                        except Exception as e:
                            #print("not possible to show feature maps image")
                            #print(self.feature_maps.shape)
                            #print(e)
                            axarr[idx, idx2].axis('off')

            

        # overwrite first image with original image
        try:
            axarr[x_axis-1, y_axis-1].imshow(self.ii.cpu().detach().numpy().transpose(1, 2, 0))
        except:
            try:
                axarr[x_axis-1, y_axis-1].imshow(self.ii.squeeze().cpu().detach().numpy().transpose(1, 2, 0))
            except:
                try: 
                    axarr[x_axis-1, y_axis-1].imshow(self.ii.squeeze(1).cpu().detach().numpy().transpose(1, 2, 0))
                except Exception as e:
                    print("not possible to show original image")
                    print(e)
        
        if self.log_dir is not None:
            tmp = self.layer_str.replace("model", "").replace(".","")
            path = os.path.join( self.log_dir, f"plt_id{self.batch_idx}_{tmp}.png" )
            fig.savefig(path)