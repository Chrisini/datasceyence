# =============================================================================
# alphabetic order misc
# =============================================================================
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
import random
import torch


# =============================================================================
# torch
# =============================================================================



class DecentFeatureMap():
    # =============================================================================
    # unit test: 
    #    examples/utest_vis_feature_map.ipynb
    # sources:
    #    https://towardsdatascience.com/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030
    #    https://towardsdatascience.com/convolutional-neural-network-feature-map-and-filter-visualization-f75012a5a49c
    #    https://discuss.pytorch.org/t/how-can-l-load-my-best-model-as-a-feature-extractor-evaluator/17254/4
    # =============================================================================

    def __init__(self, model, layer, layer_str, log_dir='', device="cpu"):
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
        #self.m = image_tensor.ms_x # but i need 'this m' ... from the filter
        #self.n = image_tensor.ns_x
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

        # print('self.feature_maps', self.feature_maps.data.shape)
        
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
        # called for each layer
        '''
        * entry image: entry_id5_0_0_0_mo3_gt2.png
        * hidden layer: hid_id5_3_8_2.png
        * last layer (global pooling - connected to class n): pool_2_3_4_gp2.png
        * activated image: cam_id5_mo3_gt2.png
        * activated image gray: camgray_id5_mo3_gt2.png

        * circle in: in_2_3_4_ep65.png
        * circle out: out_2_3_4_ep65.png
        '''
        # =============================================================================
        
        # problem - the maps are wrong!!
        
        # self.feature_maps
        
        print('feature map shape', self.feature_maps.shape)
        
        #filter_list = []
        
        # todo, we cannot use 
        # for i_map in range(self.feature_maps.shape[1]):
        # because rn we have more filters than feature maps ... idk how this even works
        # with a kernel of size 0, we still get a hid_ map
        
        # we have to make sure that empty maps are removed
        for i_map in range(self.feature_maps.shape[1]):

            if self.layer_str == 'decent1':
                m = self.model.decent1.filter_list[i_map].m_this.data.squeeze().detach().cpu().numpy().item()
                n = self.model.decent1.filter_list[i_map].n_this.data.squeeze().detach().cpu().numpy().item()
                tmp_file_name = f'hid_id{self.batch_idx}_{int(m)}_{int(n)}_{1}.png'
                #filter_list.append(f"filter_{int(m)}_{int(n)}_{1}")
            elif self.layer_str == 'decent2':
                m = self.model.decent2.filter_list[i_map].m_this.data.squeeze().detach().cpu().numpy().item()
                n = self.model.decent2.filter_list[i_map].n_this.data.squeeze().detach().cpu().numpy().item()
                tmp_file_name = f'hid_id{self.batch_idx}_{int(m)}_{int(n)}_{2}.png'
                #filter_list.append(f"filter_{int(m)}_{int(n)}_{2}")
            elif self.layer_str == 'decent3':
                m = self.model.decent3.filter_list[i_map].m_this.data.squeeze().detach().cpu().numpy().item()
                n = self.model.decent3.filter_list[i_map].n_this.data.squeeze().detach().cpu().numpy().item()
                tmp_file_name = f'hid_id{self.batch_idx}_{int(m)}_{int(n)}_{3}.png'
                #filter_list.append(f"filter_{int(m)}_{int(n)}_{3}")
            elif self.layer_str == 'decent1x1':
                m = self.model.decent1x1.filter_list[i_map].m_this.data.squeeze().detach().cpu().numpy().item()
                n = self.model.decent1x1.filter_list[i_map].n_this.data.squeeze().detach().cpu().numpy().item()
                # the class that is connected to the last layer's filters via global pooling
                # the class has the same order as the list ... i hope ...
                tmp_file_name = f'pool_id{self.batch_idx}_{int(m)}_{int(n)}_{4}_cl{i_map}.png'
                #filter_list.append(f"filter_{int(m)}_{int(n)}_{4}")
            else:
                print("DECENT WARNING: Layer not found")

            tmp_img = self.feature_maps.squeeze()[i_map].cpu().detach().numpy()
            
            # [Errno 22] Invalid argument: "examples/example_results\\lightning_logs\\dumpster\\version_13\\
            # hid_id0_Parameter containing:\ntensor([1.], device='cuda:0')_Parameter containing:\ntensor([6.], device='cuda:0')_1.png"
            
            # plt_cam_id{batch_idx}_mo{pred_max.detach().cpu().numpy().squeeze()}_gt{ground_truth.detach().cpu().numpy().squeeze()}.png
            tmp_path = os.path.join(self.log_dir, "activation_maps")
            os.makedirs(tmp_path, exist_ok=True)
            plt.imsave(os.path.join(tmp_path, tmp_file_name), tmp_img)
        
        #return filter_list
                    

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

        # print('self.feature_maps', self.feature_maps.data.shape)
        
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
        # print("amount of feature maps:", amount)
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
        # print("random_samples", random_samples)
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
            path = os.path.join(self.log_dir, f"plt_id{self.batch_idx}_{tmp}.png" )
            fig.savefig(path)