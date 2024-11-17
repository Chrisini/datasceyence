# =============================================================================
# alphabetic order misc
# =============================================================================
import math
import matplotlib.pyplot as plt
import os
import pandas as pd
# =============================================================================
# torch
# =============================================================================
import torch
import torch.nn as nn
# =============================================================================
# datasceyence
# =============================================================================
from model.decentlayer import *


class DecentNet(nn.Module):
    def __init__(self, model_kwargs, log_dir="", ckpt_path='') -> None:
        super(DecentNet, self).__init__()
        
        self.n_classes = model_kwargs["n_classes"]
        out_dim = model_kwargs["out_dim"]
        out_dim.append(self.n_classes) # out_dim = [1, 32, 48, 64, 10]     
        print("DECENT INFO: dimensions are entry, decent1, decent2, decent3, decent1x1 == out", out_dim)
        
        grid_size = model_kwargs["grid_size"]
        assert not any(i > grid_size for i in out_dim), f"filters need to be less than {grid_size}"
        self.grid_sqrt = int(math.sqrt(grid_size))
        
        self.ci_metric = model_kwargs["ci_metric"]
        
        self.new_cc_mode = model_kwargs["new_cc_mode"]
        
        self.log_dir = log_dir

        # backbone
        
        # initialise input positions of first layer
        ms_in_1 = [torch.tensor(0)]
        ns_in_1 = [torch.tensor(0)]
        if ckpt_path == '':
            assert out_dim[0] == len(ms_in_1), f"x data (out_dim[0]={out_dim[0]}) needs to match (ms_in_1={len(ms_in_1)})"
            assert out_dim[0] == len(ns_in_1), f"x data (out_dim[0]={out_dim[0]}) needs to match (ns_in_1={len(ns_in_1)})"
        self.decent1 = DecentLayer(ms_in=ms_in_1, ns_in=ns_in_1, n_filters=out_dim[1], 
                                   kernel_size=3, stride=1, padding=0, dilation=1, 
                                   model_kwargs=model_kwargs, 
                                   layer_name='decent1',
                                   ckpt_path=ckpt_path)
        
        # get position of previous layer as input for this layer
        ms_in_2,ns_in_2 = self.decent1.get_filter_positions()
        if ckpt_path == '':
            assert out_dim[1] == len(ms_in_2), f"x data (out_dim[1]={out_dim[1]}) needs to match (ms_in_2={len(ms_in_2)})"
            assert out_dim[1] == len(ns_in_2), f"x data (out_dim[1]={out_dim[1]}) needs to match (ns_in_2={len(ns_in_2)})"
        self.decent2 = DecentLayer(ms_in=ms_in_2, ns_in=ns_in_2, n_filters=out_dim[2], 
                                   kernel_size=3, stride=1, padding=0, dilation=1,
                                   model_kwargs=model_kwargs, 
                                   layer_name='decent2',
                                   ckpt_path=ckpt_path)
        
        ms_in_3,ns_in_3 = self.decent2.get_filter_positions()
        if ckpt_path == '':
            assert out_dim[2] == len(ms_in_3), f"x data (out_dim[2]={out_dim[2]}) needs to match (ms_in_3={len(ms_in_3)})"
            assert out_dim[2] == len(ns_in_3), f"x data (out_dim[2]={out_dim[2]}) needs to match (ns_in_3={len(ns_in_3)})"
        self.decent3 = DecentLayer(ms_in=ms_in_3, ns_in=ns_in_3, n_filters=out_dim[3], 
                                   kernel_size=3, stride=1, padding=0, dilation=1, 
                                   model_kwargs=model_kwargs, 
                                   layer_name='decent3',
                                   ckpt_path=ckpt_path)
        
        ms_in_1x1,ns_in_1x1 = self.decent3.get_filter_positions()
        if ckpt_path == '':
            assert out_dim[3] == len(ms_in_1x1), f"x data (out_dim[3]={out_dim[3]}) needs to match (ms_in_1x1={len(ms_in_1x1)})"
            assert out_dim[3] == len(ns_in_1x1), f"x data (out_dim[3]={out_dim[3]}) needs to match (ns_in_1x1={len(ns_in_1x1)})"
        self.decent1x1 = DecentLayer(ms_in=ms_in_1x1, ns_in=ns_in_1x1, n_filters=out_dim[-1], 
                                     kernel_size=1, stride=1, padding=0, dilation=1, 
                                     model_kwargs=model_kwargs, 
                                     layer_name='decent1x1',
                                     ckpt_path=ckpt_path)
        
        #self.tmp = torchvision.models.squeezenet1_0(torchvision.models.SqueezeNet1_0_Weights.IMAGENET1K_V1)
        #self.tmp.classifier[1] = nn.Conv2d(512, 10, kernel_size=(3,3))
        
        # head
        self.fc = nn.Linear(out_dim[-1], out_dim[-1])
    
        # activation
        self.mish1 = nn.Mish()
        self.mish2 = nn.Mish()
        self.mish3 = nn.Mish()
        self.mish1x1 = nn.Mish()
        
        # bias
        self.bias1 = nn.InstanceNorm2d(out_dim[1])
        self.bias2 = nn.InstanceNorm2d(out_dim[2])
        self.bias3 = nn.InstanceNorm2d(out_dim[3])
        self.bias1x1 = nn.InstanceNorm2d(out_dim[-1])
        
        # activation
        # self.sigmoid = nn.Sigmoid()

        # init connection cost
        """
        self.cc = []
        if self.new_cc_mode:
            self.update_new_connection_cost()
        else:
            self.update_connection_cost()
        """
        
        
        # get a position in filter list
        self.m_l2_plot = self.decent2.filter_list[0].m_this.detach().cpu().numpy()
        self.n_l2_plot = self.decent2.filter_list[0].n_this.detach().cpu().numpy()  
        
        print(self.m_l2_plot)
        print(self.n_l2_plot)
        
        """
        with open(os.path.join(self.log_dir, 'logger.txt'), 'a') as f:
                f.write("\n# plot #\n")
                for p in self.model.parameters():
                    if p.requires_grad:
                        f.write('m:' + str(self.m_l2_plot) + ', n: ' + str(self.n_l2_plot) + '\n')
        """
        # self.plot_layer_of_1_channel(current_epoch=0) - not working here, dir not created yet
        
        # placeholder for the gradients
        self.gradients = None
        
    def forward(self, x, mode="grad"):
        
        #print(x)
        
        x = self.decent1(x)
        x.data = self.mish1(x.data)
        x.data = self.bias1(x.data)
        
        #print(x)
        
        x = self.decent2(x)
        x.data = self.mish2(x.data)
        x.data = self.bias2(x.data)
        
        #print(x)
        
        x = self.decent3(x)
        x.data = self.mish3(x.data)
        x.data = self.bias3(x.data)
        
        #print(x)
        
        x = self.decent1x1(x)
        x.data = self.mish1x1(x.data)
        x.data = self.bias1x1(x.data)
        
        #print(x)
        
        # hook on the data (for gradcam or something similar)
        # https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
        if mode == 'explain':
            output = x.data.register_hook(self.activations_hook)
            #'cannot register a hook on a tensor that doesn't require gradient'
        
        
        # global max pooling for MIL
        # https://discuss.pytorch.org/t/global-max-pooling/1345
        # Global Average Pooling is a pooling operation designed to replace fully connected layers in classical CNNs. 
        # The idea is to generate one feature map for each corresponding category of the classification task in the last mlpconv layer.
        # Similar to global average pooling, to implement global max pooling in PyTorch, 
        # one needs to use the regular max pooling class with a kernel size equal to the size of the feature map at that point
        x.data = torch.nn.functional.max_pool2d(x.data, kernel_size=x.data.size()[2:])
        
        # or flatten
        x.data = x.data.reshape(x.data.size(0), -1)
        
        # we still want the fc ???
        # x.data = self.fc(x.data) 
        
        # x.data = self.sigmoid(x.data)
        
        # x.data = self.tmp(x.data)
        
        return x.data
    
    
    def activations_hook(self, grad):
        # hook for the gradients of the activations
        self.gradients = grad
    def get_activations_gradient(self):
        # method for the gradient extraction
        return self.gradients
    def get_activations(self, x):
        # method for the activation exctraction
        
        #print('0', x)
        
        x = self.decent1(x)
        x.data = self.mish1(x.data)
        x.data = self.bias1(x.data)
        #print('1', x)

        x = self.decent2(x)
        x.data = self.mish2(x.data)
        x.data = self.bias2(x.data)
        #print('2', x)
        
        x = self.decent3(x)
        x.data = self.mish3(x.data)
        x.data = self.bias3(x.data)
        #print('3', x)
        
        x = self.decent1x1(x)
        x.data = self.mish1x1(x.data)
        x.data = self.bias1x1(x.data)
        #print('1x1', x)
        
        return x.data
    
    def plot_incoming_connections(self, current_epoch=0):
        # analyse incoming conenctions
        # which kernels were pruned in this filter
        # decent3 = orange
        # decent2 = cyan
        # decent1 = pink
        
        # get each filter position that has a channel that matches
        ms = []; ns = []
        
        #print(self.decent2.filter_list)
        #print("**********************")
        #print(self.decent3.filter_list)

        
        # use first filter in the list of this layer
        this_filter = self.decent2.filter_list[0] # orange
        
        m_tmp = this_filter.m_this.detach().cpu().numpy()
        n_tmp = this_filter.n_this.detach().cpu().numpy()
        ms_tmp = this_filter.ms_in.detach().cpu().numpy()
        ns_tmp = this_filter.ns_in.detach().cpu().numpy()
                
        # visualising the previous and current layer neurons
        fig, ax = plt.subplots(figsize=(5,5))
        ax.scatter(m_tmp, n_tmp, s=100000, color='tab:cyan', alpha=0.1) # previous layer
        ax.scatter(m_tmp, n_tmp, s=50000, color='tab:cyan',alpha=0.2) # previous layer
        ax.scatter(m_tmp, n_tmp, s=25000, color='tab:cyan',alpha=0.3) # previous layer
        ax.scatter(m_tmp, n_tmp, s=500, color='tab:cyan') # previous layer
        ax.scatter(ms_tmp, ns_tmp, color='tab:pink') # next layer
        plt.xlim(0, self.grid_sqrt) # m coordinate of grid_size field
        plt.ylim(0, self.grid_sqrt) # n coordinate of grid_size field
        ax.grid() # enable grid line
        fig.savefig(os.path.join(self.log_dir, f"in_{self.ci_metric}_m{int(self.m_l2_plot[0])}_n{int(self.n_l2_plot[0])}_{str(current_epoch)}.png"))
    
    def plot_outgoing_connections(self, current_epoch=0): # plot_layer_of_1_channel
        # analyse outgoing conenctions
        # which filters in the next layer are influenced by this filter
        
        # get each filter position that has a channel that matches
        ms = []; ns = []
        
        #print(self.decent2.filter_list)
        #print("**********************")
        #print(self.decent3.filter_list)

        
        # go through all filters in this layer
        for f in self.decent3.filter_list:
            
            # if filter position in prev layer matches any channel in this layer
            if any(pair == (self.m_l2_plot, self.n_l2_plot) for pair in zip(f.ms_in.detach().cpu().numpy(), f.ns_in.detach().cpu().numpy())):
                #print('match', f.m_this, f.n_this)
                # save position of each filter in this layer
                ms.append(f.m_this.detach().cpu().numpy())
                ns.append(f.n_this.detach().cpu().numpy())
              
            if False:
                    print("nooooooooooooooo")
                    print(f.ms_in)
                    print(self.m_l2_plot)
                    print(f.ns_in)
                    print(self.n_l2_plot)

                    print((self.m_l2_plot, self.n_l2_plot))
                
        # visualising the previous and current layer neurons
        fig, ax = plt.subplots(figsize=(5,5))
        ax.scatter(self.m_l2_plot, self.n_l2_plot, s=100000, color='tab:cyan', alpha=0.1) # previous layer
        ax.scatter(self.m_l2_plot, self.n_l2_plot, s=50000, color='tab:cyan',alpha=0.2) # previous layer
        ax.scatter(self.m_l2_plot, self.n_l2_plot, s=25000, color='tab:cyan',alpha=0.3) # previous layer
        ax.scatter(self.m_l2_plot, self.n_l2_plot, s=500, color='tab:cyan') # previous layer
        ax.scatter(ms, ns, color='tab:orange') # next layer
        plt.xlim(0, self.grid_sqrt) # m coordinate of grid_size field
        plt.ylim(0, self.grid_sqrt) # n coordinate of grid_size field
        ax.grid() # enable grid line
        fig.savefig(os.path.join(self.log_dir, f"connections_{self.ci_metric}_m{int(self.m_l2_plot[0])}_n{int(self.n_l2_plot[0])}_{str(current_epoch)}.png"))
    
    """
    def update_connection_cost(self): # the old one
        # this cc loss term is absolute crap!!!
        self.cc = []
        # self.cc.append(self.decent1.run_layer_connection_cost()) # maybe not even needed ...
        
        tmp_cc, _ = self.decent2.run_layer_connection_cost()
        self.cc.append(tmp_cc)
        tmp_cc, _ = self.decent3.run_layer_connection_cost()
        self.cc.append(tmp_cc)
        tmp_cc, _ = self.decent1x1.run_layer_connection_cost()
        self.cc.append(tmp_cc)
        
        self.cc = torch.mean(torch.tensor(self.cc))
    """
        
    def get_cc_and_ci_loss_term(self):
        
        # update of cc after each pruning (happening automatically)
        # update of ci every n steps?? or is it too much?? = kernel magnitude
        
        c_term_for_each_layer = []
        
        c_term_for_layer = self.decent2.step()
        c_term_for_each_layer.append(c_term_for_layer)
        c_term_for_layer = self.decent3.step()
        c_term_for_each_layer.append(c_term_for_layer)
        c_term_for_layer = self.decent1x1.step()
        c_term_for_each_layer.append(c_term_for_layer)

        c_term_for_model = torch.mean(torch.tensor(c_term_for_each_layer))
        
        return c_term_for_model

    def update(self, current_epoch):
        # =============================================================================
        # update_every_nth_epoch
        # adapted from BIMT: https://github.com/KindXiaoming/BIMT/blob/main/mnist_3.5.ipynb
        # =============================================================================
        
        # update decent layers
        
        #self.decent1.update()
        print("decent2-"*5)
        self.decent2.update()
        print("decent3-"*5)
        self.decent3.update()
        print("decent1x1-"*5)
        self.decent1x1.update()
        
        # visualisation
        self.plot_incoming_connections(current_epoch)
        self.plot_outgoing_connections(current_epoch)
    
        # connection cost has to be calculated after pruning
        # self.cc which is updated is used for loss function
        #if self.new_cc_mode:
        #    self.update_new_connection_cost()
        #else:
        #    pass
            #self.update_connection_cost()
        
        
    def get_everything(self, current_epoch):
        
        # in this we have the channel importance!!
        d1 = self.decent1.get_everything()
        d2 = self.decent2.get_everything()
        d3 = self.decent3.get_everything()
        d1x1 = self.decent1x1.get_everything()
        
        #print(d1)
        #print(d2)
        #print(d3)
        
        df1 = pd.DataFrame(d1)
        #print(df1.head())
        
        df2 = pd.DataFrame(d2)
        #print(df2.head())
        
        df3 = pd.DataFrame(d3)
        #print(df3.head())
        
        df1x1 = pd.DataFrame(d1x1)
        
        frames = [df1, df2, df3, df1x1]
        result = pd.concat(frames)
        
        result.to_csv(os.path.join(self.log_dir, f'connections_{str(current_epoch)}.csv'), index=False)  
        
        # return d1, d2, d3, d1x1
        

