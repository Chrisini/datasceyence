import torch
import torch.nn as nn
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
import math
import numpy as np
from typing import Optional, List, Tuple, Union
from model.decentfilter import *
import scipy
from scipy.spatial.distance import cdist
import random


class DecentLayer(nn.Module):
    # =============================================================================
    # all explanations are wrong, redo!!
    # we save filters of the layer in the self.filter_list
    # each filter has a position (m_this, n_this)
    # each filter has input positions (ms_in, ns_in)
    #    - these vary between filters, as some are pruned
    # at the moment we have to loop through the filter list
    # convolution is applied to each filter separately which makes this very slow
    #
    # =============================================================================
    __constants__ = ['stride', 'padding', 'dilation', # 'groups',
                     'padding_mode', # 'n_channels', #  'output_padding', # 'n_filters',
                     'kernel_size']
    __annotations__ = {'bias': Optional[torch.Tensor]}
                
    def __init__(self, ms_in:list, ns_in:list, n_filters:int,
                 kernel_size: _size_2_t,  
                 stride: _size_2_t = 1,  
                 padding: Union[str, _size_2_t] = 0,  
                 dilation: _size_2_t = 1,
                 model_kwargs = None,
                 layer_name = None,
                 ckpt_path = '',
                 bias: bool = True,  # not in use
                 padding_mode: str = "zeros",  # not in use
                 device = None,
                 dtype = None) -> None:
        # =============================================================================
        # initialisation
        # parameters:
        #    a lot.
        # =============================================================================
        
        super().__init__()
        
        self.layer_name = layer_name
        
        # prune numbers
        self.prune_keep = model_kwargs["prune_keep"] # in each update [0.0:1.0]
        self.prune_keep_total = model_kwargs["prune_keep_total"] # total [0.0:1.0]
        
        # importance metric for pruning
        self.ci_metric = model_kwargs["ci_metric"]
        # distance metric for loss
        self.cc_metric = model_kwargs["cc_metric"]
        
        # from prev layer
        self.ms_in = ms_in
        self.ns_in = ns_in
        
        self.original_size = len(self.ms_in) * n_filters
        
        self.grid_size = model_kwargs["grid_size"]
        self.grid_sqrt = math.sqrt(self.grid_size)
        assert self.grid_sqrt == int(self.grid_sqrt), f"square root ({self.grid_sqrt}) from grid size {self.grid_size} not possible; possible exampes: 81 (9*9), 144 (12*12)"
        self.grid_sqrt = int(self.grid_sqrt)
        
        
        # =============================================================================
        # from ckpt or from scratch
        # =============================================================================
        if ckpt_path == '':
            # reset

            # use techniques from coo matrix
            self.geometry_array = np.full(self.grid_size, np.nan)
            # plus 1 here cause of to_sparse array
            self.geometry_array[0:n_filters] = range(1,n_filters+1)
            np.random.shuffle(self.geometry_array)
            self.geometry_array = self.geometry_array.reshape((self.grid_sqrt,self.grid_sqrt), order='C')
            self.geometry_array = torch.tensor(self.geometry_array)
            self.geometry_array = self.geometry_array.to_sparse(sparse_dim=2).to("cuda")

            #print(self.geometry_array)
            #print(self.geometry_array.values())

            self.filter_list = nn.ModuleList([])
            for i_filter in range(n_filters):
                # minus 1 here cause of to_sparse array
                index = (self.geometry_array.values()-1 == i_filter).nonzero(as_tuple=True)[0]
                self.m_this = self.geometry_array.indices()[0][index]
                self.n_this = self.geometry_array.indices()[1][index]
                f = DecentFilter(self.ms_in, self.ns_in, self.m_this, self.n_this, 
                                 kernel_size=kernel_size, 
                                 stride=stride, padding=padding, dilation=dilation)
                self.filter_list.append(f)
                # self.register_parameter(f"filter {i_filter}", f.weights)

                #nn.Parameter(torch.empty((1, n_channels, *kernel_size), **factory_kwargs))
                
        else:
            # from checkpoint file
            
            state_dict = torch.load(ckpt_path)            
            self.filter_list = nn.ModuleList([])
            
            # 'model.decent1.filter_list.0.ms_in'
            
            # self.layer_name -> decent1
            # self.filter_list element 0,1,2,3 -> filter_list.0 
            # self.ms_in -> ms
            
            # init the first one
            if False: # wrong
                self.m_this = state_dict['state_dict'][f'model.{self.layer_name}.filter_list.0.m_this']
                self.n_this = state_dict['state_dict'][f'model.{self.layer_name}.filter_list.0.n_this']
                self.ms_in  = torch.tensor([0])
                self.ns_in  = torch.tensor([0])
                ckpt_weight = state_dict['state_dict'][f'model.{self.layer_name}.filter_list.0.weights']

                start = 1 if self.layer_name == 'decent1' else 0
                
            for i_filter in range(0, n_filters):
                
                self.m_this = state_dict['state_dict'][f'model.{self.layer_name}.filter_list.{i_filter}.m_this']
                self.n_this = state_dict['state_dict'][f'model.{self.layer_name}.filter_list.{i_filter}.n_this']
                self.ms_in  = state_dict['state_dict'][f'model.{self.layer_name}.filter_list.{i_filter}.ms_in']
                self.ns_in  = state_dict['state_dict'][f'model.{self.layer_name}.filter_list.{i_filter}.ns_in']
                ckpt_weight = state_dict['state_dict'][f'model.{self.layer_name}.filter_list.{i_filter}.weights']
                
                print('DECENT INFO: ckpt_weight.shape', ckpt_weight.shape)

                # use filters that have at least one channel
                # todo idk whether this would work for actual training later ...
                if ckpt_weight.shape[1] != 0:

                    f = DecentFilter(self.ms_in, self.ns_in, self.m_this, self.n_this, 
                                     ckpt_weight=ckpt_weight,
                                     kernel_size=kernel_size, 
                                     stride=stride, padding=padding, dilation=dilation)
                    self.filter_list.append(f)        
                else:
                    print("DECENT INFO: ckpt weight is zero. skip ...")
            
        
        # =============================================================================
        # init the new loss function!!
        # =============================================================================
        self.new_cc_mode = model_kwargs["new_cc_mode"]
        if self.new_cc_mode:
            self.cc_for_each_filter = None
            self.ci_for_each_filter = None
            self.cc_max_of_layer = None
            self.ci_max_of_layer = None
            self.update_layer_max_connection_cost()
            self.update_layer_max_channel_importance()
    
    
    def get_filter_channel_importance(self, f):
        # =============================================================================
        # compute channel importance metric for pruning and loss
        # =============================================================================
        
        # every time todo
        ci_for_each_weight = [] # list of all weights in one filter
        for i_w in range(f.weights.shape[1]): # todo, sure this is 1 and not 0???
            # importance of a kernel in a filter in a layer
            if self.ci_metric == 'l2': # weight dependent - filter norm
                ci_for_each_weight.append(f.weights[:,i_w].norm(2).flatten()) # .detach().cpu().numpy()) # .detach().cpu().numpy()
            elif self.ci_metric == 'random':
                ci_for_each_weight.append(torch.rand(1))
            else:
                print("DECENT WARNING: no valid ci metric chosen.") 

        ci_for_each_weight = torch.cat(ci_for_each_weight)
        return ci_for_each_weight.flatten()            
     
    
    def get_filter_connection_cost(self, f):
        # =============================================================================
        # compute channel importance metric for loss
        # =============================================================================
        
        # every update todo
        mn = torch.cat([f.m_this.unsqueeze(0), f.n_this.unsqueeze(0)]).transpose(1,0)
        msns = torch.cat([f.ms_in.unsqueeze(0), f.ns_in.unsqueeze(0)]).transpose(1,0)

        cc_for_each_weight = [] # list of all distances in one filter

        # every update todo
        # normalised by grid size :)
        if self.cc_metric == 'l2':
            cc_for_each_weight = torch.cdist( x1=mn.float(), x2=msns.float(), p=2).flatten() / self.grid_sqrt # chose this one!!            
        else:
            print("DECENT WARNING: no valid cc metric chosen.")

        # no append, hence no flattened needed
        # cc_for_each_weight = torch.cat(cc_for_each_weight)
        return cc_for_each_weight.flatten()    
     
    
    def update_layer_connection_cost(self):
        # =============================================================================
        # layer connection cost
        # =============================================================================
        self.cc_for_each_filter = []
        for f in self.filter_list:
            self.cc_for_each_filter.append(self.get_filter_connection_cost(f).flatten()) 
        self.cc_for_each_filter = torch.cat(self.cc_for_each_filter)  
            
        
    def update_layer_channel_importance(self):
        # =============================================================================
        # layer channel importance
        # =============================================================================
        self.ci_for_each_filter = []
        for f in self.filter_list:
            self.ci_for_each_filter.append(self.get_filter_channel_importance(f))
        self.ci_for_each_filter = torch.cat(self.ci_for_each_filter)
    
    def update_layer_max_connection_cost(self):
        # =============================================================================
        # cc of a filter gives cc max for layer
        # =============================================================================
        
        # if empty, fill - useful for init
        if self.cc_for_each_filter is None or len(self.cc_for_each_filter) == 0:
            self.update_layer_connection_cost()
        # max
        
        self.cc_max_of_layer = torch.max(self.cc_for_each_filter.flatten())
        
    def update_layer_max_channel_importance(self):
        # =============================================================================
        # ci of a filter gives ci max for layer
        # =============================================================================
        
        # if empty, fill - useful for init
        if self.ci_for_each_filter is None or len(self.ci_for_each_filter) == 0:
            self.update_layer_channel_importance()
        # max
        self.ci_max_of_layer = torch.max(self.ci_for_each_filter.flatten())
    
    def run_swap_filter(self):
        # =============================================================================
        # not working yet
        # we swap filters within the layer
        # based on connection cost
        # filter can move a maximum of two positions per swap
        # change positions
        # change
        #print("swap here")
        #self.m_this = self.m_this # single integer
        #self.n_this = self.n_this # single integer
        # =============================================================================
        pass
    
    def run_grow_filter(self) -> None:
        # =============================================================================
        # not working yet* 100
        # introduce new filters in a layer
        # based on 
        # algorithmic growth process 
        # =============================================================================
        pass
    
    def run_grow_channel(self) -> None:
        # =============================================================================
        # not working yet
        # introduce new channel in a layer
        # based on connection cost??
        # algorithmic growth process 
        # =============================================================================
        pass
    
    def run_prune_filter(self) -> None:
        # =============================================================================
        # delete filter in a layer
        # list comprehension
        # removing empty filters aka 0 channels in a filter
        # problem if all connections to a filter in decent1x1 are pruned, then there is no connection left todo
        # =============================================================================
        
        # this prunes a filter if no weights left??
        
        prev_len = len(self.filter_list)
        # somelist = [x for x in somelist if not determine(x)]
        self.filter_list = nn.ModuleList([tmp_filter for tmp_filter in self.filter_list if tmp_filter.weights.shape[1] != 0])
        if prev_len != len(self.filter_list):
            print("DECENT INFO: filter list length: ", prev_len, "->", len(self.filter_list))
        
    
    def run_prune_channel(self, i_f:int, keep_ids:list) -> None:
        # =============================================================================
        # delete channels in a filter based on keep_ids
        # based on importance score
        # only keep "the best" weights
        # pruning based on a metric
        # nonsense?
        #    delete layer with id
        #    delete channels in each layer with id
        #    channel deactivation
        #    require_grad = False/True for each channel
        #    deactivate_ids = [1, 2, 6]
        #    self.active[deactivate_ids] = False
        #    print("weight")
        #    print(self.weight.shape)
        #    print(self.weight[:,self.active,:,:].shape)
        #    this is totally wrong - iterative will break after first iteration
        #    print()
        #    Good to hear it’s working, although I would think you’ll get an error at some point in your code, as the cuda() call creates a non-leaf tensor.
        #    self.weight = nn.Parameter(  self.weight[:,self.active,:,:] ) # .detach().cpu().numpy()
        #    self.weight = self.weight.cuda()
        #    print(self.weight.shape)
        #    print(self.active)
        #    print("prune here")
        #    for f in self.filter_list:
        #        f.update()
        # =============================================================================
        
        if False:
            for i in keep_ids:
                print(i)
                print(self.filter_list[i_f].ms_in[i])
                print(nn.Parameter(self.filter_list[i_f].ms_in[keep_ids]) )
        
        if random.randint(1, 100) == 5:
            print()
            print("DECENT INFO at random intervals")
            print(keep_ids)
            print(self.filter_list[i_f].weights[:, keep_ids, :, :].shape)
            print(self.filter_list[i_f].weights.shape)        
        
        # todo: check, this may create more parameters ...
        
        # prune weights, ms and ns based on the 'keep ids'
        # this becomes a grad here, hence turn off again with False
        #with torch.no_grad():
        # self.filter_list[i_f].weights = self.filter_list[i_f].weights[:, keep_ids, :, :]
        #self.filter_list[i_f].weights = torch.nn.Parameter(torch.ones_like(self.filter_list[i_f].weights[:, keep_ids, :, :]), requires_grad=True)
        self.filter_list[i_f].weights = nn.Parameter(self.filter_list[i_f].weights[:, keep_ids, :, :], requires_grad=True) # changed this 14.11.2024
        self.filter_list[i_f].ms_in = nn.Parameter(self.filter_list[i_f].ms_in[keep_ids], requires_grad=False) 
        self.filter_list[i_f].ns_in = nn.Parameter(self.filter_list[i_f].ns_in[keep_ids], requires_grad=False)
        
        # original_tensor = replacement_tensor.to(original_tensor.device).clone().requires_grad_(True)
        
        # nn.Parameter(torch.empty((1, self.n_weights, *self.kernel_size), **factory_kwargs))
    
    def forward(self, x) -> torch.Tensor:
        # =============================================================================
        # calculate representation x for each filter in this layer
        # =============================================================================
        
        output_list = []
        m_list = []
        n_list = []
        for f in self.filter_list:
            # output = filter(input)
            out = f(x)
            # if filter has no channels left
            if out is not None:
                output_list.append(out)
                m_list.append(f.m_this)
                n_list.append(f.n_this)
            else:
                print("DECENT WARNING: output in forward is None. skip ...")
                
        x.ms_x = m_list
        x.ns_x = n_list
        x.data = torch.cat(output_list, dim=1)
        return x
    
    def get_filter_positions(self):
        # =============================================================================
        # in use for next layer input (initialisation of the model)
        # =============================================================================
        
        ms_this = []
        ns_this = []
        for f in self.filter_list:
            ms_this.append(f.m_this)
            ns_this.append(f.n_this)
        
        return ms_this, ns_this
    
    def get_everything(self):
                
        sources = [] # source m,n,l-1
        targets = [] # target m,n,l
        target_groups = []
        values = [] # connection value = ci value of channel in target connected to ms[i], ns[i] 
        
        # for each filter
        for i_f, f in enumerate(self.filter_list):
            
            # print("ms_in shape", f.ms_in.shape)
            for i_s in range(len(f.ms_in)):
                
                s = str(int(f.ms_in[i_s].item()))+'_'+str(int(f.ns_in[i_s].item()))
                sources.append(s)

                t = str(int(f.m_this.item()))+'_'+str(int(f.n_this.item()))
                targets.append(t)

                target_groups.append(self.layer_name)
            
            # get all channel importances # .flatten()
            ci = self.get_filter_channel_importance(f).detach().cpu().numpy() # try this ... todo
            #print("CI"*50)
            #print(ci)
            values.extend(ci)
        
        # value = channel importance
        return {'source':sources, 'target':targets, 'target_group':target_groups, f'importance':values}
    
    
    def update(self):
        # =============================================================================
        # currently: calculate importance metric for the prune_channel method
        # remove channels based on self.prune_keepif new_cc_mode
        # layerwise pruning - percentage of layer
        # =============================================================================
     
        # self.update_layer_channel_importance() - is done here by hand cause i need length
        
        all_ci = []
        all_len = 0
        for f in self.filter_list: # for each filter
            all_len += len(f.ms_in) # amount of channels
            all_ci.append(self.get_filter_channel_importance(f).detach().cpu().numpy()) 
            # changed it, only one list, not a list of list anymore in case that was important, else maybe remove flatten works
        
        # DO NOT CAT OR FLATTEN all_ic !!!
        # print("i want: len becomes now amount of filters", len(all_ci), "= all_len " , all_len, ">", int(self.original_size * self.prune_keep_total))
        
        if all_len < int(self.original_size * self.prune_keep_total):
            # if n percent have been pruned, stop this layer
            print("DECENT NOTE: pruning done for this layer")
        else:
            print("DECENT NOTE: prune channels ...")
            # pruning
            keep_amount = int(all_len*self.prune_keep)
            
            #print("*"*50)
            #print("keep_amount, all len, all_ci len, thresh")
            #print(keep_amount)
            #print(all_len)
            #print(len(all_ci))
            
            # sure about the -n?? n is to keep not to remove - test todo
            
            # all_ci_flatten = [item for row in all_ci for item in row] # don't have equal lengths, so no numpy possible
            flattened = [item for sublist in all_ci for item in sublist]
            index = sorted(range(all_len), key=lambda sub: flattened[sub])[-keep_amount] # error, out of range n or -n??
            threshold_value = flattened[index] # everything above this value stays
            
            #print(threshold_value)

            for i_f, f in enumerate(self.filter_list):

                # channel importance list for this filter
                # ci = all_ci[i_f] # list of self.run_filter_channel_importance(i_f)
                
                ci = all_ci[i_f]
                
                keep_ids = np.where(ci >= threshold_value)[0] # we use a list hence we only have values in the 0=x coordinate.
                
                #print("how much we keep:")
                #print(keep_ids)

                # indices should be list/np/detached
                self.run_prune_channel(i_f, keep_ids)
                    
        self.run_prune_filter() # remove whole filter if zero channels in a filter
        
        #print("why is this called twice??")
        #print("\nCHECKING IN DECENT LAYER WHAT THIS THING SHOWS")
        #print(self.filter_list[1].weights[0])
        #print("END OF CHECKING\n")
        
        # update everything just in case
        self.update_layer_connection_cost()
        self.update_layer_max_connection_cost()
        self.update_layer_channel_importance()
        self.update_layer_max_channel_importance()
        
        
    def step(self):
        # =============================================================================
        # compute c term for this layer
        # =============================================================================
        
        # update each step
        self.update_layer_channel_importance()
        
        # calculate each step
        c_term_for_each_filter = []
        for cc, ci in zip(self.cc_for_each_filter, self.ci_for_each_filter):
            cc_normalised = cc/self.cc_max_of_layer
            ci_normalised = ci/self.ci_max_of_layer
            tmp = torch.abs(ci_normalised-cc_normalised)
            c_term_for_each_filter.append(tmp.flatten())
        
        c_term_for_each_filter = torch.cat(c_term_for_each_filter)
        c_term_for_layer = torch.mean(c_term_for_each_filter)
        
        return c_term_for_layer

        
        
       