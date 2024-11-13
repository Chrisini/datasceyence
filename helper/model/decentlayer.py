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
    #
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
            
    
    def run_layer_connection_cost(self):
        # =============================================================================
        # compute connection cost for this layer - based on distance
        # returns:
        #    connection cost for the loss function
        # notes:
        #    currently using l2 norm, doesn't work that well
        # sources:
        #    adapted from BIMT: https://github.com/KindXiaoming/BIMT/blob/main/mnist_3.5.ipynb
        #    https://stackoverflow.com/questions/74086766/how-to-find-total-cost-of-each-path-in-graph-using-dictionary-in-python
        # nonsense?
        #    i don't even know what the following comments are about ... 
        #    based on previous layer (cause I only have input ms_in, n_in information)
        #    mean( sum( of connection cost between this filter and all incoming filters
        #    need it for loss - aka all layers, all filters together
        #    need it for swapping - this layer, all filters
        #    only the active ones (we need to use the indices for that)
        #    for swapping i need ??
        # =============================================================================
        
        # connection cost list
        cc = []
        max_distance_of_layer = 0
        
        for f in self.filter_list:
            # for each filter we use the current position and all incoming positions

            #mn = torch.cat([torch.tensor(f.m_this), torch.tensor(f.n_this)])
            #print(mn.shape)
            #msns = torch.cat([torch.tensor(f.ms_in), torch.tensor(f.ns_in)]) # .transpose(1,0)
            #print(msns.shape)
            #cc.append(torch.cdist(mn.unsqueeze(dim=0), msns.transpose(1,0), 'euclidean') / 8) # number comes from 9*9 = 81 [0-8]
            #print(mn)
            #print(msns)

            # mean ( l2 norm as distance metric / normalisation term for l2 norm)
            # mean of distances
            # normalise with max=grid square root, min=0
            # mean from all non-nan values
            # 
            
            # todo - to make it fully differentiable we have to get equivalents for torch, yay
        
            mn = torch.cat([f.m_this.unsqueeze(0), f.n_this.unsqueeze(0)]).transpose(1,0)
            msns = torch.cat([f.ms_in.unsqueeze(0), f.ns_in.unsqueeze(0)]).transpose(1,0)
            mn_tmp = mn.detach().cpu().numpy()
            msns_tmp = msns.detach().cpu().numpy()
            
            # normalised by grid size :)
            if self.cc_metric == 'l1':
                tmp_distances = torch.tensor( scipy.spatial.distance.cdist(mn_tmp, msns_tmp, 'cityblock') /self.grid_sqrt )
            elif self.cc_metric == 'l2':
                tmp_distances = torch.tensor( scipy.spatial.distance.cdist(mn_tmp, msns_tmp, 'euclidean') /self.grid_sqrt )
            elif self.cc_metric == 'l2_torch':
                tmp_distances = torch.cdist( x1=mn.float(), x2=msns.float(), p=2).flatten() /self.grid_sqrt # i have no idea how this works
            elif self.cc_metric == 'linf':
                tmp_distances = torch.tensor( scipy.spatial.distance.cdist(mn_tmp, msns_tmp, 'chebyshev') /self.grid_sqrt )
            elif self.cc_metric == 'cos':
                tmp_distances = torch.tensor( scipy.spatial.distance.cdist(mn_tmp, msns_tmp, 'cosine') /self.grid_sqrt )
            elif self.cc_metric == 'jac':
                tmp_distances = torch.tensor( scipy.spatial.distance.cdist(mn_tmp, msns_tmp, 'jaccard') /self.grid_sqrt )
            elif self.cc_metric == 'cor':
                tmp_distances = torch.tensor( scipy.spatial.distance.cdist(mn_tmp, msns_tmp, 'correlation') /self.grid_sqrt )
            else:
                print("DECENT WARNING: no valid cc metric chosen.")
                tmp_distances = None
            
            #print("this should give multiple distances")
            #print(tmp_distances) # this should give multiple distances
            
            mean_value_of_filter = torch.nanmean(tmp_distances).unsqueeze(0) # to make it able to concat later - aka we make tensor(4) to tensor([4])
            cc.append(mean_value_of_filter)
            
            max_distance_of_filter = torch.max(tmp_distances)
            # print("the max value", max_distance_of_filter)
            # update max distance of layer
            if max_distance_of_layer < max_distance_of_filter:
                max_distance_of_layer = max_distance_of_filter

        # mean connection cost of a layer
        # mean from all non-nan values - todo - check why there are nan values, maybe cause of pruning??
        
        #print("cc", cc)
        
        cc = torch.cat(cc)
        #print("cc cat", cc)
        mean_value_of_layer = torch.nanmean(cc) #torch.tensor(cc)
        # normalise with max value to scale between 0 and 1
        normalised_mean_value_of_layer = mean_value_of_layer / max_distance_of_layer+1e5
        
        return mean_value_of_layer, normalised_mean_value_of_layer # , max_cc
    
    def run_layer_channel_importance(self):
        # =============================================================================
        # compute channel importance metric for pruning
        # calculate the norm of each weight in filter with id i_f
        # we need to call this in a loop to go through each filter
        # returns:
        #     ci: channel importance list of a filter
        # notes:
        #     based on l2 norm = magnitude = euclidean distance
        # nonsense?
        #    maybe the kernel trigger todo
        #    print(self.filter_list[i_f].weights.shape)
        #    print(self.filter_list[i_f].weights[:,i_w].shape)
        # =============================================================================
        
        # channel importance list
        ci = []
        
        # we get each weight in each filter
        for f in self.filter_list:
            for i_w in range(f.weights.shape[1]): # todo, sure this is 1 and not 0???
                # importance of a kernel in a layer

                if self.ci_metric == 'l1':
                    # weight dependent - filter norm
                    print("nooooooooooooooooo")
                    # ci.append(self.filter_list[i_f].weights[:,i_w].norm(2).detach().cpu().numpy())
                elif self.ci_metric == 'l2': # this is the only one working - l2
                    # weight dependent - filter norm
                    ci.append(f.weights[:,i_w].norm(2).unsqueeze(0)) # .detach().cpu().numpy()) # .detach().cpu().numpy()
                elif self.ci_metric == '':
                    # weight dependent - filter correlation
                    print("nooooooooooooooooo")
                elif self.ci_metric == '':
                    # activation-based
                    print("nooooooooooooooooo")
                elif self.ci_metric == 'mi':
                    # mutual information
                    print("nooooooooooooooooo")
                elif self.ci_metric == 'tay':
                    # Hessian matrix / Taylor
                    print("nooooooooooooooooo")
                elif self.ci_metric == '':
                    print("nooooooooooooooooo")
                elif self.ci_metric == 'random':
                    ci.append(torch.rand(1))# np.array(random.random()))
                else:
                    print("DECENT WARNING: no valid ci metric chosen.")
                    
                    
                # tmp_importance is a single value, not a list like in the distances
                # is this not already a mean??
                # mean_value_of_weight = torch.nanmean(tmp_distances)
                # tmp_importance)
                    
                # max_importance_of_weight = torch.max(tmp_importance)
                # update max distance of layer
                
        #if max_importance_of_layer < max_importance_of_weight:
        #        max_importance_of_layer = max_importance_of_weight

        try:
            ci = torch.cat(ci)
            max_importance_of_layer = torch.max(ci) # torch.max(torch.tensor(ci))
            # mean importance of a layer
            # mean from all non-nan values - todo - check why there are nan values, maybe cause of pruning??
            mean_importance_of_layer = torch.nanmean(ci) # TODO, for the future torch.tensor(
            # maybe zero should be included though ... also, since we didn't remove whole filters that do not connect to anything, maybe there are problems
            # normalise with max value to scale between 0 and 1
            normalised_mean_importance_of_layer = mean_importance_of_layer / max_importance_of_layer+1e5
        except Exception as e:
            print(ci)
            print(e)
        
        return mean_importance_of_layer, normalised_mean_importance_of_layer # ci, normalised_ci
    
    
    def run_filter_channel_importance(self, i_f:int) -> list:
        # =============================================================================
        # compute channel importance metric for pruning
        # calculate the norm of each weight in filter with id i_f
        # we need to call this in a loop to go through each filter
        # returns:
        #     ci: channel importance list of a filter
        # notes:
        #     based on l2 norm = magnitude = euclidean distance
        # nonsense?
        #    maybe the kernel trigger todo
        #    print(self.filter_list[i_f].weights.shape)
        #    print(self.filter_list[i_f].weights[:,i_w].shape)
        # =============================================================================
        
        ci = []
        
        # print("DECENT NOTE: weight shape", self.filter_list[i_f].weights.shape)
        
        for i_w in range(self.filter_list[i_f].weights.shape[1]): # todo, sure this is 1 and not 0???
            # importance of a kernel in a layer
            
            if self.ci_metric == 'l1':
                # weight dependent - filter norm
                print("nooooooooooooooooo")
                pass
                # ci.append(self.filter_list[i_f].weights[:,i_w].norm(2).detach().cpu().numpy())
            elif self.ci_metric == 'l2': # this is the only one working - l2
                # weight dependent - filter norm
                ci.append(self.filter_list[i_f].weights[:,i_w].norm(2).detach().cpu().numpy()) # .detach().cpu().numpy()
                pass
            
            elif self.ci_metric == '':
                # weight dependent - filter correlation
                print("nooooooooooooooooo")
                pass
            
            elif self.ci_metric == '':
                # activation-based
                print("nooooooooooooooooo")
                pass
                
            elif self.ci_metric == 'mi':
                # mutual information
                print("nooooooooooooooooo")
                pass
                
            elif self.ci_metric == 'tay':
                # Hessian matrix / Taylor
                print("nooooooooooooooooo")
                pass
                
            elif self.ci_metric == '':
                print("nooooooooooooooooo")
                pass
                
            elif self.ci_metric == 'random':
                ci.append( np.array(random.random()) )
                
            else:
                print("DECENT WARNING: no valid ci metric chosen.")
                
        return ci
    
    def run_swap_filter(self):
        # =============================================================================
        # not working yet
        # we swap filters within the layer
        # based on connection cost
        # filter can move a maximum of two positions per swap
        # change positions
        # change
        # =============================================================================
        print("swap here")
        self.m_this = self.m_this # single integer
        self.n_this = self.n_this # single integer
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
        self.filter_list[i_f].weights = nn.Parameter(self.filter_list[i_f].weights[:, keep_ids, :, :])
        self.filter_list[i_f].ms_in = nn.Parameter(self.filter_list[i_f].ms_in[keep_ids], requires_grad=False) # this becomes a grad here, hence turn off again with False
        #[self.filter_list[i_f].ms_in[i] for i in keep_ids] # self.ms_in[remove_ids]
        self.filter_list[i_f].ns_in = nn.Parameter(self.filter_list[i_f].ns_in[keep_ids], requires_grad=False)
        # [self.filter_list[i_f].ns_in[i] for i in keep_ids] # self.ns_in[remove_ids]
        

    
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
            # for each channel
            
            # print("ms_in shape", f.ms_in.shape)
            for i_s in range(len(f.ms_in)):
                
                s = str(int(f.ms_in[i_s].item()))+'_'+str(int(f.ns_in[i_s].item()))
                sources.append(s)

                t = str(int(f.m_this.item()))+'_'+str(int(f.n_this.item()))
                targets.append(t)

                target_groups.append(self.layer_name)
            
            # get all channel importances
            ci = np.array(self.run_filter_channel_importance(i_f)).flatten()
            #print("CI"*50)
            #print(ci)
            values.extend(ci)
            
            """
            try:
                values.append( [val.item() for tmp in ci for val in tmp] )
            except:
                try:
                    a = [val.item() for val in ci]
                    values.append(  [val for tmp in a for val in tmp])
                except:
                    print("empty channel importance??")
                    values = []
                    
            """
            
        #print("lengths note:", len(sources), len(targets), len(target_groups), len(values))
        
        # value = channel importance
        return {'source':sources, 'target':targets, 'target_group':target_groups, f'importance':values}
            
    
    def update(self):
        # =============================================================================
        # currently: calculate importance metric for the prune_channel method
        # remove channels based on self.prune_keep
        # layerwise pruning - percentage of layer
        # =============================================================================
        
        all_ci = []
        all_len = 0
        for i_f in range(len(self.filter_list)):
            all_len += len(self.filter_list[i_f].ms_in)
            # list of lists
            all_ci.append(self.run_filter_channel_importance(i_f))
            #tmp_ids = sorted(range(len(all_ci)), key=lambda sub: all_ci[sub])
          
        #print(all_len) # this is the size of the previous pruning
        #print(self.original_size)
        #print(self.prune_keep_total)
        #print(int(self.original_size * self.prune_keep_total))
        
        #self.log(f'{self.original_size}_active_channels', all_len, on_step=True, on_epoch=True)
        
        if all_len < int(self.original_size * self.prune_keep_total):
            # if n percent have been pruned, stop this layer
            print("DECENT NOTE: pruning done for this layer")
        else:
            # pruning
            n = int(all_len*self.prune_keep)
            all_ci_flatten = [item for row in all_ci for item in row] # don't have equal lengths, so no numpy possible
            index = sorted(range(all_len), key=lambda sub: all_ci_flatten[sub])[-n] # error, out of range
            threshold_value = all_ci_flatten[index]

            for i_f in range(len(self.filter_list)):

                # channel importance list for this filter
                ci = all_ci[i_f] # list of self.run_filter_channel_importance(i_f)

                #print(ci)
                #print(threshold_value)
                # torch.where()
                            
                indices = np.where(ci >= threshold_value)[0] # just need the x axis

                # indices should be list/np/detached
                self.run_prune_channel(i_f, indices)
                
                #print("prune done")
                # ci = ci[indices] # probably not useful
        
            
            # print("channel importance ci", ci)
            # keep_ids = random.sample(range(0, 8), 5)
            #keep_ids = sorted(range(len(ci)), key=lambda sub: ci[sub])[amout_remove:]
            #print(keep_ids)
            
            # delete filters with no input channels - no we need to still find, if there is a filter that is
            # not used by any later filter - for that we probably want to do the same as with sugiyama ...
            # if shape[1] == 0, then we can remove the whole filter from the list
            if False:
                try:
                    remove_list = []
                    for i_f, f in enumerate(self.filter_list):
                        if f.weights.shape[1] == 0: 
                            print("DECENT NOTE: we just removed a filter that was empty")
                            remove_list.append(i_f)
                            self.filter_list.pop(i_f)
                except Exception as e:
                    print(e)
                    
        self.run_prune_filter() # remove whole filter if zero channels in a filter
