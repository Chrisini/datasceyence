import torch
import torch.nn as nn
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple
import math
import numpy as np

class DecentFilter(nn.Module):
    # =============================================================================
    #
    # convolution happens in here
    # one filter has multiple channels (aka weights)
    # conv2d problem: https://stackoverflow.com/questions/61269421/expected-stride-to-be-a-single-integer-value-or-a-list-of-1-values-to-match-the
    #
    # =============================================================================
    
    def __init__(self, ms_in, ns_in, m_this, n_this,
                 ckpt_weight=None,
                 kernel_size=3, 
                 stride=1, 
                 padding=0, 
                 padding_mode="zeros",
                 dilation=3, 
                 device=None, 
                 dtype=None) -> None:
        
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        
        # padding
        padding = padding if isinstance(padding, str) else _pair(padding)
        valid_padding_strings = {'same', 'valid'}
        if isinstance(padding, str):
            if padding not in valid_padding_strings:
                raise ValueError(
                    "Invalid padding string {!r}, should be one of {}".format(
                        padding, valid_padding_strings))
            if padding == 'same' and any(s != 1 for s in stride):
                raise ValueError("padding='same' is not supported for strided convolutions")
        valid_padding_modes = {'zeros', 'reflect', 'replicate', 'circular'}
        if padding_mode not in valid_padding_modes:
            raise ValueError("padding_mode must be one of {}, but got padding_mode='{}'".format(
                valid_padding_modes, padding_mode))
        
         
        # convolution
        self.kernel_size = _pair(kernel_size)
        self.stride = stride
        self.padding_mode = padding_mode
        self.padding = padding
        self.dilation = _pair(dilation)
        
        #print("weight shape init")
        #print(self.weights.shape)
            
        # bias    
        if False: 
            # bias:
            # where should the bias be???
            self.bias = Parameter(torch.empty(1, **factory_kwargs))
        else:
            #self.bias = False
            # we only use bias via instance normalisation
            self.register_parameter('bias', None)
        
        
        # =============================================================================
        # from ckpt or from scratch
        # =============================================================================
        if ckpt_weight == None:
            # reset weights (and bias) in filter
            
            # positions, currently not trainable 
            # self.non_trainable_param = nn.Parameter(torch.Tensor([1.0]), requires_grad=False)
            self.ms_in = nn.Parameter(torch.Tensor(ms_in), requires_grad=False) # ms_in # list
            self.ns_in = nn.Parameter(torch.Tensor(ns_in), requires_grad=False) # ns_in # list
            self.m_this = nn.Parameter(torch.Tensor([m_this]), requires_grad=False) # m_this # single integer
            self.n_this = nn.Parameter(torch.Tensor([n_this]), requires_grad=False) # n_this # single integer
            
            # weights
            assert len(ms_in) == len(ns_in), "ms_in and ns_in are not of same length"
            self.n_weights = len(ms_in)
            # filters x channels x kernel x kernel
            # self.weights = torch.autograd.Variable(torch.randn(1,n_weights,*self.kernel_size)).to("cuda")
            # self.weights = nn.Parameter(torch.randn(1,n_weights,*self.kernel_size))
            self.weights = nn.Parameter(torch.empty((1, self.n_weights, *self.kernel_size), **factory_kwargs))
            
            # reset
            self.reset_parameters()
            
        else: 
            # load from checkpoint
            
            # positions
            self.ms_in = nn.Parameter(ms_in, requires_grad=False).to(device)
            self.ns_in = nn.Parameter(ns_in, requires_grad=False).to(device)
            self.m_this = nn.Parameter(m_this, requires_grad=False).to(device)
            self.n_this = nn.Parameter(n_this, requires_grad=False).to(device)
            
            # weights
            self.weights = nn.Parameter(ckpt_weight).to(device)
            
            
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*self.kernel_size)
        # For more details see: https://github.com/pytorch/pytorch/issues/15314#issuecomment-477448573
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))        
        
    def forward(self, x) -> torch.Tensor:
        # =============================================================================
        # remove channels in X because some channels in the filter are pruned (aka gone)
        # then we can apply convolution
        # parameters:
        #    x = batch x channels x width x height
        # returns:
        #    x_data: batch x filters x width x height
        # saves:
        #    self.weights = 1 filter x channels x kernel x kernel
        # =============================================================================
    
        # position matcher of filter (in) and image (x)
        # Find the indices (IDs) of channel pairs that exist in both the X and then filter
        common_pairs = [[i_in, i_x] for i_in, (m_in, n_in) in enumerate(zip(self.ms_in, self.ns_in)) for i_x, (m_x, n_x) in enumerate(zip(x.ms_x, x.ns_x)) if (m_in==m_x and n_in==n_x)]
        common_pairs_np = np.array(common_pairs)

        if False:
            print("length of in and x pairs for m:", len(self.ms_in), "=", len(x.ms_x), ", n:", len(self.ns_in), len(x.ns_x))
            for pair in common_pairs:
                print(f"Common pair at indices {pair}: {self.ms_in[pair[0]], tmp_ms[pair[1]]}, {self.ns_in[pair[0]], tmp_ns[pair[1]]}")
        
        try:
            f_ids = common_pairs_np[:,0]
            x_ids = common_pairs_np[:,1]
        except Exception as e:
            print("DECENT EXCEPTION: no common pairs")
            
            if False:
                print("error: no common pairs")
                print("pairs", common_pairs_np)
                print("pairs shape", common_pairs_np.shape)
                print("len ms in", len(self.ms_in))
                print("len ns in", len(self.ns_in))
                print("len ms x", len(x.ms_x))
                print("len ns x", len(x.ns_x))
                print(e)
        
            # if no common pairs: whole filter should be ignored aka no convolution   
            return None
        
        # filter data and weights based on common pairs of data and weights
        tmp_d = x.data[:, x_ids, :, :]
        tmp_w = self.weights[:, f_ids, :, :]
        
        # the final convolution
        if self.padding_mode != 'zeros':
            # this is written in c++
            x_data = nn.functional.conv2d(torch.nn.functional.pad(tmp_d, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            tmp_w, self.bias, self.stride,
                            _pair(0), self.dilation, groups=1)
        else:
            # this is written in c++
            x_data = nn.functional.conv2d(tmp_d, tmp_w, self.bias, self.stride, self.padding, self.dilation, groups=1)
        
        #print("tmp_w", tmp_w.shape)
        
        # print(x_data.shape, "- batch x filters x width x height")
        return x_data
    
    def setter(self, value, m_this, n_this):
        # preliminary, not in use
        self.weights = value # weights in this filter
        self.m_this = m_this # single integer
        self.n_this = n_this # single integer
    
    def getter(self):
        # preliminary, not in use
        return self.weights, self.m_this, self.n_this
    
    def __str__(self):
        return 'DecentFilter(weights: ' + str(self.weights.shape) + ' at position: m_this=' + str(self.m_this) + ', n_this=' + str(self.n_this) + ')' + \
    '\n with inputs: ms_in= ' + ', '.join(str(int(m.item())) for m in self.ms_in) + ', ns_in= ' + ', '.join(str(int(n.item())) for n in self.ns_in) + ')'
    __repr__ = __str__
    
        