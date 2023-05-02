import random

import torch
import torch.nn


class GaussianNoiseBlock(torch.nn.Module):
    # =============================================================================
    # Add Gaussian noise to a 4D tensor
    # sources: 
    #    https://github.com/ZHKKKe/PixelSSL/blob/master/pixelssl/nn/module/gaussian_noise.py
    # =============================================================================

    def __init__(self, std):
        super(GaussianNoiseBlock, self).__init__()
        self.std = std
        self.noise = torch.zeros(0)
        self.enable = False if self.std is None else True

    def forward(self, x):
        
        if not self.enable:
            return x

        if self.noise.shape != x.shape:
            self.noise = torch.zeros(x.shape).cuda()
        self.noise.data.normal_(0, std=random.uniform(0, self.std))

        imax = x.max(dim=3, keepdim=True)[0].max(dim=2, keepdim=True)[0].max(dim=1, keepdim=True)[0]
        imin = x.min(dim=3, keepdim=True)[0].min(dim=2, keepdim=True)[0].min(dim=1, keepdim=True)[0]
        
        # normalize to [0, 1]
        x.sub_(imin).div_(imax - imin + 1e-9)
        # add noise
        x.add_(self.noise)
        # clip to [0, 1]
        upper_bound = (x > 1.0).float()
        lower_bound = (x < 0.0).float()
        x.mul_(1 - upper_bound).add_(upper_bound)
        x.mul_(1 - lower_bound)
        # de-normalize
        x.mul_(imax - imin + 1e-9).add_(imin)

        return x