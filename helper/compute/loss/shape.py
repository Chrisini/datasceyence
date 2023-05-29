from compute.loss.template import *
    
from PIL import Image
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg    

    
import cv2 as cv
import numpy as np

import torch
from torch import nn

from scipy.ndimage.morphology import distance_transform_edt as edt


# rename this to distance - they are all working with distance map

# https://github.com/JunMa11/SegLoss

class SignedDistanceLoss(TemplateLoss):
    # SignedDistanceLoss
    # AAAI_sdf_loss
    # https://github.com/JunMa11/SegWithDistMap/blob/master/code/utils/losses.py
    
    def __init__(self, smooth=1e-5, device="cpu", n_output_neurons=2):
        super(SignedDistanceLoss, self).__init__()
        self.smooth = smooth
        self.device = device
        self.n_output_neurons = n_output_neurons
        
    def compute_sdf1_1(self, segmentation):
        # Signed distance function
        # https://github.com/JunMa11/SegWithDistMap/blob/master/code/utils/losses.py
        """
        compute the signed distance map of binary mask
        input: segmentation, shape = (batch_size, class, x, y, z)
        output: the Signed Distance Map (SDM) 
        sdm(x) = 0; x in segmentation boundary
                 -inf|x-y|; x in segmentation
                 +inf|x-y|; x out of segmentation
        """
        # print(type(segmentation), segmentation.shape)

        segmentation = segmentation.astype(np.uint8)
        if len(segmentation.shape) == 4: # 3D image
            segmentation = np.expand_dims(segmentation, 1)
        normalized_sdf = np.zeros(segmentation.shape)
        if segmentation.shape[1] == 1:
            dis_id = 0
        else:
            dis_id = 1
        for b in range(segmentation.shape[0]): # batch size
            for c in range(dis_id, segmentation.shape[1]): # class_num
                # ignore background
                posmask = segmentation[b][c]
                negmask = ~posmask
                posdis = distance(posmask)
                negdis = distance(negmask)
                boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                sdf = negdis/np.max(negdis) - posdis/np.max(posdis)
                sdf[boundary>0] = 0
                normalized_sdf[b][c] = sdf
        return normalized_sdf
    
    def sum_tensor(self, inp, axes, keepdim=False):
        axes = np.unique(axes).astype(int)
        if keepdim:
            for ax in axes:
                inp = inp.sum(int(ax), keepdim=True)
        else:
            for ax in sorted(axes, reverse=True):
                inp = inp.sum(int(ax))
        return inp
    
    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_output_neurons):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
        
    def forward(self, model_output, ground_truth, weight:int=1):
        # weight of the whole loss value
        
        ground_truth = self._one_hot_encoder(ground_truth)
        
        print('model_output.shape, ground_truth.shape', model_output.shape, ground_truth.shape)
        # ([4, 1, 112, 112, 80])
        
        axes = tuple(range(2, len(model_output.size())))
        shp_x = model_output.shape
        shp_y = ground_truth.shape

        with torch.no_grad():
            if len(shp_x) != len(shp_y):
                ground_truth = ground_truth.view((shp_y[0], 1, *shp_y[1:]))

            if all([i == j for i, j in zip(model_output.shape, ground_truth.shape)]):
                # if this is the case then ground_truth is probably already a one hot encoding
                y_onehot = ground_truth
            else:
                ground_truth = ground_truth.long()
                y_onehot = torch.zeros(shp_x)
                y_onehot = y_onehot.to(self.device)
                
                y_onehot.scatter_(1, ground_truth, 1)
                
                
            ground_truth_sdm_npy = self.compute_sdf1_1(y_onehot.cpu().numpy())
            ground_truth_sdm = torch.from_numpy(ground_truth_sdm_npy).float().to(self.device)
        
        print(y_onehot.shape)
        print(ground_truth_sdm_npy.shape)
        print(ground_truth_sdm.shape)
        
        intersect = self.sum_tensor(model_output * ground_truth_sdm, axes, keepdim=False)
        pd_sum = self.sum_tensor(model_output ** 2, axes, keepdim=False)
        ground_truth_sum = self.sum_tensor(ground_truth_sdm ** 2, axes, keepdim=False)
        
        print(intersect.shape)
        print(pd_sum.shape)
        print(ground_truth_sum.shape)
        
        try: 
            print((intersect + ground_truth_sum).shape)
        except:
            print("not working")
        
        try: 
            print((intersect + pd_sum).shape)
        except:
            print("not working")
        
        try: 
            print((pd_sum + ground_truth_sum).shape)
        except:
            print("not working")
        
        
        try: 
            print((intersect / (pd_sum + ground_truth_sum)).shape)
        except:
            print("not working")
        
        print("error appearing now")
        
        L_product = (intersect + self.smooth) / (intersect + pd_sum + ground_truth_sum)
        # print('L_product.shape', L_product.shape) (4,2)
        L_SDF_AAAI = - L_product.mean() + torch.norm(model_output - ground_truth_sdm, 1)/torch.numel(model_output)

        return L_SDF_AAAI * weight
    
    
    

"""
Hausdorff loss implementation based on paper:
https://arxiv.org/pdf/1904.10030.pdf
"""

# https://github.com/PatRyg99/HausdorffLoss/blob/master/hausdorff_loss.py

class HausdorffDTLoss(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img: np.ndarray) -> np.ndarray:
        field = np.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(
        self, model_output: torch.Tensor, ground_truth: torch.Tensor, debug=False
    ) -> torch.Tensor:
        
        
        # or this torch.amax(a, 1)
        
        _, model_output = torch.max(model_output, 1)
        
        # ground_truth = ground_truth.squeeze(1)
        model_output = model_output.unsqueeze(1)
                
        """
        Uses one binary channel: 1 - fg, 0 - bg
        model_output: (b, 1, x, y, z) or (b, 1, x, y)
        ground_truth: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert model_output.dim() == 4 or model_output.dim() == 5, "Only 2D and 3D supported"
        assert (
            model_output.dim() == ground_truth.dim()
        ), "model_outputiction and ground_truth need to be of same dimension"

        # model_output = torch.sigmoid(model_output)

        model_output_dt = torch.from_numpy(self.distance_field(model_output.detach().cpu().numpy())).float()
        ground_truth_dt = torch.from_numpy(self.distance_field(ground_truth.detach().cpu().numpy())).float()

        model_output_error = ((model_output - ground_truth) ** 2).detach().cpu()
        distance = model_output_dt ** self.alpha + ground_truth_dt ** self.alpha

        dt_field = model_output_error * distance
        loss = dt_field.mean()

        if debug:
            return (
                loss.cpu().numpy(),
                (
                    dt_field.cpu().numpy()[0, 0],
                    model_output_error.cpu().numpy()[0, 0],
                    distance.cpu().numpy()[0, 0],
                    model_output_dt.cpu().numpy()[0, 0],
                    ground_truth_dt.cpu().numpy()[0, 0],
                ),
            )

        else:
            return torch.tensor(loss)
        
        
        
# https://github.com/PatRyg99/HausdorffLoss/blob/master/hausdorff_loss.py

class HausdorffDTLossTorch(nn.Module):
    """Binary Hausdorff loss based on distance transform"""

    def __init__(self, alpha=2.0):
        super(HausdorffDTLoss, self).__init__()
        self.alpha = alpha

    @torch.no_grad()
    def distance_field(self, img:torch.tensor) -> torch.tensor:
        field = torch.zeros_like(img)

        for batch in range(len(img)):
            fg_mask = img[batch] > 0.5

            if fg_mask.any():
                bg_mask = ~fg_mask

                fg_dist = edt(fg_mask)
                bg_dist = edt(bg_mask)

                field[batch] = fg_dist + bg_dist

        return field

    def forward(
        self, model_output: torch.Tensor, ground_truth: torch.Tensor, debug=False
    ) -> torch.Tensor:
        
        
        # or this torch.amax(a, 1)
        
        _, model_output = torch.max(model_output, 1)
        
        # ground_truth = ground_truth.squeeze(1)
        model_output = model_output.unsqueeze(1)
                
        """
        Uses one binary channel: 1 - fg, 0 - bg
        model_output: (b, 1, x, y, z) or (b, 1, x, y)
        ground_truth: (b, 1, x, y, z) or (b, 1, x, y)
        """
        assert model_output.dim() == 4 or model_output.dim() == 5, "Only 2D and 3D supported"
        assert (
            model_output.dim() == ground_truth.dim()
        ), "model_outputiction and ground_truth need to be of same dimension"

        # model_output = torch.sigmoid(model_output)

        model_output_dt = torch.from_numpy(self.distance_field(model_output.detach().cpu().numpy())).float()
        ground_truth_dt = torch.from_numpy(self.distance_field(ground_truth.detach().cpu().numpy())).float()

        model_output_error = ((model_output - ground_truth) ** 2).detach().cpu()
        distance = model_output_dt ** self.alpha + ground_truth_dt ** self.alpha

        dt_field = model_output_error * distance
        loss = dt_field.mean()

        if debug:
            return (
                loss.cpu().numpy(),
                (
                    dt_field.cpu().numpy()[0, 0],
                    model_output_error.cpu().numpy()[0, 0],
                    distance.cpu().numpy()[0, 0],
                    model_output_dt.cpu().numpy()[0, 0],
                    ground_truth_dt.cpu().numpy()[0, 0],
                ),
            )

        else:
            return torch.tensor(loss)
        
        
        