from compute.loss.template import *

import torch
import torch.nn.functional
from torch.nn.modules.loss import _Loss
import numpy as np
from typing import Optional, List

def to_tensor(x, dtype=None) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    if isinstance(x, (list, tuple)):
        x = np.array(x)
        x = torch.from_numpy(x)
        if dtype is not None:
            x = x.type(dtype)
        return x
    
def soft_dice_score(
    output: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 0.0,
    eps: float = 1e-7,
    dims=None,
) -> torch.Tensor:
    assert output.size() == target.size()
    if dims is not None:
        intersection = torch.sum(output * target, dim=dims)
        cardinality = torch.sum(output + target, dim=dims)
    else:
        intersection = torch.sum(output * target)
        cardinality = torch.sum(output + target)
    dice_score = (2.0 * intersection + smooth) / (cardinality + smooth).clamp_min(eps)
    return dice_score

class DiceLoss(_Loss):
    """Dice loss for image segmentation task.
        Parameters:
            log_loss: If True, loss computed as `- log(dice_coeff)`, otherwise `1 - dice_coeff`
            smooth: Smoothness constant for dice coefficient (a)
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)
        
        # extreme values 0 and 1
        Shape
             - **model_output** - torch.Tensor of shape (B, C, H, W)
             - **ground_truth** - torch.Tensor of shape (B, H, W) or (B, C, H, W)
        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
            https://github.com/qubvel/segmentation_models.pytorch
            # changed a lot
        """
    
    def __init__(
        self,
        n_output_neurons : int = 1,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        eps: float = 1e-7,
    ):
        super(DiceLoss, self).__init__()

        self.n_output_neurons = n_output_neurons
        self.log_loss = log_loss
        self.smooth = smooth
        self.eps = eps


    def forward(self, model_output: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:

        assert ground_truth.size(0) == model_output.size(0)


        bs = ground_truth.size(0)
        dims = (0, 2)

        if self.n_output_neurons == 1: # Binary
            
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            model_output = torch.nn.functional.logsigmoid(model_output).exp()
            
            ground_truth = ground_truth.view(bs, 1, -1)
            model_output = model_output.view(bs, 1, -1)

            mask = ground_truth
            model_output = model_output * mask
            ground_truth = ground_truth * mask

        if self.n_output_neurons > 1: # Multiclass
            
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            model_output = model_output.log_softmax(dim=1).exp()
            
            ground_truth = ground_truth.view(bs, -1)
            model_output = model_output.view(bs, self.n_output_neurons, -1)

            mask = ground_truth
            model_output = model_output * mask.unsqueeze(1)

            ground_truth = torch.nn.functional.one_hot((ground_truth * mask).to(torch.long), self.n_output_neurons)  # N,H*W -> N,H*W, C
            ground_truth = ground_truth.permute(0, 2, 1) * mask.unsqueeze(1)  # N, C, H*W

        scores = self.compute_score(model_output, ground_truth.type_as(model_output), smooth=self.smooth, eps=self.eps, dims=dims)

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(model_output)`
        # for this case, however it will be a modified jaccard loss

        mask = ground_truth.sum(dims) > 0
        loss *= mask.to(loss.dtype)

        return self.aggregate_loss(loss)

    def aggregate_loss(self, loss):
        return loss.mean()

    def compute_score(self, output, target, smooth=0.0, eps=1e-7, dims=None) -> torch.Tensor:
        return soft_dice_score(output, target, smooth, eps, dims)
