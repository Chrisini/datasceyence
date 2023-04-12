from loss.template import *

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
    
def soft_jaccard_score(
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

    union = cardinality - intersection
    jaccard_score = (intersection + smooth) / (union + smooth).clamp_min(eps)
    return jaccard_score

class IoULoss(_Loss):
    def __init__(
        self,
        mode: str,
        classes: Optional[List[int]] = None,
        log_loss: bool = False,
        from_logits: bool = True,
        smooth: float = 0.0,
        eps: float = 1e-7,
    ):
        """Jaccard loss for image segmentation task.
        It supports binary, multiclass and multilabel cases
        Args:
            mode: Loss mode 'binary', 'multiclass' or 'multilabel'
            classes:  List of classes that contribute in loss computation. By default, all channels are included.
            log_loss: If True, loss computed as `- log(jaccard_coeff)`, otherwise `1 - jaccard_coeff`
            from_logits: If True, assumes input is raw logits
            smooth: Smoothness constant for dice coefficient
            eps: A small epsilon for numerical stability to avoid zero division error
                (denominator will be always greater or equal to eps)
        Shape
             - **model_output** - torch.Tensor of shape (N, C, H, W)
             - **ground_truth** - torch.Tensor of shape (N, H, W) or (N, C, H, W)
        Reference
            https://github.com/BloodAxe/pytorch-toolbelt
        """
        assert mode in {BINARY_MODE, MULTILABEL_MODE, MULTICLASS_MODE}
        super(IoULoss, self).__init__()

        self.mode = mode
        if classes is not None:
            assert mode != BINARY_MODE, "Masking classes is not supported with mode=binary"
            classes = to_tensor(classes, dtype=torch.long)

        self.classes = classes
        self.from_logits = from_logits
        self.smooth = smooth
        self.eps = eps
        self.log_loss = log_loss

    def forward(self, model_output: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:

        assert ground_truth.size(0) == model_output.size(0)

        if self.from_logits:
            # Apply activations to get [0..1] class probabilities
            # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
            # extreme values 0 and 1
            if self.mode == MULTICLASS_MODE:
                model_output = model_output.log_softmax(dim=1).exp()
            else:
                model_output = torch.nn.functional.logsigmoid(model_output).exp()

        bs = ground_truth.size(0)
        num_classes = model_output.size(1)
        dims = (0, 2)

        if self.mode == BINARY_MODE:
            ground_truth = ground_truth.view(bs, 1, -1)
            model_output = model_output.view(bs, 1, -1)

        if self.mode == MULTICLASS_MODE:
            ground_truth = ground_truth.view(bs, -1)
            model_output = model_output.view(bs, num_classes, -1)

            ground_truth = torch.nn.functional.one_hot(ground_truth, num_classes)  # N,H*W -> N,H*W, C
            ground_truth = ground_truth.permute(0, 2, 1)  # H, C, H*W

        if self.mode == MULTILABEL_MODE:
            ground_truth = ground_truth.view(bs, num_classes, -1)
            model_output = model_output.view(bs, num_classes, -1)

        scores = soft_jaccard_score(
            model_output,
            ground_truth.type(model_output.dtype),
            smooth=self.smooth,
            eps=self.eps,
            dims=dims,
        )

        if self.log_loss:
            loss = -torch.log(scores.clamp_min(self.eps))
        else:
            loss = 1.0 - scores

        # IoU loss is defined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(model_output)`
        # for this case, however it will be a modified jaccard loss

        mask = ground_truth.sum(dims) > 0
        loss *= mask.float()

        if self.classes is not None:
            loss = loss[self.classes]

        return loss.mean()