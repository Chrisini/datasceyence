from compute.loss.template import *

import torch.nn


class DiceLoss(torch.nn.Module):
    # Source: https://github.com/HiLab-git/SSL4MIS/blob/master/code/utils/losses.py
    def __init__(self, n_output_neurons):
        super(DiceLoss, self).__init__()
        self.n_output_neurons = n_output_neurons

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_output_neurons):
            temp_prob = input_tensor == i * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob)
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, model_output, ground_truth):
        ground_truth = ground_truth.float()
        smooth = 1e-5
        intersect = torch.sum(model_output * ground_truth)
        y_sum = torch.sum(ground_truth * ground_truth)
        z_sum = torch.sum(model_output * model_output)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, model_output, ground_truth, weight=None, softmax=False):
        if softmax:
            model_output = torch.softmax(model_output, dim=1)
        ground_truth = self._one_hot_encoder(ground_truth)
        
        if weight is None:
            weight = [1] * self.n_output_neurons
        assert model_output.size() == ground_truth.size(), 'predict & ground_truth shape do not match'
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_output_neurons):
            dice = self._dice_loss(model_output[:, i], ground_truth[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_output_neurons