from scipy.spatial.distance import directed_hausdorff
import torch

import numpy as np

class SymmetricHausdorffMetric():
    # currently the sym hd skips each zero distance
    # invalid: if either ground truth or model output have no positive class, aka result = 0.0
    # returns "inf" if all samples are invalid
    
    # needs tensors
    
    def __init__(self):
        pass
    
    def __call__(self, model_output: torch.Tensor, ground_truth: torch.Tensor):
        
        # model_output batch_size, number of output neurons, w, h
        # ground truth batch_size, 1, w, h
        
        _, highest_class = torch.max(model_output, 1) # remove "number of output neurons"
        _, ground_truth = torch.max(ground_truth, 1)  # remove "1"
        
        #print(highest_class[0].shape)
        #print(ground_truth[0].shape)
        
        highest_class = highest_class.detach().cpu().numpy()
        ground_truth = ground_truth.detach().cpu().numpy()
        
        hds = []
        
        for h, g in zip(highest_class, ground_truth):
        
            hd = max(directed_hausdorff(u=h, v=g)[0], 
                    directed_hausdorff(v=g, u=h)[0])
            
            # 
            
            if hd > 0: # distance of zero is impossible
                hds.append(hd)
            else:
                pass # skip those values? or should we give inf?
                # would also need to check, whether there is a positive class in the ground truth

        if not hds:
            # if empty, give value inf
            hds = [np.inf]
            
        return np.mean(hds)
    
    