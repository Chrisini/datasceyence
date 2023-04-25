from sklearn.metrics import *
from scipy.spatial.distance import directed_hausdorff

class BCE_BinSeg_CU(TemplateComputingUnit):
    
    # =============================================================================
    # 
    # =============================================================================
    
    def __init__(self, n_output_neurons=1):
        super().__init__()
        # error if head is used instead of subclass
        if type(self) == TemplateComputingUnit:
             raise Exception("<Head> must be sub-classed.")
                
        # head name
        self.head_name = type(self).__name__
        
        # store
        self.epoch_collector = {
            "metrics" : {},
            "loss" : {}
        }
            
        self.best = {
            "jaccard" : 0.0, 
            "fscore"  : 0.0,
            "sym_hd"  : 0.0}
        
        # output neurons - 1 for reg/bin-ce and 2+ for multi-class
        self.n_output_neurons = n_output_neurons
         
    def run_batch(self, configs, model_output, ground_truth, mode):
        
        
            # forward the student model
            s_resulter, s_debugger = self.models["s"].forward(self.item["img_s"])



            # calculate the supervised task constraint on the labeled data
            l_s_pred = func.split_tensor_tuple(s_pred, 0, lbs)
            l_gt = func.split_tensor_tuple(gt, 0, lbs)
            l_s_inp = func.split_tensor_tuple(self.item["img_s"], 0, lbs)

            # 'task_loss' is a tensor of 1-dim & n elements, where n == batch_size
            s_task_loss = self.s_criterion.forward(l_s_pred, l_gt, l_s_inp)
            s_task_loss = torch.mean(s_task_loss)
            self.meters.update('s_task_loss', s_task_loss.data)
        
        # Find the general (symmetric) Hausdorff distance between two 2-D arrays of coordinates:
        self.epoch_collector["metrics"]["sym_hd"] = max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
        
        
        return self.epoch_collector
    
    
    
    def run_iter(self, configs, mode):
        
        
        
        return self.epoch_collector
    
    

    
    

        
    
    

        
        
        
    
    