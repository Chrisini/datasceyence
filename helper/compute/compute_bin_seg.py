from compute.template import *

from sklearn.metrics import *
from scipy.spatial.distance import directed_hausdorff



class BCE_BinSeg_CU(TemplateComputingUnit):
    
    # =============================================================================
    # 
    # =============================================================================
    
    def __init__(self, n_output_neurons=1, mode="train"):
        super().__init__(n_output_neurons=n_output_neurons, mode=mode)
                
        # head name
        self.name = type(self).__name__
        self.mode = mode
        self.epoch = 0
        self.data_size = 0
        
        self.task_loss = None
        
        
        keys = [#'name', 'mode', # stays the same - needs to be re-saved
                #'epoch',
                #'train_size','val_size', # train val size
                # multiple data
                'loss',
                'acc', 
                'fscore', 'f_micro',
                'jac', 'prec', 'rec',
                 #'kappa', # task specific: multiclass
                 #'mse', 'mae', # task specific: regression
                'hd_sym' # task specific: segmentation
                ]
        
        # everything we want to write to the csv file
        self.epoch_collector = dict.fromkeys(keys, [])
        self.batch_collector = dict.fromkeys(keys, [])
        

        self.best = {
            "jaccard" : 0.0, 
            "fscore"  : 0.0,
            "sym_hd"  : 0.0}
        
        # output neurons - 1 for reg/bin-ce and 2+ for multi-class
        self.n_output_neurons = n_output_neurons
         
    def run_batch(self, configs, criterions, model_output, ground_truth):
        
        self.data_size += len(ground_truth)
        
        shape_loss = criterions["shape"](model_output=model_output, ground_truth=ground_truth) * 0.7
        pixel_loss = criterions["pixel"](model_output=model_output, ground_truth=ground_truth) * 0.3
        
        self.task_loss = shape_loss + pixel_loss  
        

        
        self.batch_collector["loss"].append(self.task_loss)
        print(self.batch_collector["loss"])
        
        
        # Find the general (symmetric) Hausdorff distance between two 2-D arrays of coordinates:
        self.epoch_collector["sym_hd"] = None # # max(directed_hausdorff(u, v)[0], directed_hausdorff(v, u)[0])
    
    
    def run_epoch(self):
        
        # append to epoch_collector, reset batch_collector
        for key in self.keys:
            self.epoch_collector[key].append(np.mean(self.batch_collector[key]))
            self.batch_collector[key] = []
        
        self.epoch += 1
        
        self.data_size = 0
        
        return self.epoch_collector
    
    

    
    

        
    
    

        
        
        
    
    