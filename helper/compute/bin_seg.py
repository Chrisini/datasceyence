from compute.template import *
from compute.metric.symmetric_hausdorff import *

class BCE_BinSeg_CU(TemplateComputingUnit):
    
    # =============================================================================
    # 
    # =============================================================================
    
    def __init__(self, n_output_neurons, mode="train", bin_threshold=0.5):
        super().__init__(n_output_neurons=n_output_neurons, mode=mode)
                
        # head name
        self.name = type(self).__name__
        self.mode = mode
        self.datasize = 0
        
        self.bin_threshold = bin_threshold # needed if 1 output neuron
        
        self.task_loss = None
        
        # output neurons - 1 for reg/bin-ce and 2+ for multi-class
        self.n_output_neurons = n_output_neurons
        
        
        b_keys = [  'loss',
                    'acc', 'fscore', 'fmicro', 'jac', 'prec', 'rec',
                    'symhd' # task specific: segmentation
                ]
        
        e_keys = [  'name', 'mode', 'epoch', 'datasize',
                    'loss',
                    'acc', 'fscore', 'fmicro', 'jac', 'prec', 'rec',
                    'symhd', # task specific: segmentation
                ]
        
        # everything we want to track
        self.batch_collector = dict.fromkeys(b_keys, [])
        
        # everything we want to write to the csv file
        self.epoch_collector = dict.fromkeys(e_keys, None)
        
        self.top_collector = {  'lowest_loss' : 0, 
                                'highest_fscore' : np.inf, 
                                'highest_jac' : np.inf, 
                                'highest_symhd' : np.inf # high or low good??
                             }
        
        self.symhd = SymmetricHausdorffMetric()

         
    def run_batch(self, configs, criterions, model_output, ground_truth):
        
        # data size
        self.datasize += len(ground_truth)
        
        # loss
        shape_loss = criterions["shape"](model_output=model_output, ground_truth=ground_truth) * 0.7
        pixel_loss = criterions["pixel"](model_output=model_output, ground_truth=ground_truth) * 0.3
        self.task_loss = shape_loss + pixel_loss  
        self.batch_collector["loss"].append(self.task_loss)
        
        
        if self.n_output_neurons > 1:
            _, highest_class = torch.max(model_output, 1)    
            highest_class = highest_class.detach().cpu().numpy() # [2:3].squeeze()
        else:
            highest_class = torch.sigmoid(model_output) > self.bin_threshold 
            highest_class = highest_class.detach().cpu().numpy()
            
        ground_truth = ground_truth.detach().cpu().numpy() # .squeeze()[2:3].squeeze()

        
        # Find the general (symmetric) Hausdorff distance between two 2-D arrays of coordinates:
        if True:
            self.batch_collector["symhd"].append(self.symhd(highest_class, ground_truth))

    
    def run_epoch(self, epoch):
        
        self.epoch_collector["name"] = self.name
        self.epoch_collector["mode"] = self.mode
        self.epoch_collector["datasize"] = self.datasize
        self.epoch_collector["epoch"] = epoch
        
        # save mean value to epoch_collector, reset batch_collector
        for key in self.batch_collector.keys():
            self.epoch_collector[key] = np.mean(self.batch_collector[key])
            self.batch_collector[key] = []
        
        
        
        # log all the stuff
        print(self.epoch_collector)
        
        # reset
        self.data_size = 0
            
    

    
    

        
    
    

        
        
        
    
    