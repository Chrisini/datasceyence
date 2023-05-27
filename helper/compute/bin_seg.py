from compute.template import *
from compute.metric.symmetric_hausdorff import *
from compute.metric.segmentation import *

class BCE_BinSeg_CU(TemplateComputingUnit):
    
    # =============================================================================
    # 
    # =============================================================================
    
    def __init__(self, n_output_neurons, mode="train", bin_threshold=0.5, model="", device="cpu", writer=None):
        super().__init__(n_output_neurons=n_output_neurons, mode=mode, model=model)
                
        # head name
        self.name = type(self).__name__
        self.mode = mode
        self.datasize = 0
        self.model = model
        self.device = device
        
        self.writer = writer
        
        self.bin_threshold = bin_threshold # needed if 1 output neuron
        
        self.task_loss = None
        
        # output neurons - 1 for reg/bin-ce and 2+ for multi-class
        self.n_output_neurons = n_output_neurons
        
        
        b_keys = [  'loss',
                    'acc', 'fscore', 'jac', 'prec', 'rec',
                    'symhd' # task specific: segmentation
                ]
        
        e_keys = [  'name', 'model', 'mode', 'epoch', 'datasize',
                    'loss',
                    'acc', 'fscore', 'jac', 'prec', 'rec',
                    'symhd', # task specific: segmentation
                ]
        
        # everything we want to track
        self.batch_collector = { key : [] for key in b_keys }

        
        # everything we want to write to the csv file
        self.epoch_collector = { key : None for key in e_keys }
        
        self.top_collector = {  'lowest_loss' : 0, 
                                'highest_fscore' : np.inf, 
                                'highest_jac' : np.inf, 
                                'highest_symhd' : np.inf # high or low good??
                             }
        
        self.symhd = SymmetricHausdorffMetric()
        self.segmet = SegmentationMetrics()

         
    def run_batch(self, configs, criterions, model_output, ground_truth):
        
        
        
        # data size
        self.datasize += len(ground_truth)
        
        ground_truth = ground_truth.to(self.device)
        
        # loss
        self.shape_loss = criterions["shape"](model_output=model_output, ground_truth=ground_truth)
        self.pixel_loss = criterions["pixel"](model_output=model_output, ground_truth=ground_truth)
        
        #print("shape loss", self.shape_loss)
        #print("pixel loss", self.pixel_loss)
        
        # use this for backprop
        self.task_loss = self.pixel_loss # + other + weighted
        
        # this loss should not be used for backprop
        # detach: returns a new Tensor, detached from the current graph.
        self.batch_collector["loss"].append(self.task_loss.detach().cpu().numpy())
        
        
        # Find the general (symmetric) Hausdorff distance - use original model output!
        self.batch_collector["symhd"].append(self.symhd(model_output, ground_truth))

        
        m_item = self.segmet(y_true=ground_truth, y_pred=model_output)

        """
        if self.n_output_neurons > 1:
            _, highest_class = torch.max(model_output, 1)    
            highest_class = highest_class.detach().cpu().numpy() # [2:3].squeeze()
        else:
            highest_class = torch.sigmoid(model_output) > self.bin_threshold 
            highest_class = highest_class.detach().cpu().numpy()
            
        ground_truth = ground_truth.detach().cpu().numpy() # .squeeze()[2:3].squeeze()
        """
        
        
        self.batch_collector["acc"].append(m_item["acc"])
        self.batch_collector["fscore"].append(m_item["fscore"])
        self.batch_collector["jac"].append(m_item["jac"])
        self.batch_collector["prec"].append(m_item["prec"])
        self.batch_collector["rec"].append(m_item["rec"])
        
        
    
    def run_epoch(self, i_epoch):
        
        #print("epoch", i_epoch)
        
        self.epoch_collector["name"] = self.name
        self.epoch_collector["mode"] = self.mode
        self.epoch_collector["datasize"] = self.datasize
        self.epoch_collector["epoch"] = i_epoch
        self.epoch_collector["model"] = self.model
        
        
        # save mean value to epoch_collector, reset batch_collector
        for key in self.batch_collector.keys():
            
            value = round(np.mean(self.batch_collector[key]), 3) # .123
            self.epoch_collector[key] = value
            self.batch_collector[key] = []
            
            if self.writer:
                self.writer.add_scalars("metrics/" + key, {self.model+"_"+self.mode : value}, i_epoch)
            
        if i_epoch > 5:
            if self.top_collector['lowest_loss'] < self.epoch_collector["loss"]      : self.top_collector['lowest_loss'] = self.epoch_collector["loss"]
            if self.top_collector['highest_fscore'] > self.epoch_collector["fscore"] : self.top_collector['highest_fscore'] = self.epoch_collector["fscore"] 
            if self.top_collector['highest_jac'] > self.epoch_collector["jac"]       : self.top_collector['highest_jac'] = self.epoch_collector["jac"]
            if self.top_collector['highest_symhd'] > self.epoch_collector["symhd"]   : self.top_collector['highest_symhd'] = self.epoch_collector["symhd"]
            
        # log all the stuff
        print(self.epoch_collector)
        
        self.datasize = 0    
    

    
    

        
    
    

        
        
        
    
    