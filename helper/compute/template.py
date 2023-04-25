from sklearn.metrics import *


class TemplateComputingUnit():
    
    # =============================================================================
    # Parent Class: Head (ONLY MAKE INSTANCE OF CHILD CLASSES)
    # compute batches and epochs for each output head
    # use .detach().cpu().numpy() for each model output
    # =============================================================================
    
    def __init__(self, n_output_neurons, mode="train"):
        super().__init__()
        # error if head is used instead of subclass
        if type(self) == TemplateComputingUnit:
             raise Exception("<Head> must be sub-classed.")
                
        # head name
        self.name = type(self).__name__
        self.mode = mode
        
        keys = ['name', 'mode', # stays the same - needs to be re-saved
                'epoch',
                'train_size','val_size', # train val size
                'loss',
                'acc', 
                'fscore', 'f_micro',
                'jac', 'prec', 'rec',
                'kappa', # task specific: multiclass
                'mse', 'mae', # task specific: regression
                'hd_sym' # task specific: segmentation
                ]
        
        # everything we want to write to the csv file
        self.epoch_collector = dict.fromkeys(keys)
        
        # save best scores
        self.best = {"fscore" : 0.0}
        
        # output neurons - 1 for reg/bin-ce and 2+ for multi-class
        self.n_output_neurons = n_output_neurons
        
    def run_batch(self, model_output, ground_truth):
        pass
    
    def run_epoch(self):
        pass
    
    def reset_epoch(self):
        
        for key in epoch_collecter.keys():
            self.epoch_collector[key] = None
            
        self.epoch_collector["name"] = self.name
        self.epoch_collector["mode"] = self.mode
        
        
    def log(self, csv_file_path):
        # =============================================================================
        # Write all information to a csv file (loss, accuracy, precision, recall, fscore, jaccard)
        # arguments:
        #    csv_file_path
        # returns:
        #    none
        # writes:
        #    one csv file for each computing unit aka each model head
        # =============================================================================            

        with open(csv_file_path, 'a+') as file:

            writer = csv.DictWriter(file, fieldnames=self.epoch_collector.keys())

            if not os.path.isfile(filename):
                writer.writeheader()
            
            writer.writerow(self.epoch_collector)