# =============================================================================
# Parent Class: Head (ONLY MAKE INSTANCE OF CHILD CLASSES)
# use .detach().cpu().numpy() for each model output
# todo: bad: every head is created from the start ... not only the ones needed
# =============================================================================
class TemplateComputingUnit():
    
    def __init__(self, class_names, dropout, n_output_neurons):
        super().__init__()
        # error if head is used instead of subclass
        if type(self) == Head:
             raise Exception("<Head> must be sub-classed.")
        # heads    
        self.head_name = type(self).__name__
        
        # used in resnet, DON'T DELETE
        self.dropout = dropout
        # iteration: ground truth for number, norm, ...
        self.iter_ground_truth = {}
        # iteration: model output per head
        self.iter_model_output =  {}
        
        self.losses_within_iter = {
                "train" : [],
                "val" : []
        }
        
        self.reset_iter()
        
        self.best = {"kappa" : 0.0, "fscore" : 0.0}
        
        # class names
        self.class_names = class_names
        # output neurons - 1 for reg/bin-ce and 2+ for multi-class
        if "L1_Reg_CU" in self.head_name or "BCE_BinClass_CU" in self.head_name:
            self.class_amount = 1
        else:
            self.class_amount = len(class_names)
            
            
    def run_batch(self, configs, model_output, ground_truth, mode):
        
        # used for train, then reset is needed
        # used for val, then reset is needed
        
        # for each torchtensor in model output (torch)
        tmp = torch.cat([torchtensor.unsqueeze(0) for torchtensor in model_output])
        if self.iter_model_output == {}:
            # init
            self.iter_model_output = tmp
        else:
            self.iter_model_output = torch.cat([self.iter_model_output, tmp])
        
        #print("MODEL OUTPUT ITER")
        #print(self.iter_model_output)
            
        # for each mode and for each torchtensor in groundtruth (dict of torch)
        for output_type, value in ground_truth.items():
            tmp = torch.cat([torchtensor.unsqueeze(0) for torchtensor in ground_truth[output_type]])
            if output_type not in self.iter_ground_truth.keys():    
                # init
                self.iter_ground_truth[output_type] = tmp
            else:
                self.iter_ground_truth[output_type] = torch.cat([self.iter_ground_truth[output_type], tmp])
            
            #print("GROUND TRUTH ITER")
            #print(self.iter_ground_truth[output_type])
        
        loss = self.calculate_batch_loss(configs=configs, 
                                   model_output=model_output, 
                                   ground_truth=ground_truth,
                                   mode=mode,
                                   batch_iter="Batch")
        
        return loss
    
    def run_iter(self, configs, mode):
        
        """
        loss = self.calculate_loss(configs=configs,
                                   model_output=self.iter_model_output_attached, 
                                   ground_truth=self.iter_ground_truth, 
                                   mode=mode, 
                                   batch_iter="Iter")
        """
        
        loss = self.calculate_iter_losses(configs, mode=mode, batch_iter="Iter")
        
        # set iter, because calculate metrics could be used for batch too
        metrics = self.calculate_iter_metrics(configs=configs,
                                model_output=self.iter_model_output,
                                ground_truth=self.iter_ground_truth, 
                                mode=mode, 
                                batch_iter="Iter")
        
        self.iter_model_output = {}
        self.iter_ground_truth = {}
        
        return loss, metrics
        
    
    
    def calculate_batch_loss(self):
        
        pass
    
    def calculate_iter_losses(self, configs, mode, batch_iter):
    
        loss = np.mean(self.losses_within_iter[mode])

        self.losses_within_iter = {
                "meta_train" : [],
                "meta_val" : [],
                "reader_val" : []
        }
        
        # tensorboard
        configs.writers[mode].add_scalar(f"{batch_iter} loss/{self.head_name}", 
                                         loss,
                                         configs.total_iter_counter)
        
        return loss
    
    def calculate_iter_metrics(self):
        pass
    
    
    
    def get_counter(self, configs, batch_iter=None, mode="train"):        
        # choose counter for tensorboard
        if batch_iter.lower() == "batch" and "train" in mode:
            return configs.total_train_batch_counter
        elif batch_iter.lower() == "batch" and "val" in mode:
            return configs.total_val_batch_counter
        else:
            return configs.total_iter_counter
        

        
    
    
    
    def log(self, configs, mode, loss, metrics):
        # =============================================================================
        # Write loss and metrics for good iterations to a csv file TODO        
        # =============================================================================            
        """
        write all metrics to a csv file (loss, accuracy, precision, recall, fscore, jaccard)
        arguments:
            configs: configs object
            mode: reader_val, meta_val
        returns:
            none
        """
        
        if not os.path.exists("results/metrics_csv/"):
            os.makedirs("results/metrics_csv/")
        
        filename = f"results/metrics_csv/{configs.name}_metrics.csv"
        file_exists = os.path.isfile(filename)

        with open(filename, 'a+') as metric_file:

            fieldnames = ['logtime', 'configs',
                          'head', 'mode', 'iter',
                          'train_set','val_set',
                          'loss',
                          'acc', 
                          'f', 'f_micro',
                          'jac', 'prec', 'rec',
                          'kappa', 'mse', 'mae'
                          ]

            writer = csv.DictWriter(metric_file, fieldnames=fieldnames)

            if not file_exists:
                writer.writeheader()
                
            if "mse" not in metrics.keys():
                metrics["mse"] = 0
            if "mae" not in metrics.keys():
                metrics["mae"] = 0
            if "fscore_micro" not in metrics.keys():
                metrics["fscore_micro"] = 0
            
            writer.writerow(
                {'logtime': configs.logtime, 'configs': configs.name,
                 'head': self.head_name, 'mode': mode, 'iter': configs.total_iter_counter,
                 # train eval size
                 'train_set': configs.trainset_size, 
                 'val_set': configs.valset_size,
                 # loss
                 'loss': loss,
                 # accuracy
                 'acc': metrics['accuracy'],
                 # f-score
                 'f': metrics["fscore"],
                 'f_micro': metrics["fscore_micro"],
                 # jaccard
                 'jac': metrics["jaccard"],
                 # precision
                 'prec': metrics["precision"],
                 # recall
                 'rec': metrics["recall"],
                 # kappa
                 'kappa': metrics["kappa"],
                 # mse+mae - only for regression
                 'mse': metrics["mse"],
                 'mae': metrics["mae"]
                 })