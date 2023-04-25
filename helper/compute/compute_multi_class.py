from sklearn.metrics import *
from compute.template import *

class CE_MultiClass_CU(TemplateComputingUnit):
    # =============================================================================
    #
    # Multi-class classification, Cross Entropy Loss
    #
    # =============================================================================

    def __init__(self, n_output_neurons=1):
        # =============================================================================
        # parameters:
        #    n_output_neurons: 1 for regression
        # notes:
        #   Cross entropy loss (with logits), activation function done by loss
        # =============================================================================
        
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.n_output_neurons = n_output_neurons
        
    def calculate_loss(self, configs, model_output, ground_truth, mode="train"):
        # =============================================================================
        # parameters: 
        #   configs file, model output (linear layer probabilities), ground truth (0-4)
        #   mode (train, val)
        # returns:
        #   cross entropy loss, single value
        # notes:
        #   iter not recommended
        # =============================================================================
        
        # choose ground truth
        if "meta" in mode:
            ground_truth = ground_truth["meta_phase_number"]
        elif "reader" in mode:
            ground_truth = ground_truth["reader_phase_number"]
        
        # calculate loss: ground truth to long()
        loss = self.criterion(input=model_output,
                             target=ground_truth.to(configs.device).long())
                
        # append loss for whole iteration
        self.losses_within_iter[mode].append(loss.detach().cpu().numpy())
        
        # tensorboard
        configs.writers[mode].add_scalar(f"{batch_iter} loss/{self.head_name}", 
                                         np.mean(loss.detach().cpu().numpy()),
                                         self.get_counter(configs, batch_iter, mode))
        # return single value
        return loss

    def calculate_metrics(self, configs, model_output, ground_truth, mode="meta_train", batch_iter="iter"):
        # =============================================================================
        # parameters: 
        #   configs file, model output (linear layer probabilities, attached), ground truth (0-4)
        #   mode (meta_train, meta_val, reader_val), batch_iter (iter, (batch))
        # returns:
        #   dict of metrics
        # notes:
        #   batch not recommended
        #   torch.max needs attached - do not detach before this function.
        #   how torch.max works:
        #   [0.2, 0.2, 0.5, 0.1] -> highest: pred 0.5 / class 2
        #   torch.max(input, dim)
        #   .detach().cpu().numpy() to detach
        #   .cpu().detach() vs .detach().cpu() - second is faster cause 
        #   doesn’t track gradients for cpu()
        # =============================================================================
        
        # choose ground truth
        if "meta" in mode:
            ground_truth = ground_truth["meta_phase_number"]
        elif "reader" in mode:
            ground_truth = ground_truth["reader_phase_number"]
        
        # argmax activation: highest class     
        _, highest_class = torch.max(model_output, 1)    
        highest_class = highest_class.detach().cpu().numpy()    
            
        # calculate metrics: wheighted + one micro
        metrics = {}
        metrics["kappa"] = cohen_kappa_score(y1 = ground_truth, y2 = highest_class)
        metrics["accuracy"] = balanced_accuracy_score(y_true = ground_truth, y_pred = highest_class)
        metrics["jaccard"] = jaccard_score(y_true = ground_truth, y_pred = highest_class, average="weighted")
        metrics["recall"] = recall_score(y_true = ground_truth, y_pred = highest_class, average="weighted")
        metrics["fscore"] = f1_score(y_true = ground_truth, y_pred = highest_class, average="weighted")
        metrics["fscore_micro"] = f1_score(y_true = ground_truth, y_pred = highest_class, average="micro")
        metrics["precision"] = precision_score(y_true = ground_truth, y_pred = highest_class, average="weighted")
        
            
        # tensorboard
        for key, value in metrics.items():
            configs.writers[mode].add_scalar(f"{batch_iter} {key}/{self.head_name}", 
                                             np.mean(value), 
                                             self.get_counter(configs, batch_iter, mode))

        # return dict
        return metrics
    
    def plot_confusion_matrix(self, configs, model_output, ground_truth, mode="meta_val"):
        # =============================================================================
        # parameters: 
        #   configs file, model output (linear layer probabilities), ground truth (0-4)
        #   mode (meta_train, meta_val, reader_val)
        # saves:
        #   confusion matrix plots (original - count, normalised 0-1)
        # notes:
        #   only for iter
        # =============================================================================
        
        # confusion matrix with colours
        cmap = sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)
        
        # one list entry for each class name
        labels = list(range(0,len(self.class_names)))
        
        # choose ground truth
        if "meta" in mode:
            ground_truth = ground_truth["meta_phase_number"]
        elif "reader" in mode:
            ground_truth = ground_truth["reader_phase_number"]
            
        # argmax activation: highest class
        _, highest_class = torch.max(model_output, 1)    
        highest_class = highest_class.detach().cpu().numpy()
        
        cm_dict = {}
        # original count
        cm_dict["original"] = confusion_matrix(ground_truth, highest_class, labels=labels)
        # normalised between 0 and 1
        cm_dict["normalised"] = np.divide(cm_dict["original"].astype('float'), 
                                  cm_dict["original"].sum(axis=1)[:, np.newaxis],
                                  out=np.zeros_like(cm_dict["original"].astype('float')), 
                                  where=cm_dict["original"].sum(axis=1)[:, np.newaxis] != 0)
        
        for cm_type, cm_value in cm_dict.items():
            sns.heatmap(cm_value, annot=True, cmap=cmap, vmin=0)
            plt.xlabel("Predicted labels")
            plt.ylabel("True labels")
            plt.title(f"{self.head_name}")
            plt.savefig(f"results/confusion_matrix/cm_{configs.name}_{configs.total_iter_counter}_{self.head_name}_{cm_type}.png")
            # plt.close()
        
class LSL_MultiClass_CU(CE_MultiClass_CU):
    # =============================================================================
    #
    # Child class
    # Multi-class classification (Phase)
    # Cross Entropy Loss with label smoothing
    #
    # =============================================================================

    def __init__(self, class_names, dropout):
        # =============================================================================
        # parameters:
        #   class names TODO, which format??, dropout
        # notes:
        #   Cross entropy loss (with logits and label smoothing), activation function alternative done by loss
        # =============================================================================
        
        super().__init__(class_names, dropout)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.5)   
        
        
class CDW_MultiClass_CU(CE_MultiClass_CU):
    # =============================================================================
    #
    # Child class
    # Multi-class classification (Phase)
    # Cross Entropy Loss with label smoothing
    #
    # =============================================================================

    def __init__(self, class_names, dropout):
        # =============================================================================
        # parameters:
        #
        # notes:
        #   Class distance weighted loss
        # =============================================================================
        
        super().__init__(class_names, dropout)
        self.criterion = ClassDistanceWeightedLoss(n_classes=len(classes), power=2.0, reduction="mean")   
    
    def calculate_loss(self, configs, model_output, ground_truth, mode="meta_train", batch_iter="batch"):
        # =============================================================================
        # parameters: 
        #   configs file, model output (linear layer probabilities), ground truth (0-4)
        #   mode (meta_train, meta_val, reader_val), batch_iter (batch, (iter))
        # returns:
        #   Kappa loss, single value
        # notes:
        # 
        # =============================================================================
            
        
        # calculate loss: todo
        loss = self.criterion(model_output=model_output.float(),
                              ground_truth=ground_truth)
         
        # append loss for whole iteration
        self.losses_within_iter[mode].append(loss.detach().cpu().numpy())


        # tensorboard
        configs.writers[mode].add_scalar(f"{batch_iter} loss/{self.head_name}", 
                                         np.mean(loss.detach().cpu().numpy()),
                                         self.get_counter(configs, batch_iter, mode))

        # return single value
        return loss

    
class Kappa_MultiClass_CU(CE_MultiClass_CU):
    # =============================================================================
    #
    # Child class
    # Multi-class classification (Phase)
    # QWK: Quadratic Weighted Kappa Loss
    #
    # =============================================================================
    """
    QWK loss head for multi-class phase prediction
    init arguments:
        configs: config object from config file
    
    calculate_loss returns:
        QWK loss
    """
    def __init__(self, class_names, dropout):
        # =============================================================================
        # parameters:
        #   class names TODO, which format??, dropout
        # notes:
        #   Kappa Loss, activation softmax done by loss
        # =============================================================================
        super().__init__(class_names, dropout)
        self.criterion = KappaLoss(n_classes=len(class_names))

    def calculate_loss(self, configs, model_output, ground_truth, mode="meta_train", batch_iter="batch"):
        # =============================================================================
        # parameters: 
        #   configs file, model output (linear layer probabilities), ground truth (0-4)
        #   mode (meta_train, meta_val, reader_val), batch_iter (batch, (iter))
        # returns:
        #   Kappa loss, single value
        # notes:
        #   iter not recommended as loss is dependent on batch size
        # =============================================================================
        
        # choose ground truth
        if "meta" in mode:
            ground_truth = ground_truth["meta_phase_number"]
        elif "reader" in mode:
            ground_truth = ground_truth["reader_phase_number"]
            
        
        # calculate loss: todo
        loss = self.criterion(model_output=model_output.float(),
                              ground_truth=ground_truth, 
                              device=configs.device)
         
        # append loss for whole iteration
        self.losses_within_iter[mode].append(loss.detach().cpu().numpy())


        # tensorboard
        configs.writers[mode].add_scalar(f"{batch_iter} loss/{self.head_name}", 
                                         np.mean(loss.detach().cpu().numpy()),
                                         self.get_counter(configs, batch_iter, mode))

        # return single value
        return loss
    

