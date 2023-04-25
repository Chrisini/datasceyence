from sklearn.metrics import *
from compute.template import *

class L1_Reg_CU(TemplateComputingUnit):
    # =============================================================================
    #
    # Regression, L1 Loss
    #
    # =============================================================================

    def __init__(self, n_output_neurons):
        # =============================================================================
        # parameters:
        #   class names TODO, which format??, dropout
        # notes:
        #   L1 Loss, no activation in regression
        # =============================================================================
        
        super().__init__(class_names, dropout)
        self.criterion = nn.L1Loss()

    def calculate_loss(self, configs, model_output, ground_truth, mode="train"):
        # =============================================================================
        # parameters: 
        #   configs file, model output (relu activation), ground truth (continuous: seconds or normalised)
        #   mode (train, val)
        # returns:
        #   L1 Loss, single value
        # notes:
        #   iter not recommended
        # =============================================================================
        
        # calculate loss: both values float todo
        loss = self.criterion(input=model_output.squeeze().float(), 
                              target=ground_truth.to(configs.device).float())

        # append loss for whole iteration
        self.losses_within_iter[mode].append(loss.detach().cpu().numpy())

        # tensorboard (for regression)
        configs.writers[mode].add_scalar(f"{batch_iter} loss/{self.head_name}", 
                                         np.mean(loss.detach().cpu().numpy()),
                                         self.get_counter(configs, batch_iter, mode))

        return loss

        

    def calculate_metrics(self, configs, model_output, ground_truth, mode="train"):
        # =============================================================================
        # parameters: 
        #   configs file, model output (relu activation), ground truth (continuous: seconds or normalised)
        #   mode (train, val)
        # returns:
        #   dict of metrics
        # notes:
        #   batch not recommended
        #   regression metrics
        #   conversion + classification metrics
        # =============================================================================
        
        # detach model output
        model_output = model_output.squeeze().float().detach().cpu().numpy()
        
        # calculate metrics: dict
        metrics = {}
        
        # choose ground truth and calculate regression metrics
        if "meta" in mode:
            ground_truth = ground_truth["meta_phase_norm"]
            
            # calculate metrics: regression
            metrics["mse"] = mean_absolute_error(y_true = ground_truth,
                                                 y_pred = model_output)
            metrics["mae"] = mean_squared_error(y_true = ground_truth, 
                                                y_pred = model_output)
        
            # convert ground truth to phase number
            ground_truth = norm_to_seconds(configs, ground_truth)
            ground_truth = seconds_to_phase_number(configs, ground_truth)

            
        elif mode == "reader_val":
            # no mse/mae possible
            # get ground truth as phase number
            ground_truth = ground_truth["reader_phase_number"]
            
        # convert model output to phase number
        model_output = norm_to_seconds(configs, model_output)
        model_output = seconds_to_phase_number(configs, model_output)
        
        # calculate metrics: classification metrics, wheighted + one micro
        metrics["kappa"] = cohen_kappa_score(y1 = model_output, y2 = ground_truth)
        metrics["accuracy"] = balanced_accuracy_score(y_true = ground_truth, y_pred = model_output)
        metrics["jaccard"] = jaccard_score(y_true = ground_truth, y_pred = model_output, average="weighted")
        metrics["recall"] = recall_score(y_true = ground_truth, y_pred = model_output, average="weighted")
        metrics["fscore"] = f1_score(y_true = ground_truth, y_pred = model_output, average="weighted")
        metrics["fscore_micro"] = f1_score(y_true = ground_truth, y_pred = model_output, average="micro")
        metrics["precision"] = precision_score(y_true = ground_truth, y_pred = model_output, average="weighted")
                
        # tensorboard
        for key, value in metrics.items():
            configs.writers[mode].add_scalar(f"{batch_iter} {key}/{self.head_name}", 
                                             np.mean(value), 
                                             self.get_counter(configs, batch_iter, mode))
            
        # return dict
        return metrics
    
    def plot_confusion_matrix(self, configs, model_output, ground_truth, mode="meta_val"):
        # =============================================================================
        # not implemented
        # =============================================================================
        pass

