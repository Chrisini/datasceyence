from sklearn.metrics import *




class CE_MultiClass_CU(ComputingUnit):
    # =============================================================================
    #
    # Child class
    # Multi-class classification (Phase)
    # Cross Entropy Loss
    #
    # =============================================================================

    def __init__(self, class_names, dropout):
        # =============================================================================
        # parameters:
        #   class names TODO, which format??, dropout
        # notes:
        #   Cross entropy loss (with logits), activation function alternative done by loss
        # =============================================================================
        
        super().__init__(class_names, dropout)
        self.criterion = nn.CrossEntropyLoss()
        
    def calculate_loss(self, configs, model_output, ground_truth, mode="meta_train", batch_iter="batch"):
        # =============================================================================
        # parameters: 
        #   configs file, model output (linear layer probabilities), ground truth (0-4)
        #   mode (meta_train, meta_val, reader_val), batch_iter (batch, (iter))
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
        
        # choose ground truth
        if "meta" in mode:
            ground_truth = ground_truth["meta_phase_number"]
        elif "reader" in mode:
            ground_truth = ground_truth["reader_phase_number"]
            
        
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
    

class L1_Reg_CU(ComputingUnit):
    # =============================================================================
    #
    # Child class
    # Regression (Phase)
    # L1 Loss
    #
    # =============================================================================

    def __init__(self, class_names, dropout):
        # =============================================================================
        # parameters:
        #   class names TODO, which format??, dropout
        # notes:
        #   L1 Loss, no activation in regression
        # =============================================================================
        
        super().__init__(class_names, dropout)
        self.criterion = nn.L1Loss()

    def calculate_loss(self, configs, model_output, ground_truth, mode="meta_train", batch_iter="batch"):
        # =============================================================================
        # parameters: 
        #   configs file, model output (relu activation), ground truth (continuous: seconds or normalised)
        #   mode (meta_train, meta_val, reader_val), batch_iter (batch, (iter))
        # returns:
        #   L1 Loss, single value
        # notes:
        #   iter not recommended
        #   loss empty for reader labels
        # =============================================================================
        
        # choose ground truth and calculate regression loss
        if "meta" in mode.lower():
            ground_truth = ground_truth["meta_phase_norm"]
                                    
            # calculate loss: both values float todo
            loss = self.criterion(input=model_output.squeeze().float(), 
                                  target=ground_truth.to(configs.device).float())
            
            # append loss for whole iteration
            self.losses_within_iter[mode].append(loss.detach().cpu().numpy())

            # tensorboard (for regression)
            configs.writers[mode].add_scalar(f"{batch_iter} loss/{self.head_name}", 
                                             np.mean(loss.detach().cpu().numpy()),
                                             self.get_counter(configs, batch_iter, mode))
            
        elif mode == "reader_val":
            
            # calculate loss: regression loss not possible with reader labels
            loss = 0
            
            # append loss for whole iteration
            self.losses_within_iter[mode].append(0)
            
        return loss

        

    def calculate_metrics(self, configs, model_output, ground_truth, mode="meta_train", batch_iter="iter"):
        # =============================================================================
        # parameters: 
        #   configs file, model output (relu activation), ground truth (continuous: seconds or normalised)
        #   mode (meta_train, meta_val, reader_val), batch_iter (iter, (batch))
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

class BCE_BinClass_CU(ComputingUnit):
    # =============================================================================
    #
    # Child class
    # Binary classification (Laterality)
    # BCE With Logits Loss
    #
    # =============================================================================
    

    def __init__(self, class_names, dropout):
        # =============================================================================
        # parameters:
        #   class names TODO, which format??, dropout
        # notes:
        #   BCE (with logits), activation function alternative done by loss
        # =============================================================================
        
        super().__init__(class_names, dropout)
        self.criterion = nn.BCEWithLogitsLoss()
        

    def calculate_loss(self, configs, model_output, ground_truth, mode="meta_train", batch_iter="batch"):
        # =============================================================================
        # parameters: 
        #   configs file, model output (linear layer probabilities), ground truth (0-1)
        #   mode (meta_train, meta_val, reader_val), batch_iter (batch, (iter))
        # returns:
        #   binary cross entropy loss, single value
        # notes:
        #   iter not recommended
        #   squeeze:
        #      print(model_output.squeeze().shape)
        #      print(model_output.squeeze(0).shape)
        #      print(model_output.squeeze(-1).shape) ... works
        #      print(ground_truth.shape)
        # =============================================================================
                    
        # choose ground truth
        if "meta" in mode:
            ground_truth = ground_truth["meta_lat_number"]
        elif "reader" in mode:
            ground_truth = ground_truth["reader_lat_number"]
        
        # calculate loss: model output squeeze(-1), why to device??
        loss = self.criterion(input=model_output.squeeze(-1).float(), 
                              target=ground_truth.to(configs.device).float())

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
        #   configs file, model output (linear layer probabilities, attached), ground truth (0-1)
        #   mode (meta_train, meta_val, reader_val), batch_iter (iter, (batch))
        # returns:
        #   dict of metrics
        # notes:
        #   batch not recommended
        #   sigmoid activation is used
        #   binary metrics
        # =============================================================================
        
        # choose ground truth
        if "meta" in mode:
            ground_truth = ground_truth["meta_lat_number"]
        elif "reader" in mode:
            ground_truth = ground_truth["reader_lat_number"]
        
        # sigmoid activation: for 1 output neuron, threshold 0.5 (outputs 0 or 1)
        highest_class = torch.sigmoid(model_output) > 0.5
        highest_class = highest_class.detach().cpu().numpy()
        
        # calculate metrics: binary
        metrics = {}
        metrics["kappa"] = cohen_kappa_score(y1 = ground_truth, y2 = highest_class)
        metrics["accuracy"] = balanced_accuracy_score(y_true = ground_truth, y_pred = highest_class)
        metrics["jaccard"] = jaccard_score(y_true = ground_truth, y_pred = highest_class, average="binary")
        metrics["recall"] = recall_score(y_true = ground_truth, y_pred = highest_class, average="binary")
        metrics["fscore"] = f1_score(y_true = ground_truth, y_pred = highest_class, average="binary")
        metrics["precision"] = precision_score(y_true = ground_truth, y_pred = highest_class, average="binary")
        
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
        #   configs file, model output (linear layer probabilities), ground truth (0-1)
        #   mode (meta_train, meta_val, reader_val)
        # saves:
        #   confusion matrix plots (original - count, normalised 0-1)
        # notes:
        #   only for iter
        # =============================================================================
        
        # confusion matrix with colours
        cmap = sns.color_palette("ch:s=.25,rot=-.25", as_cmap=True)

        
        labels = list(range(0,len(self.class_names)))
        
        # choose ground truth
        if "meta" in mode:
            ground_truth = ground_truth["meta_lat_number"]
        elif "reader" in mode:
            ground_truth = ground_truth["reader_lat_number"]
        
        # sigmoid activation: for 1 output neuron, threshold 0.5 (outputs 0 or 1)
        highest_class = torch.sigmoid(model_output) > 0.5
        highest_class = highest_class.detach().cpu().numpy()
        
        cm_dict = {}
        # total number
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

class CE_BinClass_CU(BCE_BinClass_CU):
    # =============================================================================
    #
    # Child class
    # Binary classification (Laterality)
    # Cross Entropy Loss
    #
    # =============================================================================

    def __init__(self, class_names, dropout):
        # =============================================================================
        # parameters:
        #   class names TODO, which format??, dropout
        # notes:
        #   Cross entropy loss (with logits), activation function alternative done by loss
        # =============================================================================
        
        super().__init__(class_names, dropout)
        self.criterion = nn.CrossEntropyLoss()

    def compute_loss_for_backprop(self, configs, model_output, ground_truth, mode="meta_train", batch_iter="Batch"):
        # =============================================================================
        # calculates the loss for a single batch
        # parameters: 
        #   configs file, model output (linear layer probabilities), ground truth (0-1)
        #   mode (meta_train, meta_val, reader_val), batch_iter (batch, (iter))
        # returns:
        #   cross entropy loss, single value
        # =============================================================================
        
        # calculate loss
        loss = self.criterion(input=model_output, 
                              target=ground_truth.to(configs.device).long())

        # append loss for whole iteration
        self.losses_within_iter[mode].append(loss.detach().cpu().numpy())
        
        # return single value
        return loss
    
    def compute_loss_for_logger(self, configs, model_output, ground_truth, mode="meta_train", batch_iter="Batch"):
        # =============================================================================
        # calculates the loss for a whole epoch
        # parameters: 
        #   configs file, model output (linear layer probabilities), ground truth (0-1)
        #   mode (meta_train, meta_val, reader_val), batch_iter (batch, (iter))
        # returns:
        #   cross entropy loss, single value
        # =============================================================================
        
        # calculate loss
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

    def compute_metrics_for_logger(self, logger, model_output, ground_truth, mode="meta_train"):
        # =============================================================================
        # calculates the metrics for a whole epoch
        # parameters: 
        #   configs file, model output (linear layer probabilities, attached), ground truth (0-1)
        # returns:
        #   dict of metrics
        # notes:
        #   binary metrics
        #   torch.max needs attached - do not detach before this function.
        #   how torch.max works:
        #   [0.2, 0.2, 0.5, 0.1] -> highest: pred 0.5 / class 2
        #   torch.max(input, dim)
        #   .detach().cpu().numpy() to detach
        #   .cpu().detach() vs .detach().cpu() - second is faster cause 
        #   doesn’t track gradients for cpu()
        # =============================================================================

        # argmax activation: highest class
        highest_pred, highest_class = torch.max(model_output, 1)    
        highest_class = highest_class.detach().cpu().numpy()
        
        # calculate metrics: binary
        metrics = {}
        metrics["acc"]    = balanced_accuracy_score(y_true=ground_truth, y_pred=highest_class)
        metrics["fscore"] = f1_score(y_true=ground_truth, y_pred=highest_class, average="binary")
        metrics["jac"]    = jaccard_score(y_true=ground_truth, y_pred=highest_class, average="binary")
        metrics["kappa"]  = cohen_kappa_score(y1=ground_truth, y2=highest_class)
        metrics["prec"]   = precision_score(y_true=ground_truth, y_pred=highest_class, average="binary")
        metrics["rec"]    = recall_score(y_true=ground_truth, y_pred=highest_class, average="binary")
        
        # tensorboard
        for key, value in metrics.items():
            logger.writers[mode].add_scalar(f"{key}/{self.head_name}", 
                                             np.mean(value), 
                                             self.get_counter(configs, None, mode))

        # return dict
        return metrics