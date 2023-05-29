from sklearn.metrics import *
import numpy as np
from scipy.spatial.distance import directed_hausdorff

import csv

import os

import torch

class TemplateComputingUnit():
    
    # =============================================================================
    # Parent Class: Head (ONLY MAKE INSTANCE OF CHILD CLASSES)
    # compute batches and epochs for each output head
    # use .detach().cpu().numpy() for each model output
    # =============================================================================
    
    def __init__(self, n_output_neurons, mode="train", model=""):
        super().__init__()
        # error if head is used instead of subclass
        if type(self) == TemplateComputingUnit:
             raise Exception("<Head> must be sub-classed.")
        
        """
        # head name
        self.name = type(self).__name__
        self.mode = mode
        
        keys = ['name', 'model', 'mode', # stays the same - needs to be re-saved
                'epoch',
                'data_size', # train set size or val set size
                'loss',
                'acc', 
                'fscore', 'f_micro',
                'jac', 'prec', 'rec',
                'kappa', # task specific: multiclass
                'mse', 'mae', # task specific: regression
                'hd_sym' # task specific: segmentation
                ]
        
        # everything we want to write to the csv file
        self.epoch_collector = { key : [] for key in keys }
        
        # save best scores
        self.best = {"fscore" : 0.0}
        
        # output neurons - 1 for reg/bin-ce and 2+ for multi-class
        self.n_output_neurons = n_output_neurons
        """
        
    def run_batch(self, model_output, ground_truth):
        pass
    
    def run_epoch(self):
        pass
    
    def reset_epoch(self):
        
        if self.top_collector['highest_fscore'] < self.epoch_collector["fscore"]:
            self.top_collector["highest_fscore"] = self.epoch_collector["fscore"]
        
        for key in self.epoch_collector.keys():
            self.epoch_collector[key] = None
            
        self.epoch_collector["name"] = self.name
        self.epoch_collector["mode"] = self.mode
        self.epoch_collector["model"] = self.model
        
        
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

            writer = csv.DictWriter(file, fieldnames=self.epoch_collector.keys(), delimiter=';')

            if not os.path.isfile(csv_file_path):
                writer.writeheader()
            
            writer.writerow(self.epoch_collector)