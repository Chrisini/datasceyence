import numpy as np
from torch.utils.data.sampler import Sampler

import copy

class MixedBatchSampler(Sampler):
    def __init__(self, original_labels, n_samples_per_class_per_batch=2):
        # =============================================================================
        # MixedBatchSampler samples n images of each class. 
        # min available images per class / numbers of samples per batch 
        # defines how many batches will be returned
        # notes: 
        #    choose random n classes - so it works for 0.5
        #    n_samples_per_class_per_batch - values from 0.1 to max class size of smallest class (e.g. 20)
        #    todo - make more if conditions to check
        # bugs: 
        #    if 1+: taking care of balance in batches in epoch
        #    if smaller than 1: not taking care of balance accross batches in epoch (this is bad)
        # sources: 
        #    https://stackoverflow.com/questions/66065272/customizing-the-batch-with-specific-elements
        # =============================================================================
        
        self.label_dict = {}
        original_labels = np.array(original_labels)
        self.unique_labels = np.unique(original_labels)
        self.n_samples_per_class_per_batch = n_samples_per_class_per_batch
        self.final_batch_size = int(self.n_samples_per_class_per_batch * len(self.unique_labels))
        if self.n_samples_per_class_per_batch < 1: # if smaller than 1
            self.n_samples_per_class_per_batch = 1
        
        # for label in unique labels
        for this_label in self.unique_labels:
            # get indices of label in original labels
            indices_list = np.squeeze(np.where(original_labels==this_label))
            # save indices into dict for each unique label
            self.label_dict[this_label] = list(indices_list)
            
    def __iter__(self):

        # reset dataset
        self.data = copy.deepcopy(self.label_dict)

        # shuffle each class list for each iteration
        for k in self.data:
            np.random.shuffle(self.data[k])
        
        batches = []

        # for each batch
        while True:

            batch = []
            # shuffle needed for n_samples_per_class_per_batch smaller than 1
            np.random.shuffle(self.unique_labels)
            # for each unique label 
            for k in self.unique_labels:
                # if more samples than n_samples_per_class_per_batch are available
                if len(self.data[k]) >= self.n_samples_per_class_per_batch:   
                    # get the next n samples
                    batch.extend(self.data[k][:self.n_samples_per_class_per_batch])
                    # and delete them from the dataset
                    del self.data[k][:self.n_samples_per_class_per_batch]
                
                # if length reached (needed for n_samples_per_class_per_batch smaller than 1) 
                if len(batch) >= self.final_batch_size:
                    break
                    
            # if not enough values available "we done now"
            if len(batch) < self.final_batch_size:
                # we ignore the incomplete batch
                break
            else:
                # we use the batch
                batches.append(batch)
            
        return iter(batches)
    
