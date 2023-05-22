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
        #    could do something with weights here, but maybe simpler to do this for the loss
        # bugs: 
        #    need to check the n_samples_per_class_per_batch value to be bigger than 1/n of n classes
        #    3 classes - n_samples_per_class_per_batch needs ti be 0.34 in order to get an image out
        # sources: 
        #    https://stackoverflow.com/questions/66065272/customizing-the-batch-with-specific-elements
        # =============================================================================
        
        self.batches = []
        
        self.label_dict = {}
        original_labels = np.array(original_labels)
        self.unique_labels = np.unique(original_labels)
        self.n_samples_per_class_per_batch = n_samples_per_class_per_batch
        self.final_batch_size = int(self.n_samples_per_class_per_batch * len(self.unique_labels))
        if self.n_samples_per_class_per_batch < 1: # if smaller than 1
            self.n_samples_per_class_per_batch = 1
        
        self.smallest_class = np.inf
        # for label in unique labels
        for this_label in self.unique_labels:
            # get indices of label in original labels
            
            # print(isinstance(np.where(original_labels==this_label), list))
            # print(isinstance(np.squeeze(np.where(original_labels==this_label), list)))
            
            
            indices_list = np.squeeze(np.where(original_labels==this_label)) # 
            
            try:
                # if it is a list
                if self.smallest_class > len(indices_list):
                    self.smallest_class = len(indices_list)
                # save indices into dict for each unique label
                self.label_dict[this_label] = list(indices_list)
            except:
                print("broken", indices_list)
                indices_list = [indices_list]
                print("converted to a list", indices_list)
                if self.smallest_class > len(indices_list):
                    self.smallest_class = len(indices_list)
                # save indices into dict for each unique label
                self.label_dict[this_label] = list(indices_list)
                
            
    def __len__(self):                
        return self.smallest_class * self.unique_labels
            
    def __iter__(self):

        # reset dataset
        self.data = copy.deepcopy(self.label_dict)

        # shuffle each class list for each iteration
        for k in self.data:
            np.random.shuffle(self.data[k])
        
        batches = []

        # for each batch
        for i in range(0, int(self.smallest_class/self.n_samples_per_class_per_batch)):

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
                
            # if enough images are available for one batch: append
            if len(batch) >= self.final_batch_size:
                # we need this to split patches in case n_samples_per_class_per_batch smaller than 1
                for i in range(0, len(batch), self.final_batch_size):
                    tmp = batch[i:i + self.final_batch_size]
                    # only append if right size
                    if len(tmp) == self.final_batch_size:
                        batches.append(tmp)
            
        return iter(batches)
    
