class MixedBatchSampler(Sampler):
    def __init__(self, original_data, n_samples_per_class_per_batch=2):
        # =============================================================================
        # source: https://stackoverflow.com/questions/66065272/customizing-the-batch-with-specific-elements
        # choose random n classes - so it works for 0.5
        # =============================================================================
        
        self.label_dict = {}
        # self.class_sizes = []
        np_array = np.array(original_data)
        for key in set(np_array):
            indices_list = np.squeeze(np.where(np_array==key))
            self.label_dict[key] = list(indices_list)
            # self.class_sizes.append(len(list(indices_list)))
            
        self.classes = list(self.label_dict.keys())
        self.n_samples_per_class_per_batch = n_samples_per_class_per_batch
        self.final_batch_size = int(self.n_samples_per_class_per_batch * len(self.classes))
        # self.n_batches = n_samples_per_class_per_iter // n_samples_per_class_per_batch
        
        if self.n_samples_per_class_per_batch < 1: # if smaller than 1
            self.n_samples_per_class_per_batch = 1

    def __iter__(self):

        # reset dataset
        self.data = copy.deepcopy(self.label_dict)

        # shuffle each class list for each iteration
        for k in self.data:
            np.random.shuffle(self.data[k])
        
        batches = []

        # for each batch
        # for _ in range(self.n_batches):
        while True:

            batch = []
            np.random.shuffle(self.classes)
            for k in self.classes:
                if len(self.data[k]) >= self.n_samples_per_class_per_batch:   
                    batch.extend(self.data[k][:self.n_samples_per_class_per_batch])
                    del self.data[k][:self.n_samples_per_class_per_batch]
                
                # if length reached
                if len(batch) >= self.final_batch_size:
                    break
                    
            # if not enough values available
            if len(batch) < self.final_batch_size:
                break

            batches.append(batch)
            
            

        return iter(batches)