from dataset.template import *
import skimage.io
import skimage.util

class MilDataset(TemplateDataset):
    # =============================================================================
    # Describe what is going on
    # parameters:
    #    parameter1: e.g. hidden vector of shape [bsz, n_views, ...].
    #    parameter2: e.g. ground truth of shape [bsz].
    # returns:
    #    parameter2: e.g. a loss scalar.
    # saves:
    #    collector of data within a class 
    # writes:
    #    csv file, png images, ...
    # notes:
    #    Whatever comes into your mind
    # sources:
    #    https...
    # =============================================================================
    
    def __init__(self, data, ids, labels, normalize=True):
        self.data = data
        self.labels = labels
        self.ids = ids

        # Modify shape of bagids if only 1d tensor
        if (len(ids.shape) == 1):
            ids.resize_(1, len(ids))

        self.bags = torch.unique(self.ids[0])

        # Normalize
        if normalize:
            std = self.data.std(dim=0)
            mean = self.data.mean(dim=0)
            self.data = (self.data - mean)/std

    def __len__(self):
        return len(self.bags)

    def __getitem__(self, index):
        data = self.data[self.ids[0] == self.bags[index]]
        bagids = self.ids[:, self.ids[0] == self.bags[index]]
        labels = self.labels[index]

        return data, bagids, labels

    def n_features(self):
        return self.data.size(1)
