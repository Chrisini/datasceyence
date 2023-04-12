from loss.template import *

class KappaLoss(TemplateLoss):
    # =============================================================================
    # Kappa Loss
    #
    # QWK loss function as described in https://arxiv.org/pdf/1612.00775.pdf
    # adapted from: https://www.kaggle.com/gennadylaptev/qwk-loss-for-pytorch
    # init arguments:
    #    n_classes: number of multi-class classification
    # forward arguments:
    #    model_output: from linear layer, a tensor with probability predictions, [batch_size, n_classes]
    #                    softmax activation is applied in the forward function
    #    ground truth: a tensor with ???
    #    device: gpu/cpu
    # forward returns:
    #    QWK loss
    # =============================================================================

    def __init__(self, n_classes=5):
        super(KappaLoss, self).__init__()
        self.soft = torch.nn.Softmax(dim=1)
        self.n_classes = n_classes
        self.eps = 1e-10 # epsilon to avoid nan
        
        # symmetric weight matrix (doesn't have to be symmetric, but this one is)
        self.W = np.zeros([self.n_classes, self.n_classes])
        for i in range(self.n_classes):
            for j in range(self.n_classes):
                self.W[i, j] = ((i - j) ** 2)


    def forward(self, model_output, ground_truth, device="cpu"):
        
        # probabilities, with softmax activation function, to not get zero as kappa loss result
        # it doesn't work without softmax
        model_output = self.soft(model_output).type(torch.FloatTensor)
        
        # one hot encoded ground truth
        ground_truth = torch.nn.functional.one_hot(ground_truth, num_classes=self.n_classes)
        ground_truth = ground_truth.type(torch.FloatTensor)

        # weight matrix aka W to torch and to device
        W = torch.from_numpy(self.W.astype(np.float32)).to(device)

        # confusion matrix aka O
        cm = torch.matmul(ground_truth.t(), model_output)

        # expected matrics aka E
        E = torch.matmul(ground_truth.sum(dim=0).view(-1, 1), model_output.sum(dim=0).view(1, -1)) / cm.sum() 

        # calculated kappa loss, variations with 1- and log are mentioned in papers
        result = ( (W*cm.to(device)).sum() / (W*E.to(device)).sum() + self.eps )

        return result


