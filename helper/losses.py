# =============================================================================
# Loss functions:
# * KappaLoss
# * CDWLoss (3 versions)
# * CORN loss
# =============================================================================

class KappaLoss(nn.Module):
    # =============================================================================
    # Kappa Loss (Helper Class)
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


# =============================================================================
# testing the loss function
# =============================================================================
n_classes = 3
criterion = KappaLoss(n_classes=n_classes)

model_outputs = [  torch.tensor([[-0.1, -0.1, 5], [4, -0.1, -0.1], [-0.1, 6, -0.1], [-0.1, -0.1, 3]]), # good
                    torch.tensor([[0.1, 0.1, 0.9], [0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]]), # middle
                    torch.tensor([[5, 0.1, 0.9], [0.9, 3, 0.1], [0.1, -0.3, 0.7], [0.7, 0.1, 0.1]])  # bad
]

ground_truth = torch.tensor([2, 0, 1, 2])

for model_output in model_outputs:
    loss = criterion(model_output=model_output, ground_truth=ground_truth)
    print('\n', '='*50)
    print(loss)




class CDWLossChrisy(nn.Module):
    # =============================================================================
    # Class Distance Weighted Cross Entropy
    # MIUA2022 conference
    # working
    # =============================================================================
    def __init__(self, n_classes=5, alpha=2):
        super(CDWLossChrisy, self).__init__()
        self.soft = torch.nn.Softmax(dim=1)
        self.n_classes = n_classes
        self.alpha = alpha

    def forward(self, model_output, ground_truth, device="cpu"):    

        S=self.soft(model_output)
        log_part = torch.log(1-S)        

        W = torch.zeros_like(model_output)
        for i, target_item in enumerate(ground_truth):
            W[i] = torch.tensor([abs(k - target_item) for k in range(self.n_classes)])
        W.pow_(self.alpha)
        
        image_loss = log_part * W
        batch_loss = image_loss.sum(dim=0)

        return(-torch.mean(batch_loss)/4)

# =============================================================================
# testing the loss function
# =============================================================================
n_classes = 3
criterion = CDWLossChrisy(n_classes=n_classes)

model_outputs = [  torch.tensor([[-0.1, -0.1, 5], [4, -0.1, -0.1], [-0.1, 6, -0.1], [-0.1, -0.1, 3]]), # good
                    torch.tensor([[0.1, 0.1, 0.9], [0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]]), # middle
                    torch.tensor([[5, 0.1, 0.9], [0.9, 3, 0.1], [0.1, -0.3, 0.7], [0.7, 0.1, 0.1]])  # bad
]

ground_truth = torch.tensor([2, 0, 1, 2])

for model_output in model_outputs:
    loss = criterion(model_output=model_output, ground_truth=ground_truth)
    print('\n', '='*50)
    print(loss)



# new loss function by Ram
class CDWLossRam(nn.Module):
    # =============================================================================
    # Class Distance Weighted Cross Entropy
    # MIUA2022 conference
    # working
    # =============================================================================
    def __init__(self, n_classes=5, alpha=2):
        super(CDWLossRam, self).__init__()
        self.soft = torch.nn.Softmax(dim=1)
        self.n_classes = n_classes
        self.alpha = alpha

    def forward(self, model_output, ground_truth, device="cuda"):

        weight = torch.zeros_like(model_output) # |i - c|^alpha
        for i, target_item in enumerate(ground_truth):
            weight[i] = torch.tensor([abs(k - target_item) for k in range(self.n_classes)])

        weight.pow_(self.alpha)

        ground_truth_ = torch.nn.functional.one_hot(ground_truth, num_classes=self.n_classes)
        first_term = torch.log(1 - self.soft(model_output)).reshape(model_output.shape).to(torch.float32) # log (1 - S)
        # second_term = torch.pow(torch.abs(weight - ground_truth_), self.alpha).reshape(self.n_classes, model_output.shape[0]).to(torch.float32) 
        #loss = -torch.mean(torch.flatten(torch.matmul(first_term, weight)))
        loss = -torch.mean(torch.flatten(first_term * weight))

        return loss

# =============================================================================
# testing the loss function
# =============================================================================
n_classes = 3
criterion = CDWLossRam(n_classes=n_classes)

model_outputs = [  torch.tensor([[-0.1, -0.1, 5], [4, -0.1, -0.1], [-0.1, 6, -0.1], [-0.1, -0.1, 3]]), # good
                    torch.tensor([[0.1, 0.1, 0.9], [0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]]), # middle
                    torch.tensor([[5, 0.1, 0.9], [0.9, 3, 0.1], [0.1, -0.3, 0.7], [0.7, 0.1, 0.1]])  # bad
]

ground_truth = torch.tensor([2, 0, 1, 2])

for model_output in model_outputs:
    loss = criterion(model_output=model_output, ground_truth=ground_truth)
    print('\n', '='*50)
    print(loss)




class ClassDistanceWeightedLoss(torch.nn.Module):
    """
    Source: https://github.com/GorkemP/labeled-images-for-ulcerative-colitis/blob/main/utils/loss.py
    Instead of calculating the confidence of true class, this class takes into account the confidences of
    non-ground-truth classes and scales them with the neighboring distance.
    Paper: "Class Distance Weighted Cross-Entropy Loss for Ulcerative Colitis Severity Estimation" (https://arxiv.org/abs/2202.05167)
    It is advised to experiment with different power terms. When searching for new power term, linearly increasing
    it works the best due to its exponential effect.
    """

    def __init__(self, n_classes: int, alpha: float = 2., reduction: str = "mean"):
        super(ClassDistanceWeightedLoss, self).__init__()
        self.n_classes = n_classes
        self.power = alpha
        self.reduction = reduction

    def forward(self, model_output: torch.Tensor, ground_truth: torch.Tensor) -> torch.Tensor:
        
        input_sm = model_output.softmax(dim=1)

        weight_matrix = torch.zeros_like(input_sm)
        for i, target_item in enumerate(ground_truth):
            weight_matrix[i] = torch.tensor([abs(k - target_item) for k in range(self.n_classes)])

        weight_matrix.pow_(self.power)

        # TODO check here, stop here if a nan value and debug it
        reverse_probs = (1 - input_sm).clamp_(min=1e-4)

        log_loss = -torch.log(reverse_probs)
        if torch.sum(torch.isnan(log_loss) == True) > 0:
            print("nan detected in forward pass")

        loss = log_loss * weight_matrix
        loss_sum = torch.sum(loss, dim=1)

        if self.reduction == "mean":
            loss_reduced = torch.mean(loss_sum)
        elif self.reduction == "sum":
            loss_reduced = torch.sum(loss_sum)
        else:
            raise Exception("Undefined reduction type: " + self.reduction)

        return loss_reduced

# =============================================================================
# testing the loss function
# =============================================================================

n_classes = 3
criterion = ClassDistanceWeightedLoss(n_classes=n_classes)

model_outputs = [  torch.tensor([[-0.1, -0.1, 5], [4, -0.1, -0.1], [-0.1, 6, -0.1], [-0.1, -0.1, 3]]), # good
                    torch.tensor([[0.1, 0.1, 0.9], [0.9, 0.1, 0.1], [0.1, 0.9, 0.1], [0.1, 0.1, 0.9]]), # middle
                    torch.tensor([[5, 0.1, 0.9], [0.9, 3, 0.1], [0.1, -0.3, 0.7], [0.7, 0.1, 0.1]])  # bad
]

ground_truth = torch.tensor([2, 0, 1, 2])

for model_output in model_outputs:    
    loss = criterion(model_output=model_output, ground_truth=ground_truth)
    print('\n', '='*50)
    print(loss)



def corn_loss(logits, y_train, num_classes):
    """Computes the CORN loss described in our forthcoming
    'Deep Neural Networks for Rank Consistent Ordinal
    Regression based on Conditional Probabilities'
    manuscript.
    Parameters
    ----------
    logits : torch.tensor, shape=(num_examples, num_classes-1)
        Outputs of the CORN layer.
    y_train : torch.tensor, shape=(num_examples)
        Torch tensor containing the class labels.
    num_classes : int
        Number of unique class labels (class labels should start at 0).
    Returns
    ----------
        loss : torch.tensor
        A torch.tensor containing a single loss value.
    Examples
    ----------
    >>> import torch
    >>> from coral_pytorch.losses import corn_loss
    >>> # Consider 8 training examples
    >>> _  = torch.manual_seed(123)
    >>> X_train = torch.rand(8, 99)
    >>> y_train = torch.tensor([0, 1, 2, 2, 2, 3, 4, 4])
    >>> NUM_CLASSES = 5
    >>> #
    >>> #
    >>> # def __init__(self):
    >>> corn_net = torch.nn.Linear(99, NUM_CLASSES-1)
    >>> #
    >>> #
    >>> # def forward(self, X_train):
    >>> logits = corn_net(X_train)
    >>> logits.shape
    torch.Size([8, 4])
    >>> corn_loss(logits, y_train, NUM_CLASSES)
    tensor(0.7127, grad_fn=<DivBackward0>)
    """

    logsigmoid = nn.LogSigmoid()

    sets = []
    for i in range(num_classes-1):
        label_mask = y_train > i-1
        label_tensor = (y_train[label_mask] > i).to(torch.int64)
        sets.append((label_mask, label_tensor))

    num_examples = 0
    losses = 0.
    for task_index, s in enumerate(sets):
        train_examples = s[0]
        train_labels = s[1]

        if len(train_labels) < 1:
            continue

        num_examples += len(train_labels)
        pred = logits[train_examples, task_index]

        loss = -torch.sum(logsigmoid(pred)*train_labels
                          + (logsigmoid(pred) - pred)*(1-train_labels))
        losses += loss

    return losses/num_examples

def corn_label_from_logits(logits): # to call: predicted_labels = corn_label_from_logits(logits).float()
    """
    Returns the predicted rank label from logits for a
    network trained via the CORN loss.
    Parameters
    ----------
    logits : torch.tensor, shape=(n_examples, n_classes)
        Torch tensor consisting of logits returned by the
        neural net.
    Returns
    ----------
    labels : torch.tensor, shape=(n_examples)
        Integer tensor containing the predicted rank (class) labels
    Examples
    ----------
    >>> # 2 training examples, 5 classes
    >>> logits = torch.tensor([[14.152, -6.1942, 0.47710, 0.96850],
    ...                        [65.667, 0.303, 11.500, -4.524]])
    >>> corn_label_from_logits(logits)
    tensor([1, 3])
    """
    probas = torch.sigmoid(logits)
    probas = torch.cumprod(probas, dim=1)
    predict_levels = probas > 0.5
    predicted_labels = torch.sum(predict_levels, dim=1)
    return predicted_labels

class CornLoss(torch.nn.Module):
    # =============================================================================
    # CORN (Helper Class)
    # https://github.com/Raschka-research-group/coral-pytorch
    # from https://github.com/Raschka-research-group/coral-pytorch/blob/main/coral_pytorch/losses.py
    # and https://raschka-research-group.github.io/coral-pytorch/tutorials/pure_pytorch/CORN_mnist/
    # Todo: Adding the Corn layer to the network. Model output should come from this layer.
    # =============================================================================
    """
    Computes the CORN loss described in our forthcoming
    'Deep Neural Networks for Rank Consistent Ordinal
    Regression based on Conditional Probabilities'
    manuscript.
    Parameters
    ----------
    num_classes : int
        Number of unique class labels (class labels should start at 0).
    Examples
    ----------
    >>> import torch
    >>> from coral_pytorch.losses import corn_loss
    >>> # Consider 8 training examples
    >>> _  = torch.manual_seed(123)
    >>> X_train = torch.rand(8, 99)
    >>> y_train = torch.tensor([0, 1, 2, 2, 2, 3, 4, 4])
    >>> NUM_CLASSES = 5
    >>> #
    >>> #
    >>> # def __init__(self):
    >>> corn_net = torch.nn.Linear(99, NUM_CLASSES-1)
    >>> #
    >>> #
    >>> # def forward(self, X_train):
    >>> logits = corn_net(X_train)
    >>> logits.shape
    torch.Size([8, 4])
    >>> corn_loss(logits, y_train, NUM_CLASSES)
    tensor(0.7127, grad_fn=<DivBackward0>)
    """
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
        
    def forward(self, model_output, ground_truth):
        """
        Parameters
        ----------
        model_output : torch.tensor, shape=(num_examples, n_classes-1)
            Outputs of the CORN layer.
        ground_truth : torch.tensor, shape=(num_examples)
            Torch tensor containing the class labels.
        Returns
        ----------
        loss : torch.tensor
            A torch.tensor containing a single loss value.
        """
        return corn_loss(model_output, ground_truth, num_classes=self.n_classes)


# =============================================================================
# testing the loss function
# =============================================================================
n_classes = 3
criterion = CornLoss(n_classes=n_classes)

# with 2 output neurons
model_outputs = [  torch.tensor([[2, 1], [-0.3, 0.0], [0.8, 0.1], [2, 0.8]]), # good
                    torch.tensor([[0.1, 0.8], [0.1, 0.7], [0.8, 0.2], [0.7, 0.8]]), # mid
                    torch.tensor([[0.2, 0.1], [2, 0.3], [0.7, 0.8], [0.0, 0.3]]) # bad
]

ground_truth = torch.tensor([2, 0, 1, 2])

for model_output in model_outputs:
    loss = criterion(model_output=model_output, ground_truth=ground_truth)
    print('\n', '='*50)
    print(loss)