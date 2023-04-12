from loss.template import *

class CornLoss(TemplateLoss):
    # =============================================================================
    # CORN
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
    """
    def __init__(self, n_classes):
        super(CornLoss, self).__init__()
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
        return self.corn_loss(model_output, ground_truth, num_classes=self.n_classes)
    
    def corn_loss(self, logits, y_train, num_classes):
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
        """

        logsigmoid = torch.nn.LogSigmoid()

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
    
    def get_predicted_label(self, logits):
        # to generate labels for metrics
        # to call: predicted_labels = get_label(logits).float()
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
        """
        probas = torch.sigmoid(logits)
        probas = torch.cumprod(probas, dim=1)
        predict_levels = probas > 0.5
        predicted_labels = torch.sum(predict_levels, dim=1)
        return predicted_labels

    


