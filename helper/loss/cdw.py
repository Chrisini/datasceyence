from loss.template import *

class CDWLossChrisy(TemplateLoss):
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


# new loss function by Ram
class CDWLossRam(TemplateLoss):
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

class ClassDistanceWeightedLoss(TemplateLoss):
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
    



