from compute.loss.template import *

class SupConLoss(TemplateLoss):
    # =============================================================================
    # The goal of the supervised contrastive loss is:
    # - pull together points belonging to the same class in the embedding space
    # - push apart clusters of samples from different classes. 
    # The loss function is supervised, hence we are leveraging label information

    # --- no idea whether it is doing this?? ---
    # The loss function considers many positives per anchor in addition to many negatives 
    # (as opposed to self-supervised contrastive learning which uses only a single positive)
    # These positives are drawn from samples of the same class as the anchor,
    # rather than being data augmentations of the anchor, as done in self-supervised learning.

    # Author: Yonglong Tian (yonglong@mit.edu)
    # https://github.com/HobbitLong/SupContrast/blob/master/losses.py
    # Date: May 07, 2020
    # Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    # =============================================================================

    def __init__(self, temperature=0.07,
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        
    def print(self):
        pass
        # print("features", features)

        # print("features.shape", features.shape)
        
        # print("batch_size", batch_size)
        
        # print("contrast_count", contrast_count)
        
        # print("labels", labels)
        # print("labels.shape", labels.shape)
        
        # print("mask", mask)
        # print("contrast_feature", contrast_feature)
        # print("contrast_feature.shape", contrast_feature.shape)
        
        # print("anchor_dot_contrast", anchor_dot_contrast)
        # print("anchor_dot_contrast.shape", anchor_dot_contrast.shape)
        # print("logits", logits)
        # print("logits.shape", logits.shape)
        # print("mask", mask)
        # print("mask.shape", mask.shape)
        # print("logits_mask", logits_mask)
        # print("logits_mask.shape", logits_mask.shape)
        # print("exp_logits", exp_logits)
        # print("exp_logits.shape", exp_logits.shape)
        # print("log_prob", log_prob)
        # print("log_prob.shape", log_prob.shape)
        # print("mean_log_prob_pos", mean_log_prob_pos)
        # print("loss", loss)
        # print("loss.shape", loss.shape)
        # print("loss view", loss)
        

    def forward(self, features, labels):
        # =============================================================================
        # Compute loss for model.
        # parameters:
        #    features: hidden vector of shape [bsz, n_views, ...].
        #    labels: ground truth of shape [bsz].
        # returns:
        #    A loss scalar.
        # notes:
        # =============================================================================
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
    
        contrast_count = features.shape[1]

        # create mask
        labels = labels.contiguous().view(-1, 1)

        if labels.shape[0] != batch_size:
            raise ValueError('Num of labels does not match num of features')
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        
        # mask-out self-contrast cases
        # inverse identity matrix, size [batchsize*contrast count, batchsize*contrast count]
        # 0 1 1
        # 1 0 1
        # 1 1 0
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        
        return loss