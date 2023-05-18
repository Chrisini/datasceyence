from compute.loss.template import *
    
from PIL import Image
from scipy.ndimage import distance_transform_edt as distance
from skimage import segmentation as skimage_seg    

# https://github.com/JunMa11/SegLoss

class ShapeLoss(TemplateLoss):
    # SignedDistanceLoss
    # AAAI_sdf_loss
    # https://github.com/JunMa11/SegWithDistMap/blob/master/code/train_LA_AAAISDF.py
    
    def __init__(self, smooth=1e-5, weight=10, device="cpu"):
        super(ShapeLoss, self).__init__()
        self.smooth = smooth
        self.weight = weight
        self.device = device
        
    def compute_sdf(self, img_gt, out_shape):
        # Signed distance function
        # https://github.com/JunMa11/SegWithDistMap/blob/master/code/train_LA_AAAISDF.py
        """
        compute the signed distance map of binary mask
        input: segmentation, shape = (batch_size, x, y, z)
        output: the Signed Distance Map (SDM) 
        sdf(x) = 0; x in segmentation boundary
                 -inf|x-y|; x in segmentation
                 +inf|x-y|; x out of segmentation
        normalize sdf to [-1,1]
        """

        img_gt = img_gt.astype(np.uint8)
        normalized_sdf = np.zeros(out_shape)

        for b in range(out_shape[0]): # batch size
            for c in range(out_shape[1]):
                posmask = img_gt[b].astype(np.bool_)
                if posmask.any():
                    negmask = ~posmask
                    posdis = distance(posmask)
                    negdis = distance(negmask)
                    boundary = skimage_seg.find_boundaries(posmask, mode='inner').astype(np.uint8)
                    sdf = (negdis-np.min(negdis))/(np.max(negdis)-np.min(negdis)) - (posdis-np.min(posdis))/(np.max(posdis)-np.min(posdis))
                    sdf[boundary==1] = 0
                    normalized_sdf[b][c] = sdf
                    assert np.min(sdf) == -1.0, print(np.min(posdis), np.max(posdis), np.min(negdis), np.max(negdis))
                    assert np.max(sdf) ==  1.0, print(np.min(posdis), np.min(negdis), np.max(posdis), np.max(negdis))

        return normalized_sdf
        
    def forward(self, model_output, ground_truth):
        
        # print('model_output.shape, ground_truth.shape', model_output.shape, ground_truth.shape)
        # ([4, 1, 112, 112, 80])
        
        with torch.no_grad():
            s = model_output.shape
            ground_truth = self.compute_sdf(ground_truth.cpu().numpy(), s)
            ground_truth = torch.from_numpy(ground_truth).float().to(self.device)
        
        # compute eq (4)
        intersect = torch.sum(model_output * ground_truth)
        pd_sum = torch.sum(model_output ** 2)
        gt_sum = torch.sum(ground_truth ** 2)
        L_product = (intersect + self.smooth) / (intersect + pd_sum + gt_sum + self.smooth)
        # print('L_product.shape', L_product.shape) (4,2)
        L_SDF_AAAI = - L_product + torch.norm(model_output - ground_truth, 1)/torch.numel(model_output)

        return L_SDF_AAAI * self.weight