# DatascEYEnce

Collection of my deep learning in ophthalmology projects - built as a framework

Inspired by:
* [PixelSSL](https://github.com/ZHKKKe/PixelSSL) 

## Function calls

Generally, class functions with prefix **"run_"** can be called. For any sort of logging or plotting, call **"log"** functions

## Explanation of modules

| Component | Description |
| --- | --- |
| **helper** | Helper modules |
| **helper.transform** | Data augmentation |
| **helper.dataset** | Dataset |
| **helper.sampler** | Sampler |
| **helper.model** | Models |
| **helper.model.block** | Modules of the models |
| **helper.compute.loss** | Loss functions |
| **helper.compute.metrics** | Metrics (fscore, jac, rec, acc, kappa, ...) |
| **examples** | Unit tests for helper modules |
| **configs** | Settings |
| **data** | csv data preparation for dataloader | 

## Loss functions in **helper.compute.loss**
| Component | Loss+Source | Type | examples.unittest_ |
| --- | --- | --- | --- |
| **loss.corn** | [Conditional Ordinal Regression for NN](https://github.com/Raschka-research-group/coral-pytorch) | ordinal regression | unittest_loss_ordered_class |
| **loss.cdw** | [Class-Distance Loss](https://github.com/GorkemP/labeled-images-for-ulcerative-colitis/blob/main/utils/loss.py) | ordinal regression | unittest_loss_ordered_class |
| **loss.kappa** | [Kappa Loss](https://www.kaggle.com/gennadylaptev/qwk-loss-for-pytorch) | ordinal regression | unittest_loss_ordered_class |
| **loss.dice** | [Dice/F1-score Loss](https://github.com/qubvel/segmentation_models.pytorch) | region-based loss | unittest_loss_seg |
| **loss.shape** | [AAAI sdf loss](https://github.com/JunMa11/SegWithDistMap/blob/master/code/train_LA_AAAISDF.py) | Distance Transform Maps - shape based loss - boundary-based loss | unittest_loss_seg |
| **loss.iou** | [IoU Loss/Jaccard](https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/losses/jaccard.py) | iou | unittest_loss_seg |
| **loss.supcon** | [Supervised Contrastive Loss](https://github.com/HobbitLong/SupContrast/blob/master/losses.py) | representation learning | not tested yet |

## Models in **helper.models**
| Component | Model+Source | Type | examples.unittest_ |
| --- | --- | --- | --- |
| **model.decent_block** | ready to use decent model | ? | not tested yet |
| **model.block.shuffle_block** | ShuffleNet+MLP, ShuffleNet+conv1x1, ShuffleNet+linear layer | encoder block(s), early | not tested yet |
| **model.block.late_block** | late ResNet layers | encoder block(s), late | not tested yet |
| **model.block.fusion_block** | conv1x1 | fusion between blocks or between block(s) and head(s) | not tested yet |
| **model.block.head_block** | linear layer or seg head | head(s) | not tested yet |
| **model.block.noise_block** | Gaussian Noise layer | ? | not tested yet |

## Data samplers in in **helper.sampler**
| Component | Sampler+Source | Type | examples.unittest_ |
| --- | --- | --- | --- |
| **sampler.mixed_batch** | Mixed Batch Sampling | equally sampled images for each class in a batch | unittest_sampler_mixed_batch |


## Explainability, Interpretability and Visualisation in **helper.visualisation**
| Component | Method+Source | Type | examples.unittest_ |
| --- | --- | --- | --- |
| **visualisation.deepdream**| [DeepDream](https://github.com/juanigp/Pytorch-Deep-Dream/blob/master/Deep_Dream.ipynb) | layer ?? | unittest_vis_deepdream |
| **visualisation.feature_map**| Feature Map | layer ?? | unittest_vis_feature_map |
| **visualisation.guided_backprop** | [Guided Backpropagation](https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/guided_backprop.py)| | |


## Data
Guide on how to write/change the data script
* must contain: img_path (absolute path)
* must contain: mode (train, val, test)
* can contain: lbl_whatever, lbl_whatever2, ... (for numeric labels)
* can contain: msk_whatever, msk_whatever2, ... (for masks, absolute path)

## Commenting style 

### for each class:

```
# =============================================================================
#
# General information, sources, ...
#
# =============================================================================
```

### for each function:

```
# =============================================================================
# Describe what is going on
# parameters:
#    parameter1: e.g. hidden vector of shape [bsz, n_views, ...].
#    parameter2: e.g. ground truth of shape [bsz].
# returns:
#    parameter2: e.g. a loss scalar.
# writes:
#    csv file, png images, ...
# notes:
#    Whatever comes into your mind
# sources:
#    https...
# =============================================================================
```

### for sections:

```
# =============================================================================
# Info about this part
# =============================================================================
```


## Save and load checkpoints
```
state = {
    'name': self.experiment_name,
    'epoch': epoch, 
    's_model': self.models["s"].state_dict(),
    't_model': self.models["t"].state_dict(),
    's_optim': self.optimisers["s"].state_dict(),
    's_lrer': self.lrers["s"].state_dict()
}
```