# medical_ai

Collection of my deep learning in ophthalmology projects - built as a framework

Inspired by:
* [PixelSSL](https://github.com/ZHKKKe/PixelSSL) 

Generally, class functions with prefix **"run_"** can be called. For any sort of logging or plotting, call **"log"** functions


## Explanation of modules

| Component | Description |
| --- | --- |
| **helper** | Helper modules |
| **helper.transform** | Data augmentation |
| **helper.dataset** | Dataset |
| **helper.datasampler** | Sampler |
| **helper.compute.loss** | Loss functions |
| **helper.compute.metrics** | Metrics (fscore, jac, rec, acc, kappa, ...) |
| **examples** | Unit tests for helper modules |
| **configs** | Settings |
| **data** | csv data preparation for dataloader | 

## Loss functions in **helper.compute.loss**
| Component | Loss+Source | Type | examples.unittest_ |
| --- | --- | --- | --- |
| **loss.corn** | [Conditional Ordinal Regression for NN](https://github.com/Raschka-research-group/coral-pytorch) | ordinal regression | unittest_ordered_class_loss |
| **loss.cdw** | [Class-Distance Loss](https://github.com/GorkemP/labeled-images-for-ulcerative-colitis/blob/main/utils/loss.py) | ordinal regression | unittest_ordered_class_loss |
| **loss.kappa** | [Kappa Loss](https://www.kaggle.com/gennadylaptev/qwk-loss-for-pytorch) | ordinal regression | unittest_ordered_class_loss |
| **loss.dice** | [Dice/F1-score Loss](https://github.com/qubvel/segmentation_models.pytorch) | region-based loss | unittest_seg_loss |
| **loss.shape** | [AAAI sdf loss](https://github.com/JunMa11/SegWithDistMap/blob/master/code/train_LA_AAAISDF.py) | Distance Transform Maps - shape based loss - boundary-based loss | unittest_seg_loss |
| **loss.iou** | [IoU Loss/Jaccard](https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/losses/jaccard.py) | iou | unittest_seg_loss |
| **loss.supcon** | [Supervised Contrastive Loss](https://github.com/HobbitLong/SupContrast/blob/master/losses.py) | representation learning | not tested yet |

## Models in **helper.models**
| Component | Model+Source | Type | examples.unittest_ |
| --- | --- | --- | --- |
| **model.early_block** | ShuffleNet+MLP, ShuffleNet+conv1x1, ShuffleNet+linear layer | encoder block(s), early | not tested yet |
| **model.late_block** | late ResNet layers | encoder block(s), late | not tested yet |
| **model.fusion_block** | conv1x1 | fusion between blocks or between block(s) and head(s) | not tested yet |
| **model.head_block** | linear layer or seg head | head(s) | not tested yet |

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