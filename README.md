# medical_ai


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
| **model.baseline** | ResNet50 | classification | not tested yet |
| **model.decent_block** | ShuffleNet + MLP, ShuffleNet + conv1x1, ResNet layers, U-Net | encoder block(s), early vs late | not tested yet |
| **model.decent_fusion** | conv1x1 | fusion between blocks or between block(s) and head(s) | not tested yet |
| **model.decent_head** | linear layer or seg head | head(s) | not tested yet |
| **model.decent_net** | combined block(s), fusion (optional) and head | final classification or segmentation network | not tested yet |
