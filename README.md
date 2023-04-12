# medical_ai


| Component | Description |
| --- | --- |
| **helper** | Helper modules |
| **helper.transform** | Data augmentation |
| **helper.dataset** | Dataset |
| **helper.datasampler** | Sampler |
| **helper.loss** | Loss functions |
| **examples** | Unit tests for helper modules |
| **configs** | Settings |


## Loss functions in **helper.loss**
| Component | Loss+Source | Type | examples. |
| --- | --- | --- | --- |
| **loss.corn** | (Conditional Ordinal Regression for NN)["https://github.com/Raschka-research-group/coral-pytorch"] | ordinal regression | unittest_ordered_class_loss |
| **loss.cdw** | (Class-Distance Loss)["https://github.com/GorkemP/labeled-images-for-ulcerative-colitis/blob/main/utils/loss.py"] | ordinal regression | unittest_ordered_class_loss |
| **loss.kappa** | (Kappa Loss)["https://www.kaggle.com/gennadylaptev/qwk-loss-for-pytorch"] | ordinal regression | unittest_ordered_class_loss |
| **loss.dice** | (Dice/F1-score Loss)["https://github.com/qubvel/segmentation_models.pytorch"] | region-based loss | unittest_seg_loss |
| **loss.shape** | (AAAI sdf loss) ["https://github.com/JunMa11/SegWithDistMap/blob/master/code/train_LA_AAAISDF.py"] | Distance Transform Maps - shape based loss - boundary-based loss | unittest_seg_loss |
| **loss.iou** | (IoU Loss/Jaccard)["https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/losses/jaccard.py"] | iou | unittest_seg_loss |