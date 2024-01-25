# DatascEYEnce

Collection of my deep learning in ophthalmology projects - built as a framework.

Covering theory here: https://variint.github.io/datasceyence

Main sources:
* [PixelSSL](https://github.com/ZHKKKe/PixelSSL) 
* [SSL4MIS](https://github.com/HiLab-git/SSL4MIS)

## Todos
* [ ] need to change transforms - has to be for every item key that contains "img" not is equal to ... (same with "msk")


## Function calls

Generally, class functions with prefix **"run_"** can be called. For any sort of logging or plotting, call **"log"** functions

Every helper code inherits its template class

## Main files

| Component | Description |  | 
| --- | --- | --- |
| **run_bimt** | just testing a baseline: Brain-inspired modular training: https://github.com/KindXiaoming/BIMT/blob/main/mnist_3.5.ipynb |  | 
| **run_decentnet** | currently working on a modular convnet |  | 
| **run_meanteacher** | my master thesis - unsupervised domain adaptation and semi-supervised learning |  | 


## Explanation of dirs

| Component | Description | Called by | Notes |
| --- | --- | --- |
| **configs** | Settings: .ini files | | (currently not in use) |
| **data_prep** | csv data preparation for dataloader | prepare_.ipybn | 
| **examples** | Unit tests for helper modules and general code testing | utest_.ipybn | 
| **helper** | Helper modules | - |
| **helper.compute** | Compute Unit - for each head (seg/class/reg, bin/multi), computes loss and metrics | run_.ipybn | 
| **helper.compute.loss** | Loss functions, called in compute |
| **helper.compute.metrics** | Metrics (fscore, jac, rec, acc, kappa, ...) | used in compute |
| **helper.dataset** | Dataset | run_.ipybn | 
| **helper.dataset.transform** | Data augmentation | used by dataset |
| **helper.model** | Models | run_.ipybn | 
| **helper.model.block** | Modules of the models | used by model |
| **helper.sampler** | Sampler | run_.ipybn | 
| **helper.visualisation** | explainability, interpretability, plots | run_.ipybn | 


## Loss functions in **helper.compute.loss**
| Component | Loss+Source | Type | examples.utest_ |
| --- | --- | --- | --- |
| **loss.corn** | [Conditional Ordinal Regression for NN](https://github.com/Raschka-research-group/coral-pytorch) | ordinal regression | utest_loss_ordered_class |
| **loss.cdw** | [Class-Distance Loss](https://github.com/GorkemP/labeled-images-for-ulcerative-colitis/blob/main/utils/loss.py) | ordinal regression | utest_loss_ordered_class |
| **loss.kappa** | [Kappa Loss](https://www.kaggle.com/gennadylaptev/qwk-loss-for-pytorch) | ordinal regression | utest_loss_ordered_class |
| **loss.dice** | [Dice/F1-score Loss](https://github.com/qubvel/segmentation_models.pytorch) | region-based loss | utest_loss_seg |
| **loss.shape** | [AAAI sdf loss](https://github.com/JunMa11/SegWithDistMap/blob/master/code/train_LA_AAAISDF.py) | Distance Transform Maps - shape based loss - boundary-based loss | utest_loss_seg |
| **loss.iou** | [IoU Loss/Jaccard](https://github.com/qubvel/segmentation_models.pytorch/blob/master/segmentation_models_pytorch/losses/jaccard.py) | iou | utest_loss_seg |
| **loss.supcon** | [Supervised Contrastive Loss](https://github.com/HobbitLong/SupContrast/blob/master/losses.py) | representation learning | not tested yet |

## Models in **helper.models**
| Component | Model+Source | Type | examples.utest_ |
| --- | --- | --- | --- |
| **model.decentblock** | ready to use decent model | ? | not tested yet |
| **model.block.shuffle_block** | ShuffleNet+MLP, ShuffleNet+conv1x1, ShuffleNet+linear layer | encoder block(s), early | not tested yet |
| **model.block.late_block** | late ResNet layers | encoder block(s), late | not tested yet |
| **model.block.fusion_block** | conv1x1 | fusion between blocks or between block(s) and head(s) | not tested yet |
| **model.block.head_block** | linear layer or seg head | head(s) | not tested yet |
| **model.block.noise_block** | Gaussian Noise layer | ? | not tested yet |

## Data samplers in in **helper.sampler**
| Component | Sampler+Source | Type | examples.utest_ |
| --- | --- | --- | --- |
| **sampler.mixed_batch** | Mixed Batch Sampling | equally sampled images for each class in a batch | utest_sampler_mixed_batch |


## Explainability, Interpretability and Visualisation in **helper.visualisation**
| Component | Method+Source | Type | examples.utest_ |
| --- | --- | --- | --- |
| **visualisation.deepdream**| [DeepDream](https://github.com/juanigp/Pytorch-Deep-Dream/blob/master/Deep_Dream.ipynb) | layer ?? | utest_vis_deepdream |
| **visualisation.feature_map**| Feature Map | layer ?? | utest_vis_feature_map |
| **visualisation.guided_backprop** | [Guided Backpropagation](https://github.com/utkuozbulak/pytorch-cnn-visualizations/blob/master/src/guided_backprop.py)| | |


## Data
Guide on how to write/change the data script
* must contain: img_path (absolute path)
* must contain: mode (train, val, test)
* can contain: lbl_whatever, lbl_whatever2, ... (for numeric labels)
* can contain: msk_path_whatever, msk_path_whatever2, ... (for masks, absolute path)
* mask convention: 0 = background (black), 1 or 255 = class (white)
* sep=';' for to_csv and read_csv

<img src="readme/example_mask.bmp" width="100">


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
# saves:
#    collector of data within a class 
# writes:
#    csv file, png images, ...
# notes:
#    Whatever comes into your mind
# sources:
#    https...
# nonsense?
#    comments i don't want to delete but i have no idea what i wanted to say
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
# student - teacher architectures such as mean teacher
state = {
    'name': self.experiment_name,
    'epoch': epoch, 
    's_model': self.models["s"].state_dict(),
    't_model': self.models["t"].state_dict(),
    's_optim': self.optimisers["s"].state_dict(),
    's_lrer': self.lrers["s"].state_dict()
}
```
