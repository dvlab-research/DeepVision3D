# Indoor Object Detection

This repo contains the code and configuration files for indoor object detection.

## Related Papers
* A Unified Query-based Paradigm for Point Cloud Understanding ([paper](https://arxiv.org/pdf/2203.01252.pdf))

## Acknowledgements
* This implementation heavily relies on [MMDetection3D](https://github.com/open-mmlab/mmdetection3d/tree/master/data). We thank the MMDetection3D team for providing such a well-organized codebase.

## Results and Models

`Note.` All models below follow EQ-Paradigm and use a [Q-Net](../EQNet/eqnet/models/query_producer/qnet.py) to enable a free combination
between backbones and heads. `(*)` means the improvement compared to the model with its original backbone network without Q-Net.
### ScanNet Dataset models
|                                             |Backbone| mAP@0.25 | mAP@0.5  | download | 
|---------------------------------------------|:----------|:-------:|:-------:|:---------:|
| [VoteNet](configs/eq_paradigm/votenet/ptbased_backbone/eqvotenet_ptbased_backbone_scannet-3d-18class.py) |PointNet++ (SSG)| 64.7 (+2.4) | 45.3 (+5.4) | [model](https://drive.google.com/file/d/1dwrh74z3jaoamm0iSi2mtFRxHbDUHp1Q/view?usp=sharing) | 
| [GroupFree3d](configs/eq_paradigm/groupfree/ptbased_backbone/eqgroupfree_ptbased_backbone_scannet-3d-18class-L6-O256.py)|PointNet++ (SSG)  | 67.7 (+1.4) | 51.0 (+3.2) | [model](https://drive.google.com/file/d/1dxTLVgHSFWbhghuLS6v-xGBgVzI-Pk46/view?usp=sharing)|
| [VoteNet](configs/eq_paradigm/votenet/vxbased_backbone/eqvotenet_vxbased_backbone_scannet-3d-18class.py)| SparseConvNet | 58.1 | 39.5 | [model](https://drive.google.com/file/d/1LAEPJciq_51f-shzRmuY21ydMLgog1b1/view?usp=sharing) |
| [GroupFree3d](configs/eq_paradigm/groupfree/vxbased_backbone/eqgroupfree_vxbased_backbone_scannet-3d-18class-L6-O256.py) |SparseConvNet| 60.0 | 41.3 | [model](https://drive.google.com/file/d/1bKXzPO3ywzfsVlOR4FqmA0U5H1a9p7M-/view?usp=sharing) |


### SUN RGB-D Dataset models
|                                             |Backbone| mAP@0.25 | mAP@0.5  | download | 
|---------------------------------------------|:----------|:-------:|:-------:|:---------:|
| [VoteNet](configs/eq_paradigm/votenet/ptbased_backbone/eqvotenet_ptbased_backbone_sunrgbd-3d-10class.py) |PointNet++ (SSG)| 61.3 (+1.6) | 39.9 (+4.2) | [model](https://drive.google.com/file/d/1eoDZ3ZAyeFmGkQVZiMHygVhpXMDveYBm/view?usp=sharing) | 
| [VoteNet](configs/eq_paradigm/votenet/vxbased_backbone/eqvotenet_vxbased_backbone_sunrgbd-3d-10class.py) | SparseConvNet | 60.0 | 40.3 | [model](https://drive.google.com/file/d/12uYNdtnRlt7zu8b5jcx9czA-7h5OcHjK/view?usp=sharing) | 

### More models on other datasets
Models on other backbones and datasets will be released soon.

## Getting Started
### Data preprocessing
Please refer to [MMDetection3D](https://github.com/open-mmlab/mmdetection3d/tree/master/data) for data preparation before the first run.

### Training
We take the VoteNet with PointNet++ (SSG) backbone on ScanNet dataset as an example here.
```
cd /path/to/DeepVision3D/MMDetection3D

# training configuration.
NUM_GPUS=4
CONFIG_FILE=configs/eq_paradigm/votenet/ptbased_backbone/eqvotenet_ptbased_backbone_scannet-3d-18class.py

# training script.
tools/dist_train.sh ${CONFIG_FILE} ${NUM_GPUS}
```

### Testing
```
cd /path/to/DeepVision3D/MMDetection3D

# testing configuration.
NUM_GPUS=1
CONFIG_FILE=configs/eq_paradigm/votenet/ptbased_backbone/eqvotenet_ptbased_backbone_scannet-3d-18class.py
CHECKPOINT=/path/to/eqvotenet_ptbased_backbone_scannet-3d-18class.pth

# testing script.
bash tools/dist_test.sh ${CONFIG_FILE} ${CHECKPOINT} ${NUM_GPUS} --eval mAP
```