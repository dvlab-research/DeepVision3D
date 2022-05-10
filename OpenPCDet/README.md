# Outdoor Object Detection

This repo contains the code and configuration files for outdoor object detection.

## Related Papers
* A Unified Query-based Paradigm for Point Cloud Understanding ([paper](https://arxiv.org/pdf/2203.01252v3.pdf))

## Acknowledgements
* This implementation heavily relies on [OpenPCDet](https://github.com/open-mmlab/OpenPCDet). We thank the OpenPCDet team for providing such a well-organized codebase.

## Important Notes
* All LiDAR-based models are trained with 4 3090 GPUs with 24Gb memory and are available for download. Please mind your GPU memory cost during training.

## Results and Models

`Note.` All models below follow EQ-Paradigm and use a [Q-Net](../EQNet/eqnet/models/query_producer/qnet.py) to enable a free combination
between backbones and heads. `(*)` means the improvement compared to the model with its original backbone network without Q-Net.
### KITTI Dataset models
|                                             |Backbone| Car@R40 | Ped.@R40 | Cyc.@R40  | download | 
|---------------------------------------------|:----------|:-------:|:-------:|:-------:|:---------:|
| [SECOND](tools/cfgs/kitti_models/eq_paradigm/vxbased_backbone/second.yaml) |SparseConvNet| 81.72 (+0.06) | 53.32 (+2.20) | 66.58 (+3.22) | [model](https://drive.google.com/file/d/1wqIq6lEGLLQ5uY-AT-765zucOS1rgB_0/view?usp=sharing) | 
| [PointRCNN](tools/cfgs/kitti_models/eq_paradigm/vxbased_backbone/pointrcnn.yaml)|SparseConvNet  | 82.52 | 60.85 | 72.76 | [model](https://drive.google.com/file/d/188TAuZZzkpE0TnZ5JG5q81ZO_Y8_hAUp/view?usp=sharing)|
| [PVRCNN](tools/cfgs/kitti_models/eq_paradigm/vxbased_backbone/pv_rcnn.yaml)| SparseConvNet | 85.31 (+0.56) | 62.00 (+7.58) | 75.22 (+4.80) | [model](https://drive.google.com/file/d/1QIgfqnA1tbg5EH3hWoa28AwekIzmF24t/view?usp=sharing) |
| [VoxelRCNN](tools/cfgs/kitti_models/eq_paradigm/vxbased_backbone/proposal_grid_rcnn.yaml) |SparseConvNet| 85.39 | 60.85 | 74.23 | [model](https://drive.google.com/file/d/1xwfnD1QcaxLVySK6S2yibwXTYMmcZm5X/view?usp=sharing) |
| [SECOND](tools/cfgs/kitti_models/eq_paradigm/ptbased_backbone/second.yaml) |PointNet++ (MSG) |82.46 | 50.05 | 61.96 | [model](https://drive.google.com/file/d/1rG15izGVX66lGFyrL0VVfNj8fqAr-r7P/view?usp=sharing)| 
| [PointRCNN](tools/cfgs/kitti_models/eq_paradigm/ptbased_backbone/pointrcnn.yaml) | PointNet++ (MSG) | 84.42 (+4.17) | 58.09 (+3.76) | 72.93 (+1.38) | [model](https://drive.google.com/file/d/1u-vEufcnz8NEHPiuXODGUsSPTNGx625a/view?usp=sharing) |
| [PVRCNN](tools/cfgs/kitti_models/eq_paradigm/ptbased_backbone/pv_rcnn.yaml)| PointNet++ (MSG) | 84.94 | 54.77 | 73.43 | [model](https://drive.google.com/file/d/1vfEkIcHrS-Pv_DVAaqYIZ08r7XmTWPsR/view?usp=sharing) |
| [VoxelRCNN](tools/cfgs/kitti_models/eq_paradigm/ptbased_backbone/proposal_grid_rcnn.yaml) |PointNet++ (MSG)| 84.83 | 53.84 | 65.10 | [model](https://drive.google.com/file/d/1F9ySITGXHHNKDrLFCh4mUJT-4bf7BCdc/view?usp=sharing) |
* `NOTE.` The provided models above are slightly different from those in the original EQ-Paradigm paper with fewer parameters, faster inference speed
 yet comparable performance. If you wish to reproduce the results mentioned in the paper, please refer to the
 [supplementary materials](https://drive.google.com/file/d/1fEi-_OmyDuu4ToA7LUBd0MRKanXmIHnY/view?usp=sharing) to correspondingly adjust the hyper-parameters like the number of channels and neighbors in Q-Net.

### Other datasets
Models on other datasets will be supported soon.

## Getting Started
### Data preprocessing
Please refer to [OpenPCDet](https://github.com/open-mmlab/OpenPCDet/blob/master/docs/GETTING_STARTED.md) for data preparation before the first run.

### Training
We take the SECOND with voxel-based SparseConvNet backbone as an example here.
```
cd /path/to/DeepVision3D/OpenPCDet && cd tools

# training configuration.
NUM_GPUS=4
CONFIG_FILE=cfgs/kitti_models/eq_paradigm/vxbased_backbone/second.yaml

# training script.
bash scripts/dist_train.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE} --accumulated_step 2
```

### Testing
```
cd /path/to/DeepVision3D/OpenPCDet && cd tools

# testing configuration.
NUM_GPUS=2
CONFIG_FILE=cfgs/kitti_models/eq_paradigm/vxbased_backbone/second.yaml
CKPT=/path/to/second_vxbased_backbone.pth
BATCH_SIZE=8

# testing script.
bash scripts/dist_test.sh ${NUM_GPUS} --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```