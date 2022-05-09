# Semantic Segmentation

This repo contains the code and configuration files for point cloud semantic segmentation.


## Related Papers
* A Unified Query-based Paradigm for Point Cloud Understanding ([paper](https://arxiv.org/pdf/2203.01252.pdf))

## Results and Models
`Note.` All models below are trained with 8 1080TI GPU, follow EQ-Paradigm, and use a [Q-Net](../EQNet/eqnet/models/query_producer/qnet.py) to enable a free combination
between backbones and heads. `(*)` means the improvement compared to the model with its original backbone network without Q-Net.

### ScanNet Semantic Segmentation Model
|                                             |Backbone| mIoU | mAcc | allAcc | download
|---------------------------------------------|:----------|:-------:|:---------:|:---------:|:---------:|
| [EQNet](config/eqnet_scannet.yaml) | SparseConvNet | 75.1 (+2.2) | 82.7 (+1.9) | 91.1 (+0.7) | [model](https://drive.google.com/file/d/1152aLDOhoLff5EEMzW2cj0Z1vK0FpXB6/view?usp=sharing) |

### S3DIS Semantic Segmentation Model
To be released soon.

### More models
Performance of other backbones supported in this codebase will be released soon.


## Getting Started
### ScanNet V2
* `Data preparation:` Download **ScanNet v2** [here](https://github.com/ScanNet/ScanNet) and preprocess the data.
    ```
    cd /path/to/DeepVision3D/DVSegmentation/data/scannetv2
    python prepare_data.py --scannet_path /path/to/ScanNet --split [train/val/test]
    ```

* `Training: ` You can train on ScanNet v2 with following codes.
    ```
    cd /path/to/DeepVision3D/DVSegmentation
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash train_segmentation.sh 8 --config config/eqnet_scannet.yaml
    ```

* `Testing: `
    ```
    python test_segmentation.py --config config/eqnet_scannet.yaml --set NECK.QUERY_POSITION_CFG.SELECTION_FUNCTION _get_point_query_position
    ```
    For testing our provided model:
    ```
    CHECKPOINT=/path/to/eqnet_scannet_v2-000000600.pth
    python test_segmentation.py --config config/eqnet_scannet.yaml --pretrain ${CHECKPOINT} --set NECK.QUERY_POSITION_CFG.SELECTION_FUNCTION _get_point_query_position
    ```

