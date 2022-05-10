# Shape Classification

This repo contains the code and configuration files for shape classification.


## Related Papers
* A Unified Query-based Paradigm for Point Cloud Understanding ([paper](https://arxiv.org/pdf/2203.01252v3.pdf))

## Acknowledgements
* This implementation is modified from [Pytorch_PointNet++](https://github.com/yanx27/Pointnet_Pointnet2_pytorch). We thank the author for providing such a well-organized codebase.

## Results and Models
`Note.` All models below are trained with 1 1080TI GPU, follow EQ-Paradigm and use a [Q-Net](../EQNet/eqnet/models/query_producer/qnet.py) to enable a free combination
between backbones and heads. `(*)` means the improvement compared to the model with its original backbone network without Q-Net.

### ModelNet40 Classification Model
|                                             |Backbone| Accuracy | download | 
|---------------------------------------------|:----------|:-------:|:---------:|
| [EQ-PointNet++](config/eqpointnet2.yaml)  |PointNet++ (SSG)| 93.18 (+0.98) | [model](https://drive.google.com/file/d/1AFkq0a2tN4N0359-oTpjeBysAHeaidot/view?usp=sharing) | 

### More models
Performance of other backbones supported in this codebase will be released soon.

## Getting Started
### Data Preparation
Download alignment **ModelNet** [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `data/modelnet40_normal_resampled/`.

### Training
You can run different modes with following codes. 
* If you want to use offline processing of data, you can use `--process_data` in the first run. You can download pre-processd data [here](https://drive.google.com/drive/folders/1_fBYbDO3XSdRt3DSbEBe41r5l9YpIGWF?usp=sharing) and save it in `data/modelnet40_normal_resampled/`.
```
cd /path/to/DeepVision3D/DVClassification
python train_classification.py --config config/eqpointnet2.yaml
```

### Testing
```
cd /path/to/DeepVision3D/DVClassification
python test_classification.py --config config/eqpointnet2.yaml
```
For testing our provided model:
```
cd /path/to/DeepVision3D/DVClassification
python test_classification.py --config config/eqpointnet2.yaml --ckpt /path/to/eqpointnet2_modelnet40.pth
```
