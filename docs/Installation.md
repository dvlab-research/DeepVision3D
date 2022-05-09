# Installing DeepVision3D ToolBox

## Prerequisites
* Linux or macOS
* Python 3.8+
* PyTorch 1.3+
* CUDA 10.2 / CUDA 11
* GCC 5+

## Installation
(1) Clone this repository.
```
git clone https://github.com/dvlab-research/DeepVision3D
```
(2) Build anaconda Environment.
```
conda create -n deepvision3d python=3.8 -y
conda activate deepvision3d
```
(3) Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).

`E.g. 1` To install PyTorch 1.10.1 with a CUDA version of 10.2:
```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=10.2 -c pytorch
```

`E.g. 2` To install PyTorch 1.10.1 with a CUDA version of 11.3:
```
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

(4) Install requirements.
```
cd /path/to/DeepVision3D
pip install -r requirements.txt
```

(5) Build [MMCV](https://mmcv.readthedocs.io/en/latest/).
```
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/{cu_version}/{torch_version}/index.html
```
Please specify `{cu_version}` and `{torch_version}` to your required version. 

`E.g. 1` To install mmcv on PyTorch 1.10.0 with CUDA 10.2:
```
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu102/torch1.10.0/index.html
```

`E.g. 2` To install mmcv on PyTorch 1.10.0 with CUDA 11.1:
```
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.10.0/index.html
```
More information can be found at [here](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)

(6) Build [MMDetection3D](https://github.com/open-mmlab/mmdetection3d)
* Install dependencies:
    ```
    pip install mmdet==2.22.0
    pip install mmsegmentation==0.22.0
    ``` 
* Build MMDetection3D.
    ```
    cd /path/to/DeepVision3D/MMDetection3D
    python setup.py develop
    ```
  > **_NOTE:_**  If you are using RTX3090 GPUs, please add `TORCH_CUDA_ARCH_LIST=8.0+PTX` like:
  >  ```
  >  cd /path/to/DeepVision3D/MMDetection3D
  >  TORCH_CUDA_ARCH_LIST=8.0+PTX python setup.py develop
  >  ```

(7) Build [SparseConv](https://github.com/traveller59/spconv) library and [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) (We use `spconv v1.2.1`).
* Clone spconv v1.2.1:
    ```
    git clone -b v1.2.1 https://github.com/traveller59/spconv --recursive
    ```
* Install cmake & libboost:
    ```
    conda install -c statiskit libboost-dev && conda install -c anaconda cmake
    ```
* Build spconv library:
    ```
    cd spconv && python setup.py bdist_wheel
    ```
  > **_NOTE:_**  If you are using RTX3090 GPUs, please add `TORCH_CUDA_ARCH_LIST=8.0+PTX`:
  >  ```
  >  cd spconv && TORCH_CUDA_ARCH_LIST=8.0+PTX python setup.py bdist_wheel
  >  ```

* Install spconv:
  ```
  cd ./dist && pip install spconv-1.2.1-cp37-cp37m-linux_x86_64.whl
  ```
* Install OpenPCDet:
  ```
  cd /path/to/DeepVision3D/OpenPCDet && python setup.py develop
  ```
  > **_NOTE:_**  If you are using RTX3090 GPUs, please add `TORCH_CUDA_ARCH_LIST=8.0+PTX`:
  >  ```
  >  cd /path/to/DeepVision3D/OpenPCDet && TORCH_CUDA_ARCH_LIST=8.0+PTX python setup.py develop
  >  ```

(8) Build EQNet.
```
cd /path/to/DeepVision3D/EQNet && python setup.py develop
```
  > **_NOTE:_**  If you are using RTX3090 GPUs, please add `TORCH_CUDA_ARCH_LIST=8.0+PTX`:
  >  ```
  >  cd /path/to/DeepVision3D/EQNet && TORCH_CUDA_ARCH_LIST=8.0+PTX python setup.py develop
  >  ```

(9) Build DVSegmentation.
* Install Google Hashmap:
  ```
  conda install -c bioconda google-sparsehash
  ```
* Build DVSegmentation (Please specify the `include_dirs` in [setup.py](../DVSegmentation/ops/setup.py) to your anaconda include path `i.e. /path/to/anaconda3/envs/deepvision3d/include/`):
  ```
  cd /path/to/DeepVision3D/DVSegmentation/ops && python setup.py install
  ```
    > **_NOTE:_**  If you are using RTX3090 GPUs, please add `TORCH_CUDA_ARCH_LIST=8.0+PTX`:
    >  ```
    >  cd /path/to/DeepVision3D/DVSegmentation/ops && TORCH_CUDA_ARCH_LIST=8.0+PTX python setup.py install
    >  ```
