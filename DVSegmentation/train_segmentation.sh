#!/usr/bin/env bash

NGPUS=$1

python -m torch.distributed.launch --nproc_per_node=${NGPUS} train_segmentation.py --launcher pytorch ${@:2}

# CUDA_VISIBLE_DEVICES=0,1 ./train_segmentation.sh 2 --config config/eqnet_scannet.yaml