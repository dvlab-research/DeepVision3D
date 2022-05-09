# Copyright (c) OpenMMLab. All rights reserved.
import sys

from mmdet.models.backbones import SSDVGG, HRNet, ResNet, ResNetV1d, ResNeXt
from .dgcnn import DGCNNBackbone
from .dla import DLANet
from .multi_backbone import MultiBackbone
from .nostem_regnet import NoStemRegNet
from .pointnet2_sa_msg import PointNet2SAMSG
from .pointnet2_sa_ssg import PointNet2SASSG
from .second import SECOND

from eqnet.models.support_producer import __mmdet3d_all__

__all__ = [
    'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet', 'NoStemRegNet',
    'SECOND', 'DGCNNBackbone', 'PointNet2SASSG', 'PointNet2SAMSG',
    'MultiBackbone', 'DLANet'
]

for k, v in __mmdet3d_all__.items():
    setattr(sys.modules[__name__], k, v)
    __all__.append(k)