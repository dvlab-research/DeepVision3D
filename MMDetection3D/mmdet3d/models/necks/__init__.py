# Copyright (c) OpenMMLab. All rights reserved.
import sys

from mmdet.models.necks.fpn import FPN
from .dla_neck import DLANeck
from .imvoxel_neck import OutdoorImVoxelNeck
from .pointnet2_fp_neck import PointNetFPNeck
from .second_fpn import SECONDFPN

from eqnet.models.query_producer import __mmdet3d_all__

__all__ = [
    'FPN', 'SECONDFPN', 'OutdoorImVoxelNeck', 'PointNetFPNeck', 'DLANeck'
]

for k, v in __mmdet3d_all__.items():
    setattr(sys.modules[__name__], k, v)
    __all__.append(k)