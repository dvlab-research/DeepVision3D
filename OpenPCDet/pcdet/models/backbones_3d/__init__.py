from .pointnet2_backbone import PointNet2MSG
from .spconv_backbone import VoxelBackBone8x, VoxelResBackBone8x
from .spconv_unet import UNetV2

from eqnet.models import support_producer

__all__ = {
    'VoxelBackBone8x': VoxelBackBone8x,
    'UNetV2': UNetV2,
    'PointNet2MSG': PointNet2MSG,
    'VoxelResBackBone8x': VoxelResBackBone8x,
}

__all__.update(support_producer.__pcdet_all__)
