"""
Embedding stage network for EQ-Paradigm or ED-Paradigm models (also known as backbone)

Current available models:
PointNet++ (Adapted from OpenPCDet code base):

DV-Lab Research Backbones:
Focal Sparse Conv.
Stratified transformer.
"""

from .pointnet2_backbone import PCDetPointNet2Backbone, MMDet3DPointNet2Backbone, DVClsPointNet2Backbone
from .spconv_backbone import PCDetVoxelBackbone, MMDet3DVoxelBackbone, DVSegVoxelBackbone

# pcdet.
__pcdet_all__ = {
    'EQPointNet2Backbone': PCDetPointNet2Backbone,
    'EQVoxelBackbone': PCDetVoxelBackbone,
}

# mmdet3d.
__mmdet3d_all__ = {
    'MMDet3DPointNet2Backbone': MMDet3DPointNet2Backbone,
    'MMDet3DVoxelBackbone': MMDet3DVoxelBackbone,
}

# dv_cls.
__dvcls__all__ = {
    'EQPointNet2Backbone': DVClsPointNet2Backbone,
}

# dv_seg.
__dvseg__all__ = {
    'EQVoxelBackbone': DVSegVoxelBackbone,
}