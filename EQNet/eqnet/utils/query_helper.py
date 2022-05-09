"""
A helper function for generating query positions.
"""
import torch
import numpy as np

from eqnet.utils import utils
from pcdet.utils import common_utils


def obtain_bev_query_position(t, point_cloud_range, voxel_size):
    """ According to the point cloud range and target voxel size to generate
        query positions for BEV map.

    :param t: a tensor for providing target device.
    :param point_cloud_range: (xmin, ymin, zmin, xmax, ymax, zmax)
    :param voxel_size: (x_s, y_s, z_s)
    :return:
        query_positions: (N, 3)
    """
    x_range, y_range = (point_cloud_range[3] - point_cloud_range[0]), (point_cloud_range[4] - point_cloud_range[1])
    x_size, y_size = voxel_size[0], voxel_size[1]

    dx, dy = int(np.round(x_range / x_size)), int(np.round(y_range / y_size))
    assert dx * x_size == x_range and dy * y_size == y_range, 'The point cloud range must be divisible by voxel size.'

    faked_features = t.new_ones((dx, dy))
    dense_idx = faked_features.nonzero()  # (dx * dy, 2) [x_idx, y_idx]
    voxel_size = torch.from_numpy(np.array(voxel_size)).float().to(t.device)
    base_xy = (dense_idx + 0.5) * voxel_size[:2]

    point_cloud_range = torch.tensor(point_cloud_range, device=t.device).float()
    bev_xy = base_xy + point_cloud_range[:2]
    bev_z = bev_xy.new_zeros(bev_xy.shape[0], 1)
    bev_xyz = torch.cat([bev_xy, bev_z], dim=-1)  # -1, 3
    return bev_xyz


def obtain_bev_query_indices(bev_xyz, voxel_size, point_cloud_range):
    """ Cast the generated bev_xyz to bev_indices, for utilizing spconv.api to cast them
        into dense representation.

    :param bev_xyz: A float tensor with shape [batch_size, pts_num, 3]
    :param voxel_size: A float tensor with shape [3]: {x, y, z}
    :param point_cloud_range: {x_min, y_min, z_min, x_max, y_max, z_max}
    :return:
        bev_xyz: A float tensor with shape [batch_size, pts_num, 3]
        bev_indices: A float tensor with shape [batch_size * pts_num, 4],
            with bev_indices, we can easily use spconv.api to generate dense representation.
    """
    batch_size, pts_num = bev_xyz.shape[:2]
    bev_indices = utils.cast_centers_to_indices(bev_xyz, voxel_size, point_cloud_range)

    bev_xyz = utils.cast_indices_to_centers(bev_indices, voxel_size, point_cloud_range)
    bev_xyz = bev_xyz.view(batch_size, pts_num, 3)
    return bev_xyz, bev_indices


def obtain_proposal_grid_query_position(rois, roi_grid_size):
    """ According to the rois: (N, 7){x, y, z, l, w, h, ry} to
    generate their grids as query positions.
    Modified from: https://github.com/open-mmlab/OpenPCDet/blob/master/pcdet/models/roi_heads/voxelrcnn_head.py

    :param rois: A float tensor with [N, 7] {x, y, z, l, w, h, ry}
        indicating proposals in 3D world.
    :param roi_grid_size:
    :return:
        global_roi_grid_points: [N, roi_grid_size ** 3, 3]
    """
    rois = rois.view(-1, rois.shape[-1])
    batch_size_rcnn = rois.shape[0]

    # get dense grid points.
    faked_features = rois.new_ones((roi_grid_size, roi_grid_size, roi_grid_size))
    dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
    dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (N, roi_grid_size ** 3, 3)

    # get relative coordinates to local center
    local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
    local_roi_grid_points = ((dense_idx + 0.5) / roi_grid_size * local_roi_size.unsqueeze(dim=1)
                                  - (local_roi_size.unsqueeze(dim=1) / 2))  # (B, 6x6x6, 3)

    global_roi_grid_points = common_utils.rotate_points_along_z(
        local_roi_grid_points.clone(), rois[:, 6]
    ).squeeze(dim=1)
    global_center = rois[:, 0:3].clone()
    global_roi_grid_points += global_center.unsqueeze(dim=1)
    return global_roi_grid_points
