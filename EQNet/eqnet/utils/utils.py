import torch


def obtain_spatial_shape(voxel_size, point_cloud_range):
    """

    :param voxel_size: A float tensor with shape [3]: {x, y, z}
    :param point_cloud_range: {x_min, y_min, z_min, x_max, y_max, z_max}
    :return:
    """
    spatial_shape = (point_cloud_range[3:6] - point_cloud_range[0:3]) / voxel_size
    return spatial_shape[[2, 1, 0]]


def cast_centers_to_indices(voxel_centers, voxel_size, point_cloud_range):
    """ A helper function for transferring voxel center coordinates to voxel indices.

    :param voxel_centers: A float tensor with shape [batch_size, pts_num, 3]
    :param voxel_size: A float tensor with shape [3]: {x, y, z}
    :param point_cloud_range: {x_min, y_min, z_min, x_max, y_max, z_max}
    """
    batch_size, pts_num = voxel_centers.shape[:2]
    spatial_shape = obtain_spatial_shape(voxel_size, point_cloud_range)

    point_cloud_range = torch.tensor(point_cloud_range, device=voxel_centers.device).float()
    voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float()
    spatial_shape = torch.tensor(spatial_shape, device=voxel_centers.device).int()

    voxel_indices = (voxel_centers - point_cloud_range[:3]) // voxel_size
    voxel_indices = voxel_indices[..., [2, 1, 0]].int()
    voxel_indices = torch.minimum(
        torch.maximum(voxel_indices, torch.zeros_like(voxel_indices)), spatial_shape - 1)

    batch_idx = torch.arange(batch_size, device=voxel_indices.device).view(
        -1, 1).repeat(1, pts_num).view(-1, 1).int()
    voxel_indices = torch.cat([batch_idx, voxel_indices.view(-1, 3)], dim=-1)
    return voxel_indices


def cast_indices_to_centers(voxel_indices, voxel_size, point_cloud_range):
    """ An inverse function of cast_centers_to_indices.

    :param voxel_indices: A float tensor with shape [-1, 4]: {batch_index, x, y, z}
    :param voxel_size: A float tensor with shape [3]: {x, y, z}
    :param point_cloud_range: {x_min, y_min, z_min, x_max, y_max, z_max}
    """
    voxel_size = torch.tensor(voxel_size, device=voxel_indices.device).float()
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_indices.device).float()

    assert voxel_indices.shape[1] == 4
    voxel_centers = voxel_indices[:, [3, 2, 1]].contiguous().float()  # (xyz)
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers