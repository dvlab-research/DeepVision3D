def reorganize_input_for_transformer(batch_size, point_indices, point_features,
                                     max_valid_num=40000):
    """ Cast support features
         from stack representation: [N1 + N2 + ..., C] (features) & [N1 + N2 + ..., 4] (coords)
         to batch representation: [B, M, C] & [B, M, 3] for using {attention_mapper} function.

    :param batch_size
    :param point_indices: A integer tensor with shape [N1 + N2 + ..., 4],
        indicating {batch_index, x_index, y_index, z_index}
    :param point_features: A float tensor with shape [N1 + N2 + ..., C]
        indicating features obtained by spconv.
    :param max_valid_num
    :return:
        batch_features: A tensor with shape [B, M, C]
        batch_coords: A tensor with shape [B, M, 3]
        batch_masks: A tensor with shape [B, M], indicating whether a point is a padding point:
            0 (False): valid / 1 (True): invalid.
    """
    features, coords = [], []
    max_points_num = 0
    for k in range(batch_size):
        cur_mask = (point_indices[:, 0] == k)
        max_points_num = max(cur_mask.sum().item(), max_points_num)
        features.append(point_features[cur_mask])
        coords.append(point_indices[cur_mask, 1:4])
    max_points_num = min(max_points_num, max_valid_num)

    batch_features = point_features.new_zeros((batch_size, max_points_num, point_features.shape[-1]))  # (B, M, C)
    batch_coords = point_indices.new_zeros((batch_size, max_points_num, 3))  # (B, M, 3)
    batch_masks = point_indices.new_ones((batch_size, max_points_num)).bool()  # (B, M)
    for k in range(batch_size):
        cur_points_num = min(features[k].shape[0], max_points_num)
        batch_features[k, :cur_points_num] = features[k][:cur_points_num]
        batch_coords[k, :cur_points_num] = coords[k][:cur_points_num]
        batch_masks[k, :cur_points_num] = 0  # 0: valid point, 1: ignored padding points.
    return batch_features, batch_coords, batch_masks