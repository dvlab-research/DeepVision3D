"""
A module for producing auxiliary supervision for intermediate results from qnet.
"""
import torch
from mmcv.cnn import ConvModule
from torch import nn as nn

from mmdet3d.models.builder import build_loss
from mmdet.core import multi_apply


class AuxModule(nn.Module):
    """Aux Module. (An adaption version of vote module.)

    For all intermediate output from qnet:
        * Generate the offset prediction.
        * Generate the classfication.
        * Compute loss for them.

    Args:
        in_channels (int): Number of channels of intermediate features.
        gt_per_seed (int): Number of ground truth votes generated
            from each query point.
        conv_channels (tuple[int]): Out channels of vote
            generating convolution.
        conv_cfg (dict): Config of convolution.
            Default: dict(type='Conv1d').
        norm_cfg (dict): Config of normalization.
            Default: dict(type='BN1d').
        offset_loss (dict): Config of offset auxiliary supervision.
        class_loss (dict): Config of class auxiliary supervision.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 gt_per_seed=3,
                 conv_channels=(16, 16),
                 conv_cfg=dict(type='Conv1d'),
                 norm_cfg=dict(type='BN1d'),
                 act_cfg=dict(type='ReLU'),
                 offset_loss=None,
                 class_loss=None):
        super().__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.gt_per_seed = gt_per_seed

        if offset_loss is not None:
            self.offset_loss = build_loss(offset_loss)
        if class_loss is not None:
            self.class_loss = build_loss(class_loss)

        # 1. Define offset generation network.
        prev_channels = in_channels
        offset_conv_list = list()
        for k in range(len(conv_channels)):
            offset_conv_list.append(
                ConvModule(
                    prev_channels,
                    conv_channels[k],
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    bias=True,
                    inplace=True))
            prev_channels = conv_channels[k]
        self.offset_conv = nn.Sequential(*offset_conv_list)
        self.offset_head = nn.Conv1d(prev_channels, 3, 1)

        # define two head.
        prev_channels = in_channels
        class_conv_list = list()
        for k in range(len(conv_channels)):
            class_conv_list.append(
                ConvModule(
                    prev_channels,
                    conv_channels[k],
                    1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg,
                    bias=True,
                    inplace=True))
            prev_channels = conv_channels[k]
        self.class_conv = nn.Sequential(*class_conv_list)
        self.class_head = nn.Conv1d(prev_channels, 1, 1)

    # assign targets. (Adapt from vote_head.get_targets)
    def get_targets(self,
                    points,
                    with_rot,
                    gt_bboxes_3d,
                    gt_labels_3d,
                    pts_semantic_mask=None,
                    pts_instance_mask=None,):
        """Generate targets of aux head..

        Args:
            points (list[torch.Tensor]): Points of each batch.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth \
                bboxes of each batch.
            gt_labels_3d (list[torch.Tensor]): Labels of each batch.
            pts_semantic_mask (None | list[torch.Tensor]): Point-wise semantic
                label of each batch.
            pts_instance_mask (None | list[torch.Tensor]): Point-wise instance
                label of each batch.

        Returns:
            tuple[torch.Tensor]: Targets of vote head.
        """
        # find empty example
        valid_gt_masks = list()
        gt_num = list()
        for index in range(len(gt_labels_3d)):
            if len(gt_labels_3d[index]) == 0:
                fake_box = gt_bboxes_3d[index].tensor.new_zeros(
                    1, gt_bboxes_3d[index].tensor.shape[-1])
                gt_bboxes_3d[index] = gt_bboxes_3d[index].new_box(fake_box)
                gt_labels_3d[index] = gt_labels_3d[index].new_zeros(1)
                valid_gt_masks.append(gt_labels_3d[index].new_zeros(1))
                gt_num.append(1)
            else:
                valid_gt_masks.append(gt_labels_3d[index].new_ones(
                    gt_labels_3d[index].shape))
                gt_num.append(gt_labels_3d[index].shape[0])

        if pts_semantic_mask is None:
            pts_semantic_mask = [None for i in range(len(gt_labels_3d))]
            pts_instance_mask = [None for i in range(len(gt_labels_3d))]
        with_rot = [with_rot for i in range(len(gt_labels_3d))]

        vote_targets, vote_target_masks = multi_apply(self.get_targets_single, points, with_rot,
                                                      gt_bboxes_3d, gt_labels_3d,
                                                      pts_semantic_mask, pts_instance_mask)

        vote_targets = torch.stack(vote_targets)
        vote_target_masks = torch.stack(vote_target_masks)

        return vote_targets, vote_target_masks

    def get_targets_single(self,
                           points,
                           with_rot,
                           gt_bboxes_3d,
                           gt_labels_3d,
                           pts_semantic_mask=None,
                           pts_instance_mask=None,):
        """Generate targets of aux module for single batch.

        Args:
            points (torch.Tensor): Points of each batch.
            with_rot: bbox_coder.with_rot
            gt_bboxes_3d (:obj:`BaseInstance3DBoxes`): Ground truth \
                boxes of each batch.
            gt_labels_3d (torch.Tensor): Labels of each batch.
            pts_semantic_mask (None | torch.Tensor): Point-wise semantic
                label of each batch.
            pts_instance_mask (None | torch.Tensor): Point-wise instance
                label of each batch.

        Returns:
            tuple[torch.Tensor]: Targets of vote head.
        """
        assert with_rot or pts_semantic_mask is not None

        gt_bboxes_3d = gt_bboxes_3d.to(points.device)

        # generate votes target
        num_points = points.shape[0]
        if with_rot:
            vote_targets = points.new_zeros([num_points, 3 * self.gt_per_seed])
            vote_target_masks = points.new_zeros([num_points],
                                                 dtype=torch.long)
            vote_target_idx = points.new_zeros([num_points], dtype=torch.long)
            box_indices_all = gt_bboxes_3d.points_in_boxes_all(points[:, :3].contiguous())
            for i in range(gt_labels_3d.shape[0]):
                box_indices = box_indices_all[:, i]
                indices = torch.nonzero(
                    box_indices, as_tuple=False).squeeze(-1)
                selected_points = points[indices]
                vote_target_masks[indices] = 1
                vote_targets_tmp = vote_targets[indices]
                votes = gt_bboxes_3d.gravity_center[i].unsqueeze(
                    0) - selected_points[:, :3]

                for j in range(self.gt_per_seed):
                    column_indices = torch.nonzero(
                        vote_target_idx[indices] == j,
                        as_tuple=False).squeeze(-1)
                    vote_targets_tmp[column_indices,
                                     int(j * 3):int(j * 3 +
                                                    3)] = votes[column_indices]
                    if j == 0:
                        vote_targets_tmp[column_indices] = votes[
                            column_indices].repeat(1, self.gt_per_seed)

                vote_targets[indices] = vote_targets_tmp
                vote_target_idx[indices] = torch.clamp(
                    vote_target_idx[indices] + 1, max=2)
        elif pts_semantic_mask is not None:
            vote_targets = points.new_zeros([num_points, 3])
            vote_target_masks = points.new_zeros([num_points],
                                                 dtype=torch.long)

            for i in torch.unique(pts_instance_mask):
                indices = torch.nonzero(
                    pts_instance_mask == i, as_tuple=False).squeeze(-1)
                if pts_semantic_mask[indices[0]] < self.num_classes:
                    selected_points = points[indices, :3]
                    center = 0.5 * (
                            selected_points.min(0)[0] + selected_points.max(0)[0])
                    vote_targets[indices, :] = center - selected_points
                    vote_target_masks[indices] = 1
            vote_targets = vote_targets.repeat((1, self.gt_per_seed))
        else:
            raise NotImplementedError

        return vote_targets, vote_target_masks

    def forward(self, query_pos, query_features):
        """forward.

        Args:
            query_pos (torch.Tensor): Coordinate of the seed
                query_position in shape (B, N, 3).
            query_features (torch.Tensor): Features of the seed points in shape
                (H, B, N, C).

        Returns:
            tuple[torch.Tensor]:
                - vote_x: Voted xyz based on the query positions with shape
                    (B, H, N, 3)
                - class_x: fg/bg classification prediction for each query position
                    with shape (B, H, N, 1)
        """
        h, bs, q_num, c_num = query_features.shape

        # bs, h, q_num, c_num.
        query_features = query_features.permute(1, 0, 2, 3).contiguous()

        # generate vote_x.
        offset_x = self.offset_conv(query_features.view(-1, c_num, 1))
        offset_x = self.offset_head(offset_x)
        offset_x = offset_x.view(bs, h, q_num, 3)
        vote_x = (query_pos.view(bs, 1, q_num, 3) + offset_x).contiguous()

        # generate class_x.
        class_x = self.class_conv(query_features.view(-1, c_num, 1))
        class_x = self.class_head(class_x)
        class_x = class_x.view(bs, h, q_num, 1)
        return vote_x, class_x

    def get_loss(self, query_pos,
                 vote_x, class_x,
                 query_indices,
                 vote_targets_mask, vote_targets):
        """Calculate loss for aux_module. (Similar as get_loss in vote_module.)

        Args:
            query_pos (torch.Tensor): Coordinates of query positions. (bs, q_num, 3)
            vote_x (torch.Tensor): Coordinate of the vote points. (bs, h, q_num, 3)
            query_indices (torch.Tensor): Indices of query positions in raw points. (bs, q_num)
            vote_targets_mask (torch.Tensor): Mask of valid vote targets.
                (bs, npoint), where npoint equals to the number of raw points.
            vote_targets (torch.Tensor): Targets of votes.
                (bs, npoint, 3), where npoint equals to the number of raw points.

        Returns:
            torch.Tensor: Weighted vote loss.
        """
        bs, h, q_num = vote_x.shape[:3]

        # get query_gt_vote_mask.
        # bs, q_num.
        query_gt_vote_mask = torch.gather(vote_targets_mask, 1, query_indices).float()
        # bs, h, q_num.
        query_gt_vote_mask = query_gt_vote_mask.unsqueeze(1).repeat(1, h, 1).contiguous()

        # get query_gt_vote.
        query_indices_expand = query_indices.unsqueeze(-1).repeat(
            1, 1, 3 * self.gt_per_seed)  # bs, q_num, 3 * self.gt_per_seed
        query_gt_vote = torch.gather(vote_targets, 1, query_indices_expand)
        query_gt_vote += query_pos.repeat(1, 1, self.gt_per_seed)
        # bs, h, q_num, 3 * self.gt_per_seed
        query_gt_vote = query_gt_vote.unsqueeze(1).repeat(1, h, 1, 1).contiguous()

        # get query_vote_loss.
        weight = query_gt_vote_mask / (torch.sum(query_gt_vote_mask) + 1e-6)
        distance = self.offset_loss(
            vote_x.view(bs * h * q_num, 1, 3),
            query_gt_vote.view(bs * h * q_num, -1, 3),
            dst_weight=weight.view(bs * h * q_num, 1))[1]
        offset_loss = torch.sum(torch.min(distance, dim=1)[0])

        # get query_class_loss.
        class_loss = self.class_loss(
            class_x.view(-1),
            query_gt_vote_mask.float().view(-1)
        )
        return offset_loss, class_loss
