import torch
import numpy as np

from ...utils import box_utils
from ...utils import loss_utils
from .point_head_simple import PointHeadSimple
from ...ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu


class PointHeadAuxLoss(PointHeadSimple):
    """
    Auxiliary loss for each Q-Decoder layer.
    """
    def __init__(self, num_class, input_channels, model_cfg, **kwargs):
        input_channels = model_cfg.get('INPUT_CHANNELS', input_channels)
        super().__init__(model_cfg=model_cfg, num_class=num_class, input_channels=input_channels)
        self.aux_loss_weight = self.model_cfg.get(
            'AUX_LOSS_WEIGHT', None)

        # input_keys
        self.point_coords_key = self.model_cfg.POINT_COORDS_KEY
        self.point_feature_key = self.model_cfg.POINT_FEATURE_KEY
        # assign target keys.
        target_assignment_cfg = self.model_cfg.TARGET_ASSIGNMENT_CFG
        self.gt_extra_width = target_assignment_cfg.get('GT_EXTRA_WIDTH')
        # subsample aux-supp queries.
        aux_supervision_cfg = self.model_cfg.AUX_SUPERVISION_CFG
        self.aux_supp_query_num = aux_supervision_cfg.QUERY_NUM
        self.aux_supp_pos_ratio = aux_supervision_cfg.POS_RATIO
        self.aux_supp_hardneg_ratio = aux_supervision_cfg.HARDNEG_RATIO
        self.aux_supp_hardneg_width = aux_supervision_cfg.HARDNEG_WIDTH

        self.reg_layers = self.make_fc_layers(
            fc_cfg=self.model_cfg.REG_FC,
            input_channels=input_channels,
            output_channels=3
        )  # produce auxiliary offset supervision on each point feature.

    # 1. assign target for query position.
    # 2. random sub-sampling.
    def assign_targets(self, input_dict):
        """ Assign auxiliary groundtruth label for query position.
        :param:
            input_dict:
                batch_size:
                gt_boxes: (B, M, 8)
                self.point_coords_key: (B, N, 3)
        """
        batch_size = input_dict['batch_size']
        gt_boxes = input_dict['gt_boxes'][..., :7].contiguous().clone()  # bs, num_gt, 7
        query_positions = input_dict[self.point_coords_key]  # bs, num_point, 3

        tgt_gt_cls = gt_boxes.new_zeros(query_positions.shape[0], query_positions.shape[1]).long()  # bs, num_point
        tgt_gt_offset = torch.zeros_like(query_positions)  # bs, num_point, 3

        for i in range(batch_size):
            cur_gt = gt_boxes[i]  # -1, 7
            # Filter padding gt_box.
            k = cur_gt.__len__() - 1
            while k >= 0 and (cur_gt[k, :7] == 0).all():  # a padding box
                k -= 1
            if k < 0:
                continue  # no valid gt_box.
            cur_gt = cur_gt[:k + 1]
            cur_gt = box_utils.enlarge_box3d(cur_gt, extra_width=self.gt_extra_width)

            # gt_cls: num_query_points
            cur_assignment = points_in_boxes_gpu(query_positions[i:i+1].contiguous(),
                                                 cur_gt.unsqueeze(0).contiguous()).squeeze(0)
            fg_mask = (cur_assignment >= 0).long()
            tgt_gt_cls[i] = fg_mask

            # gt_offset: num_query_points, 3
            fg_ctr = cur_gt[cur_assignment.long()][..., :3].contiguous()
            tgt_gt_offset[i] = fg_ctr - query_positions[i]

        targets_dict = dict()
        targets_dict['tgt_gt_cls'] = tgt_gt_cls
        targets_dict['tgt_gt_offset'] = tgt_gt_offset
        return targets_dict

    def _random_select_aux_supervised_query(self, input_dict):
        """Randomly select fg points, hard bg points and easy bg points for aux supervision.
        Procedure:
            1. Find all positive interior points. (Positive pairs)
            2. Randomly select hard negative and easy negative points according to a ratio.

        :return selected_mask: A bool tensor with shape [B, N] indicating whether a point
                                is a valid point after subsampling.
        """
        batch_size = input_dict['batch_size']
        gt_boxes = input_dict['gt_boxes'][..., :7].contiguous().clone()  # bs, num_gt, 7
        query_positions = input_dict[self.point_coords_key]  # bs, num_point, 3
        pts_num = query_positions.shape[1]

        total_num = self.aux_supp_query_num
        pos_ratio = self.aux_supp_pos_ratio
        pos_num = int(total_num * pos_ratio)
        hardneg_ratio = self.aux_supp_hardneg_ratio
        hardneg_num = int(total_num * hardneg_ratio)
        hardneg_width = np.array(self.aux_supp_hardneg_width)
        hardneg_width = torch.from_numpy(hardneg_width).float().to(gt_boxes.device)

        gt_extra_width = self.gt_extra_width

        if total_num == -1:
            # no subsampling.
            batch_mask = gt_boxes.new_ones(batch_size, pts_num)
        else:
            batch_mask = gt_boxes.new_zeros(batch_size, pts_num)
            for i in range(batch_size):
                cur_gt = gt_boxes[i]  # -1, 7
                # Filter padding gt_box.
                k = cur_gt.__len__() - 1
                while k >= 0 and (cur_gt[k, :7] == 0).all():  # a padding box
                    k -= 1
                if k < 0:  # no valid gt_box.
                    # randomly select total_num query for aux training.
                    rand_num = torch.from_numpy(np.random.permutation(pts_num)).type_as(batch_mask).long()
                    batch_mask[i][rand_num[:total_num]] = 1
                else:
                    # has valid box.
                    cur_gt = cur_gt[:k + 1]
                    cur_gt = box_utils.enlarge_box3d(cur_gt, gt_extra_width)

                    expand_gt = cur_gt.clone()
                    expand_gt[..., 3:6] = hardneg_width
                    expand_gt[..., -1] = 0.  # Set the orientation to 0.

                    # randomly select fg.
                    strict_fg_assignment = points_in_boxes_gpu(
                        query_positions[i:i+1].contiguous(), cur_gt.unsqueeze(0).contiguous()
                    ).squeeze(0)
                    box_fg_flag = (strict_fg_assignment >= 0)
                    box_fg_idx = torch.where(box_fg_flag)[0]
                    real_fg_num = box_fg_flag.int().sum().item()
                    rand_num = torch.from_numpy(np.random.permutation(real_fg_num)).type_as(batch_mask).long()
                    fg_selection_num = min(real_fg_num, pos_num)
                    box_fg_idx = box_fg_idx[rand_num[:fg_selection_num]]
                    batch_mask[i, box_fg_idx] = 1

                    # randomly select hard bg.
                    hard_bg_assignment = points_in_boxes_gpu(
                        query_positions[i:i+1].contiguous(), expand_gt.unsqueeze(0).contiguous()
                    ).squeeze(0)
                    # hard_negative: not in bounding box and in extended bounding box.
                    hard_negative_flag = (~box_fg_flag) & (hard_bg_assignment >= 0)
                    hard_negative_idx = torch.where(hard_negative_flag)[0]
                    real_headneg_num = hard_negative_flag.int().sum().item()
                    rand_num = torch.from_numpy(np.random.permutation(real_headneg_num)).type_as(batch_mask).long()
                    hard_neg_selection_num = min(real_headneg_num, hardneg_num)
                    hard_negative_idx = hard_negative_idx[rand_num[:hard_neg_selection_num]]
                    batch_mask[i, hard_negative_idx] = 1

                    # Compute easy bg_num.
                    fg_hard_selection_num = batch_mask[i].int().sum().item()
                    easy_neg_selection_num = total_num - fg_hard_selection_num
                    # Get easy negative selection.
                    positive_flag = box_fg_flag | hard_negative_flag
                    easy_negative_flag = ~positive_flag
                    easy_negative_idx = torch.where(easy_negative_flag)[0]
                    # random selection.
                    easy_neg_num = easy_negative_flag.int().sum().item()
                    rand_num = torch.from_numpy(np.random.permutation(easy_neg_num)).type_as(batch_mask).long()
                    easy_neg_selection_num = min(easy_neg_selection_num, easy_neg_num)
                    easy_negative_idx = easy_negative_idx[rand_num[:easy_neg_selection_num]]
                    batch_mask[i, easy_negative_idx] = 1

        return batch_mask

    def _select_content_by_mask(self, t, mask):
        # t: A float tensor with shape [b, n, c]
        # mask: A int tensor with shape [b, n]
        collect_t = []
        b = t.shape[0]
        for i in range(b):
            cur_t = t[i]
            cur_mask = mask[i]
            collect_t.append(cur_t[cur_mask > 0])
        return torch.stack(collect_t, dim=0)

    def get_cls_layer_loss(self, tb_dict=None, index=0):
        tgt_gt_cls = self.forward_ret_dict['tgt_gt_cls'].view(-1)
        aux_cls_preds = self.forward_ret_dict['aux_cls_preds'][index].view(-1, self.num_class)

        positives = (tgt_gt_cls > 0).float()
        pos_normalizer = positives.sum(dim=0).float()
        cls_weights = torch.ones_like(positives)
        cls_weights = cls_weights / torch.clamp(pos_normalizer, min=1.0)

        one_hot_targets = tgt_gt_cls.new_zeros(*list(tgt_gt_cls.shape), self.num_class + 1)
        one_hot_targets.scatter_(-1, (tgt_gt_cls * (tgt_gt_cls >= 0).long()).unsqueeze(dim=-1).long(), 1.0)
        one_hot_targets = one_hot_targets[..., 1:]
        cls_loss_src = self.cls_loss_func(aux_cls_preds, one_hot_targets, weights=cls_weights)

        point_loss_cls = cls_loss_src.sum()

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        point_loss_cls = point_loss_cls * loss_weights_dict['aux_cls_weight']
        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'aux_loss_cls': point_loss_cls.item(),
            'aux_point_pos_num': pos_normalizer.item()
        })
        return point_loss_cls, tb_dict

    def get_reg_layer_loss(self, tb_dict=None, index=0):
        tgt_gt_offset = self.forward_ret_dict['tgt_gt_offset'].view(-1, 3)
        aux_reg_preds = self.forward_ret_dict['aux_reg_preds'][index].view(-1, 3)
        tgt_gt_cls = self.forward_ret_dict['tgt_gt_cls'].view(-1)

        positives = (tgt_gt_cls > 0).float()
        pos_normalizer = positives.sum(dim=0).float()

        aug_reg_loss = loss_utils.WeightedSmoothL1Loss.smooth_l1_loss(
            aux_reg_preds - tgt_gt_offset, beta=1.0 / 9.0).sum(-1) * positives

        loss_weights_dict = self.model_cfg.LOSS_CONFIG.LOSS_WEIGHTS
        aug_reg_loss = (aug_reg_loss.sum() / torch.clamp(pos_normalizer, min=1.0) *
                        loss_weights_dict['aux_reg_weight'])

        if tb_dict is None:
            tb_dict = {}
        tb_dict.update({
            'aux_loss_reg': aug_reg_loss.item(),
        })
        return aug_reg_loss, tb_dict

    def get_loss(self, tb_dict=None):
        tb_dict = {} if tb_dict is None else tb_dict

        aux_cls_preds = self.forward_ret_dict['aux_cls_preds']
        aux_head_num = len(aux_cls_preds)

        total_loss = 0.

        if self.aux_loss_weight is not None:
            assert len(self.aux_loss_weight) == aux_head_num, 'The length of aux_loss_weight should be consistent ' \
                                                              'with aux_head_num, which is %d' % aux_head_num
        for i in range(aux_head_num):
            if self.aux_loss_weight is not None:
                cur_loss_weight = self.aux_loss_weight[i]
            else:
                cur_loss_weight = 1.0

            point_loss_cls, tb_dict = self.get_cls_layer_loss(tb_dict, i)
            point_loss_cls = point_loss_cls * cur_loss_weight

            point_loss_reg, tb_dict = self.get_reg_layer_loss(tb_dict, i)
            point_loss_reg = point_loss_reg * cur_loss_weight

            total_loss = total_loss + point_loss_reg + point_loss_cls
        return total_loss, tb_dict

    def forward(self, batch_dict):
        """
        Args:
            input_dict:
                batch_size:
                gt_boxes: (B, M, 8)
                self.point_coords_key: (B, N, 3)
                self.point_feature_key: (H, B, N, C)
        """
        if self.training:
            # 1. subsample points.
            chosen_mask = self._random_select_aux_supervised_query(batch_dict)
            # h, b, n, c -> b, n, c, h -> mask chosen -> h, b, n, c.
            batch_dict[self.point_feature_key] = self._select_content_by_mask(
                batch_dict[self.point_feature_key].permute(1, 2, 3, 0).contiguous(),
                chosen_mask).permute(3, 0, 1, 2).contiguous()
            batch_dict[self.point_coords_key] = self._select_content_by_mask(
                batch_dict[self.point_coords_key], chosen_mask)

            point_features = batch_dict[self.point_feature_key]  # h, b, n, c
            nh, nb, np, nc = point_features.size()

            aux_cls_preds = self.cls_layers(point_features.view(-1, nc))  # -1, num_class
            aux_cls_preds = torch.split(aux_cls_preds.view(nh, nb * np, aux_cls_preds.shape[-1]), 1, dim=0)

            aux_reg_preds = self.reg_layers(point_features.view(-1, nc))
            aux_reg_preds = torch.split(aux_reg_preds.view(nh, nb * np, aux_reg_preds.shape[-1]), 1, dim=0)

            ret_dict = {
                'aux_cls_preds': aux_cls_preds,
                'aux_reg_preds': aux_reg_preds,
            }
            # assign target.
            ret_dict.update(self.assign_targets(batch_dict))
            self.forward_ret_dict = ret_dict
        return batch_dict
