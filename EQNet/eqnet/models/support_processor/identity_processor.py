"""
Identity processor for pre-processing support features.
"""
import torch.nn as nn

from eqnet.utils.support_helper import reorganize_input_for_transformer


class IDENTITY_PROCESSOR(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg

        # key from embedding stage network.
        self.max_valid_num = self.model_cfg.get('MAX_VALID_NUM')
        self.features_source = self.model_cfg.get('FEATURES_SOURCE')

        # input_channels from different feature source.
        self.inp_channels = self.model_cfg.get('INPUT_CHANNELS')

        # target output channel.
        target_chn = self.model_cfg.get('TARGET_CHANNEL')
        if not isinstance(target_chn, list):
            self.target_chn = [target_chn] * len(self.inp_channels)
        else:
            self.target_chn = target_chn


    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                multi_scale_support_sets:
                    features_source_key-1:
                        *support_features: M1 + M2 + ..., C
                        *support_points: M1 + M2 + ..., 4: {batch_index, x, y, z}
                    features_source_key-2:
                        *support_features: M1 + M2 + ..., C
                        *support_points: M1 + M2 + ..., 4: {batch_index, x, y, z}
                    ...

        Returns:
            support_features: a list of (B, N, C)
            support_points: a list of (B, N, 3)
            support_mask: a list of (B, N), 0: valid / 1: invalid padding key.

        """
        support_sets = batch_dict['multi_scale_support_sets']
        batch_size = batch_dict['batch_size']

        support_features, support_points, support_mask = [], [], []
        for i, src_name in enumerate(self.features_source):
            cur_sets = support_sets[src_name]

            cur_features = cur_sets['support_features']
            cur_coords = cur_sets['support_points']

            reorg_features, reorg_coords, reorg_masks = reorganize_input_for_transformer(
                batch_size, cur_coords, cur_features, self.max_valid_num)

            support_features.append(reorg_features)
            support_points.append(reorg_coords)
            support_mask.append(reorg_masks)

        ret_dict = {
            'support_features': support_features,  # a list of (B, M, C)
            'support_points': support_points,  # a list of (B, M, 3)
            'support_mask': support_mask,  # a lost of (B, M)
        }
        return ret_dict