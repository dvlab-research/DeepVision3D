"""
MLP processor for pre-processing support features.

For each support features from the embedding stage network, it simply apply:
    * a Linear layer for mapping channels to target channel;
    * a LayerNorm.
"""
import torch.nn as nn

from eqnet.utils.support_helper import reorganize_input_for_transformer


class MLP_PROCESSOR(nn.Module):
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg

        # key from embedding stage network.
        self.max_valid_num = self.model_cfg.get('MAX_VALID_NUM')
        self.features_source = self.model_cfg.get('FEATURES_SOURCE')

        # input_channels from different feature source.
        self.inp_channels = self.model_cfg.get('INPUT_CHANNELS')
        assert len(self.inp_channels) == len(self.features_source)
        # mapping layer num.
        mapping_mlp_depth = self.model_cfg.get('MAPPING_MLP_DEPTH', 1)
        if not isinstance(mapping_mlp_depth, list):
            self.mapping_mlp_depth = [mapping_mlp_depth] * len(self.inp_channels)
        else:
            self.mapping_mlp_depth = mapping_mlp_depth
        assert len(self.mapping_mlp_depth) == len(self.inp_channels)
        # target output channel.
        target_chn = self.model_cfg.get('TARGET_CHANNEL')
        if not isinstance(target_chn, list):
            self.target_chn = [target_chn] * len(self.inp_channels)
        else:
            self.target_chn = target_chn
        assert len(self.target_chn) == len(self.inp_channels)

        # align channels from input to target.
        self.mapping_mlp = nn.ModuleList()
        # Linear layer + pre-LayerNorm.
        self.prenorm_layer = nn.ModuleList()
        for i in range(len(self.inp_channels)):
            # Build downsample layers.
            mapping_mlp = []
            for j in range(self.mapping_mlp_depth[i] - 1):
                mapping_mlp.extend([
                    nn.Linear(self.inp_channels[i], self.inp_channels[i], bias=False),
                    nn.BatchNorm1d(self.inp_channels[i]),
                    nn.ReLU()
                ])
            mapping_mlp.extend([
                nn.Linear(self.inp_channels[i], self.target_chn[i], bias=False),
                nn.BatchNorm1d(self.target_chn[i]),
                nn.ReLU()
            ])
            self.mapping_mlp.append(nn.Sequential(*mapping_mlp))

            prenorm_layer = nn.Sequential(
                nn.Linear(self.target_chn[i], self.target_chn[i]),
                nn.LayerNorm(self.target_chn[i])
            )
            self.prenorm_layer.append(prenorm_layer)

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
            cur_features = self.mapping_mlp[i](cur_features)
            cur_features = self.prenorm_layer[i](cur_features)

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