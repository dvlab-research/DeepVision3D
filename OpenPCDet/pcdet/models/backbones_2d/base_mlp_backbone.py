"""
Compared to base_bev_backbone:

Use multiple interpolation function or decoding layer for decoding.
"""

import torch.nn as nn

class BaseMLPBackbone(nn.Module):
    def __init__(self, model_cfg, input_channels):
        super().__init__()
        self.model_cfg = model_cfg

        mlp_channels = self.model_cfg.MLP_CHANNELS
        mlp_channels = [input_channels] + mlp_channels
        mlp_layer_num = len(mlp_channels) - 1

        mlp_layers = []
        for i in range(mlp_layer_num):
            mlp_layers.extend([
                nn.Conv2d(mlp_channels[i], mlp_channels[i+1],
                          1, padding=0, bias=False),
                nn.BatchNorm2d(mlp_channels[i+1], eps=1e-3, momentum=0.01),
                nn.ReLU(),
            ])
        self.mlp_layers = nn.Sequential(*mlp_layers)

        self.num_bev_features = mlp_channels[-1]

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                spatial_features
        Returns:
        """
        spatial_features = data_dict['spatial_features']
        x = spatial_features
        x = self.mlp_layers(x)
        data_dict['spatial_features_2d'] = x

        return data_dict