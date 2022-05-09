import torch
import torch.nn as nn
import numpy as np

class RPE(nn.Module):
    def __init__(self, nhead, point_cloud_range, quan_size=0.02):
        '''
        Args:
            point_cloud_range: (6), [xmin, ymin, zmin, xmax, ymax, zmax]
        '''
        super().__init__()

        self.nhead = nhead
        point_cloud_range = np.array(point_cloud_range)
        point_cloud_range = point_cloud_range[3:6] - point_cloud_range[0:3]
        point_cloud_range = (point_cloud_range**2).sum()**0.5
        self.max_len = int(point_cloud_range // quan_size + 1)
        self.grid_size = quan_size

        self.pos_embed = nn.Embedding(self.max_len, self.nhead)
        nn.init.uniform_(self.pos_embed.weight)

    def forward(self, batch_rel_coords):
        """
        Args:
            batch_rel_coords: (B, N, 3)
        Returns
            pos_embedding: (B, N, nhead)
        """
        dist = torch.norm(batch_rel_coords, dim=-1)  # (B, N)
        dist = dist / self.grid_size

        idx1 = dist.long()
        idx2 = idx1 + 1
        w1 = idx2.type_as(dist) - dist
        w2 = dist - idx1.type_as(dist)

        idx1[idx1 >= self.max_len] = self.max_len - 1
        idx2[idx2 >= self.max_len] = self.max_len - 1

        embed1 = self.pos_embed(idx1)  # (B, N, nhead)
        embed2 = self.pos_embed(idx2)  # (B, N, nhead)

        embed = embed1 * w1.unsqueeze(-1) + embed2 * w2.unsqueeze(-1)  # (B, N, nhead)

        return embed