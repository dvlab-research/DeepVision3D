import torch
from torch.autograd import Function, Variable

from eqnet.ops.knn import knn_cuda


class KNNQuery(Function):
    r"""KNN (CUDA) based on heap data structure.
    Modified from `PAConv <https://github.com/CVMI-Lab/PAConv/tree/main/
    scene_seg/lib/pointops/src/knnquery_heap>`_.

    Find k-nearest points.
    """
    @staticmethod
    def forward(ctx, nsample: int,
                xyz: torch.Tensor, xyz_cnt: torch.Tensor,
                new_xyz: torch.Tensor, new_xyz_cnt: torch.Tensor):
        """
        KNN Indexing
        input: nsample: int32, Number of neighbor
               xyz: (M1 + M2 + ..., 3) coordinates of the features
               xyz_cnt: [M1, M2, ...]
               new_xyz: (N1 + N2 + ..., 3) centriods
               new_xyz_cnt: [N1, N2, ...]
            output: idx: (N1 + N2 + ..., nsample)
        """
        assert nsample > 0

        assert xyz.is_contiguous()
        assert xyz_cnt.is_contiguous()
        assert new_xyz.is_contiguous()
        assert new_xyz_cnt.is_contiguous()

        b = xyz_cnt.shape[0]
        xyz_num = xyz.shape[0]

        new_xyz_num = new_xyz.shape[0]
        idx = xyz.new_zeros((new_xyz_num, nsample)).int()
        dist2 = xyz.new_zeros((new_xyz_num, nsample)).float()

        knn_cuda.knn_wrapper(
            b, xyz_num, new_xyz_num, nsample,
            xyz, xyz_cnt,
            new_xyz, new_xyz_cnt,
            idx, dist2)

        ctx.mark_non_differentiable(idx)
        return idx

    @staticmethod
    def backward(ctx):
        return None, None, None, None, None

knn_query = KNNQuery.apply