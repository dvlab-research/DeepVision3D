import torch
from torch.autograd import Function

import PG_OP

class Voxelization_Idx(Function):
    @staticmethod
    def forward(ctx, coords, batchsize, mode=4):
        '''
        :param ctx:
        :param coords:  long (N, dimension + 1) or (N, dimension) dimension = 3
        :param batchsize
        :param mode: int 4=mean
        :param dimension: int
        :return: output_coords:  long (M, dimension + 1) (M <= N)
        :return: output_map: int M * (maxActive + 1)
        :return: input_map: int N
        '''
        assert coords.is_contiguous()
        N = coords.size(0)
        output_coords = coords.new()

        input_map = torch.IntTensor(N).zero_()
        output_map = input_map.new()

        PG_OP.voxelize_idx(coords, output_coords, input_map, output_map, batchsize, mode)
        return output_coords, input_map, output_map


    @staticmethod
    def backward(ctx, a=None, b=None, c=None):
        return None

voxelization_idx = Voxelization_Idx.apply


class Voxelization(Function):
    @staticmethod
    def forward(ctx, feats, map_rule, mode=4):
        '''
        :param ctx:
        :param map_rule: cuda int M * (maxActive + 1)
        :param feats: cuda float N * C
        :return: output_feats: cuda float M * C
        '''
        assert map_rule.is_contiguous()
        assert feats.is_contiguous()
        N, C = feats.size()
        M = map_rule.size(0)
        maxActive = map_rule.size(1) - 1

        output_feats = torch.cuda.FloatTensor(M, C).zero_()

        ctx.for_backwards = (map_rule, mode, maxActive, N)

        PG_OP.voxelize_fp(feats, output_feats, map_rule, mode, M, maxActive, C)
        return output_feats


    @staticmethod
    def backward(ctx, d_output_feats):
        map_rule, mode, maxActive, N = ctx.for_backwards
        M, C = d_output_feats.size()

        d_feats = torch.cuda.FloatTensor(N, C).zero_()

        PG_OP.voxelize_bp(d_output_feats.contiguous(), d_feats, map_rule, mode, M, maxActive, C)
        return d_feats, None, None

voxelization = Voxelization.apply


class PointRecover(Function):
    @staticmethod
    def forward(ctx, feats, map_rule, nPoint):
        '''
        :param ctx:
        :param feats: cuda float M * C
        :param map_rule: cuda int M * (maxActive + 1)
        :param nPoint: int
        :return: output_feats: cuda float N * C
        '''
        assert map_rule.is_contiguous()
        assert feats.is_contiguous()
        M, C = feats.size()
        maxActive = map_rule.size(1) - 1

        output_feats = torch.cuda.FloatTensor(nPoint, C).zero_()

        ctx.for_backwards = (map_rule, maxActive, M)

        PG_OP.point_recover_fp(feats, output_feats, map_rule, M, maxActive, C)

        return output_feats

    @staticmethod
    def backward(ctx, d_output_feats):
        map_rule, maxActive, M = ctx.for_backwards
        N, C = d_output_feats.size()

        d_feats = torch.cuda.FloatTensor(M, C).zero_()

        PG_OP.point_recover_bp(d_output_feats.contiguous(), d_feats, map_rule, M, maxActive, C)

        return d_feats, None, None

point_recover = PointRecover.apply