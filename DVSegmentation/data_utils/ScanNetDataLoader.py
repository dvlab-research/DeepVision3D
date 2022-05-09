import os, sys, glob, numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import SharedArray as SA

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

from .data_utils  import dataAugment, elastic, create_shared_memory
from ops.ops import voxelization_idx

class ScanNet(Dataset):
    def __init__(self, cfg, file_names, aug_options, test=False):
        '''
        aug_options: [jit, flip, rot, elastic, crop, rgb_aug]
        '''
        super(ScanNet, self).__init__()

        self.cache = cfg.cache

        self.full_scale = cfg.full_scale
        self.scale = cfg.scale
        self.max_npoint = cfg.max_npoint

        self.file_names = file_names
        if not self.cache:
            self.files = [torch.load(i) for i in self.file_names]

        self.jit, self.flip, self.rot, self.elas, self.crop, self.rgb_aug = aug_options

        self.test = test


    def crop_by_npoint(self, xyz):
        xyz_offset = xyz.copy()
        valid_idxs = (xyz_offset.min(1) >= 0)
        assert valid_idxs.sum() == xyz.shape[0]

        full_scale = np.array([self.full_scale[1]] * 3)
        room_range = xyz.max(0) - xyz.min(0)
        while(valid_idxs.sum() > self.max_npoint):
            offset = np.clip(full_scale - room_range + 0.001, None, 0) * np.random.rand(3)
            xyz_offset = xyz + offset
            valid_idxs = (xyz_offset.min(1) >= 0) * ((xyz_offset < full_scale).sum(1) == 3)
            full_scale[:2] -= 32

        return xyz_offset, valid_idxs


    def __getitem__(self, id):
        fn = self.file_names[id].split('/')[-1].split('.')[0]
        if self.cache:
            xyz_origin = SA.attach("shm://{}_xyz".format(fn)).copy()
            rgb = SA.attach("shm://{}_rgb".format(fn)).copy()
            if not self.test:
                label = SA.attach("shm://{}_label".format(fn)).copy()
        else:
            xyz_origin = self.files[id][0]
            rgb = self.files[id][1]
            if not self.test:
                label = self.files[id][2]

        # coord augment
        xyz = dataAugment(xyz_origin, self.jit, self.flip, self.rot)
        xyz = xyz * self.scale
        if self.elas:
            xyz = elastic(xyz, 6 * self.scale // 50, 40 * self.scale / 50)
            xyz = elastic(xyz, 20 * self.scale // 50, 160 * self.scale / 50)
        xyz -= xyz.min(0)
        if self.crop:
            while True:
                xyz_offset, valid_idxs = self.crop_by_npoint(xyz)
                if valid_idxs.sum() > 500:
                    break
            xyz, rgb = xyz_offset[valid_idxs], rgb[valid_idxs]
            if not self.test:
                label = label[valid_idxs]

        xyz_float = xyz / self.scale

        # rgb augment
        if self.rgb_aug:
            rgb += (np.random.randn(3) * 0.1)

        item = {'xyz': xyz, 'xyz_float': xyz_float, 'rgb': rgb}
        if not self.test:
            item.update({'label': label})

        item['item_id'] = id
        item['item_fn'] = fn

        return item

    def __len__(self):
        return len(self.file_names)


class MyDataset:
    def __init__(self, cfg, test=False):
        self.cfg = cfg

        self.data_root = cfg.data_root
        self.dataset = cfg.dataset
        self.filename_suffix = cfg.filename_suffix

        self.batch_size = cfg.batch_size
        self.train_workers = cfg.train_workers
        self.val_workers = cfg.train_workers

        self.dist = cfg.dist
        self.local_rank = cfg.local_rank

        self.cache = cfg.cache

        self.train_flip = cfg.train_flip
        self.train_rot = cfg.train_rot
        self.train_jit = cfg.train_jit
        self.train_elas = cfg.train_elas
        self.train_crop = getattr(cfg, 'train_crop', True)
        self.val_crop = cfg.val_crop

        self.full_scale = cfg.full_scale

        if test:
            self.test_split = cfg.split  # val or test

            cfg.batch_size = 1
            self.batch_size = 1
            self.test_workers = cfg.test_workers


    def trainLoader(self):
        self.train_file_names = sorted(glob.glob(os.path.join(self.data_root, self.dataset, 'train', '*' + self.filename_suffix)))
        if self.cache:
            num_gpus = 1 if not self.dist else torch.cuda.device_count()
            rk = self.local_rank % torch.cuda.device_count()
            create_shared_memory(self.train_file_names[rk::num_gpus], wlabel=True)

        train_set = ScanNet(
            self.cfg, self.train_file_names,
            [self.train_jit, self.train_flip, self.train_rot, self.train_elas, self.train_crop, True],
        )

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_set) if self.dist else None
        self.train_data_loader = DataLoader(train_set, batch_size=self.batch_size, collate_fn=self.get_batch_data, num_workers=self.train_workers,
                                            shuffle=(self.train_sampler is None), sampler=self.train_sampler, drop_last=True, pin_memory=True, worker_init_fn=self._worker_init_fn_)


    def valLoader(self):
        self.val_file_names = sorted(glob.glob(os.path.join(self.data_root, self.dataset, 'val', '*' + self.filename_suffix)))
        if self.cache:
            num_gpus = 1 if not self.dist else torch.cuda.device_count()
            rk = self.local_rank % torch.cuda.device_count()
            create_shared_memory(self.val_file_names[rk::num_gpus], wlabel=True)

        val_set = ScanNet(
            self.cfg, self.val_file_names,
            [False, True, True, False, self.val_crop, False],
        )

        self.val_sampler = torch.utils.data.distributed.DistributedSampler(val_set) if self.dist else None
        self.val_data_loader = DataLoader(val_set, batch_size=self.batch_size, collate_fn=self.get_batch_data, num_workers=self.val_workers,
                                          shuffle=False, sampler=self.val_sampler, drop_last=False, pin_memory=True, worker_init_fn=self._worker_init_fn_)


    def testLoader(self):
        assert self.test_split in ['val', 'test']
        self.test_file_names = sorted(glob.glob(os.path.join(self.data_root, self.dataset, self.test_split, '*' + self.filename_suffix)))
        if self.cache:
            create_shared_memory(self.test_file_names, wlabel=self.test_split=='val')
        test_set = ScanNet(
            self.cfg, self.test_file_names,
            [False, True, True, False, False, False],
            test=(self.test_split == 'test'),
        )

        self.test_data_loader = DataLoader(test_set, batch_size=1, collate_fn=self.get_batch_data, num_workers=self.test_workers,
                                           shuffle=False, drop_last=False, pin_memory=True, worker_init_fn=self._worker_init_fn_)


    def _worker_init_fn_(self, worker_id):
        torch_seed = torch.initial_seed()
        np_seed = torch_seed // 2 ** 32 - 1
        np.random.seed(np_seed)


    def get_batch_data(self, item_list):

        def cat_batch_data(item_list, key, dtype=None, add_batch_idx=False, match_dim=False):
            values = []
            for i, item in enumerate(item_list):
                value = torch.from_numpy(item[key])
                if dtype: value = value.to(dtype)
                if add_batch_idx:
                    value = torch.cat([torch.ones(value.shape[0], 1).to(value) * i, value], 1)
                values.append(value)
            if match_dim:  # for [(N1, C1, ...), (N2, C2, ...), ...], match the dimension in dim 1
                max_dim = max([v.shape[1] for v in values])
                for i, v in enumerate(values):
                    v_shape = list(v.shape)
                    v_shape[1] = max_dim - v_shape[1]
                    values[i] = torch.cat((v, v.new_zeros(v_shape)), 1)
            values = torch.cat(values, 0)
            return values

        def combine_batch_data(item_list, key):
            values = []
            for i, item in enumerate(item_list):
                values.append(item[key])
            return values

        test = not ('label' in item_list[0])

        locs = cat_batch_data(item_list, 'xyz', dtype=torch.long, add_batch_idx=True)  # long (N, 1 + 3)
        locs_float = cat_batch_data(item_list, 'xyz_float', dtype=torch.float32)  # float (N, 3)
        feats = cat_batch_data(item_list, 'rgb')  # float (N, C)
        if not test:
            labels = cat_batch_data(item_list, 'label', dtype=torch.long)           # long (N)

        fns, fids = combine_batch_data(item_list, 'item_fn'), combine_batch_data(item_list, 'item_id')

        # voxelize
        voxel_locs, p2v_map, v2p_map = voxelization_idx(locs, self.batch_size, 4)

        # spatial_shape
        spatial_shape = np.clip((locs.max(0)[0][1:] + 1).numpy(), self.full_scale[0], None)  # long (3)

        # batch_offsets
        batch_offsets = [0]
        for item in item_list: batch_offsets.append(batch_offsets[-1] + item['xyz'].shape[0])
        batch_offsets = torch.tensor(batch_offsets, dtype=torch.int32)  # int (B+1)

        batch_data = {'locs': locs, 'voxel_locs': voxel_locs, 'p2v_map': p2v_map, 'v2p_map': v2p_map,
                      'locs_float': locs_float, 'feats': feats,
                      'offsets': batch_offsets, 'spatial_shape': spatial_shape,
                      'file_names': fns, 'id': fids}
        if not test:
            batch_data['labels'] = labels

        return batch_data
