import torch
import torch.optim as optim
import time, sys, os, random, glob, argparse, yaml
from yaml import Loader
from tensorboardX import SummaryWriter
import numpy as np
from easydict import EasyDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import torch.multiprocessing as mp
import torch.distributed as dist
import subprocess

from data_utils import __dataset__all__
from utils import utils
from utils.lr import initialize_scheduler
from eqnet.utils.config import cfg_from_yaml_file, merge_new_config, namespace_to_cfg, cfg_from_list

def get_parser():
    parser = argparse.ArgumentParser(description='Point Cloud Segmentation')
    parser.add_argument('--config', type=str, default='config/eqnet_scannet.yaml', help='path to config file')

    parser.add_argument('--pretrain', type=str, default='', help='path to pretrain model')

    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=17777)

    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER, help='set extra config keys if needed')

    args = parser.parse_args()
    args = namespace_to_cfg(args)
    assert args.config is not None

    cfg = EasyDict()
    cfg_from_yaml_file(args.config, cfg)

    # rewrite some default arguments in args (if have).
    cfg = merge_new_config(args, cfg)

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    cfg.exp_path = os.path.join('exp', cfg.dataset, cfg.config.split('/')[-1][:-5])
    return cfg


def init_dist_pytorch(backend='nccl'):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    num_gpus = torch.cuda.device_count()
    assert cfg.batch_size % num_gpus == 0, 'Batch size should be matched with GPUS: (%d, %d)' % (cfg.batch_size, num_gpus)
    cfg.batch_size = cfg.batch_size // num_gpus

    torch.cuda.set_device(cfg.local_rank)

    print('[PID {}] rank: {} world_size: {}'.format(os.getpid(), cfg.local_rank, num_gpus))
    dist.init_process_group(backend=backend)


def init_dist_slurm(backend='nccl'):
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)

    addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(cfg.tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

    total_gpus = dist.get_world_size()
    assert cfg.batch_size % total_gpus == 0, 'Batch size should be matched with GPUS: (%d, %d)' % (cfg.batch_size, total_gpus)
    cfg.batch_size = cfg.batch_size // total_gpus


def init():
    if cfg.local_rank == 0:
        # logger
        global logger
        from utils.log import get_logger
        logger = get_logger(cfg)

        # log the config
        logger.info(cfg)

        # summary writer
        global writer
        writer = SummaryWriter(cfg.exp_path)

    # random seed
    random.seed(cfg.manual_seed)
    np.random.seed(cfg.manual_seed)
    torch.manual_seed(cfg.manual_seed)
    torch.cuda.manual_seed_all(cfg.manual_seed)


def train_epoch(train_loader, model, model_fn, optimizer, scheduler, epoch):
    iter_time, data_time = utils.AverageMeter(), utils.AverageMeter()
    am_dict = {}

    model.train()
    start_epoch = time.time()
    end = time.time()
    for i, batch in enumerate(train_loader):
        data_time.update(time.time() - end)

        # forward
        loss, _, visual_dict, meter_dict = model_fn(batch, model, epoch)

        # meter dict
        for k, v in meter_dict.items():
            if k not in am_dict.keys():
                am_dict[k] = utils.AverageMeter()
            am_dict[k].update(v[0], v[1])

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # adjust learning rate
        lrs = scheduler.get_last_lr()
        scheduler.step()

        # time and print
        current_iter = (epoch - 1) * len(train_loader) + i + 1
        max_iter = cfg.epochs * len(train_loader)
        remain_iter = max_iter - current_iter

        iter_time.update(time.time() - end)
        end = time.time()

        remain_time = remain_iter * iter_time.avg
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))

        if cfg.local_rank == 0:
            sys.stdout.write(
                "epoch: {}/{} iter: {}/{} lr: {:.4e} loss: {:.4f}({:.4f}) data_time: {:.2f}({:.2f}) iter_time: {:.2f}({:.2f}) remain_time: {remain_time}\n".format
                (epoch, cfg.epochs, i + 1, len(train_loader), lrs[0], am_dict['loss'].val, am_dict['loss'].avg,
                 data_time.val, data_time.avg, iter_time.val, iter_time.avg, remain_time=remain_time))
            if (i == len(train_loader) - 1): print()

            writer.add_scalar('lr', lrs[0], current_iter)

    if cfg.local_rank == 0:
        logger.info("epoch: {}/{}, train loss: {:.4f}, time: {}s".format(epoch, cfg.epochs, am_dict['loss'].avg, time.time() - start_epoch))

        f = utils.checkpoint_save(model, optimizer, cfg.exp_path, cfg.config.split('/')[-1][:-5], epoch, cfg.save_freq, epochs=cfg.epochs)
        logger.info('Saving {}'.format(f))

        iou_dict = {}
        for k in am_dict.keys():
            if k in visual_dict.keys():
                writer.add_scalar(k + '_train', am_dict[k].avg, epoch)
            if k in ['intersection', 'union', 'target']:
                iou_dict[k] = am_dict[k].sum

        if 'intersection' in iou_dict:
            miou = np.mean(iou_dict['intersection'] / (iou_dict['union'] + 1e-10))
            macc = np.mean(iou_dict['intersection'] / (iou_dict['target'] + 1e-10))
            allacc = sum(iou_dict['intersection']) / (sum(iou_dict['target']) + 1e-10)
            writer.add_scalar('miou_train', miou, epoch)
            writer.add_scalar('macc_train', macc, epoch)
            writer.add_scalar('allacc_train', allacc, epoch)
            logger.info('miou: {:.4f} macc: {:.4f} allacc: {:.4f}'.format(miou, macc, allacc))


def eval_epoch(val_loader, model, model_fn, epoch):
    if cfg.local_rank == 0:
        logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    am_dict = {}

    with torch.no_grad():
        model.eval()
        start_epoch = time.time()
        for i, batch in enumerate(val_loader):
            # forward
            loss, preds, visual_dict, meter_dict = model_fn(batch, model, epoch)

            # merge(allreduce) multi-gpu
            if cfg.dist:
                for k, v in visual_dict.items():
                    count = meter_dict[k][1]
                    v = v * count
                    count = loss.new_tensor([count], dtype=torch.long)
                    dist.all_reduce(v), dist.all_reduce(count)
                    count = count.item()
                    v = v / count

                    visual_dict[k] = v
                    meter_dict[k] = (float(v), count)

            # meter dict
            for k, v in meter_dict.items():
                if k not in am_dict.keys():
                    am_dict[k] = utils.AverageMeter()
                if k in ['intersection', 'union', 'target'] and cfg.dist:
                    cnt_list = torch.from_numpy(v[0]).cuda()
                    dist.all_reduce(cnt_list)
                    am_dict[k].update(cnt_list.cpu().numpy(), v[1])
                else:
                    am_dict[k].update(v[0], v[1])

            # print
            if cfg.local_rank == 0:
                sys.stdout.write("\riter: {}/{} loss: {:.4f}({:.4f})".format(i + 1, len(val_loader), am_dict['loss'].val, am_dict['loss'].avg))
                if (i == len(val_loader) - 1): print()

        if cfg.local_rank == 0:
            logger.info("epoch: {}/{}, val loss: {:.4f}, time: {}s".format(epoch, cfg.epochs, am_dict['loss'].avg, time.time() - start_epoch))

            iou_dict = {}
            for k in am_dict.keys():
                if k in visual_dict.keys():
                    writer.add_scalar(k + '_eval', am_dict[k].avg, epoch)
                if k in ['intersection', 'union', 'target']:
                    iou_dict[k] = am_dict[k].sum

            if 'intersection' in iou_dict:
                miou = np.mean(iou_dict['intersection'] / (iou_dict['union'] + 1e-10))
                macc = np.mean(iou_dict['intersection'] / (iou_dict['target'] + 1e-10))
                allacc = sum(iou_dict['intersection']) / (sum(iou_dict['target']) + 1e-10)
                writer.add_scalar('miou_eval', miou, epoch)
                writer.add_scalar('macc_eval', macc, epoch)
                writer.add_scalar('allacc_eval', allacc, epoch)
                logger.info('miou: {:.4f} macc: {:.4f} allacc: {:.4f}'.format(miou, macc, allacc))




if __name__ == '__main__':
    # config
    global cfg
    cfg = get_parser()

    # init
    if cfg.launcher == 'pytorch':
        init_dist_pytorch(backend='nccl')
        cfg.dist = True
    elif cfg.launcher == 'slurm':
        init_dist_slurm(backend='nccl')
        cfg.dist = True
        cfg.local_rank = dist.get_rank()
    else:
        cfg.dist = False
    init()

    # model
    if cfg.local_rank == 0:
        logger.info('=> creating model ...')

    from models.models import Semantic as Network, model_fn_decorator
    model = Network(cfg)
    model_fn = model_fn_decorator(cfg)

    use_cuda = torch.cuda.is_available()
    if cfg.local_rank == 0:
        logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda
    model = model.cuda()

    if cfg.dist:
        num_gpus = torch.cuda.device_count()
        if cfg.sync_bn:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(cfg.local_rank % num_gpus)
        local_rank = cfg.local_rank % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    # optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    # dataset
    dataset = __dataset__all__[cfg.dataset](cfg)
    dataset.trainLoader()
    dataset.valLoader()
    if cfg.local_rank == 0:
        logger.info('Training samples: {}'.format(len(dataset.train_data_loader.dataset)))
        logger.info('Validation samples: {}'.format(len(dataset.val_data_loader.dataset)))
    cfg.iter_per_epoch_train = len(dataset.train_data_loader)
    cfg.max_iter = cfg.iter_per_epoch_train * cfg.epochs

    # resume
    start_epoch, f = utils.checkpoint_restore(model, cfg.exp_path, cfg.config.split('/')[-1][:-5],
                                dist=cfg.dist, gpu=cfg.local_rank % torch.cuda.device_count(), optimizer=optimizer)
    if cfg.local_rank == 0:
        logger.info('Restore from {}'.format(f) if len(f) > 0 else 'Start from epoch {}'.format(start_epoch))

    # lr_scheduler
    scheduler = initialize_scheduler(optimizer, cfg, last_epoch=start_epoch - 2, scheduler_epoch=cfg.scheduler_epoch)

    # train and val
    for epoch in range(start_epoch, cfg.epochs + 1):
        if cfg.dist:
            dataset.train_sampler.set_epoch(epoch)
        train_epoch(dataset.train_data_loader, model, model_fn, optimizer, scheduler, epoch)

        if cfg.validation:
            if utils.is_multiple(epoch, cfg.save_freq) or utils.is_power2(epoch) or utils.is_last(epoch, cfg.epochs):
                if cfg.dist:
                    dataset.val_sampler.set_epoch(epoch)
                eval_epoch(dataset.val_data_loader, model, model_fn, epoch)

