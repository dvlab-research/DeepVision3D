import torch
import time
import numpy as np
import random
import os, sys
import argparse
from easydict import EasyDict

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

from data_utils import __dataset__all__
from utils import utils
from eqnet.utils.config import cfg_from_yaml_file, merge_new_config, namespace_to_cfg, cfg_from_list

def get_parser():
    parser = argparse.ArgumentParser(description='Point Cloud Segmentation')
    parser.add_argument('--config', type=str, default='config/eqnet_scannet.yaml', help='path to config file')

    parser.add_argument('--pretrain', type=str, default='', help='path to pretrain model')
    parser.add_argument('--fix_seed', type=bool, default=False, help='whether fix random seed during testing.')

    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')

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

def init():
    global result_dir
    result_dir = os.path.join(cfg.exp_path, 'result', 'epoch{}_reps{}'.format(cfg.test_epoch, cfg.test_reps), cfg.split)

    global semantic_label_idx, semantic_names
    if cfg.dataset == 'scannetv2':
        semantic_label_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]
        semantic_names = np.array(
            ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture',
             'counter', 'desk', 'curtain', 'refridgerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture'])
    elif cfg.dataset == 's3dis':
        semantic_label_idx = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
        semantic_names = np.array(
            ['ceiling', 'floor', 'wall', 'beam', 'column', 'window', 'door', 'chair', 'table', 'bookcase', 'sofa',
             'board', 'clutter'])

    global logger
    from utils.log import get_logger
    logger = get_logger(cfg)

    logger.info(cfg)

    if cfg.fix_seed:
        random.seed(cfg.manual_seed)
        np.random.seed(cfg.manual_seed)
        torch.manual_seed(cfg.manual_seed)
        torch.cuda.manual_seed_all(cfg.manual_seed)


def test(model, model_fn, dataset, epoch):
    logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    with torch.no_grad():
        model = model.eval()
        start = time.time()

        pred_scores = []
        for rep in range(cfg.test_reps):
            intersection, union, target = np.zeros(cfg.classes, dtype=np.int32), np.zeros(cfg.classes, dtype=np.int32), np.zeros(cfg.classes, dtype=np.int32)
            for i, batch in enumerate(dataset.test_data_loader):
                N = batch['feats'].shape[0]
                test_scene_name = dataset.test_file_names[int(batch['id'][0])].split('/')[-1].split('.')[0]

                start1 = time.time()
                preds = model_fn(batch, model, epoch)
                end1 = time.time() - start1

                # get prediction
                semantic_scores = preds['semantic']

                if rep == 0:
                    pred_scores.append(semantic_scores.cpu())
                else:
                    pred_scores[i] += semantic_scores.cpu()
                semantic_pred = pred_scores[i].max(1)[1]
                semantic_np = semantic_pred.cpu().numpy()

                # prepare for eval
                if cfg.eval:
                    i, u, t = utils.intersectionAndUnion(semantic_np, batch['labels'].cpu().numpy(), cfg.classes, cfg.ignore_label)
                    intersection += i
                    union += u
                    target += t

                # save files
                start3 = time.time()
                if cfg.save_semantic:
                    os.makedirs(os.path.join(result_dir, 'semantic'), exist_ok=True)
                    np.save(os.path.join(result_dir, 'semantic', test_scene_name + '.npy'), semantic_np)

                end3 = time.time() - start3
                end = time.time() - start
                start = time.time()

                # print
                logger.info("rep: {}/{} instance iter: {}/{} {} point_num: {} time: total {:.2f}s inference {:.2f}s save {:.2f}s".format(
                    rep + 1, cfg.test_reps, batch['id'][0] + 1, len(dataset.test_file_names), test_scene_name, N, end, end1, end3))

            # eval
            if cfg.eval:
                iou_class = intersection / (union + 1e-10)
                accuracy_class = intersection / (target + 1e-10)
                utils.print_iou_acc_class(iou_class, accuracy_class, semantic_names, logger)
                mIoU = np.mean(iou_class)
                mAcc = np.mean(accuracy_class)
                allAcc = sum(intersection) / (sum(target) + 1e-10)
                logger.info('rep: {}/{}: mIoU/mAcc/allAcc {:.2f}/{:.2f}/{:.2f}.'.format(rep + 1, cfg.test_reps, mIoU * 100, mAcc * 100, allAcc * 100))


if __name__ == '__main__':
    global cfg
    cfg = get_parser()
    cfg.task = 'test'
    cfg.dist = False
    init()

    # model
    logger.info('=> creating model ...')
    logger.info('Classes: {}'.format(cfg.classes))

    from models.models import Semantic as Network, model_fn_decorator
    model = Network(cfg)
    model_fn = model_fn_decorator(cfg, test=True)

    use_cuda = torch.cuda.is_available()
    logger.info('cuda available: {}'.format(use_cuda))
    assert use_cuda
    model = model.cuda()

    logger.info('#classifier parameters (model): {}'.format(sum([x.nelement() for x in model.parameters()])))

    # load model
    _, f = utils.checkpoint_restore(model, cfg.exp_path, cfg.config.split('/')[-1][:-5], cfg.test_epoch, dist=False, f=cfg.pretrain)
    logger.info('Restore from {}'.format(f))

    # data
    dataset = __dataset__all__[cfg.dataset](cfg, test=True)
    dataset.testLoader()
    logger.info('Testing samples ({}): {}'.format(cfg.split, len(dataset.test_file_names)))

    # evaluate
    test(model, model_fn, dataset, cfg.test_epoch)
