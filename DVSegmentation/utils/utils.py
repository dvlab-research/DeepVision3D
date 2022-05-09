import numpy as np
import os
import glob
import torch

def intersectionAndUnion(output, target, K, ignore_index=255):
    assert (output.ndim in [1, 2, 3])
    assert output.shape == target.shape
    output = output.reshape(output.size).copy()
    target = target.reshape(target.size)
    output[np.where(target == ignore_index)[0]] = ignore_index
    intersection = output[np.where(output == target)[0]]
    area_intersection, _ = np.histogram(intersection, bins=np.arange(K+1))
    area_output, _ = np.histogram(output, bins=np.arange(K+1))
    area_target, _ = np.histogram(target, bins=np.arange(K+1))
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target

class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * float(n)
        self.count += n
        self.avg = self.sum / float(self.count)

def checkpoint_restore(model, exp_path, exp_name, epoch=0, dist=False, f='', gpu=0, optimizer=None):
    if not f:
        if epoch > 0:
            f = os.path.join(exp_path, exp_name + '-%09d'%epoch + '.pth')
            assert os.path.isfile(f)
        else:
            f = sorted(glob.glob(os.path.join(exp_path, exp_name + '-*.pth')))
            if len(f) > 0:
                f = f[-1]
                epoch = int(f[len(exp_path) + len(exp_name) + 2 : -4])

    if len(f) > 0:
        map_location = {'cuda:0': 'cuda:{}'.format(gpu)} if gpu > 0 else None
        state = torch.load(f, map_location=map_location)

        checkpoint = state if not (isinstance(state, dict) and 'state_dict' in state) else state['state_dict']
        for k, v in checkpoint.items():
            if 'module.' in k:
                checkpoint = {k[len('module.'):]: v for k, v in checkpoint.items()}
            break
        if dist:
            model.module.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)

        if optimizer is not None:
            if isinstance(state, dict) and 'optimizer' in state:
                optim_dict = optimizer.state_dict()
                optim_dict['param_groups'] = state['optimizer']['param_groups']
                optim_dict['state'] = state['optimizer']['state']
                optimizer.load_state_dict(optim_dict)

    return epoch + 1, f


def is_power2(num):
    return num != 0 and ((num & (num - 1)) == 0)


def is_multiple(num, multiple):
    return num != 0 and num % multiple == 0


def is_last(num, total_num, ratio=0.98):
    return num > int(total_num * ratio)


def checkpoint_save(model, optimizer, exp_path, exp_name, epoch, save_freq=16, last_ratio=0.98, epochs=512):
    f = os.path.join(exp_path, exp_name + '-%09d'%epoch + '.pth')
    state = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict()
    }
    torch.save(state, f)

    #remove previous checkpoints
    epoch = epoch - 1
    fd = os.path.join(exp_path, exp_name + '-%09d'%epoch + '.pth')
    if os.path.isfile(fd):
        if not is_multiple(epoch, save_freq) and not is_last(epoch, epochs, last_ratio):
            os.remove(fd)

    return f

def print_iou_acc_class(iou_class, acc_class, semantic_names, logger=None):
    sep     = ""
    col1    = ":"
    lineLen = 64

    if logger is None:
        print("")
        print("#" * lineLen)
    else:
        logger.info("")
        logger.info("#" * lineLen)
    line  = ""
    line += "{:<15}".format("what"      ) + sep + col1
    line += "{:>15}".format("iou"        ) + sep
    line += "{:>15}".format("acc"    ) + sep
    if logger is None:
        print(line)
        print("#" * lineLen)
    else:
        logger.info(line)
        logger.info("#" * lineLen)

    for (li,label_name) in enumerate(semantic_names):
        iou = iou_class[li]
        acc = acc_class[li]
        line  = "{:<15}".format(label_name) + sep + col1
        line += sep + "{:>15.3f}".format(iou ) + sep
        line += sep + "{:>15.3f}".format(acc ) + sep
        if logger is None:
            print(line)
        else:
            logger.info(line)

    if logger is None:
        print("-"*lineLen)
    else:
        logger.info("-"*lineLen)
    line  = "{:<15}".format("average") + sep + col1
    line += "{:>15.3f}".format(np.mean(iou_class))  + sep
    line += "{:>15.3f}".format(np.mean(acc_class))  + sep
    if logger is None:
        print(line)
        print("")
    else:
        logger.info(line)
        logger.info("")