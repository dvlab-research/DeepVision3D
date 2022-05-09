import numpy as np
import math
import scipy
import os
import SharedArray as SA
import torch
import scipy.ndimage
import scipy.interpolate

def dataAugment(xyz, jitter=False, flip=False, rot=False):
    m = np.eye(3)
    if jitter:
        m += np.random.randn(3, 3) * 0.1
    if flip:
        m[0][0] *= np.random.randint(0, 2) * 2 - 1  # flip x randomly
    if rot:
        theta = np.random.rand() * 2 * math.pi
        m = np.matmul(m, [[math.cos(theta), math.sin(theta), 0],
                          [-math.sin(theta), math.cos(theta), 0],
                          [0, 0, 1]])  # rotation
    return np.matmul(xyz, m)


def elastic(x, gran, mag):
    blur0 = np.ones((3, 1, 1)).astype('float32') / 3
    blur1 = np.ones((1, 3, 1)).astype('float32') / 3
    blur2 = np.ones((1, 1, 3)).astype('float32') / 3

    bb = (np.abs(x).max(0).astype(np.int32) // gran + 3).astype(np.int32)
    noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
    noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    ax = [np.linspace(-(b - 1) * gran, (b - 1) * gran, b) for b in bb]
    interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]

    def g(x_):
        return np.hstack([i(x_)[:, None] for i in interp])

    return x + g(x) * mag


def sa_create(name, var):
    x = SA.create(name, var.shape, dtype=var.dtype)
    x[...] = var[...]
    x.flags.writeable = False
    return x


def create_shared_memory(file_names, wlabel=True):
    for i, fname in enumerate(file_names):
        fn = fname.split('/')[-1].split('.')[0]
        if not os.path.exists("/dev/shm/{}_xyz".format(fn)):
            print("[PID {}] {} {}".format(os.getpid(), i, fn))
            if wlabel:
                xyz, rgb, label = torch.load(fname)
            else:
                xyz, rgb = torch.load(fname)
            sa_create("shm://{}_xyz".format(fn), xyz)
            sa_create("shm://{}_rgb".format(fn), rgb)
            if wlabel:
                sa_create("shm://{}_label".format(fn), label)
