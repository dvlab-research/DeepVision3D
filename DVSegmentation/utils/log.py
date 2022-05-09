import logging
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)


def create_logger(log_file, rank=0):
    logger = logging.getLogger(__name__)
    level = logging.DEBUG if rank == 0 else logging.ERROR
    logger.setLevel(level)

    log_format = '[%(asctime)s  %(levelname)s  %(filename)s  line %(lineno)d  %(process)d]  %(message)s'
    formatter = logging.Formatter(log_format)

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    logging.basicConfig(level=level, format=log_format, filename=log_file)
    return logger


def get_logger(cfg):
    if cfg.task == 'train':
        log_file = os.path.join(
            cfg.exp_path,
            'train-{}.log'.format(time.strftime("%Y%m%d_%H%M%S", time.localtime()))
        )
    elif cfg.task == 'test':
        log_file = os.path.join(
            cfg.exp_path, 'result', 'epoch{}_reps{}'.format(cfg.test_epoch, cfg.test_reps),
            cfg.split, 'test-{}.log'.format(time.strftime("%Y%m%d_%H%M%S", time.localtime()))
        )
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = create_logger(log_file, cfg.local_rank)
    logger.info('************************ Start Logging ************************')
    return logger