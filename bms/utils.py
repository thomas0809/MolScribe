import os
import random
import numpy as np
import torch
import math
import time
from tensorboardX import SummaryWriter
from bms.tokenizer import *


FORMAT_INFO = {
    "inchi": {
        "name": "InChI_text",
        "tokenizer": "tokenizer_inchi.json",
        "max_len": 300
    },
    "atomtok": {
        "name": "SMILES_atomtok",
        "tokenizer": "tokenizer_smiles_atomtok.json",
        "max_len": 256
    },
    "nodes": {
        "max_len": 384
    }
}


def init_logger(log_file='train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger


def init_summary_writer(save_path):
    # summary = SummaryWriter(os.path.join(save_path, datetime.now().strftime("%Y%m%d-%H%M%S")))
    summary = SummaryWriter(save_path)
    return summary


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def print_rank_0(message):
    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            print(message, flush=True)
    else:
        print(message, flush=True)


