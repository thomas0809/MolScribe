import os
import random
import numpy as np
import torch
import math
import time
import datetime
import json
from json import encoder


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
    "nodes": {"max_len": 384},
    "atomtok_coords": {"max_len": 480},
    "chartok_coords": {"max_len": 480}
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
    from tensorboardX import SummaryWriter
    summary = SummaryWriter(save_path)
    return summary


def save_args(args):
    dt = datetime.datetime.strftime(datetime.datetime.now(), "%y%m%d-%H%M")
    path = os.path.join(args.save_path, f'train_{dt}.log')
    with open(path, 'w') as f:
        for k, v in vars(args).items():
            f.write(f"**** {k} = *{v}*\n")
    return


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


class EpochMeter(AverageMeter):
    def __init__(self):
        super().__init__()
        self.epoch = AverageMeter()

    def update(self, val, n=1):
        super().update(val, n)
        self.epoch.update(val, n)


class LossMeter(EpochMeter):
    def __init__(self):
        self.subs = {}
        super().__init__()

    def reset(self):
        super().reset()
        for k in self.subs:
            self.subs[k].reset()

    def update(self, loss, losses, n=1):
        loss = loss.item()
        super().update(loss, n)
        losses = {k: v.item() for k, v in losses.items()}
        for k, v in losses.items():
            if k not in self.subs:
                self.subs[k] = EpochMeter()
            self.subs[k].update(v, n)


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


def to_device(data, device):
    if torch.is_tensor(data):
        return data.to(device)
    if type(data) is list:
        return [to_device(v, device) for v in data]
    if type(data) is dict:
        return {k: to_device(v, device) for k, v in data.items()}


def round_floats(o):
    if isinstance(o, float):
        return round(o, 3)
    if isinstance(o, dict):
        return {k: round_floats(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [round_floats(x) for x in o]
    return o


def format_df(df):
    def _dumps(obj):
        if obj is None:
            return obj
        return json.dumps(round_floats(obj)).replace(" ", "")
    for field in ['node_coords', 'node_symbols', 'edges']:
        if field in df.columns:
            df[field] = [_dumps(obj) for obj in df[field]]
    return df
