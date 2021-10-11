import os
import random
import numpy as np
import torch
import math
import time
import json
import copy
from datetime import datetime
import multiprocessing
import Levenshtein
import rdkit
import rdkit.Chem as Chem
import selfies as sf
rdkit.RDLogger.DisableLog('rdApp.*')

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
    "selfies": {
        "name": "SELFIES_tok",
        "tokenizer": "tokenizer_selfies.json",
        "max_len": 120
    }
}


def is_valid(str_, format_='atomtok'):
    if format_ == 'atomtok':
        mol = Chem.MolFromSmiles(str_)
    elif format_ == 'inchi':
        if not str_.startswith('InChI=1S'):
            str_ = f"InChI=1S/{str_}"
        mol = Chem.MolFromInchi(str_)
    else:
        raise NotImplemented
    return mol is not None
    

def convert_smiles_to_inchi(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        inchi = Chem.MolToInchi(mol)
    except:
        inchi = None
    return inchi


def batch_convert_smiles_to_inchi(smiles_list, num_workers=16):
    with multiprocessing.Pool(num_workers) as p:
        inchi_list = p.map(convert_smiles_to_inchi, smiles_list)
    n_success = sum([x is not None for x in inchi_list])
    r_success = n_success / len(inchi_list)
    inchi_list = [x if x else 'InChI=1S/H2O/h1H2' for x in inchi_list]
    return inchi_list, r_success


def batch_convert_selfies_to_inchi(selfies_list, num_workers=16):
    with multiprocessing.Pool(num_workers) as p:
        smiles_list = p.map(sf.decoder, selfies_list)
    return batch_convert_smiles_to_inchi(smiles_list)


def canonicalize_smiles(smiles):
    rlist = ['R', 'R1', 'R2', 'R3', 'R4', 'R5', 'X', 'Ar']
    rdict = {}
    for i, symbol in enumerate(rlist):
        rdict[f'[{symbol}]'] = f'[*:{i}]'
    for a, b in rdict.items():
        smiles = smiles.replace(a, b)
    try:
        canon_smiles = Chem.CanonSmiles(smiles)
    except:
        canon_smiles = smiles
    return canon_smiles


def get_canon_smiles_score(gold_smiles, pred_smiles):
    gold_canon_smiles = np.array([canonicalize_smiles(smiles) for smiles in gold_smiles])
    pred_canon_smiles = np.array([canonicalize_smiles(smiles) for smiles in pred_smiles])
    return (gold_canon_smiles == pred_canon_smiles).mean()


def merge_inchi(inchi1, inchi2):
    replaced = 0
    inchi1 = copy.deepcopy(inchi1)
    for i in range(len(inchi1)):
        if inchi1[i] == 'InChI=1S/H2O/h1H2':
            inchi1[i] = inchi2[i]
            replaced += 1
    return inchi1, replaced


def get_score(y_true, y_pred):
    scores = []
    exact_match = 0
    for true, pred in zip(y_true, y_pred):
        score = Levenshtein.distance(true, pred)
        scores.append(score)
        exact_match += int(true == pred)
    avg_score = np.mean(scores)
    exact_match = exact_match / len(y_true)
    return avg_score, exact_match


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


