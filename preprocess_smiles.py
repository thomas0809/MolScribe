import os
import re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
import torch
import argparse

from bms.utils import Tokenizer

def main(args):
    smiles_tokenizer = args.tokenizer
    print(f"Loading training data from data/train_smiles_{smiles_tokenizer}.csv")
    train = pd.read_csv(f'data/train_smiles_{smiles_tokenizer}.csv')
    print(f'train.shape: {train.shape}')
    target_dir = f'data/smiles_{smiles_tokenizer}'
    os.makedirs(target_dir, exist_ok=True)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train['smiles'].values)
    torch.save(tokenizer, f'{target_dir}/tokenizer.pth')
    print(f'Saved tokenizer to {target_dir}/tokenizer.pth')

    lengths = []
    tk0 = tqdm(train['smiles'].values, total=len(train))
    for text in tk0:
        seq = tokenizer.text_to_sequence(text)
        length = len(seq) - 2
        lengths.append(length)
    train['smiles_length'] = lengths
    train.to_pickle(f'{target_dir}/train.pkl')
    print(f'Saved preprocessed {target_dir}/train.pkl')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer",
        "-t",
        type=str,
        choices=["atomtok", "spe_chembl"],
        default="atomtok",
        help="Which tokenizer to use."
    )

    args = parser.parse_args()
    main(args)
