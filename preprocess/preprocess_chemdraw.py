import os
import re
import argparse
import numpy as np
import pandas as pd
import selfies as sf
from collections import Counter
import multiprocessing
from tqdm.auto import tqdm
tqdm.pandas()

import torch
from sklearn.model_selection import StratifiedKFold
import rdkit.Chem as Chem

from bms.utils import Tokenizer, batch_convert_smiles_to_inchi


# ====================================================
# main
# ====================================================
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--tokenizer', action='store_true')
    args = parser.parse_args()
    
    # ====================================================
    # Data Loading
    # ====================================================
    PATH = 'data/molbank/chemdraw-data/'
    
#     for split in ['train', 'valid', 'test']:
#         with open(PATH + f'files/std-files/std-src-{split}.txt') as f:
#             lines = [s.strip() for s in f.readlines()]
#             ids = [s.replace('.png', '') for s in lines]
#             file_path = [PATH + f'images/std-images/{s}' for s in lines]
#         with open(PATH + f'files/std-files/std-tgt-{split}.txt') as f:
#             smiles_tok = [s.strip() for s in f.readlines()]
#             smiles = [s.replace(' ', '') for s in smiles_tok]
#         inchi, r_success = batch_convert_smiles_to_inchi(smiles)
#         print(split, f'SMILES to InChI: {r_success:.4f}')
#         df = pd.DataFrame({
#             'image_id': ids,
#             'file_path': file_path, 
#             'SMILES': smiles, 
#             'SMILES_atomtok': smiles_tok, 
#             'InChI': inchi
#         })
#         df.to_csv(PATH + f'std_{split}.csv', index=False)
#         print('Max length:', max([len(s.split()) for s in smiles_tok]))
#         if split == 'train':
#             train_df = df

    for split in ['train', 'valid', 'test']:
        df = pd.read_csv(f'/data/rsg/chemistry/jiang_guo/chemai/literature-ie/image-omr/molbank/chemdraw-data/{split}.csv')
        mask = ['abb-images' not in s for s in df['image_path'].values]
        df = df[mask]
        file_path = [PATH + s for s in df['image_path'].values]
        ids = [s.split('/')[-1].replace('.png', '') for s in file_path]
        smiles_tok = df['smiles_mittok'].values
        smiles = [s.replace(' ', '') for s in smiles_tok]
        inchi, r_success = batch_convert_smiles_to_inchi(smiles)
        print(split, f'SMILES to InChI: {r_success:.4f}')
        df = pd.DataFrame({
            'image_id': ids,
            'file_path': file_path, 
            'SMILES': smiles, 
            'SMILES_atomtok': smiles_tok, 
            'InChI': inchi
        })
        df.to_csv(PATH + f'{split}.csv', index=False)
        print('Max length:', max([len(s.split()) for s in smiles_tok]))
        if split == 'train':
            train_df = df
    
    # ====================================================
    # create tokenizer
    # ====================================================
    if args.tokenizer:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(train_df['SMILES_atomtok'].values)
        tokenizer.save(PATH + 'tokenizer_smiles_mittok.json')
        print('Saved tokenizer_smiles_atomtok')
        print(f"vocab: {len(tokenizer.stoi)}")
        
        
#     for split in ['valid', 'test']:
#         with open(PATH + f'files/sty_nodup-src-{split}.txt') as f:
#             lines = [s.strip() for s in f.readlines()]
#             ids = [s.split('/')[-1].replace('.png', '') for s in lines]
#             file_path = [PATH + f'images/{s}' for s in lines]
#         with open(PATH + f'files/sty_nodup-tgt-{split}.txt') as f:
#             smiles_tok = [s.strip() for s in f.readlines()]
#             smiles = [s.replace(' ', '') for s in smiles_tok]
#         inchi, r_success = batch_convert_smiles_to_inchi(smiles)
#         print(split, f'SMILES to InChI: {r_success:.4f}')
#         df = pd.DataFrame({
#             'image_id': ids,
#             'file_path': file_path, 
#             'SMILES': smiles, 
#             'SMILES_atomtok': smiles_tok, 
#             'InChI': inchi
#         })
#         df.to_csv(PATH + f'sty_{split}.csv', index=False)
#         print('Max length:', max([len(s.split()) for s in smiles_tok]))
    
        
if __name__ == '__main__':
    main()