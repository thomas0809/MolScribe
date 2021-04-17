import os
import re
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
tqdm.pandas()
import torch
from sklearn.model_selection import StratifiedKFold

from bms.utils import Tokenizer

# ====================================================
# Preprocess functions
# ====================================================
def split_form(form):
    string = ''
    for i in re.findall(r"[A-Z][^A-Z]*", form):
        elem = re.match(r"\D+", i).group()
        num = i.replace(elem, "")
        if num == "":
            string += f"{elem} "
        else:
            string += f"{elem} {str(num)} "
    return string.rstrip(' ')

def split_form2(form):
    string = ''
    for i in re.findall(r"[a-z][^a-z]*", form):
        elem = i[0]
        num = i.replace(elem, "").replace('/', "")
        num_string = ''
        for j in re.findall(r"[0-9]+[^0-9]*", num):
            num_list = list(re.findall(r'\d+', j))
            assert len(num_list) == 1, f"len(num_list) != 1"
            _num = num_list[0]
            if j == _num:
                num_string += f"{_num} "
            else:
                extra = j.replace(_num, "")
                num_string += f"{_num} {' '.join(list(extra))} "
        string += f"/{elem} {num_string}"
    return string.rstrip(' ')

# ====================================================
# main
# ====================================================
def main():
    # ====================================================
    # Data Loading
    # ====================================================
    train = pd.read_csv('data/train_labels.csv')
    print(f'train.shape: {train.shape}')
    
    # ====================================================
    # preprocess train.csv
    # ====================================================
    train['InChI_1'] = train['InChI'].progress_apply(lambda x: x.split('/')[1])
    train['InChI_text'] = train['InChI_1'].progress_apply(split_form) + ' ' + \
                            train['InChI'].apply(lambda x: '/'.join(x.split('/')[2:])).progress_apply(split_form2).values
    
    # ====================================================
    # SMILES
    # ====================================================
    smiles = pd.read_csv('data/train_smiles.csv')
    train['SMILES'] = smiles['smiles']
    smiles_atomtok = pd.read_csv('data/train_smiles_atomtok.csv')
    train['SMILES_atomtok'] = smiles_atomtok['smiles']
    smiles_spe = pd.read_csv('data/train_smiles_spe_chembl.csv')
    train['SMILES_spe'] = smiles_spe['smiles']
    
    # ====================================================
    # create tokenizer
    # ====================================================
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train['InChI_text'].values)
    torch.save(tokenizer, 'data/tokenizer_inchi.pth')
    print('Saved tokenizer_inchi')
    print(f"tokenizer.stoi: {tokenizer.stoi}")
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train['SMILES_atomtok'].values)
    torch.save(tokenizer, 'data/tokenizer_smiles_atomtok.pth')
    print('Saved tokenizer_smiles_atomtok')
    print(f"tokenizer.stoi: {tokenizer.stoi}")
    
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train['SMILES_spe'].values)
    torch.save(tokenizer, 'data/tokenizer_smiles_spe.pth')
    print('Saved tokenizer_smiles_spe')
    print(f"tokenizer.stoi: {tokenizer.stoi}")

#     train.to_pickle('data/train.pkl')
#     print('Saved preprocessed train.pkl')
    
    folds = train.copy()
    Fold = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    for n, (train_index, val_index) in enumerate(Fold.split(folds, [1] * len(folds))):
        folds.loc[val_index, 'fold'] = int(n)
    folds['fold'] = folds['fold'].astype(int)
    
    trn_idx = folds[folds['fold'] != 0].index
    val_idx = folds[folds['fold'] == 0].index
    train_folds = folds.loc[trn_idx].reset_index(drop=True)
    valid_folds = folds.loc[val_idx].reset_index(drop=True)
    
    train_folds.to_csv('data/train_folds.csv', index=False)
    valid_folds.to_csv('data/valid_folds.csv', index=False)


if __name__ == '__main__':
    main()