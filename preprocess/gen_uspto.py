import urllib.request
import zipfile
import os
import re
import sys
import glob
import shutil
import multiprocessing
import pandas as pd
from tqdm import tqdm
import rdkit
import rdkit.Chem as Chem
rdkit.RDLogger.DisableLog('rdApp.*')
from SmilesPE.pretokenizer import atomwise_tokenizer
sys.path.append('.')
from bms.utils import NodeTokenizer
from bms.chemistry import RGROUP_SYMBOLS, SUBSTITUTIONS
from collections import Counter
import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')
import numpy as np


BASE = '/scratch/yujieq/uspto_grant_red/'
BASE_MOL = '/scratch/yujieq/uspto_mol/'


# Download data
def _download_file(url, output):
    if not os.path.exists(output):
        urllib.request.urlretrieve(url, output)

def download():
    for year in range(2002, 2011):
        url = f"https://bulkdata.uspto.gov/data/patent/grant/redbook/{year}/"
        f = urllib.request.urlopen(url)
        content = f.read().decode('utf-8')
        print(url)
        zip_files = re.findall(r"href=\"(I*\d\d\d\d\d\d\d\d(.ZIP|.zip|.tar))\"", content)
        print(zip_files)
        path = os.path.join(BASE, str(year))
        os.makedirs(path, exist_ok=True)
        args = []
        for file, ext in zip_files:
            output = os.path.join(path, file)
            args.append((url + file, output))
        with multiprocessing.Pool(32) as p:
            p.starmap(_download_file, args)


# Unzip
def is_zip(file):
    return file[-4:] in ['.zip', '.ZIP']

def unzip():
    for year in range(2009, 2010):
        path = os.path.join(BASE, str(year))
        for datefile in sorted(os.listdir(path)):
            if is_zip(datefile):
                if datefile < 'I20080930':
                    continue
                with zipfile.ZipFile(os.path.join(path, datefile), 'r') as zipobj:
                    zipobj.extractall(path)
                date = datefile[:-4]
                molpath = os.path.join(BASE_MOL, str(year), date)
                cnt = 0
                total = 0
                for file in glob.glob(f"{path}/project/pdds/ICEwithdraw/{date}/**/US*.ZIP"):
                    total += 1
                    with zipfile.ZipFile(file, 'r') as zipobj:
                        filelist = zipobj.namelist()
                        if any([name[-4:] in ['.mol', '.MOL'] for name in filelist]):
                            zipobj.extractall(molpath)
                            cnt += 1
                print(datefile, f"{cnt} / {total} have molecules")


# Filter
def parse_mol_file(mol_file):
    with open(mol_file) as f:
        mol_data = f.read()
        superatom = [(int(i)-1, symb) for i, symb in re.findall(r'A\s+(\d+)\s+(\S+)\s', mol_data)]
        lines = mol_data.split('\n')
        coords = []
        edges = []
        for i, line in enumerate(lines):
            if line.endswith("V2000"):
                tokens = line.split()
                num_atoms = int(tokens[0])
                num_bonds = int(tokens[1])
                for atom_line in lines[i + 1:i + 1 + num_atoms]:
                    atom_tokens = atom_line.strip().split()
                    coords.append([float(atom_tokens[0]), float(atom_tokens[1])])
                for bond_line in lines[i + 1 + num_atoms:i + 1 + num_atoms + num_bonds]:
                    bond_tokens = bond_line.strip().split()
                    start, end, bond_type, stereo = [int(token) for token in bond_tokens[:4]]
                    etype = bond_type if bond_type <= 4 else 1
                    if bond_type == 1:
                        if stereo == 1:
                            etype = 5
                        if stereo == 6:
                            etype = 6
                    edges.append((start - 1, end - 1, etype))
                break
    return superatom, coords, edges

def convert_mol_to_smiles(mol_file, debug=False):
    try:
        mol = Chem.MolFromMolFile(mol_file, sanitize=False)
        smiles = Chem.MolToSmiles(mol)
        atom_order = mol.GetProp('_smilesAtomOutputOrder')
        atom_order = eval(atom_order)  # str -> List[int], since the Prop is a str
        reverse_map = np.argsort(atom_order)
        if mol.GetNumAtoms() < 3 or mol.GetNumAtoms() > 100:
            return None, None, None, None
        superatoms, coords, edges = parse_mol_file(mol_file)
        coords = np.array(coords)
        coords = coords[atom_order].tolist()
        edges = [(int(reverse_map[start]), int(reverse_map[end]), etype) for (start, end, etype) in edges]
        pseudo_smiles = smiles
        if len(superatoms) > 0:
            superatoms = {int(reverse_map[atom_idx]): symb for atom_idx, symb in superatoms}
            tokens = atomwise_tokenizer(smiles)
            atom_idx = 0
            for i, t in enumerate(tokens):
                if t.isalpha() or t[0] == '[' or t == '*':
                    if atom_idx in superatoms:
                        symb = superatoms[atom_idx]
                        tokens[i] = f"[{symb}]"
                    atom_idx += 1
            pseudo_smiles = ''.join(tokens)
        return smiles, pseudo_smiles, coords, edges
    except Exception as e:
        if debug:
            raise e
        return None, None, None, None

def canonical_smiles(smiles):
    try:
        return Chem.CanonSmiles(smiles)
    except Exception as e:
        return smiles


def filter():
    mol_path = []
    img_path = []
    for file in sorted(glob.glob(f"{BASE_MOL}**/*.MOL", recursive=True)):
        if os.path.exists(file[:-4] + '.TIF'):
            mol_path.append(file)
            img_path.append(mol_path[-1].replace('.MOL', '.TIF'))
    print(len(mol_path))
    # for path in mol_path[:10]:
    #     print(convert_mol_to_smiles(path, debug=True))
    with multiprocessing.Pool(32) as p:
        results = p.map(convert_mol_to_smiles, mol_path, chunksize=64)
    print('Convert finish')
    smiles_list, pseudo_smiles_list, coords_list, edges_list = zip(*results)
    img_path = ['uspto_mol/'+os.path.relpath(p, BASE_MOL) for p in img_path]
    mol_path = ['uspto_mol/'+os.path.relpath(p, BASE_MOL) for p in mol_path]
    df = pd.DataFrame({'file_path': img_path,
                       'mol_path': mol_path,
                       'raw_SMILES': smiles_list,
                       'SMILES': pseudo_smiles_list,
                       'node_coords': [json.dumps(coords).replace(" ", "") for coords in coords_list],
                       'edges': [json.dumps(edges).replace(" ", "") for edges in edges_list]})
    bool_list = []
    seen_set = set()
    test_df = pd.read_csv('/Mounts/rbg-storage1/users/yujieq/molscribe/data/molbank/Img2Mol/staker/staker.csv')
    with multiprocessing.Pool(32) as p:
        test_smiles = p.map(canonical_smiles, test_df['SMILES'], chunksize=64)
    seen_set.update(test_smiles)
    test_df = pd.read_csv('/Mounts/rbg-storage1/users/yujieq/molscribe/data/molbank/Img2Mol/USPTO.csv')
    with multiprocessing.Pool(32) as p:
        test_smiles = p.map(canonical_smiles, test_df['SMILES'], chunksize=64)
    seen_set.update(test_smiles)
    for s in smiles_list:
        if s is None:
            bool_list.append(False)
        else:
            if s in seen_set:
                bool_list.append(False)
            else:
                bool_list.append(True)
                seen_set.add(s)
    # df.to_csv('USPTO_full.csv', index=False)
    print("Save csv")
    df = df[bool_list]
    print(len(df))
    df.to_csv('USPTO_train.csv', index=False)
    for file_path, mol_path in tqdm(zip(df['file_path'], df['mol_path'])):
        src_file_path = '/scratch/yujieq/' + file_path
        src_mol_path = '/scratch/yujieq/' + mol_path
        dest_file_path = '/Mounts/rbg-storage1/users/yujieq/molscribe/data/molbank/' + file_path
        dest_mol_path = '/Mounts/rbg-storage1/users/yujieq/molscribe/data/molbank/' + mol_path
        if not os.path.exists(dest_file_path):
            os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
            shutil.copy(src_file_path, dest_file_path)
            shutil.copy(src_mol_path, dest_mol_path)



# Vocab
def gen_vocab():
    uspto_df = pd.read_csv('data/molbank/uspto_mol/train.csv')
    pubchem_df = pd.read_csv('data/molbank/pubchem/train_200k.csv')
    smiles_list = uspto_df['SMILES'].tolist() + pubchem_df['SMILES'].tolist()
    counter = Counter()
    for smiles in smiles_list:
        tokens = atomwise_tokenizer(smiles)
        for t in tokens:
            counter[t] += 1
    with open('tokens.json', 'w') as f:
        json.dump(counter, f)
    sorted_counter = sorted(counter.items(), key=lambda x: x[1], reverse=True)
    texts = [k for k, v in sorted_counter[:2000]]
    missed = 0
    for symbol in RGROUP_SYMBOLS:
        if f'[{symbol}]' not in counter:
            texts.append(f'[{symbol}]')
            missed += 1
    for sub in SUBSTITUTIONS:
        for symbol in sub.abbrvs:
            if f'[{symbol}]' not in counter:
                texts.append(f'[{symbol}]')
                missed += 1
    print(missed)
    tokenizer = NodeTokenizer()
    tokenizer.fit_atom_symbols(texts)
    print(len(tokenizer))
    tokenizer.save('vocab_uspto.json')


if __name__ == "__main__":
    if sys.argv[1] == 'download':
        download()
    elif sys.argv[1] == 'unzip':
        unzip()
    elif sys.argv[1] == 'filter':
        filter()
    elif sys.argv[1] == 'vocab':
        gen_vocab()
