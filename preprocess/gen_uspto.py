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
def get_superatom(mol_file):
    with open(mol_file) as f:
        mol_data = f.read()
    return [(int(i)-1, symb) for i, symb in re.findall(r'A\s+(\d+)\s+(\S+)\s', mol_data)]

def convert_mol_to_smiles(mol_file, debug=False):
    try:
        mol = Chem.MolFromMolFile(mol_file, sanitize=False)
        smiles = Chem.MolToSmiles(mol)
        if mol.GetNumAtoms() < 3 or mol.GetNumAtoms() > 100:
            return None, None
        superatoms = get_superatom(mol_file)
        pseudo_smiles = smiles
        if len(superatoms) > 0:
            mappings = []
            cnt = 1
            mw = Chem.RWMol(mol)
            for atom_idx, symb in superatoms:
                atom = Chem.Atom("*")
                while f"[{cnt}*]" in pseudo_smiles:
                    cnt += 1
                atom.SetIsotope(cnt)
                mw.ReplaceAtom(atom_idx, atom)
                mappings.append((f"[{cnt}*]", f"[{symb}]"))
                cnt += 1
            pseudo_smiles = Chem.MolToSmiles(mw)
            for placeholder, symb in mappings:
                pseudo_smiles = pseudo_smiles.replace(placeholder, symb)
        return smiles, pseudo_smiles
    except Exception as e:
        if debug:
            raise e
        return None, None

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
    # for path in mol_path[:100]:
    #     print(convert_mol_to_smiles(path, debug=True))
    with multiprocessing.Pool(32) as p:
        results = p.map(convert_mol_to_smiles, mol_path, chunksize=64)
    smiles_list, pseudo_smiles_list = zip(*results)
    img_path = ['uspto_mol/'+os.path.relpath(p, BASE_MOL) for p in img_path]
    mol_path = ['uspto_mol/'+os.path.relpath(p, BASE_MOL) for p in mol_path]
    df = pd.DataFrame({'file_path': img_path,
                       'mol_path': mol_path,
                       'SMILES': smiles_list,
                       'pseudo_SMILES': pseudo_smiles_list})
    bool_list = []
    seen_set = set()
    test_df = pd.read_csv('/Mounts/rbg-storage1/users/yujieq/bms/data/molbank/Img2Mol/staker/staker.csv')
    with multiprocessing.Pool(32) as p:
        test_smiles = p.map(canonical_smiles, test_df['SMILES'], chunksize=64)
    seen_set.update(test_smiles)
    test_df = pd.read_csv('/Mounts/rbg-storage1/users/yujieq/bms/data/molbank/Img2Mol/USPTO.csv')
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
    df.to_csv('USPTO_full.csv', index=False)
    df = df[bool_list]
    df.to_csv('USPTO_train.csv', index=False)
    for file_path, mol_path in tqdm(zip(df['file_path'], df['mol_path'])):
        src_file_path = '/scratch/yujieq/' + file_path
        src_mol_path = '/scratch/yujieq/' + mol_path
        dest_file_path = '/Mounts/rbg-storage1/users/yujieq/bms/data/molbank/' + file_path
        dest_mol_path = '/Mounts/rbg-storage1/users/yujieq/bms/data/molbank/' + mol_path
        if not os.path.exists(dest_file_path):
            os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
            shutil.copy(src_file_path, dest_file_path)
            shutil.copy(src_mol_path, dest_mol_path)
    print(len(df))


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
