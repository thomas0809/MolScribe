import os
import sys
sys.path.append('./')
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import multiprocessing
import rdkit.Chem as Chem
from indigo import Indigo
from indigo.renderer import IndigoRenderer


indigo = Indigo()
renderer = IndigoRenderer(indigo)
indigo.setOption('render-background-color', '1,1,1')
indigo.setOption('render-stereo-style', 'none')
indigo.setOption('render-label-mode', 'hetero')

def generate_image(obj):
    i, row = obj
    path = row['file_path']
    if 'abb-images' in path:
        return
    dirname, filename = os.path.split(path)
    os.makedirs(dirname, exist_ok=True)
    try:
        smiles = row['SMILES']
        if smiles is None or type(smiles) is not str:
            smiles = ''
        mol = indigo.loadMolecule(smiles)
        renderer.renderToFile(mol, path)
        success = True
    except:
        print(path, row['SMILES'])
        success = False
        if os.path.exists(path):
            os.remove(path)
#         img = np.array([[[255,255,255]] * 10] * 10)
#         cv2.imwrite(path, img)
    return success


# Generate ChemDraw molecules

# for split in ['train', 'valid', 'test']:
#     print(split)
#     df = pd.read_csv(f'data/molbank/chemdraw-data/{split}.csv')
#     print('chemdraw:', len(df))
#     def convert_path(path):
#         return path.replace('chemdraw-data', 'indigo-data')
#     df['file_path'] = df['file_path'].apply(convert_path)
#     with multiprocessing.Pool(16) as pool:
#         success = pool.map(generate_image, list(df.iterrows()))
#     df = df[success]
#     print('indigo', len(df))
#     df.to_csv(f'data/molbank/indigo-data/{split}.csv')


# Generate PubChem molecules

# for split in ['valid', 'test']:
#     print(split)
#     df = pd.read_csv(f'data/molbank/pubchem/{split}_raw.csv')
#     print('pubchem', len(df))
#     df.rename(columns={'pubchem_cid': 'image_id'}, inplace=True)
#     df['file_path'] = [f'data/molbank/pubchem/images/{id}.png' for id in df['image_id'].values]
#     with multiprocessing.Pool(16) as pool:
#         success = pool.map(generate_image, list(df.iterrows()))
#     df = df[success]
#     print('indigo', len(df))
#     df.to_csv(f'data/molbank/pubchem/{split}.csv', index=False)


# Generate Zinc molecules

# for split in ['valid', 'test']:
#     print(split)
#     df = pd.read_csv(f'data/molbank/zinc/{split}_raw.csv')
#     print('zinc', len(df))
#     # df = df[:20000]
#     df.rename(columns={'zinc_id': 'image_id'}, inplace=True)
#     df['file_path'] = [f'data/molbank/zinc/images/{id}.png' for id in df['image_id'].values]
#     with multiprocessing.Pool(16) as pool:
#         success = pool.map(generate_image, list(df.iterrows()))
#     df = df[success]
#     print('indigo', len(df))
#     df.to_csv(f'data/molbank/zinc/{split}.csv', index=False)


# Generate USPTO test

df = pd.read_csv('data/molbank/uspto_test/raw.csv')
print('uspto_test', len(df))
def convert_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        smiles = Chem.MolToSmiles(mol, kekuleSmiles=True)
    except:
        smiles = ""
    return smiles
df['SMILES'] = [convert_smiles(smiles) for smiles in df['SMILES']]
print('valid', sum([smiles is not None and type(smiles) is str and len(smiles) > 0 for smiles in df['SMILES']]))
df['file_path'] = [f'uspto_test/chemdraw/{id}.png' for id in df['image_id']]
df.to_csv('data/molbank/uspto_test/uspto_chemdraw.csv', index=False)
df['file_path'] = [f'uspto_test/indigo/{id}.png' for id in df['image_id']]
with multiprocessing.Pool(16) as pool:
    success = pool.map(generate_image, list(df.iterrows()))
print('indigo', sum(success))
df.to_csv('data/molbank/uspto_test/uspto_indigo.csv', index=False)
