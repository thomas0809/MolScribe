import os
import shutil
import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import multiprocessing
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
        mol = indigo.loadMolecule(row['SMILES'])
        renderer.renderToFile(mol, path)
        success = True
    except:
        print(split, path, row['SMILES'])
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

for split in ['valid', 'test']:
    print(split)
    df = pd.read_csv(f'data/molbank/pubchem/{split}_raw.csv')
    print('pubchem', len(df))
    df.rename(columns={'pubchem_cid': 'image_id'}, inplace=True)
    df['file_path'] = [f'data/molbank/pubchem/images/{id}.png' for id in df['image_id'].values]
    with multiprocessing.Pool(16) as pool:
        success = pool.map(generate_image, list(df.iterrows()))
    df = df[success]
    print('indigo', len(df))
    df.to_csv(f'data/molbank/pubchem/{split}.csv', index=False)
