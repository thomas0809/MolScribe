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
    except:
        print(split, path, row['SMILES'])
        img = np.array([[255,255,255] * 10] * 10)
        cv2.imwrite(path, img)
    return


for split in ['train', 'valid', 'test']:
    print(split)
    df = pd.read_csv(f'data/molbank/chemdraw-data/{split}.csv')
    def convert_path(path):
        return path.replace('chemdraw-data', 'indigo-data')
    df['file_path'] = df['file_path'].apply(convert_path)
    df.to_csv(f'data/molbank/indigo-data/{split}.csv')
    with multiprocessing.Pool(16) as pool:
        pool.map(generate_image, list(df.iterrows()))
