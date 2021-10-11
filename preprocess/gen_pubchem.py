import random
from tqdm import tqdm
import pandas as pd
import multiprocessing
import rdkit
from rdkit.Chem import PandasTools
rdkit.RDLogger.DisableLog('rdApp.*')

BASE = '/scratch/yujieq/pubchem/'
OUTPUT = 'data/molbank/pubchem/'


def load_sdf(name):
    frame = PandasTools.LoadSDF(BASE + name)
    return frame


filenames = []

for i in range(0, 150, 5):
    name = f'Compound_{i:04d}00001_{i+5:04d}00000.sdf'
    filenames.append(name)

with multiprocessing.Pool(30) as p:
    data = p.map(load_sdf, filenames)

for name, frame in zip(filenames, data):
    print(name, len(frame))

df = pd.concat(data, ignore_index=True)
print(len(df))
df = df.sample(frac=1).reset_index(drop=True)
# df.to_csv(OUTPUT + 'pubchem.csv', index=False)

df = df[['PUBCHEM_COMPOUND_CID', 'PUBCHEM_IUPAC_INCHI', 'PUBCHEM_OPENEYE_ISO_SMILES', 'PUBCHEM_HEAVY_ATOM_COUNT']]
df.rename(columns={'PUBCHEM_COMPOUND_CID': 'pubchem_cid', 'PUBCHEM_IUPAC_INCHI': 'InChI',
                   'PUBCHEM_OPENEYE_ISO_SMILES': 'SMILES', 'PUBCHEM_HEAVY_ATOM_COUNT': 'num_atoms'},
          inplace=True)

n_train = 10000000
n_dev = 50000
n_test = 50000

train_df = df.loc[:n_train]
train_df.to_csv(OUTPUT + 'train.csv', index=False)

dev_df = df.loc[n_train:n_train+n_dev]
dev_df.to_csv(OUTPUT + 'valid.csv', index=False)

test_df = df.loc[n_train+n_dev:n_train+n_dev+n_test]
test_df.to_csv(OUTPUT + 'test.csv', index=False)

