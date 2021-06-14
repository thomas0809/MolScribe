import os
import json
import random
import numpy as np
import pandas as pd
import multiprocessing

import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import Draw
rdkit.RDLogger.DisableLog('rdApp.*')

from SmilesPE.pretokenizer import atomwise_tokenizer


class Molecule:
    def __init__(self, smiles):
        self.smiles = smiles
        self.atomtok = ' '.join(atomwise_tokenizer(smiles))
        self.inchi = None
        self.init = True
        self.image_failed = False
        
    def to_mol(self):
        self.init = False
        if self.inchi is not None:
            mol = Chem.MolFromInchi(self.inchi)
            if mol is not None:
                return mol
        try:
            mol = Chem.MolFromSmiles(self.smiles)
            self.inchi = Chem.MolToInchi(mol)
            self.is_valid = True
            return mol
        except:
            self.inchi = 'InChI=1S/H2O/h1H2'
            self.is_valid = False
            return Chem.MolFromInchi('InChI=1S/H2O/h1H2')

        
def convert_molecule(mol):
    if mol:
        mol.to_mol()
    return mol


def get_self_training_data(test_df, beam_file, tokenizer, thres=0.3):
    molecules = []
    with open(beam_file) as f:
        for line in f:
            pred = json.loads(line)
            if pred['score'][0] - pred['score'][1] < thres:
                molecules.append(None)
            else:
                molecules.append(Molecule(pred['text'][0]))
    with multiprocessing.Pool(8) as p:
        molecules = p.map(convert_molecule, molecules)
    flags = []
    inchi_list = []
    atomtok_list = []
    for mol in molecules:
        if mol and mol.is_valid and mol.inchi:
            flags.append(True)
            inchi_list.append(mol.inchi)
            atomtok_list.append(mol.atomtok)
        else:
            flags.append(False)
    df = test_df.loc[flags]
    df['InChI'] = inchi_list
    df['SMILES_atomtok'] = atomtok_list
    return df
