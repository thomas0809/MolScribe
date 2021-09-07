import rdkit.Chem as Chem 
from rdkit.Chem import AllChem 
from substitutions import substitutions
import re
from random import random

'''
Defines a function convert() that takes a SMILES string and returns
a new SMILES string where some groups have been converted to their
shorthand form. The new strings are **not** valid SMILES strings,
but are appropriate for pasting/drawing in ChemDraw.

The conversions are made stochastically with very crude 
probabilities that seemed appropriate based on how common certain
abbreviations are. These are all definede in substitutions.py

There needs to be some work done on alkyl side chains to prevent
longer chains from being written as CCCCC[Et], since that isn't
ever done in practice. Maybe fixable with recursive SMARTS.
'''

substitutions_loaded = []

def load_substitutions():
    global substitutions_loaded
    # Pre-load using MolFromSmarts
    for i, (replacement, smarts, p) in enumerate(substitutions):
        core_query = Chem.MolFromSmarts(smarts)
        core_replacement = Chem.MolFromSmiles('[*:{}]'.format(i))
        substitutions_loaded.append((
            replacement, core_query, core_replacement, p
        ))

def convert(smi):
    if not smi:
        raise ValueError('Need to enter a SMILES string')
    mol = Chem.MolFromSmiles(smi)
    if not mol:
        raise ValueError('Could not parse input SMILES string {}'.format(smi))
    if not substitutions_loaded:
        load_substitutions()
    Chem.Kekulize(mol) ###
    # Apply *all* definitions
    for (replacement, core_query, core_replacement, p) in substitutions_loaded:
        if random() < p:
            mol = Chem.ReplaceSubstructs(mol, core_query, core_replacement, True)[0]
    # Get new smiles
    new_smi = Chem.MolToSmiles(mol, True, kekuleSmiles=True) ###
    # Find which definitions were made (using atom map num placeholder)
    # and replace the SMILES strings
    replacement_ids = [int(x) for x in re.findall('\:([[0-9]+)\]', new_smi)]
    for replacement_id in replacement_ids:
        to_sub = '[*:{}]'.format(replacement_id)
        new_smi = new_smi.replace(to_sub,
            substitutions_loaded[replacement_id][0])
    return new_smi

import sys
if __name__ == '__main__':
    for smi in open(sys.argv[1]):
        smi = smi.strip().replace(" ", "")
        try:
            new_smi = convert(smi)
            if smi != new_smi:
                print("{}\t{}".format(smi, new_smi))
        except Exception as e:
            sys.stderr.write(e)

# if __name__ == '__main__':
#     prompt = ''
#     while prompt != 'done':
#         try:
#             prompt = raw_input('SMILES: ')
#             smi = prompt.strip()
#             if smi == 'done':
#                 break
#             new_smi = convert(smi)
#             if smi != new_smi:
#                 print('------> {}'.format(new_smi))
#             else:
#                 print('{} unchanged'.format(smi))
#         except Exception as e:
#             print(e)
