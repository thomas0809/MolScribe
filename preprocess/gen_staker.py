import sys
import pandas as pd
import rdkit
from rdkit import Chem
rdkit.RDLogger.DisableLog('rdApp.*')

from SmilesPE.pretokenizer import atomwise_tokenizer
sys.path.append('.')
from bms.chemistry import RGROUP_SYMBOLS


data = []

with open('data/molbank/Img2Mol/staker/smiles.txt') as f:
    invalid = 0
    invalid_tokens = []
    for line in f:
        idx, smiles = line.strip().split(',', 1)
        tokens = atomwise_tokenizer(smiles)
        for j, token in enumerate(tokens):
            if token[0] == '[' and token[-1] == ']':
                symbol = token[1:-1]
                if symbol[0] == 'R' and symbol[1:].isdigit():
                    tokens[j] = f'[{symbol[1:]}*]'
                elif symbol in RGROUP_SYMBOLS:
                    tokens[j] = '*'
                elif Chem.AtomFromSmiles(token) is None:
                    invalid_tokens.append(token)
                    tokens[j] = '*'
                # TODO: expand abbreviations in the groundtruth e.g. Et
        smiles = ''.join(tokens)
        data.append({
            'image_id': idx,
            'file_path': f'data/molbank/Img2Mol/staker/images/{idx}.png',
            'SMILES': smiles
        })
        if Chem.MolFromSmiles(smiles, sanitize=False) is None:
            invalid += 1
            print(smiles)
    print(invalid_tokens)
    print(invalid)

df = pd.DataFrame(data)
df.to_csv(f'data/molbank/Img2Mol/staker/staker.csv', index=False)
