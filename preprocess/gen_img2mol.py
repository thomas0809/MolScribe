import os
import re
import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
rdkit.RDLogger.DisableLog('rdApp.*')
from SmilesPE.pretokenizer import atomwise_tokenizer


def parse_mol_file(mol_file):
    with open(mol_file, encoding="utf8", errors='ignore') as f:
        mol_data = f.read()
        superatom = [(int(i)-1, symb) for i, symb in re.findall(r'A\s+(\d+)\s+(\S+)\s', mol_data)]
        coords = []
        edges = []
    return superatom, coords, edges


def process(dataset):
    data = []
    image_path = f"data/molbank/Img2Mol/OCSR_Review/assets/images/{dataset}/"
    ref_path = f"data/molbank/Img2Mol/OCSR_Review/assets/reference/{dataset}_mol_ref/"
    n_valid = 0
    for image_file in os.listdir(image_path):
        if image_file.endswith('.png'):
            name = image_file[:-4]
            if os.path.exists(ref_path + f"{name}.sdf"):
                path = ref_path + f"{name}.sdf"
            elif os.path.exists(ref_path + f"{name}.mol"):
                path = ref_path + f"{name}.mol"
            elif os.path.exists(ref_path + f"{name}.MOL"):
                path = ref_path + f"{name}.MOL"
            else:
                print(name + ' ref not exists')
                continue
            superatoms, coords, edges = parse_mol_file(path)
            # if len(superatom) > 0:
            #     continue
            try:
                mol = Chem.MolFromMolFile(path, sanitize=False, strictParsing=False)
                smiles = Chem.MolToSmiles(mol)
                n_valid += 1
                if len(superatoms) > 0:
                    atom_order = mol.GetProp('_smilesAtomOutputOrder')
                    atom_order = eval(atom_order)  # str -> List[int], since the Prop is a str
                    reverse_map = np.argsort(atom_order)
                    superatoms = {int(reverse_map[atom_idx]): symb for atom_idx, symb in superatoms}
                    tokens = atomwise_tokenizer(smiles)
                    atom_idx = 0
                    for i, t in enumerate(tokens):
                        if t.isalpha() or t[0] == '[' or t == '*':
                            if atom_idx in superatoms:
                                symb = superatoms[atom_idx]
                                tokens[i] = f"[{symb}]"
                            atom_idx += 1
                    smiles = ''.join(tokens)
            except:
                print(path)
                smiles = ''
            data.append({
                'image_id': name,
                'file_path': image_path + image_file,
                'SMILES': smiles
            })
    print(dataset)
    print('valid', n_valid)
    print('total', len(data))
    df = pd.DataFrame(data)
    df.to_csv(f'data/molbank/Img2Mol/{dataset}.csv', index=False)


# process('CLEF')
process('JPO')
# process('UOB')
# process('USPTO')
