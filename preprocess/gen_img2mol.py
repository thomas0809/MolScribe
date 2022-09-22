import os
import re
import pandas as pd
import rdkit
from rdkit import Chem
rdkit.RDLogger.DisableLog('rdApp.*')


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
            # superatom, coords, edges = parse_mol_file(path)
            # if len(superatom) > 0:
            #     continue
            try:
                mol = Chem.MolFromMolFile(path, sanitize=False, strictParsing=False)
                smiles = Chem.MolToSmiles(mol)
                n_valid += 1
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
# process('JPO')
# process('UOB')
process('USPTO')
