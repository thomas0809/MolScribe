import os
import pandas as pd
import rdkit
from rdkit import Chem
rdkit.RDLogger.DisableLog('rdApp.*')


def process(dataset):
    data = []
    image_path = f"data/molbank/Img2Mol/OCSR_Review/assets/images/{dataset}/"
    ref_path = f"data/molbank/Img2Mol/OCSR_Review/assets/reference/{dataset}_mol_ref/"
    n_valid = 0
    for image_file in os.listdir(image_path):
        if image_file.endswith('.png'):
            name = image_file[:-4]
            if os.path.exists(ref_path + f"{name}.sdf"):
                ref = Chem.SDMolSupplier(ref_path + f"{name}.sdf")
            elif os.path.exists(ref_path + f"{name}.mol"):
                ref = Chem.SDMolSupplier(ref_path + f"{name}.mol")
            else:
                print(name + ' ref not exists')
                continue
            smiles = ""
            for mol in ref:
                if mol is None:
                    continue
                n_valid += 1
                smiles = Chem.MolToSmiles(mol)
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
process('UOB')
# process('USPTO')