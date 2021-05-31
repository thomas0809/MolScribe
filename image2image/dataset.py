import os
import cv2
import numpy as np
import random
import torch
from torch.utils.data import DataLoader, Dataset
import multiprocessing
import albumentations as A
import rdkit
import rdkit.Chem as Chem
from rdkit.Chem import Draw
rdkit.RDLogger.DisableLog('rdApp.*')

from bms.dataset import get_transforms
from bms.utils import print_rank_0

cv2.setNumThreads(2)


class Molecule:
    def __init__(self, smiles):
        self.smiles = smiles
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

    def to_image(self, height=384, width=384):
        mol = self.to_mol()
        try:
            assert mol is not None
            img = Draw.MolsToGridImage([mol], subImgSize=(height, width), molsPerRow=1)
            return np.array(img)
        except:
            self.inchi = 'InChI=1S/H2O/h1H2'
            self.image_failed = True
            return self.to_image(height, width)

    
def convert_molecule(beam):
    for mol in beam:
        mol.to_mol()
    return beam


class TrainDataset(Dataset):
    def __init__(self, args, df, beam_file, tokenizer, split='train'):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.file_paths = df['file_path'].values
        self.height = args.input_size
        self.width = args.input_size
        if split in ['train', 'valid']:
            self.beam_size = args.train_beam_size
        else:
            self.beam_size = args.beam_size
        self.split = split
        self.labelled = (split == 'train' or split == 'valid')
        if self.labelled:
            self.inchi = df['InChI'].values
            self.smiles = df['SMILES'].values
        # Read the beam search results
        print_rank_0("Read beam search results...")
        with open(beam_file) as f:
            beams = []
            idx = -1
            for line in f:
                if line.startswith('ID'):
                    idx += 1
                    if idx >= len(df):
                        break
                    beams.append([])
                    continue
                mol = Molecule(line.strip())
                beams[idx].append(mol)
        self.beams = beams
        self.beam_converted = [False] * len(beams)
        # Image transforms
        self.transform = get_transforms(args, split == 'train')
        self.fix_transform = A.Compose([A.Transpose(p=1), A.VerticalFlip(p=1)])

    def __len__(self):
        return len(self.df)
    
    def process_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        h, w, _ = image.shape
        if h > w:
            image = self.fix_transform(image=image)['image']
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image
    
    def get_beam(self, idx):
        # Test set: do not change
        if not self.labelled:
            return self.beams[idx]
        
        if not self.beam_converted[idx]:
            self.beam_converted[idx] = True
            gold_smiles = self.smiles[idx]
            gold_mol = Molecule(gold_smiles)
            gold_mol.to_mol()
            beam = self.beams[idx]
            new_beam = [gold_mol]
            for pred_mol in beam:
                pred_mol.to_mol()
                # Only keep valid molecules which are different to the ground truth
                if pred_mol.inchi != gold_mol.inchi and pred_mol.is_valid:
                    new_beam.append(pred_mol)
            if len(new_beam) == 1:
                new_beam.append(Molecule('C'))
            # Valid set: pad the beam
            if self.split == 'valid':
                # Repeat other molecules
                while len(new_beam) < self.beam_size:
                    new_beam += new_beam[1:]
                new_beam = new_beam[:self.beam_size]
            self.beams[idx] = new_beam
        
        if self.split == 'valid':
            return self.beams[idx][:self.beam_size]
        
        beam = [self.beams[idx][0]] + random.choices(self.beams[idx][1:], k=self.beam_size-1)
        return beam
        
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)
        image = self.process_image(image)
        beam = self.get_beam(idx)
#         for mol in beam:
#             print(mol.smiles, mol.inchi)
#             x = mol.to_image(self.height, self.width)
        beam_image = torch.stack([self.process_image(mol.to_image(self.height, self.width)) for mol in beam])
        return idx, image, beam_image
    
    def get_num_image_failed(self):
        count = 0
        for beam in self.beams:
            for mol in beam:
                count += mol.image_failed
        return count

    def save_images(self, save_path, num_images=1):
        for i in range(num_images):
            path = os.path.join(save_path, str(i))
            os.makedirs(path, exist_ok=True)
            _, image, beam_image = self.__getitem__(i)
            def _convert(image):
                return np.clip(image, 0, 1) * 255
            cv2.imwrite(os.path.join(path, 'input.png'), _convert(image.permute(1, 2, 0).numpy()))
            for j, img in enumerate(beam_image):
                cv2.imwrite(os.path.join(path, f'beam_{j}.png'), _convert(img.permute(1, 2, 0).numpy()))
            with open(os.path.join(path, 'molecule.txt'), 'w') as f:
                f.write(self.inchi[i] + '\n')
                f.write('-'*20 + '\n')
                for mol in self.beams[i]:
                    f.write(mol.inchi + '\n')
                f.write('\n')
                f.write(self.smiles[i] + '\n')
                f.write('-'*20 + '\n')
                for mol in self.beams[i]:
                    f.write(mol.smiles + '\n')
        return

