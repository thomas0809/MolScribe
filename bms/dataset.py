import os
import cv2
import time
import random
import string
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import albumentations as A
from albumentations.pytorch import ToTensorV2

from indigo import Indigo
from indigo.renderer import IndigoRenderer

from bms.augment import SafeRotate, CropWhite, NormalizedGridDistortion
from bms.utils import PAD_ID, FORMAT_INFO, print_rank_0
from bms.chemistry import get_num_atoms, RGROUP_SYMBOLS, SUBSTITUTIONS


cv2.setNumThreads(1)

INDIGO_HYGROGEN_PROB = 0.2
INDIGO_RGROUP_PROB = 0.5
INDIGO_COMMENT_PROB = 0.3
INDIGO_DEARMOTIZE_PROB = 0.5


def get_transforms(input_size, augment=True, rotate=True, debug=False):
    trans_list = []
    if augment and rotate:
        trans_list.append(SafeRotate(limit=90, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255)))
    trans_list.append(CropWhite(pad=5))
    if augment:
        trans_list += [
            # NormalizedGridDistortion(num_steps=10, distort_limit=0.3),
            A.Downscale(scale_min=0.15, scale_max=0.3, interpolation=3),
            A.Blur(),
            A.GaussNoise()  # TODO GaussNoise applies clip [0,1]
        ]
    trans_list.append(A.Resize(input_size, input_size))
    if not debug:
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        trans_list += [
            A.ToGray(p=1),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ]
    return A.Compose(trans_list, keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))


def add_functional_group(indigo, mol, debug=False):
    # Delete functional group and add a pseudo atom with its abbrv
    substitutions = [sub for sub in SUBSTITUTIONS]
    random.shuffle(substitutions)
    for sub in substitutions:
        query = indigo.loadSmarts(sub.smarts)
        matcher = indigo.substructureMatcher(mol)
        matched_atoms_ids = set()
        for match in matcher.iterateMatches(query):
            if random.random() < sub.probability or debug:
                atoms = []
                atoms_ids = set()
                for item in query.iterateAtoms():
                    atom = match.mapAtom(item)
                    atoms.append(atom)
                    atoms_ids.add(atom.index())
                if len(matched_atoms_ids.intersection(atoms_ids)) > 0:
                    continue
                abbrv = random.choice(sub.abbrvs)
                superatom = mol.addAtom(abbrv)
                for atom in atoms:
                    for nei in atom.iterateNeighbors():
                        if nei.index() not in atoms_ids:
                            if nei.symbol() == 'H':
                                # indigo won't match explicit hydrogen, so remove them explicitly
                                atoms_ids.add(nei.index())
                            else:
                                superatom.addBond(nei, nei.bond().bondOrder())
                for id in atoms_ids:
                    mol.getAtom(id).remove()
                matched_atoms_ids = matched_atoms_ids.union(atoms_ids)
    return mol


def add_explicit_hydrogen(indigo, mol):
    atoms = []
    for atom in mol.iterateAtoms():
        try:
            hs = atom.countImplicitHydrogens()
            if hs > 0:
                atoms.append((atom, hs))
        except:
            continue
    if len(atoms) > 0 and random.random() < INDIGO_HYGROGEN_PROB:
        atom, hs = random.choice(atoms)
        for i in range(hs):
            h = mol.addAtom('H')
            h.addBond(atom, 1)
    return mol


def add_rgroup(indigo, mol, smiles):
    atoms = []
    for atom in mol.iterateAtoms():
        try:
            hs = atom.countImplicitHydrogens()
            if hs > 0:
                atoms.append(atom)
        except:
            continue
    if len(atoms) > 0 and '*' not in smiles and random.random() < INDIGO_RGROUP_PROB:
        atom = random.choice(atoms)
        symbol = random.choice(RGROUP_SYMBOLS)
        if symbol == 'Ar':
            # 'Ar' has to be 'Ar ', otherwise indigo will fail later
            r = mol.addAtom('Ar ')
        else:
            r = mol.addAtom(symbol)
        r.addBond(atom, 1)
        # new_smiles = mol.canonicalSmiles()
        # assert '*' in new_smiles
        # new_smiles = new_smiles.split(' ')[0].replace('*', f'[{symbol}]')
        # smiles = new_smiles
    return mol


def generate_output_smiles(indigo, mol):
    # TODO: if using mol.canonicalSmiles(), explicit H will be removed
    smiles = mol.smiles()
    mol = indigo.loadMolecule(smiles)
    if '*' in smiles:
        part_a, part_b = smiles.split(' ', maxsplit=1)
        assert part_b[:2] == '|$' and part_b[-2:] == '$|'
        part_b = part_b[2:-2].replace(' ', '')
        symbols = [t for t in part_b.split(';') if len(t) > 0]
        output = ''
        cnt = 0
        for i, c in enumerate(part_a):
            if c != '*':
                output += c
            else:
                output += f'[{symbols[cnt]}]'
                cnt += 1
        return mol, output
    else:
        if ' ' in smiles:
            # special cases with extension
            smiles = smiles.split(' ')[0]
        return mol, smiles


def add_comment(indigo):
    if random.random() < INDIGO_COMMENT_PROB:
        indigo.setOption('render-comment', str(random.randint(1, 20)) + random.choice(string.ascii_letters))
        indigo.setOption('render-comment-font-size', random.randint(40, 60))
        indigo.setOption('render-comment-alignment', random.choice([0, 0.5, 1]))
        indigo.setOption('render-comment-position', random.choice(['top', 'bottom']))
        indigo.setOption('render-comment-offset', random.randint(2, 30))


def get_graph(mol, img, shuffle_nodes=False):
    mol.layout()
    coords, symbols = [], []
    index_map = {}
    atoms = [atom for atom in mol.iterateAtoms()]
    if shuffle_nodes:
        random.shuffle(atoms)
    for i, atom in enumerate(atoms):
        x, y = atom.coords()
        coords.append([x, y])
        symbols.append(atom.symbol())
        index_map[atom.index()] = i
    n = len(symbols)
    edges = np.zeros((n, n), dtype=int)
    for bond in mol.iterateBonds():
        s = index_map[bond.source().index()]
        t = index_map[bond.destination().index()]
        # 1/2/3/4 : single/double/triple/aromatic
        edges[s, t] = bond.bondOrder()
        edges[t, s] = bond.bondOrder()
        if bond.bondStereo() in [5, 6]:
            edges[s, t] = bond.bondStereo()
            edges[t, s] = 11 - bond.bondStereo()
    graph = {
        'coords': coords,
        'symbols': symbols,
        'edges': edges,
        'num_atoms': len(symbols)
    }
    return graph


def generate_indigo_image(smiles, mol_augment=True, default_option=False, shuffle_nodes=False, debug=False):
    indigo = Indigo()
    renderer = IndigoRenderer(indigo)
    indigo.setOption('render-output-format', 'png')
    indigo.setOption('render-background-color', '1,1,1')
    indigo.setOption('render-stereo-style', 'none')
    indigo.setOption('render-label-mode', 'hetero')
    if not default_option:
        thickness = random.uniform(0.5, 1.5)   # limit the sum of the following two parameters to be smaller than 4
        indigo.setOption('render-relative-thickness', thickness)
        indigo.setOption('render-bond-line-width', random.uniform(1, 4 - thickness))
        indigo.setOption('render-font-family', random.choice(['Arial', 'Times', 'Courier', 'Helvetica']))
        indigo.setOption('render-label-mode', random.choice(['hetero', 'terminal-hetero']))
        indigo.setOption('render-implicit-hydrogens-visible', random.choice([True, False]))
        if random.random() < 0.1:
            indigo.setOption('render-stereo-style', 'old')
    if debug:
        indigo.setOption('render-atom-ids-visible', True)

    try:
        mol = indigo.loadMolecule(smiles)
        orig_smiles = smiles
        if mol_augment:
            if random.random() < INDIGO_DEARMOTIZE_PROB:
                mol.dearomatize()
            else:
                mol.aromatize()
            smiles = mol.canonicalSmiles()
            add_comment(indigo)
            mol = add_explicit_hydrogen(indigo, mol)
            mol = add_rgroup(indigo, mol, smiles)
            mol = add_functional_group(indigo, mol, debug)
            mol, smiles = generate_output_smiles(indigo, mol)

        buf = renderer.renderToBuffer(mol)
        img = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), 1)  # decode buffer to image
        # img = np.repeat(np.expand_dims(img, 2), 3, axis=2)  # expand to RGB
        graph = get_graph(mol, img, shuffle_nodes)
        success = True
    except Exception:
        if debug:
            raise Exception
        img = None  # np.array([[[255., 255., 255.]] * 10] * 10).astype(np.float32)
        graph = {}
        success = False
    return img, smiles, graph, success


class TrainDataset(Dataset):
    def __init__(self, args, df, tokenizer, split='train', dynamic_indigo=False):
        super().__init__()
        self.df = df
        self.args = args
        self.tokenizer = tokenizer
        if 'file_path' in df.columns:
            self.file_paths = df['file_path'].values
            if not self.file_paths[0].startswith(args.data_path):
                self.file_paths = [os.path.join(args.data_path, path) for path in df['file_path']]
        self.split = split
        self.smiles = df['SMILES'].values if 'SMILES' in df.columns else None
        self.labelled = (split == 'train')
        if self.labelled:
            self.formats = args.formats
            self.labels = {}
            for format_ in self.formats:
                if format_ in ['atomtok', 'inchi']:
                    field = FORMAT_INFO[format_]['name']
                    if field in df.columns:
                        self.labels[format_] = df[field].values
        if args.load_graph_path:
            self.load_graph = True
            file = df.attrs['file'].split('/')[-1]
            file = os.path.join(args.load_graph_path, f'prediction_{file}')
            self.pred_graph_df = pd.read_csv(file)
        else:
            self.load_graph = False
        self.transform = get_transforms(args.input_size,
                                        augment=(self.labelled and args.augment),
                                        rotate=args.rotate)
        # self.fix_transform = A.Compose([A.Transpose(p=1), A.VerticalFlip(p=1)])
        self.dynamic_indigo = (dynamic_indigo and split == 'train')
        
    def __len__(self):
        return len(self.df)

    def image_transform(self, image, coords=[]):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # .astype(np.float32)
        augmented = self.transform(image=image, keypoints=coords)
        image = augmented['image']
        if coords:
            _, height, width = image.shape
            coords = np.array(augmented['keypoints'])
            coords[:, 0] = coords[:, 0] / width
            coords[:, 1] = coords[:, 1] / height
            return image, coords
        return image

    def __getitem__(self, idx):
        ref = {}
        if self.dynamic_indigo:
            begin = time.time()
            image, smiles, graph, success = generate_indigo_image(
                self.smiles[idx],
                mol_augment=self.args.mol_augment,
                default_option=self.args.default_option,
                shuffle_nodes=self.args.shuffle_nodes)
            # raw_image = image
            end = time.time()
            if not success:
                return idx, None, {}
            image, coords = self.image_transform(image, graph['coords'])
            graph['coords'] = coords
            ref['time'] = end - begin
            if 'atomtok' in self.formats:
                max_len = FORMAT_INFO['atomtok']['max_len']
                label = self.tokenizer['atomtok'].text_to_sequence(smiles, tokenized=False)
                ref['atomtok'] = torch.LongTensor(label[:max_len])
            if 'nodes' in self.formats:
                max_len = FORMAT_INFO['nodes']['max_len']
                label = torch.LongTensor(self.tokenizer['nodes'].nodes_to_sequence(graph))
                ref['nodes'] = torch.LongTensor(label[:max_len])
            if 'edges' in self.formats and 'atomtok_coords' not in self.formats:
                ref['edges'] = torch.tensor(graph['edges'])
            if 'graph' in self.formats or self.args.patch:
                graph_ref = {
                    'coords': torch.tensor(graph['coords']),
                    'labels': torch.tensor(self.tokenizer['graph'].symbols_to_labels(graph['symbols'])),
                    'edges': torch.tensor(graph['edges'])
                }
                ref['graph'] = graph_ref
            if 'grid' in self.formats:
                ref['grid'] = torch.tensor(self.tokenizer['grid'].nodes_to_grid(graph))
            if 'atomtok_coords' in self.formats:
                self._process_atomtok_coords(ref, smiles, graph['coords'], graph['edges'],
                                             continuous_coords=self.args.continuous_coords,
                                             mask_ratio=self.args.mask_ratio)
            return idx, image, ref
        else:
            file_path = self.file_paths[idx]
            image = cv2.imread(file_path)
            image = self.image_transform(image)
            if self.labelled:
                smiles = self.smiles[idx]
                if 'atomtok' in self.formats:
                    max_len = FORMAT_INFO['atomtok']['max_len']
                    label = self.tokenizer['atomtok'].text_to_sequence(smiles, False)
                    ref['atomtok'] = torch.LongTensor(label[:max_len])
                if 'atomtok_coords' in self.formats:
                    self._process_atomtok_coords(ref, smiles, continuous_coords=self.args.continuous_coords,
                                                 mask_ratio=1)
            return idx, image, ref

    def _process_atomtok_coords(self, ref, smiles, coords=None, edges=None, continuous_coords=False, mask_ratio=0):
        max_len = FORMAT_INFO['atomtok_coords']['max_len']
        tokenizer = self.tokenizer['atomtok_coords']
        if continuous_coords:
            label, indices = tokenizer.smiles_coords_to_sequence(smiles)
        else:
            label, indices = tokenizer.smiles_coords_to_sequence(smiles, coords, mask_ratio=self.args.mask_ratio)
        ref['atomtok_coords'] = torch.LongTensor(label[:max_len])
        indices = [i for i in indices if i < max_len]
        ref['atom_indices'] = torch.LongTensor(indices)
        if continuous_coords:
            if coords is not None:
                ref['coords'] = torch.tensor(coords)
            else:
                ref['coords'] = torch.ones(len(indices), 2) * -1.
        if edges is not None:
            ref['edges'] = torch.tensor(edges)[:len(indices), :len(indices)]
        else:
            ref['edges'] = torch.ones(len(indices), len(indices), dtype=torch.long) * (-100)


class AuxTrainDataset(Dataset):

    def __init__(self, args, train_df, aux_df, tokenizer):
        super().__init__()
        self.train_dataset = TrainDataset(args, train_df, tokenizer, dynamic_indigo=args.dynamic_indigo)
        self.aux_dataset = TrainDataset(args, aux_df, tokenizer, dynamic_indigo=False)

    def __len__(self):
        return len(self.train_dataset) * 2

    def __getitem__(self, idx):
        if idx < len(self.train_dataset):
            return self.train_dataset[idx]
        else:
            worker_info = torch.utils.data.get_worker_info()
            n = len(self.aux_dataset)
            if worker_info is None:
                idx = (idx + random.randrange(n)) % n
            else:
                per_worker = int(len(self.aux_dataset) / worker_info.num_workers)
                worker_id = worker_info.id
                start = worker_id * per_worker
                idx = start + (idx + random.randrange(per_worker)) % per_worker
            return self.aux_dataset[idx]


def pad_images(imgs):
    # B, C, H, W
    max_shape = [0, 0]
    for img in imgs:
        for i in range(len(max_shape)):
            max_shape[i] = max(max_shape[i], img.shape[-1-i])
    stack = []
    for img in imgs:
        pad = []
        for i in range(len(max_shape)):
            pad = pad + [0, max_shape[i] - img.shape[-1-i]]
        stack.append(F.pad(img, pad, value=0))
    return torch.stack(stack)


def bms_collate(batch):
    ids = []
    imgs = []
    batch = [ex for ex in batch if ex[1] is not None]
    formats = list(batch[0][2].keys())
    seq_formats = [k for k in formats if k in ['atomtok', 'inchi', 'nodes', 'atomtok_coords', 'atom_indices']]
    refs = {key: [[], []] for key in seq_formats}
    for ex in batch:
        ids.append(ex[0])
        imgs.append(ex[1])
        ref = ex[2]
        for key in seq_formats:
            refs[key][0].append(ref[key])
            refs[key][1].append(torch.LongTensor([len(ref[key])]))
    # Sequence
    for key in seq_formats:
        # this padding should work for atomtok_with_coords too, each of which has shape (length, 4)
        refs[key][0] = pad_sequence(refs[key][0], batch_first=True, padding_value=PAD_ID)
        refs[key][1] = torch.stack(refs[key][1]).reshape(-1, 1)
    # Time
    # if 'time' in formats:
    #     refs['time'] = [ex[2]['time'] for ex in batch]
    if 'reweight_coef' in formats:
        refs['reweight_coef'] = torch.tensor([ex[2]['reweight_coef'] for ex in batch])
    # Graph
    if 'graph' in formats:
        refs['graph'] = [ex[2]['graph'] for ex in batch]
    # Grid
    if 'grid' in formats:
        refs['grid'] = torch.stack([ex[2]['grid'] for ex in batch])
    # Coords
    if 'coords' in formats:
        refs['coords'] = pad_sequence([ex[2]['coords'] for ex in batch], batch_first=True, padding_value=-1.)
    # Edges
    if 'edges' in formats:
        edges_list = [ex[2]['edges'] for ex in batch]
        max_len = max([len(edges) for edges in edges_list])
        refs['edges'] = torch.stack(
            [F.pad(edges, (0, max_len-len(edges), 0, max_len-len(edges)), value=-100) for edges in edges_list], dim=0)
    return ids, pad_images(imgs), refs

