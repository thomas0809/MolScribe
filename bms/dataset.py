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

from bms.augment import ExpandSafeRotate, CropWhite, ResizePad
from bms.utils import PAD_ID, FORMAT_INFO, print_rank_0
from bms.chemistry import get_substitutions, RGROUP_SYMBOLS

from SmilesPE.pretokenizer import atomwise_tokenizer
from typing import Dict, List

cv2.setNumThreads(1)

INDIGO_HYGROGEN_PROB = 0.1
INDIGO_RGROUP_PROB = 0.2
INDIGO_COMMENT_PROB = 0.3
INDIGO_DEARMOTIZE_PROB = 0.5


def get_transforms(args, labelled=True):
    trans_list = []
    if labelled and args.augment:
        if not args.nodes and not args.patch:
            trans_list.append(
                ExpandSafeRotate(limit=90, border_mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_NEAREST,
                                 value=(255, 255, 255))
            )
        trans_list += [
            A.Downscale(scale_min=0.25, scale_max=0.5),
            A.Blur(),
            # A.GaussNoise()  # TODO GaussNoise applies clip [0,1]
        ]
    # trans_list.append(CropWhite(pad=3))
    if args.multiscale:
        if labelled:
            trans_list.append(A.LongestMaxSize([224, 256, 288, 320, 352, 384, 416, 448, 480]))
        else:
            trans_list.append(A.LongestMaxSize(480))
    else:
        trans_list.append(A.Resize(args.input_size, args.input_size))
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    trans_list += [
        A.ToGray(p=1),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ]
    return A.Compose(trans_list)


# Deprecaetd
def add_functional_group_as_sgroup(indigo, mol, debug=False):
    substitutions = get_substitutions()
    random.shuffle(substitutions)
    matcher = indigo.substructureMatcher(mol)
    matched_atoms = set()
    for abbrvs, smarts, p in substitutions:
        query = indigo.loadSmarts(smarts)
        for match in matcher.iterateMatches(query):
            if random.random() < p or debug:
                overlap = False
                for item in query.iterateAtoms():
                    atom = match.mapAtom(item).index()
                    if atom in matched_atoms:
                        overlap = True
                        break
                    matched_atoms.add(atom)
                if overlap:
                    continue
                abbrv = random.choice(abbrvs)
                mol.createSGroup("SUP", match, abbrv)
    return mol, matched_atoms


def add_functional_group(indigo, mol, debug=False):
    # Delete functional group and add a pseudo atom with its abbrv
    substitutions = get_substitutions()
    random.shuffle(substitutions)
    for abbrvs, smarts, p in substitutions:
        query = indigo.loadSmarts(smarts)
        matcher = indigo.substructureMatcher(mol)
        matched_atoms_ids = set()
        for match in matcher.iterateMatches(query):
            if random.random() < p or debug:
                atoms = []
                atoms_ids = set()
                for item in query.iterateAtoms():
                    atom = match.mapAtom(item)
                    atoms.append(atom)
                    atoms_ids.add(atom.index())
                if len(matched_atoms_ids.intersection(atoms_ids)) > 0:
                    continue
                abbrv = random.choice(abbrvs)
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


def add_explicit_hydrogen(indigo, mol, smiles):
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
    return mol, smiles


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
        new_smiles = mol.canonicalSmiles()
        assert '*' in new_smiles
        new_smiles = new_smiles.split(' ')[0].replace('*', f'[{symbol}]')
        smiles = new_smiles
    return mol, smiles


def add_comment(indigo):
    if random.random() < INDIGO_COMMENT_PROB:
        indigo.setOption('render-comment', str(random.randint(1, 20)) + random.choice(string.ascii_letters))
        indigo.setOption('render-comment-font-size', random.randint(40, 60))
        indigo.setOption('render-comment-alignment', random.choice([0, 0.5, 1]))
        indigo.setOption('render-comment-position', random.choice(['top', 'bottom']))
        indigo.setOption('render-comment-offset', random.randint(2, 30))


def get_graph(mol, img, shuffle_nodes=False):

    def normalize_nodes(nodes):
        x, y = nodes[:, 0], nodes[:, 1]
        minx, maxx = min(x), max(x)
        miny, maxy = min(y), max(y)
        x = (x - minx) / max(maxx - minx, 1e-6)
        y = (maxy - y) / max(maxy - miny, 1e-6)
        return np.stack([x, y], axis=1)

    mol.layout()
    coords, symbols = [], []
    index_map = {}
    atoms = [atom for atom in mol.iterateAtoms()]
    if shuffle_nodes:
        random.shuffle(atoms)
    height, width, _ = img.shape
    for i, atom in enumerate(atoms):
        x, y = atom.coords()
        coords.append([y/height, x/width])
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
        'edges': edges
    }
    return graph, img


def generate_indigo_image(smiles, mol_augment=True, shuffle_nodes=False, debug=False):
    indigo = Indigo()
    renderer = IndigoRenderer(indigo)
    indigo.setOption('render-output-format', 'png')
    indigo.setOption('render-background-color', '1,1,1')
    indigo.setOption('render-stereo-style', 'none')
    indigo.setOption('render-superatom-mode', 'collapse')
    indigo.setOption('render-relative-thickness', random.uniform(1, 2))
    indigo.setOption('render-bond-line-width', random.uniform(1, 3))
    indigo.setOption('render-label-mode', random.choice(['hetero', 'terminal-hetero']))
    indigo.setOption('render-implicit-hydrogens-visible', random.choice([True, False]))
    if debug:
        indigo.setOption('render-atom-ids-visible', True)

    try:
        mol = indigo.loadMolecule(smiles)
        orig_smiles = smiles
        if mol_augment:
            if random.random() < INDIGO_DEARMOTIZE_PROB:
                mol.dearomatize()
            smiles = mol.canonicalSmiles()
            # add_comment(indigo)
            mol, smiles = add_explicit_hydrogen(indigo, mol, smiles)
            mol, smiles = add_rgroup(indigo, mol, smiles)
            # mol = add_functional_group(indigo, mol, debug)
            mol = indigo.loadMolecule(smiles)  # reload, important!

        buf = renderer.renderToBuffer(mol)
        img = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), 1)  # decode buffer to image
        # img = np.repeat(np.expand_dims(img, 2), 3, axis=2)  # expand to RGB
        graph, img = get_graph(mol, img, shuffle_nodes)
        success = True
    except Exception:
        if debug:
            raise Exception
        img = None  # np.array([[[255., 255., 255.]] * 10] * 10).astype(np.float32)
        graph = {}
        success = False
    return img, smiles, graph, success


def pair_label_with_coords(label: List[int], label_length: int, smiles: str, graph: Dict
                           ) -> torch.Tensor:
    def _is_atom(_token: str):
        return _token.isalpha() or _token.startswith("[")

    tokens = ["<sos>"]
    tokens += atomwise_tokenizer(smiles)
    tokens.append("<eos>")
    tokens = tokens[:label_length]              # in case len(tokens) > max_len

    coords = iter(graph["coords"])              # List[float, float]
    label_with_coords = []
    for l, t in zip(label, tokens):
        label_with_coord = [float(l)]
        if _is_atom(t):
            try:
                label_with_coord.append(float(True))
                label_with_coord.extend(next(coords))
            except StopIteration:
                print_rank_0(f"coords has shape {len(graph['coords'])}, "
                             f"which is shorter than tokens len ({len(tokens)})")
                print_rank_0(f"SMILES: {smiles}")
                print_rank_0(f"tokens: {tokens}")
                exit(0)
        else:
            label_with_coord.extend([float(False), 0.0, 0.0])
        label_with_coords.append(label_with_coord)
    label_with_coords = torch.tensor(label_with_coords, dtype=torch.float)

    return label_with_coords


class TrainDataset(Dataset):
    def __init__(self, args, df, tokenizer, split='train'):
        super().__init__()
        self.df = df
        self.args = args
        self.tokenizer = tokenizer
        if 'file_path' in df.columns:
            self.file_paths = df['file_path'].values
        self.split = split
        self.labelled = (split == 'train')
        if self.labelled:
            self.formats = args.formats
            self.smiles = df['SMILES'].values
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
        self.transform = get_transforms(args, self.labelled)
        self.fix_transform = A.Compose([A.Transpose(p=1), A.VerticalFlip(p=1)])
        self.dynamic_indigo = (args.dynamic_indigo and split == 'train')
        self.reset()
        
    def __len__(self):
        return len(self.df)

    def reset(self):
        self.failed = 0
        self.total = 0

    def __getitem__(self, idx):
        if self.dynamic_indigo:
            begin = time.time()
            image, smiles, graph, success = generate_indigo_image(
                self.smiles[idx],
                mol_augment=self.args.mol_augment,
                shuffle_nodes=self.args.shuffle_nodes)
            end = time.time()
            raw_image = image
            self.total += 1
            if not success:
                self.failed += 1
                return idx, None, {}
        else:
            file_path = self.file_paths[idx]
            image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # h, w, _ = image.shape
        # if h > w:
        #     image = self.fix_transform(image=image)['image']
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        if self.labelled:
            ref = {}
            if self.dynamic_indigo:
                ref['time'] = end - begin
                if 'atomtok' in self.formats:
                    max_len = FORMAT_INFO['atomtok']['max_len']
                    label = torch.LongTensor(self.tokenizer['atomtok'].text_to_sequence(smiles, tokenized=False)[:max_len])
                    label_length = torch.LongTensor([len(label)])
                    ref['atomtok'] = (label, label_length)
                if 'nodes' in self.formats:
                    max_len = FORMAT_INFO['nodes']['max_len']
                    label = torch.LongTensor(self.tokenizer['nodes'].nodes_to_sequence(graph)[:max_len])
                    label_length = torch.LongTensor([len(label)])
                    ref['nodes'] = (label, label_length)
                if 'edges' in self.formats:
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
                if "atomtok_with_coords" in self.formats:
                    max_len = FORMAT_INFO['atomtok']['max_len']
                    label = self.tokenizer['atomtok'].text_to_sequence(smiles, tokenized=False)[:max_len]
                    label_length = len(label)
                    label_with_coords = pair_label_with_coords(label, label_length, smiles, graph)
                    ref["atomtok_with_coords"] = (label_with_coords, torch.LongTensor([len(label)]))
            else:
                for format_ in self.formats:
                    label = self.labels[format_][idx]
                    label = self.tokenizer[format_].text_to_sequence(label)
                    label = torch.LongTensor(label)
                    label_length = len(label)
                    label_length = torch.LongTensor([label_length])
                    ref[format_] = (label, label_length)
            return idx, image, ref
        else:
            ref = {}
            if self.load_graph:
                row = self.pred_graph_df.iloc[idx]
                ref['graph'] = {
                    'coords': torch.tensor(eval(row['node_coords'])),
                    'labels': torch.LongTensor(self.tokenizer['graph'].symbols_to_labels(eval(row['node_symbols']))),
                    # 'edges': torch.tensor(eval(row['edges']))
                }
            return idx, image, ref


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
    seq_formats = [k for k in formats if k not in ['edges', 'graph', 'grid', 'time']]
    refs = {key: [[], []] for key in seq_formats}
    for ex in batch:
        ids.append(ex[0])
        imgs.append(ex[1])
        ref = ex[2]
        for key in seq_formats:
            refs[key][0].append(ref[key][0])
            refs[key][1].append(ref[key][1])
    # Sequence
    for key in seq_formats:
        # this padding should work for atomtok_with_coords too, each of which has shape (length, 4)
        refs[key][0] = pad_sequence(refs[key][0], batch_first=True, padding_value=PAD_ID)
        refs[key][1] = torch.stack(refs[key][1]).reshape(-1, 1)
    # Time
    if 'time' in formats:
        refs['time'] = [ex[2]['time'] for ex in batch]
    # Graph
    if 'graph' in formats:
        refs['graph'] = [ex[2]['graph'] for ex in batch]
    # Grid
    if 'grid' in formats:
        refs['grid'] = torch.stack([ex[2]['grid'] for ex in batch])
    # Edges
    if 'edges' in formats:
        edges_list = [ex[2]['edges'] for ex in batch]
        max_len = max([len(edges) for edges in edges_list])
        refs['edges'] = torch.stack(
            [F.pad(edges, (0, max_len-len(edges), 0, max_len-len(edges)), value=-100) for edges in edges_list], dim=0)
    return ids, pad_images(imgs), refs

