import cv2
import random
import string
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import albumentations as A
from albumentations.pytorch import ToTensorV2

from indigo import Indigo
from indigo.renderer import IndigoRenderer

from bms.augment import ExpandSafeRotate, CropWhite, ResizePad
from bms.utils import PAD_ID, FORMAT_INFO
from bms.chemistry import get_substitutions, RGROUP_SYMBOLS

cv2.setNumThreads(1)

INDIGO_HYGROGEN_PROB = 0.1
INDIGO_RGROUP_PROB = 0.2
INDIGO_COMMENT_PROB = 0.3
INDIGO_DEARMOTIZE_PROB = 0.5


def get_transforms(args, labelled=True):
    trans_list = []
    if labelled and args.augment:
        if not args.nodes:
            trans_list.append(
                ExpandSafeRotate(limit=90, border_mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_NEAREST,
                                 value=(255, 255, 255))
            )
        trans_list += [
            A.Downscale(scale_min=0.25, scale_max=0.5),
            A.Blur(),
            A.GaussNoise()
        ]
    trans_list.append(CropWhite(pad=3))
    trans_list.append(A.Resize(args.input_size, args.input_size))
    if args.cycada:
        mean = std = [0.5, 0.5, 0.5]
    else:
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


def get_graph(mol):

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
    for i, atom in enumerate(mol.iterateAtoms()):
        x, y, z = atom.xyz()
        coords.append([x, y])
        symbols.append(atom.symbol())
        index_map[atom.index()] = i
    coords = normalize_nodes(np.array(coords))
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
    return graph


def generate_indigo_image(smiles, mol_augment=True, debug=False):
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
            add_comment(indigo)
            mol, smiles = add_explicit_hydrogen(indigo, mol, smiles)
            mol, smiles = add_rgroup(indigo, mol, smiles)
            mol = add_functional_group(indigo, mol, debug)
        buf = renderer.renderToBuffer(mol)
        img = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), 0)  # decode buffer to image
        img = np.repeat(np.expand_dims(img, 2), 3, axis=2)  # expand to RGB
        graph = get_graph(mol)
        success = True
    except Exception:
        if debug:
            raise Exception
        img = np.array([[[255., 255., 255.]] * 10] * 10).astype(np.float32)
        graph = {}
        success = False
    return img, smiles, graph, success


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
                if format_ in ['nodes', 'edges']:
                    continue
                field = FORMAT_INFO[format_]['name']
                if field in df.columns:
                    self.labels[format_] = df[field].values
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
            image, smiles, graph, success = generate_indigo_image(self.smiles[idx])
            self.total += 1
            if not success and idx != 0:
                self.failed += 1
                return self.__getitem__(0)
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
                if 'atomtok' in self.formats:
                    max_len = FORMAT_INFO['atomtok']['max_len']
                    label = torch.LongTensor(self.tokenizer['atomtok'].text_to_sequence(smiles, tokenized=False)[:max_len])
                    label_length = torch.LongTensor([len(label)])
                    ref['atomtok'] = (label, label_length)
                max_len = FORMAT_INFO['nodes']['max_len']
                label = torch.LongTensor(self.tokenizer['nodes'].nodes_to_sequence(graph)[:max_len])
                label_length = torch.LongTensor([len(label)])
                ref['nodes'] = (label, label_length)
                ref['edges'] = torch.LongTensor(graph['edges'])
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
            return idx, image


# Deprecated
class TestDataset(Dataset):
    def __init__(self, args, df):
        super().__init__()
        self.df = df
        self.file_paths = df['file_path'].values
        self.transform = get_transforms(args)
        self.fix_transform = A.Compose([A.Transpose(p=1), A.VerticalFlip(p=1)])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        h, w, _ = image.shape
        if h > w:
            image = self.fix_transform(image=image)['image']
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        return image


def bms_collate(batch):
    ids = []
    imgs = []
    formats = [k for k in batch[0][2].keys() if k != 'edges']
    refs = {key: [[], []] for key in formats}
    refs['edges'] = []
    for data_point in batch:
        ids.append(data_point[0])
        imgs.append(data_point[1])
        ref = data_point[2]
        for key in formats:
            refs[key][0].append(ref[key][0])
            refs[key][1].append(ref[key][1])
        refs['edges'].append(ref['edges'])
    for key in formats:
        refs[key][0] = pad_sequence(refs[key][0], batch_first=True, padding_value=PAD_ID)
        refs[key][1] = torch.stack(refs[key][1]).reshape(-1, 1)
    max_len = max([len(edges) for edges in refs['edges']])
    refs['edges'] = torch.stack(
        [F.pad(edges, (0, max_len-len(edges), 0, max_len-len(edges)), value=-100) for edges in refs['edges']],
        dim=0)
    return torch.LongTensor(ids), torch.stack(imgs), refs

