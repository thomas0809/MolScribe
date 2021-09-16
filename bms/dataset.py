import cv2
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import albumentations as A
from albumentations.pytorch import ToTensorV2

from indigo import Indigo
from indigo.renderer import IndigoRenderer

from bms.augment import ExpandSafeRotate, CropWhite, ResizePad
from bms.utils import PAD_ID, FORMAT_INFO

cv2.setNumThreads(1)


def get_transforms(args, labelled=True):
    trans_list = []
    if labelled and args.augment:
    # if args.augment:
        trans_list.append(ExpandSafeRotate(limit=90, border_mode=cv2.BORDER_CONSTANT, interpolation=cv2.INTER_NEAREST, value=(255,255,255)))
        trans_list += [
            A.Downscale(scale_min=0.25, scale_max=0.5),
            A.Blur(),
            A.GaussNoise()
        ]
    if not args.no_crop_white:
        trans_list.append(CropWhite(pad=3))
    if args.resize_pad:
        trans_list.append(ResizePad(args.input_size, args.input_size, interpolation=cv2.INTER_NEAREST))
    else:
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


def add_sgroup(indigo, mol, substitutions):
    matcher = indigo.substructureMatcher(mol)
    for abbrv, smarts, p in substitutions:
        query = indigo.loadSmarts(smarts)
        for match in matcher.iterateMatches(query):
            if random.random() < p:
                mol.createSGroup("SUP", match, abbrv)
    return mol


def generate_indigo_image(smiles, substitutions):
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
    try:
        mol = indigo.loadMolecule(smiles)
        mol = add_sgroup(indigo, mol, substitutions)
        buf = renderer.renderToBuffer(mol)
        # decode buffer to image
        img = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), 0)
        # expand to RGB
        img = np.repeat(np.expand_dims(img, 2), 3, axis=2)
        success = True
    except:
        img = np.array([[[255, 255, 255]] * 10] * 10)
        success = False
    return img, success


class TrainDataset(Dataset):
    def __init__(self, args, df, tokenizer, split='train'):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.file_paths = df['file_path'].values
        self.split = split
        self.labelled = (split == 'train')
        if self.labelled:
            self.formats = args.formats
            self.smiles = df['SMILES'].values
            self.labels = {}
            for format_ in self.formats:
                field = FORMAT_INFO[format_]['name']
                self.labels[format_] = df[field].values
        self.transform = get_transforms(args, self.labelled)
        self.fix_transform = A.Compose([A.Transpose(p=1), A.VerticalFlip(p=1)])
        self.dynamic_indigo = (args.dynamic_indigo and split == 'train')
        if self.dynamic_indigo:
            from bms.substitutions import get_indigo_substitutions
            self.substitutions = get_indigo_substitutions()
        
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if self.dynamic_indigo:
            image, success = generate_indigo_image(self.smiles[idx], self.substitutions)
            if not success and idx != 0:
                return self.__getitem__(0)
        else:
            file_path = self.file_paths[idx]
            image = cv2.imread(file_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        h, w, _ = image.shape
        if h > w:
            image = self.fix_transform(image=image)['image']
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
        if self.labelled:
            ref = {}
            for format_ in self.formats:
                label = self.labels[format_][idx]
                label = self.tokenizer[format_].text_to_sequence(label)
                label = torch.LongTensor(label)
                label_length = len(label)
                label_length = torch.LongTensor([label_length])
                ref[format_] = (label, label_length)
            return image, ref
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
    imgs = []
    formats = batch[0][1].keys()
    refs = {key: [[],[]] for key in formats}
    for data_point in batch:
        imgs.append(data_point[0])
        ref = data_point[1]
        for key in formats:
            refs[key][0].append(ref[key][0])
            refs[key][1].append(ref[key][1])
    for key in refs:
        refs[key][0] = pad_sequence(refs[key][0], batch_first=True, padding_value=PAD_ID)
        refs[key][1] = torch.stack(refs[key][1]).reshape(-1, 1)
    return torch.stack(imgs), refs


