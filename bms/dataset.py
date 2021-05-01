import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
import albumentations as A
from albumentations.pytorch import ToTensorV2

from bms.utils import PAD_ID, FORMAT_INFO

cv2.setNumThreads(2)


def get_transforms(args):
    return A.Compose([
        A.Resize(args.input_size, args.input_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


class TrainDataset(Dataset):
    def __init__(self, args, df, tokenizer, labelled=True, multi_task=False):
        super().__init__()
        self.df = df
        self.tokenizer = tokenizer
        self.file_paths = df['file_path'].values
        if labelled:
            self.formats = [args.format]
            if multi_task:
                self.formats.append('inchi')
            self.labels = {}
            for format_ in self.formats:
                field = FORMAT_INFO[format_]['name']
                self.labels[format_] = df[field].values
        else:
            self.labels = None
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
        if self.labels is not None:
            ref = {}
            for format_ in self.formats:
                label = self.labels[format_][idx]
                label = self.tokenizer.text_to_sequence(label)
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


