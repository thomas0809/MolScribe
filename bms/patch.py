import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


class ImagePatch(nn.Module):

    def __init__(self, args, patch_size, num_classes):
        super().__init__()
        self.args = args
        self.patch_size = patch_size
        self.embedding = nn.Embedding(num_classes, patch_size * patch_size * 3)

    def add_patches(self, image, coords, labels):
        C, H, W = image.shape
        device = image.device
        labels = labels.to(device)
        patch_size = self.patch_size
        patches = self.embedding(labels).view(-1, 3, patch_size, patch_size)
        pad_patches = []
        for i, (x, y) in enumerate(coords.tolist()):
            if random.random() < 0.05:
                continue
            if random.random() < 0.05:
                patch = random.choice(patches)
            else:
                patch = patches[i]
            x, y = np.random.normal((x * H, y * W), (4, 4))
            x = min(round(x), H - patch_size // 2)
            y = min(round(y), W - patch_size // 2)
            x1 = max(x - patch_size // 2, 0)
            y1 = max(y - patch_size // 2, 0)
            # image[:, x1:x1+patch_size, y1:y1+patch_size] += patch
            pad_patches.append(F.pad(patch, (y1, W - y1 - patch_size, x1, H - x1 - patch_size)))
        if len(pad_patches) > 0:
            image = image + torch.stack(pad_patches).sum(0)
        return image

    def forward(self, image, graph):
        B, C, H, W = image.shape
        for b in range(B):
            image[b] = self.add_patches(image[b], graph[b]['coords'], graph[b]['labels'])
        return image
