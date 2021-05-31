import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class Image2ImageModel(nn.Module):
    
    def __init__(self, encoder1, encoder2):
        super().__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        assert encoder1.n_features == encoder2.n_features
        self.encoder_dim = encoder1.n_features
        self.attention1 = Image2ImageAttention(self.encoder_dim, self.encoder_dim)
        self.attention2 = Image2ImageAttention(self.encoder_dim, self.encoder_dim)

    def forward(self, image, beam_image):
        batch_size, beam_size, c, h, w = beam_image.size()
        feature1 = self.encoder1(image)
        feature1 = feature1.view(batch_size, -1, self.encoder_dim)
        feature2 = self.encoder2(beam_image.view(-1, c, h, w))
        feature2 = feature2.view(batch_size, beam_size, -1, self.encoder_dim)
#         encoder_dim = feature1.size(-1)
#         feature1 = feature1.view(batch_size, -1, 1)
#         feature2 = feature2.view(batch_size, beam_size, -1)
#         logits = (feature2 @ feature1).squeeze(-1)  # batch_size * beam_size
        feature1 = feature1.unsqueeze(1).expand(-1, beam_size, -1, -1)
        feature2 = feature2.view(batch_size, beam_size, -1, self.encoder_dim)
        attn_feature1 = self.attention1(feature1, feature2)
        attn_feature1 = torch.mean(attn_feature1, dim=2)  # batch * beam * encoder_dim
        attn_feature2 = self.attention2(feature2, feature1)
        attn_feature2 = torch.mean(attn_feature2, dim=2)  # batch * beam * encoder_dim
        logits = (attn_feature1.unsqueeze(2) @ attn_feature2.unsqueeze(3)).squeeze(-1).squeeze(-1)
        return logits
    
    
class Image2ImageAttention(nn.Module):
    
    def __init__(self, encoder_dim, attention_dim):
        super().__init__()
        self.query = nn.Linear(encoder_dim, attention_dim)
        self.key = nn.Linear(encoder_dim, attention_dim)
        self.value = nn.Linear(encoder_dim, encoder_dim)
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
    
    def forward(self, feature1, feature2):
        batch_size, beam_size, num_pixel, encoder_dim = feature1.size()
        q = self.query(feature1)
        k = self.key(feature2).transpose(2, 3)
        v = self.value(feature2)
        attn = (q @ k) / math.sqrt(self.attention_dim)
        attn = F.softmax(attn, dim=-1)
        feature = (attn @ v)  # batch * beam * num_pixel * encoder_dim
        return feature