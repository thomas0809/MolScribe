import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


def accuracy(logits, labels):
    preds = logits.argmax(1)
    return (preds == labels).float().mean().item()
    

class Image2ImageModel(nn.Module):
    
    def __init__(self, encoder1, encoder2, temperature=1.0):
        super().__init__()
        self.encoder1 = encoder1
        self.encoder2 = encoder2
        assert encoder1.n_features == encoder2.n_features
        self.encoder_dim = encoder1.n_features
        self.attention1 = Image2ImageAttention(self.encoder_dim, self.encoder_dim)
        self.attention2 = Image2ImageAttention(self.encoder_dim, self.encoder_dim)
        self.temperature = temperature
        
    def compute_global_logits(self, feature1, feature2):
        feature1 = torch.mean(feature1, dim=1)  # batch * encoder_dim
        feature2 = torch.mean(feature2, dim=2)  # batch * beam * encoder_dim
        feature1 = torch.cat(GatherLayer.apply(feature1), 0)
        feature2 = torch.cat(GatherLayer.apply(feature2), 0)
        batch_size, beam_size, encoder_dim = feature2.size()
        cos_similarity = nn.CosineSimilarity(dim=2)
        logits = cos_similarity(
            feature1.unsqueeze(1).expand(-1, batch_size*beam_size, -1),
            feature2.view(-1, encoder_dim).unsqueeze(0).expand(batch_size, -1, -1))  # batch * (batch*beam)
        return logits / self.temperature

    def compute_local_logits(self, feature1, feature2):
        batch_size, beam_size, _, encoder_dim = feature2.size()
        feature1 = feature1.unsqueeze(1).expand(-1, beam_size, -1, -1)
        attn_feature1 = self.attention1(feature1, feature2)
        attn_feature1 = torch.mean(attn_feature1, dim=2)  # batch * beam * encoder_dim
        attn_feature2 = self.attention2(feature2, feature1)
        attn_feature2 = torch.mean(attn_feature2, dim=2)  # batch * beam * encoder_dim
#         logits = (attn_feature1.unsqueeze(2) @ attn_feature2.unsqueeze(3)).squeeze(-1).squeeze(-1)
        cos_similarity = nn.CosineSimilarity(dim=2)
        logits = cos_similarity(attn_feature1, attn_feature2)
        return logits / self.temperature

    def forward(self, image, beam_image, criterion=None):
        batch_size, beam_size, c, h, w = beam_image.size()
        feature1 = self.encoder1(image)
        feature1 = feature1.view(batch_size, -1, self.encoder_dim)
        feature2 = self.encoder2(beam_image.view(-1, c, h, w))
        feature2 = feature2.view(batch_size, beam_size, -1, self.encoder_dim)
        local_logits = self.compute_local_logits(feature1, feature2)
        if criterion is None:
            return None, local_logits
        else:
            global_logits = self.compute_global_logits(feature1, feature2)
            return self.compute_loss(global_logits, local_logits, criterion)
    
    def compute_loss(self, global_logits, local_logits, criterion):
        batch_size = global_logits.size(0)
        beam_size = global_logits.size(1) // batch_size
        labels = torch.arange(batch_size, dtype=torch.long, device=global_logits.device) * beam_size
        global_loss = criterion(global_logits, labels)
        global_acc = accuracy(global_logits, labels)
        # glolbal and local logits have different sizes
        batch_size, beam_size = local_logits.size()
        labels = torch.zeros((batch_size,), dtype=torch.long, device=local_logits.device)
        local_loss = criterion(local_logits, labels)
        local_acc = accuracy(local_logits, labels)
        loss = global_loss + local_loss
        return loss, global_acc, local_acc
        
    
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
    

class GatherLayer(torch.autograd.Function):
    '''Gather tensors from all process, supporting backward propagation.
    '''

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) \
            for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        input, = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out
