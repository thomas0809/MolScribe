import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm
import math


class Encoder(nn.Module):
    def __init__(self, model_name='resnet18', pretrained=False):
        super().__init__()
        self.model_name = model_name
        if model_name.startswith('resnet'):
            self.cnn = timm.create_model(model_name, pretrained=pretrained)
            self.n_features = self.cnn.num_features  # encoder_dim
            self.cnn.global_pool = nn.Identity()
            self.cnn.fc = nn.Identity()
        elif model_name.startswith('swin'):
            self.transformer = timm.create_model(model_name, pretrained=pretrained)
            self.n_features = self.transformer.num_features
            self.transformer.head = nn.Identity()
        else:
            raise NotImplemented

    def forward(self, x):
        if self.model_name.startswith('resnet'):
            features = self.cnn(x)
            features = features.permute(0, 2, 3, 1)
        elif self.model_name.startswith('swin'):
            def swin_features(transformer, x):
                x = transformer.patch_embed(x)
                if transformer.absolute_pos_embed is not None:
                    x = x + transformer.absolute_pos_embed
                x = transformer.pos_drop(x)
                x = transformer.layers(x)
                x = transformer.norm(x)  # B L C
                return x
            features = swin_features(self.transformer, x)
        else:
            raise NotImplemented
        return features


class Attention(nn.Module):
    """
    Attention network for calculate attention value
    """
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: input size of encoder network
        :param decoder_dim: input size of decoder network
        :param attention_dim: input size of attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)
        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder network with attention network used for training
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, max_len, vocab_size, tokenizer, n_layer=1, encoder_dim=512, dropout=0.5):
        """
        :param attention_dim: input size of attention network
        :param embed_dim: input size of embedding network
        :param decoder_dim: input size of decoder network
        :param vocab_size: total number of characters used in training
        :param encoder_dim: input size of encoder network
        :param dropout: dropout rate
        """
        super(DecoderWithAttention, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.tokenizer = tokenizer
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
        self.embedding = nn.Embedding(vocab_size, embed_dim)  # embedding layer
        self.dropout = nn.Dropout(p=dropout)
        self.decode_step = nn.LSTMCell(embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.n_layer = n_layer
        if n_layer > 1:
            self.decode_layers = nn.ModuleList([
                nn.LSTMCell(decoder_dim, decoder_dim, bias=True) for i in range(n_layer-1)
            ])
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights()  # initialize some layers with the uniform distribution

    def init_weights(self):
        self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune=True):
        for p in self.embedding.parameters():
            p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        hh = [h for i in range(self.n_layer-1)]
        cc = [c for i in range(self.n_layer-1)]
        return h, c, hh, cc
    
    def lstm_step(self, x, h, c, hh, cc, batch_size=-1):
        if batch_size == -1:
            batch_size = h.size(0)
        h, c = self.decode_step(x, (h[:batch_size], c[:batch_size]))
        x = h
        for i in range(self.n_layer-1):
            hh[i], cc[i] = self.decode_layers[i](x, (hh[i][:batch_size], cc[i][:batch_size]))
            x = hh[i]
        preds = self.fc(self.dropout(x))
        return preds, h, c, hh, cc

    def forward(self, encoder_out, encoded_captions=None, caption_lengths=None):
        """
        :param encoder_out: output of encoder network
        :param encoded_captions: transformed sequence from character to integer
        :param caption_lengths: length of transformed sequence
        """
        device = encoder_out.device
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]
        # embedding transformed sequence for vector
        embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)
        # initialize hidden state and cell state of LSTM cell
        h, c, hh, cc = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        # set decode length by caption length - 1 because of omitting start token
        decode_lengths = (caption_lengths - 1).tolist()
#         max(decode_lengths)
        predictions = torch.zeros(batch_size, self.max_len, vocab_size).to(device)
        alphas = torch.zeros(batch_size, self.max_len, num_pixels).to(device)
        # predict sequence
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
#             h, c = self.decode_step(
#                 torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
#                 (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
#             preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            x = torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1)
            preds, h, c, hh, cc = self.lstm_step(x, h, c, hh, cc, batch_size_t)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
        decode_lengths = torch.Tensor(decode_lengths).to(device)
        return predictions, encoded_captions, decode_lengths

    def predict(self, encoder_out):
        device = encoder_out.device
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)
        # embed start tocken for LSTM input
        start_tockens = torch.ones(batch_size, dtype=torch.long).to(device) * self.tokenizer.stoi["<sos>"]
        embeddings = self.embedding(start_tockens)
        # initialize hidden state and cell state of LSTM cell
        h, c, hh, cc = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)
        predictions = torch.zeros(batch_size, self.max_len, vocab_size).to(device)
        # predict sequence
        for t in range(self.max_len):
            attention_weighted_encoding, alpha = self.attention(encoder_out, h)
            gate = self.sigmoid(self.f_beta(h))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
#             h, c = self.decode_step(
#                 torch.cat([embeddings, attention_weighted_encoding], dim=1),
#                 (h, c))  # (batch_size_t, decoder_dim)
#             preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            x = torch.cat([embeddings, attention_weighted_encoding], dim=1)
            preds, h, c, hh, cc = self.lstm_step(x, h, c, hh, cc)
            predictions[:, t, :] = preds
            if np.argmax(preds.detach().cpu().numpy()) == self.tokenizer.stoi["<eos>"]:
                break
            embeddings = self.embedding(torch.argmax(preds, -1))
        return predictions


class PositionalEncoding(nn.Module):
    """
    Position encoding used with Transformer
    """
    def __init__(self, d_model, max_len, dropout=0.1):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class DecoderWithTransformer(nn.Module):
    """
    Decoder network with Transformer.
    """
    def __init__(self, d_model, n_head, num_layers, max_len, vocab_size, device, dropout):
        super(DecoderWithTransformer, self).__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=n_head)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

        self.init_weights()  # initialize some layers with the uniform distribution

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        self.device = device

    def init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-init_range, init_range)

    def generate_square_subsequent_mask(self, sz):
        """
        Generate a square mask for the sequence.
        The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, memory, tgt, tgt_mask=None):
        """
        :param memory: output of encoder network (batch_size, H, W, C)
        :param tgt: target sequence of token indexes (batch_size, T)
        :param tgt_mask: attention mask, generate on-the-fly if None
        """
        batch_size = memory.shape[0]
        assert memory.shape[-1] == self.d_model
        memory = memory.view(batch_size, -1, memory.shape[-1]).permute(1, 0, 2)

        tgt = tgt.transpose(0, 1)
        tgt = self.embedding(tgt)
        tgt = self.pos_encoder(tgt) # / torch.sqrt(self.d_model)

        if tgt_mask is None:
            tgt_mask = self.generate_square_subsequent_mask(tgt.shape[0]).to(self.device)

        output = self.transformer_decoder(tgt, memory, tgt_mask=tgt_mask)
        output = self.fc(output) # seq_len, batch_size, vocab_size

        return output.permute(1, 0, 2)

    def predict(self, memory, tokenizer):
        batch_size = memory.shape[0]
        memory = memory.view(batch_size, -1, memory.shape[-1]).permute(1, 0, 2)

        predictions = torch.zeros((batch_size, self.max_len), dtype=torch.long).to(self.device)
        predictions[:, 0] = torch.tensor([tokenizer.stoi["<sos>"]] * batch_size).to(self.device)
        for i in range(1, self.max_len):
            # there should be a smarter way of doing this
            # we should definitely cache the encodings of the past tokens
            pred = self.embedding(predictions[:,:i].transpose(0, 1))
            pred = self.pos_encoder(pred)
            output = self.transformer_decoder(pred, memory)
            output = self.fc(output)

            predictions[:, i] = output[-1].argmax(1).reshape(-1)

        return predictions

