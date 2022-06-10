import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from bms.utils import FORMAT_INFO, SOS_ID, EOS_ID, PAD_ID, MASK_ID, to_device
from bms.inference import GreedySearch, BeamSearch
from bms.transformer import TransformerDecoder, Embeddings
from bms.chemistry import is_valid_mol, get_edge_prediction


class Encoder(nn.Module):
    def __init__(self, args, pretrained=False):
        super().__init__()
        model_name = args.encoder
        self.model_name = model_name
        if model_name.startswith('resnet'):
            self.model_type = 'resnet'
            self.cnn = timm.create_model(model_name, pretrained=pretrained)
            self.n_features = self.cnn.num_features  # encoder_dim
            self.cnn.global_pool = nn.Identity()
            self.cnn.fc = nn.Identity()
        elif model_name.startswith('swin'):
            self.model_type = 'swin'
            self.transformer = timm.create_model(model_name, pretrained=pretrained, pretrained_strict=False,
                                                 use_checkpoint=args.use_checkpoint)
            self.n_features = self.transformer.num_features
            self.transformer.head = nn.Identity()
        elif 'efficientnet' in model_name:
            self.model_type = 'efficientnet'
            self.cnn = timm.create_model(model_name, pretrained=pretrained)
            self.n_features = self.cnn.num_features
            self.cnn.global_pool = nn.Identity()
            self.cnn.classifier = nn.Identity()
        else:
            raise NotImplemented

    def swin_forward(self, transformer, x):
        x = transformer.patch_embed(x)
        if transformer.absolute_pos_embed is not None:
            x = x + transformer.absolute_pos_embed
        x = transformer.pos_drop(x)

        def layer_forward(layer, x, hiddens):
            for blk in layer.blocks:
                if not torch.jit.is_scripting() and layer.use_checkpoint:
                    x = torch.utils.checkpoint.checkpoint(blk, x)
                else:
                    x = blk(x)
            H, W = layer.input_resolution
            B, L, C = x.shape
            hiddens.append(x.view(B, H, W, C))
            if layer.downsample is not None:
                x = layer.downsample(x)
            return x, hiddens

        hiddens = []
        for layer in transformer.layers:
            x, hiddens = layer_forward(layer, x, hiddens)
        x = transformer.norm(x)  # B L C
        hiddens[-1] = x.view_as(hiddens[-1])
        return x, hiddens

    def forward(self, x, refs=None):
        if self.model_type in ['resnet', 'efficientnet']:
            features = self.cnn(x)
            features = features.permute(0, 2, 3, 1)
            hiddens = []
        elif self.model_type == 'swin':
            if 'patch' in self.model_name:
                features, hiddens = self.swin_forward(self.transformer, x)
            else:
                features, hiddens = self.transformer(x)
        else:
            raise NotImplemented
        return features, hiddens


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


class LstmDecoder(nn.Module):
    """
    Decoder network with attention network used for training
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, max_len, tokenizer, n_layer=1, encoder_dim=512, dropout=0.5):
        """
        :param attention_dim: input size of attention network
        :param embed_dim: input size of embedding network
        :param decoder_dim: input size of decoder network
        :param vocab_size: total number of characters used in training
        :param encoder_dim: input size of encoder network
        :param dropout: dropout rate
        """
        super(LstmDecoder, self).__init__()
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.embed_dim = embed_dim
        self.decoder_dim = decoder_dim
        self.max_len = max_len
        self.vocab_size = len(tokenizer)
        self.tokenizer = tokenizer
        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network
        self.embedding = nn.Embedding(self.vocab_size, embed_dim)  # embedding layer
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
        self.fc = nn.Linear(decoder_dim, self.vocab_size)  # linear layer to find scores over vocabulary
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
        max_len = max(decode_lengths)
        predictions = torch.zeros(batch_size, max_len, vocab_size, device=device)
        alphas = torch.zeros(batch_size, max_len, num_pixels, device=device)
        # predict sequence
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            x = torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1)
            preds, h, c, hh, cc = self.lstm_step(x, h, c, hh, cc, batch_size_t)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha
        return predictions, encoded_captions[:, 1:]

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
            x = torch.cat([embeddings, attention_weighted_encoding], dim=1)
            preds, h, c, hh, cc = self.lstm_step(x, h, c, hh, cc)
            predictions[:, t, :] = preds
            if np.all(np.argmax(preds.detach().cpu().numpy(), -1) == self.tokenizer.stoi["<eos>"]):
                break
            embeddings = self.embedding(torch.argmax(preds, -1))
        return predictions
    
    def decode(self, encoder_out, beam_size=1, n_best=1):
        """An alternative to `predict`, decoding with greedy or beam search.
        """
        memory_bank = encoder_out # rename
        if encoder_out.dim() == 4: # for resnet encoders
            memory_bank = memory_bank.view(encoder_out.size(0), -1, encoder_out.size(-1))  # (batch_size, memory_length, encoder_dim)
        batch_size, memory_length, encoder_dim = memory_bank.shape

        if beam_size == 1:
            decode_strategy = GreedySearch(
                pad=PAD_ID,
                bos=SOS_ID,
                eos=EOS_ID,
                batch_size=batch_size,
                min_length=0,
                return_attention=False,
                max_length=self.max_len,
                sampling_temp=1,
                keep_topk=1)
        else:
            decode_strategy = BeamSearch(
                pad=PAD_ID,
                bos=SOS_ID,
                eos=EOS_ID,
                batch_size=batch_size,
                beam_size=beam_size,
                n_best=n_best,
                min_length=0,
                return_attention=False,
                max_length=self.max_len)

        # fill in first decoding step as self.bos (self.alive_seq)
        _, memory_bank = decode_strategy.initialize(memory_bank)
        # initialize hidden state and cell state of LSTM cell
        h, c, hh, cc = self.init_hidden_state(memory_bank)

        # for beam search
        # if fn_map_state is not None:
        #     self.map_state(fn_map_state)

        for step in range(self.max_len):
            decoder_input = decode_strategy.current_predictions
            embeddings = self.embedding(decoder_input)

            # Decoder forward
            attention_weighted_encoding, alpha = self.attention(memory_bank, h)
            gate = self.sigmoid(self.f_beta(h))
            attention_weighted_encoding = gate * attention_weighted_encoding
            x = torch.cat([embeddings, attention_weighted_encoding], dim=1)
            preds, h, c, hh, cc = self.lstm_step(x, h, c, hh, cc)

            # convert preds to log_probs (critic for beam-search)
            log_probs = F.log_softmax(preds, dim=-1)

            decode_strategy.advance(log_probs, alpha)
            any_finished = decode_strategy.is_finished.any()
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices
            if beam_size > 1 or any_finished:
                memory_bank = memory_bank.index_select(0, select_indices)
                h = h.index_select(0, select_indices)
                c = c.index_select(0, select_indices)
                hh = [x.index_select(0, select_indices) for x in hh]
                cc = [x.index_select(0, select_indices) for x in cc]

        return (decode_strategy.predictions, decode_strategy.scores)
    

class TransformerDecoderBase(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.enc_trans_layer = nn.Sequential(
            nn.Linear(args.encoder_dim, args.dec_hidden_size)
            # nn.LayerNorm(args.dec_hidden_size, eps=1e-6)
        )
        self.enc_pos_emb = nn.Embedding(144, args.encoder_dim) if args.enc_pos_emb else None

        self.decoder = TransformerDecoder(
            num_layers=args.dec_num_layers,
            d_model=args.dec_hidden_size,
            heads=args.dec_attn_heads,
            d_ff=args.dec_hidden_size * 4,
            copy_attn=False,
            self_attn_type="scaled-dot",
            dropout=args.hidden_dropout,
            attention_dropout=args.attn_dropout,
            max_relative_positions=args.max_relative_positions,
            aan_useffn=False,
            full_context_alignment=False,
            alignment_layer=0,
            alignment_heads=0,
            pos_ffn_activation_fn='gelu'
        )

    def enc_transform(self, encoder_out):
        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        max_len = encoder_out.size(1)
        device = encoder_out.device
        if self.enc_pos_emb:
            pos_emb = self.enc_pos_emb(torch.arange(max_len, device=device)).unsqueeze(0)
            encoder_out = encoder_out + pos_emb
        encoder_out = self.enc_trans_layer(encoder_out)
        return encoder_out


class TransformerDecoderAR(TransformerDecoderBase):
    """Autoregressive Transformer Decoder"""

    def __init__(self, args, tokenizer):
        super().__init__(args)
        self.tokenizer = tokenizer
        self.vocab_size = len(self.tokenizer)
        self.output_layer = nn.Linear(args.dec_hidden_size, self.vocab_size, bias=True)
        self.embeddings = Embeddings(
            word_vec_size=args.dec_hidden_size,
            word_vocab_size=self.vocab_size,
            word_padding_idx=PAD_ID,
            position_encoding=True,
            dropout=args.hidden_dropout)

    def dec_embedding(self, tgt, step=None):
        pad_idx = self.embeddings.word_padding_idx
        tgt_pad_mask = tgt.data.eq(pad_idx).transpose(1, 2)  # [B, 1, T_tgt]
        emb = self.embeddings(tgt, step=step)
        assert emb.dim() == 3  # batch x len x embedding_dim
        return emb, tgt_pad_mask

    def forward(self, encoder_out, labels, label_lengths):
        """Training mode"""
        batch_size, max_len, _ = encoder_out.size()
        memory_bank = self.enc_transform(encoder_out)

        tgt = labels.unsqueeze(-1)  # (b, t, 1)
        tgt_emb, tgt_pad_mask = self.dec_embedding(tgt)
        dec_out, *_ = self.decoder(tgt_emb=tgt_emb, memory_bank=memory_bank, tgt_pad_mask=tgt_pad_mask)

        logits = self.output_layer(dec_out)    # (b, t, h) -> (b, t, v)
        return logits[:, :-1], labels[:, 1:], dec_out

    def decode(self, encoder_out, beam_size: int, n_best: int, min_length: int = 1, max_length: int = 256,
               labels=None):
        """Inference mode. Autoregressively decode the sequence. Only greedy search is supported now. Beam search is
        out-dated. The labels is used for partial prediction, i.e. part of the sequence is given. In standard decoding,
        labels=None."""
        batch_size, max_len, _ = encoder_out.size()
        memory_bank = self.enc_transform(encoder_out)
        orig_labels = labels

        if beam_size == 1:
            decode_strategy = GreedySearch(
                sampling_temp=0.0, keep_topk=1, batch_size=batch_size, min_length=min_length, max_length=max_length,
                pad=PAD_ID, bos=SOS_ID, eos=EOS_ID,
                return_attention=False, return_hidden=True)
        else:
            decode_strategy = BeamSearch(
                beam_size=beam_size, n_best=n_best, batch_size=batch_size, min_length=min_length, max_length=max_length,
                pad=PAD_ID, bos=SOS_ID, eos=EOS_ID,
                return_attention=False)

        # adapted from onmt.translate.translator
        results = {
            "predictions": None,
            "scores": None,
            "attention": None
        }

        # (2) prep decode_strategy. Possibly repeat src objects.
        _, memory_bank = decode_strategy.initialize(memory_bank=memory_bank)

        # (3) Begin decoding step by step:
        for step in range(decode_strategy.max_length):
            tgt = decode_strategy.current_predictions.view(-1, 1, 1)
            if labels is not None:
                label = labels[:, step].view(-1, 1, 1)
                mask = label.eq(MASK_ID).long()
                tgt = tgt * mask + label * (1-mask)
            tgt_emb, tgt_pad_mask = self.dec_embedding(tgt)
            dec_out, dec_attn, *_ = self.decoder(tgt_emb=tgt_emb, memory_bank=memory_bank,
                                                 tgt_pad_mask=tgt_pad_mask, step=step)

            attn = dec_attn.get("std", None)

            dec_logits = self.output_layer(dec_out)            # [b, t, h] => [b, t, v]
            dec_logits = dec_logits.squeeze(1)
            log_probs = F.log_softmax(dec_logits, dim=-1)

            if self.tokenizer.output_constraint:
                output_mask = [self.tokenizer.get_output_mask(id) for id in tgt.view(-1).tolist()]
                output_mask = torch.tensor(output_mask, device=log_probs.device)
                log_probs.masked_fill_(output_mask, -10000)

            label = labels[:, step + 1] if labels is not None and step + 1 < labels.size(1) else None
            decode_strategy.advance(log_probs, attn, dec_out, label)
            any_finished = decode_strategy.is_finished.any()
            if any_finished:
                decode_strategy.update_finished()
                if decode_strategy.done:
                    break

            select_indices = decode_strategy.select_indices
            if any_finished:
                # Reorder states.
                memory_bank = memory_bank.index_select(0, select_indices)
                if labels is not None:
                    labels = labels.index_select(0, select_indices)
                self.map_state(lambda state, dim: state.index_select(dim, select_indices))

        # TODO (zhening)
        #  decode_strategy.scores is a single score for each sequence.
        #  Add results['token_scores'], a list of scores for all steps.
        results["scores"] = decode_strategy.scores
        results["predictions"] = decode_strategy.predictions
        results["attention"] = decode_strategy.attention
        results["hidden"] = decode_strategy.hidden
        if orig_labels is not None:
            for i in range(batch_size):
                pred = results["predictions"][i][0]
                label = orig_labels[i][1:len(pred)+1]
                mask = label.eq(MASK_ID).long()
                pred = pred[:len(label)]
                results["predictions"][i][0] = pred * mask + label * (1-mask)

        return results["predictions"], results['scores'], results["hidden"]

    # adapted from onmt.decoders.transformer
    def map_state(self, fn):
        def _recursive_map(struct, batch_dim=0):
            for k, v in struct.items():
                if v is not None:
                    if isinstance(v, dict):
                        _recursive_map(v)
                    else:
                        struct[k] = fn(v, batch_dim)
        if self.decoder.state["cache"] is not None:
            _recursive_map(self.decoder.state["cache"])


class TransformerDecoderNAR(TransformerDecoderBase):

    def __init__(self, args, num_classes):
        super(TransformerDecoderNAR, self).__init__(args)
        self.dec_len = args.dec_num_queries
        dec_dim = args.dec_hidden_size
        self.query_embedding = nn.Embedding(self.dec_len, dec_dim)
        self.coords_mlp = nn.Sequential(
            nn.Linear(dec_dim, dec_dim), nn.GELU(),
            # nn.Linear(dec_dim, dec_dim), nn.GELU(),
            nn.Linear(dec_dim, 2)
        )
        self.class_mlp = nn.Sequential(
            nn.Linear(dec_dim, dec_dim), nn.GELU(),
            # nn.Linear(dec_dim, dec_dim), nn.GELU(),
            nn.Linear(dec_dim, num_classes)
        )
        self.edges_mlp = nn.Sequential(
            nn.Linear(dec_dim * 2, dec_dim), nn.GELU(),
            # nn.Linear(dec_dim, dec_dim), nn.GELU(),
            nn.Linear(dec_dim, 7)
        )

    def forward(self, encoder_out):
        batch_size, max_len, _ = encoder_out.size()
        device = encoder_out.device
        memory_bank = self.enc_transform(encoder_out)
        tgt_emb = self.query_embedding(torch.arange(self.dec_len, device=device))
        tgt_emb = tgt_emb.unsqueeze(0).expand(batch_size, -1, -1)

        dec_out, _, dec_hiddens = self.decoder(tgt_emb=tgt_emb, memory_bank=memory_bank, future=True)
        b, l, h = dec_out.size()

        def _get_predictions(hidden):
            coords_pred = self.coords_mlp(hidden)  # (b, t, h) -> (b, t, 2)
            class_pred = self.class_mlp(hidden)    # (b, t, h) -> (b, t, c)
            x = torch.cat([hidden.unsqueeze(2).expand(b, l, l, h),
                           hidden.unsqueeze(1).expand(b, l, l, h)], dim=3)
            edges_pred = self.edges_mlp(x)
            return {"coords": coords_pred, "logits": class_pred, "edges": edges_pred}

        predictions = _get_predictions(dec_out)
        predictions["aux"] = []
        for hidden in dec_hiddens[:-1]:
            predictions["aux"].append(_get_predictions(hidden))

        return predictions

    def decode(self, encoder_out):
        preds = self.forward(encoder_out)
        batch_size = encoder_out.size(0)
        outputs = []
        for i in range(batch_size):
            output = {}
            labels = preds["logits"][i].argmax(-1)
            ids = labels.ne(PAD_ID).nonzero().view(-1)  # len
            output["labels"] = labels[ids]
            output["coords"] = preds["coords"][i, ids]
            edges = preds["edges"][i].argmax(-1)
            output["edges"] = edges[ids][:, ids]
            output["edges"] = torch.zeros((len(ids), len(ids)), dtype=torch.int)
            outputs.append(output)
        return outputs


class GraphPredictor(nn.Module):

    def __init__(self, decoder_dim, coords=False):
        super(GraphPredictor, self).__init__()
        self.coords = coords
        self.mlp = nn.Sequential(
            nn.Linear(decoder_dim * 2, decoder_dim), nn.GELU(),
            nn.Linear(decoder_dim, 7)
        )
        if coords:
            self.coords_mlp = nn.Sequential(
                nn.Linear(decoder_dim, decoder_dim), nn.GELU(),
                nn.Linear(decoder_dim, 2)
            )

    def forward(self, hidden, indices=None):
        b, l, dim = hidden.size()
        if indices is None:
            index = [i for i in range(3, l, 3)]
            hidden = hidden[:, index]
        else:
            batch_id = torch.arange(b).unsqueeze(1).expand_as(indices).reshape(-1)  #.to(device)
            indices = indices.view(-1)
            hidden = hidden[batch_id, indices].view(b, -1, dim)
        b, l, dim = hidden.size()
        results = {}
        hh = torch.cat([hidden.unsqueeze(2).expand(b, l, l, dim), hidden.unsqueeze(1).expand(b, l, l, dim)], dim=3)
        results['edges'] = self.mlp(hh).permute(0, 3, 1, 2)
        if self.coords:
            results['coords'] = self.coords_mlp(hidden)
        return results


def conv3x3(in_planes, out_planes, stride=1, has_bias=False):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=has_bias)


def conv3x3_bn_relu(in_planes, out_planes, stride=1):
    return nn.Sequential(
            conv3x3(in_planes, out_planes, stride),
            nn.SyncBatchNorm(out_planes, eps=1e-6),
            nn.ReLU())


class FeaturePyramidNetwork(nn.Module):

    def __init__(self, hidden_dims, num_classes):
        super(FeaturePyramidNetwork, self).__init__()
        fpn_dim = hidden_dims[0]
        self.fpn_in = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims):
            self.fpn_in.append(nn.Sequential(
                nn.Conv2d(h_dim, fpn_dim, kernel_size=1, bias=False),
                nn.SyncBatchNorm(fpn_dim, eps=1e-6),
                nn.ReLU()
            ))
        self.fpn_out = nn.ModuleList()
        for i, h_dim in enumerate(hidden_dims[:-1]):
            self.fpn_out.append(nn.Sequential(
                conv3x3_bn_relu(fpn_dim, fpn_dim, 1)
            ))
        self.conv_fusion = conv3x3_bn_relu(len(hidden_dims) * fpn_dim, fpn_dim, 1)
        self.class_mlp = nn.Sequential(
            conv3x3_bn_relu(fpn_dim, fpn_dim, 1),
            nn.Conv2d(fpn_dim, num_classes, kernel_size=1, bias=True)
        )

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, hiddens):
        hiddens = [x.permute(0, 3, 1, 2).contiguous() for x in hiddens]
        f = self.fpn_in[-1](hiddens[-1])
        fpn_feature_list = [f]
        for i in reversed(range(len(hiddens) - 1)):
            conv_x = hiddens[i]
            conv_x = self.fpn_in[i](conv_x)  # lateral branch
            f = self._upsample_add(f, conv_x)
            fpn_feature_list.append(self.fpn_out[i](f))
        fpn_feature_list.reverse()  # [P2 - P5]

        output_size = fpn_feature_list[0].size()[2:]
        fusion_list = [fpn_feature_list[0]]
        for i in range(1, len(fpn_feature_list)):
            fusion_list.append(F.interpolate(
                fpn_feature_list[i], output_size, mode='bilinear', align_corners=False))
        fusion_out = torch.cat(fusion_list, 1)
        x = self.conv_fusion(fusion_out)  # B H W C

        class_pred = self.class_mlp(x)  # B C H W
        return class_pred


class Decoder(nn.Module):
    """This class is a wrapper for different decoder architectures, and support multiple decoders."""

    def __init__(self, args, tokenizer):
        super(Decoder, self).__init__()
        self.args = args
        self.formats = args.formats
        self.tokenizer = tokenizer
        decoder = {}
        for format_ in args.formats:
            if format_ == 'graph':
                decoder[format_] = TransformerDecoderNAR(args, tokenizer[format_].len_symbols())
            elif format_ == 'grid':
                dim = args.encoder_dim
                hidden_dims = [dim // 8, dim // 4, dim // 2, dim]
                decoder[format_] = FeaturePyramidNetwork(hidden_dims, tokenizer[format_].len_symbols())
            elif format_ == 'edges':
                decoder['edges'] = GraphPredictor(args.dec_hidden_size, coords=args.continuous_coords)
            else:
                if args.decoder == 'lstm':
                    decoder[format_] = LstmDecoder(
                        attention_dim=args.attention_dim, embed_dim=args.embed_dim, encoder_dim=args.encoder_dim,
                        decoder_dim=args.decoder_dim, max_len=FORMAT_INFO[format_]['max_len'], dropout=args.dropout,
                        n_layer=args.decoder_layer, tokenizer=tokenizer[format_])
                    args.dec_hidden_size = args.decoder_dim
                else:
                    decoder[format_] = TransformerDecoderAR(args, tokenizer[format_])
        self.decoder = nn.ModuleDict(decoder)

    def forward(self, encoder_out, hiddens, refs):
        """Training mode. Compute the logits with teacher forcing."""
        results = {}
        refs = to_device(refs, encoder_out.device)
        for format_ in self.formats:
            if format_ == 'graph':
                output = self.decoder['graph'](encoder_out)
                results['graph'] = (output, refs['graph'])
            elif format_ == 'grid':
                output = self.decoder['grid'](hiddens)
                results['grid'] = (output, refs['grid'])
            elif format_ == 'edges':
                if 'nodes' in results:
                    dec_out = results['nodes'][2]
                    predictions = self.decoder['edges'](dec_out)  # b 7 l l
                elif 'atomtok_coords' in results:
                    dec_out = results['atomtok_coords'][2]
                    predictions = self.decoder['edges'](dec_out, indices=refs['atom_indices'][0])
                else:
                    raise NotImplemented
                targets = {'edges': refs['edges']}
                if 'coords' in predictions:
                    targets['coords'] = refs['coords']
                results['edges'] = (predictions, targets)
            else:
                labels, label_lengths = refs[format_]
                results[format_] = self.decoder[format_](encoder_out, labels, label_lengths)
        return results

    def decode(self, encoder_out, hiddens, refs=None, beam_size=1, n_best=1):
        """Inference mode. Call each decoder's decode method (if required), convert the output format (e.g. token to
        sequence)."""
        results = {}
        predictions = {}
        beam_predictions = {}
        if refs is not None:
            refs = to_device(refs, encoder_out.device)
        for format_ in self.formats:
            if format_ == 'graph':
                outputs = self.decoder['graph'].decode(encoder_out)
                results['graph'] = outputs
                def _convert(x):
                    x = {k: v.tolist() for k, v in x.items()}
                    x['symbols'] = self.tokenizer['graph'].labels_to_symbols(x['labels'])
                    x.pop('labels')
                    return x
                predictions['graph'] = [_convert(pred) for pred in outputs]
            elif format_ == 'grid':
                outputs = self.decoder['grid'](hiddens)
                results['grid'] = outputs
                predictions['grid'] = [self.tokenizer['grid'].grid_to_nodes(grid.tolist())
                                       for grid in outputs.argmax(1)]
            elif format_ == 'edges':
                if 'nodes' in results:
                    dec_out = results['nodes'][2]  # batch x n_best x len x dim
                    # outputs = [[self.decoder['edges'](h.unsqueeze(0))['edges'].argmax(1).squeeze(0) for h in hs]
                    #            for hs in dec_out]
                    outputs = [[F.softmax(self.decoder['edges'](h.unsqueeze(0))['edges'].squeeze(0).permute(1, 2, 0), dim=2) for h in hs]
                               for hs in dec_out]
                    predictions['edges'] = [pred[0].tolist() for pred in outputs]
                elif 'atomtok_coords' in results:
                    dec_out = results['atomtok_coords'][2]  # batch x n_best x len x dim
                    predictions['edges'] = []
                    for i in range(len(dec_out)):
                        hidden = dec_out[i][0].unsqueeze(0)  # 1 * len * dim
                        indices = torch.LongTensor(predictions['atomtok_coords'][i]['indices']).unsqueeze(0)  # 1 * k
                        pred = self.decoder['edges'](hidden, indices)  # k * k
                        # predictions['edges'].append(pred['edges'].argmax(1).squeeze(0).tolist())
                        predictions['edges'].append(F.softmax(pred['edges'].squeeze(0).permute(1, 2, 0), dim=2).tolist())
                        if 'coords' in pred:
                            predictions['atomtok_coords'][i]['coords'] = pred['coords'].squeeze(0).tolist()
                else:
                    raise NotImplemented
                predictions['edges'] = [get_edge_prediction(prob) for prob in predictions['edges']]
                # results['edges'] = outputs     # batch x n_best x len x len
            # TODO (zhening)
            #  The following all rely on TransformerDecoderAR. Try to keep compatibility.
            elif format_ == 'nodes':
                max_len = FORMAT_INFO['nodes']['max_len']
                results['nodes'] = self.decoder['nodes'].decode(encoder_out, beam_size, n_best, max_length=max_len)
                outputs, scores, *_ = results['nodes']
                beam_preds = [[self.tokenizer['nodes'].sequence_to_smiles(x.tolist()) for x in pred] for pred in outputs]
                beam_predictions['nodes'] = (beam_preds, scores)
                predictions['nodes'] = [pred[0] for pred in beam_preds]
            elif format_ == 'atomtok_coords':
                labels = refs['atomtok_coords'][0] if self.args.predict_coords else None
                max_len = FORMAT_INFO['atomtok_coords']['max_len']
                results[format_] = self.decoder[format_].decode(encoder_out, beam_size, n_best, max_length=max_len,
                                                                labels=labels)
                outputs, scores, *_ = results[format_]
                beam_preds = [[self.tokenizer[format_].sequence_to_smiles(x.tolist()) for x in pred]
                              for pred in outputs]
                beam_predictions[format_] = (beam_preds, scores)
                predictions[format_] = [pred[0] for pred in beam_preds]
            else:
                max_len = FORMAT_INFO[format_]['max_len']
                results[format_] = self.decoder[format_].decode(encoder_out, beam_size, n_best, max_length=max_len)
                outputs, scores, *_ = results[format_]
                beam_preds = [[self.tokenizer[format_].predict_caption(x.tolist()) for x in pred] for pred in outputs]
                beam_predictions[format_] = (beam_preds, scores)
                def _pick_valid(preds, format_):
                    """Pick the top valid prediction from n_best outputs
                    """
                    best = preds[0]  # default
                    if self.args.check_validity:
                        for i, p in enumerate(preds):
                            if is_valid_mol(p, format_):
                                best = p
                                break
                    return best
                predictions[format_] = [_pick_valid(pred, format_) for pred in beam_preds]
        return predictions, beam_predictions
