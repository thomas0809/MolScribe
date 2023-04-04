import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

import timm

from .utils import FORMAT_INFO, to_device
from .tokenizer import SOS_ID, EOS_ID, PAD_ID, MASK_ID
from .inference import GreedySearch, BeamSearch
from .transformer import TransformerDecoder, Embeddings


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

        logits = self.output_layer(dec_out)  # (b, t, h) -> (b, t, v)
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
                tgt = tgt * mask + label * (1 - mask)
            tgt_emb, tgt_pad_mask = self.dec_embedding(tgt)
            dec_out, dec_attn, *_ = self.decoder(tgt_emb=tgt_emb, memory_bank=memory_bank,
                                                 tgt_pad_mask=tgt_pad_mask, step=step)

            attn = dec_attn.get("std", None)

            dec_logits = self.output_layer(dec_out)  # [b, t, h] => [b, t, v]
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

        results["scores"] = decode_strategy.scores  # fixed to be average of token scores
        results["token_scores"] = decode_strategy.token_scores
        results["predictions"] = decode_strategy.predictions
        results["attention"] = decode_strategy.attention
        results["hidden"] = decode_strategy.hidden
        if orig_labels is not None:
            for i in range(batch_size):
                pred = results["predictions"][i][0]
                label = orig_labels[i][1:len(pred) + 1]
                mask = label.eq(MASK_ID).long()
                pred = pred[:len(label)]
                results["predictions"][i][0] = pred * mask + label * (1 - mask)

        return results["predictions"], results['scores'], results["token_scores"], results["hidden"]

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
            batch_id = torch.arange(b).unsqueeze(1).expand_as(indices).reshape(-1)
            indices = indices.view(-1)
            hidden = hidden[batch_id, indices].view(b, -1, dim)
        b, l, dim = hidden.size()
        results = {}
        hh = torch.cat([hidden.unsqueeze(2).expand(b, l, l, dim), hidden.unsqueeze(1).expand(b, l, l, dim)], dim=3)
        results['edges'] = self.mlp(hh).permute(0, 3, 1, 2)
        if self.coords:
            results['coords'] = self.coords_mlp(hidden)
        return results


def get_edge_prediction(edge_prob):
    if not edge_prob:
        return [], []
    n = len(edge_prob)
    if n == 0:
        return [], []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(5):
                edge_prob[i][j][k] = (edge_prob[i][j][k] + edge_prob[j][i][k]) / 2
                edge_prob[j][i][k] = edge_prob[i][j][k]
            edge_prob[i][j][5] = (edge_prob[i][j][5] + edge_prob[j][i][6]) / 2
            edge_prob[i][j][6] = (edge_prob[i][j][6] + edge_prob[j][i][5]) / 2
            edge_prob[j][i][5] = edge_prob[i][j][6]
            edge_prob[j][i][6] = edge_prob[i][j][5]
    prediction = np.argmax(edge_prob, axis=2).tolist()
    score = np.max(edge_prob, axis=2).tolist()
    return prediction, score


class Decoder(nn.Module):
    """This class is a wrapper for different decoder architectures, and support multiple decoders."""

    def __init__(self, args, tokenizer):
        super(Decoder, self).__init__()
        self.args = args
        self.formats = args.formats
        self.tokenizer = tokenizer
        decoder = {}
        for format_ in args.formats:
            if format_ == 'edges':
                decoder['edges'] = GraphPredictor(args.dec_hidden_size, coords=args.continuous_coords)
            else:
                decoder[format_] = TransformerDecoderAR(args, tokenizer[format_])
        self.decoder = nn.ModuleDict(decoder)
        self.compute_confidence = args.compute_confidence

    def forward(self, encoder_out, hiddens, refs):
        """Training mode. Compute the logits with teacher forcing."""
        results = {}
        refs = to_device(refs, encoder_out.device)
        for format_ in self.formats:
            if format_ == 'edges':
                if 'atomtok_coords' in results:
                    dec_out = results['atomtok_coords'][2]
                    predictions = self.decoder['edges'](dec_out, indices=refs['atom_indices'][0])
                elif 'chartok_coords' in results:
                    dec_out = results['chartok_coords'][2]
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

    def decode(self, encoder_out, hiddens=None, refs=None, beam_size=1, n_best=1):
        """Inference mode. Call each decoder's decode method (if required), convert the output format (e.g. token to
        sequence). Beam search is not supported yet."""
        results = {}
        predictions = []
        for format_ in self.formats:
            if format_ in ['atomtok', 'atomtok_coords', 'chartok_coords']:
                max_len = FORMAT_INFO[format_]['max_len']
                results[format_] = self.decoder[format_].decode(encoder_out, beam_size, n_best, max_length=max_len)
                outputs, scores, token_scores, *_ = results[format_]
                beam_preds = [[self.tokenizer[format_].sequence_to_smiles(x.tolist()) for x in pred]
                              for pred in outputs]
                predictions = [{format_: pred[0]} for pred in beam_preds]
                if self.compute_confidence:
                    for i in range(len(predictions)):
                        # -1: y score, -2: x score, -3: symbol score
                        indices = np.array(predictions[i][format_]['indices']) - 3
                        if format_ == 'chartok_coords':
                            atom_scores = []
                            for symbol, index in zip(predictions[i][format_]['symbols'], indices):
                                atom_score = (np.prod(token_scores[i][0][index - len(symbol) + 1:index + 1])
                                              ** (1 / len(symbol))).item()
                                atom_scores.append(atom_score)
                        else:
                            atom_scores = np.array(token_scores[i][0])[indices].tolist()
                        predictions[i][format_]['atom_scores'] = atom_scores
                        predictions[i][format_]['average_token_score'] = scores[i][0]
            if format_ == 'edges':
                if 'atomtok_coords' in results:
                    atom_format = 'atomtok_coords'
                elif 'chartok_coords' in results:
                    atom_format = 'chartok_coords'
                else:
                    raise NotImplemented
                dec_out = results[atom_format][3]  # batch x n_best x len x dim
                for i in range(len(dec_out)):
                    hidden = dec_out[i][0].unsqueeze(0)  # 1 * len * dim
                    indices = torch.LongTensor(predictions[i][atom_format]['indices']).unsqueeze(0)  # 1 * k
                    pred = self.decoder['edges'](hidden, indices)  # k * k
                    prob = F.softmax(pred['edges'].squeeze(0).permute(1, 2, 0), dim=2).tolist()  # k * k * 7
                    edge_pred, edge_score = get_edge_prediction(prob)
                    predictions[i]['edges'] = edge_pred
                    if self.compute_confidence:
                        predictions[i]['edge_scores'] = edge_score
                        predictions[i]['edge_score_product'] = np.sqrt(np.prod(edge_score)).item()
                        predictions[i]['overall_score'] = predictions[i][atom_format]['average_token_score'] * \
                                                          predictions[i]['edge_score_product']
                        predictions[i][atom_format].pop('average_token_score')
                        predictions[i].pop('edge_score_product')
        return predictions
