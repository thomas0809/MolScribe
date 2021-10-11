"""Ensemble decoding.

Decodes using multiple models simultaneously,
combining their prediction distributions by averaging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from bms.model import Encoder
from bms.model import DecoderWithAttention, MultiTaskDecoder
from bms.utils import FORMAT_INFO, SOS_ID, EOS_ID, PAD_ID
from bms.inference import GreedySearch, BeamSearch

import copy


class EnsembleEncoder(nn.Module):
    def __init__(self, model_encoders):
        super().__init__()
        self.model_encoders = nn.ModuleList(model_encoders)

    def forward(self, src):
        features = [model_encoder(src) for model_encoder in self.model_encoders]

        return features


class EnsembleMultiTaskDecoder(nn.Module):
    def __init__(self, model_decoders, formats):
        super().__init__()
        self.model_decoders = nn.ModuleList(model_decoders)
        self.formats = []
        for format_ in formats:
            if all([format_ in x.decoder for x in self.model_decoders]):
                self.formats.append(format_)
            else:
                print(f"{format_} not supported for all checkpoints (skipped).")

    def decode(self, encoder_outs, beam_size=1, n_best=1):
        """
        Args:
            :encoder_outs: ensembled encoder outputs
        """
        results = {}
        encoder_outs = [x.view(x.size(0), -1, x.size(-1)) for x in encoder_outs]
        batch_size, memory_length, encoder_dim = encoder_outs[0].shape

        for format_ in self.formats:
            # clone memory_banks
            memory_banks = copy.deepcopy(encoder_outs)
            max_len = FORMAT_INFO[format_]['max_len']
            if beam_size == 1:
                decode_strategy = GreedySearch(
                    pad=PAD_ID, bos=SOS_ID, eos=EOS_ID,
                    batch_size=batch_size,
                    min_length=0,
                    return_attention=False,
                    max_length=max_len,
                    sampling_temp=1,
                    keep_topk=1)
            else:
                decode_strategy = BeamSearch(
                    pad=PAD_ID, bos=SOS_ID, eos=EOS_ID,
                    batch_size=batch_size,
                    beam_size=beam_size,
                    n_best=n_best,
                    min_length=0,
                    return_attention=False,
                    max_length=max_len)

            fn_map_state, memory_banks[0] = decode_strategy.initialize(memory_banks[0])
            if fn_map_state is not None: # just for beam search
                for i in range(1, len(memory_banks)):
                    memory_banks[i] = fn_map_state(memory_banks[i], dim=0)

            hs, cs, hhs, ccs = [], [], [], []
            for memory_bank, mt_decoder in zip(memory_banks, self.model_decoders):
                h, c, hh, cc = mt_decoder.decoder[format_].init_hidden_state(memory_bank)
                hs.append(h)
                cs.append(c)
                hhs.append(hh)
                ccs.append(cc)

            # decoding
            for step in range(max_len):
                decoder_input = decode_strategy.current_predictions

                # Decoder forward
                log_probs = []
                alphas = []
                for i, mt_decoder in enumerate(self.model_decoders):
                    decoder = mt_decoder.decoder[format_]
                    embeddings = decoder.embedding(decoder_input)
                    attention_weighted_encoding, alpha = decoder.attention(memory_banks[i], hs[i])
                    gate = decoder.sigmoid(decoder.f_beta(hs[i]))
                    attention_weighted_encoding = gate * attention_weighted_encoding
                    x = torch.cat([embeddings, attention_weighted_encoding], dim=1)
                    preds, hs[i], cs[i], hhs[i], ccs[i] = decoder.lstm_step(
                        x, hs[i], cs[i], hhs[i], ccs[i])
                    log_probs.append(F.log_softmax(preds, dim=-1))
                    alphas.append(alpha)

                avg_log_probs = torch.stack(log_probs).mean(0)
                avg_alphas = torch.stack(alphas).mean(0)
                decode_strategy.advance(avg_log_probs, avg_alphas)
                any_finished = decode_strategy.is_finished.any()
                if any_finished:
                    decode_strategy.update_finished()
                    if decode_strategy.done:
                        break

                select_indices = decode_strategy.select_indices
                if beam_size > 1 or any_finished:
                    memory_banks = [mb.index_select(0, select_indices) for mb in memory_banks]
                    hs = [h.index_select(0, select_indices) for h in hs]
                    cs = [c.index_select(0, select_indices) for c in cs]
                    hhs = [[x.index_select(0, select_indices) for x in hh] for hh in hhs]
                    ccs = [[x.index_select(0, select_indices) for x in cc] for cc in ccs]

            results[format_] = (decode_strategy.predictions, decode_strategy.scores)

        return results


def get_ensemble_model(args, tokenizer, device, ckpts):
    encoders = []
    decoders = []

    for ckpt in ckpts:
        encoder = Encoder(
            ckpt['encoder'],
            pretrained=True,
            use_checkpoint=args.use_checkpoint)
        decoder = MultiTaskDecoder(
            formats=ckpt['formats'],
            attention_dim=args.attention_dim * args.decoder_scale,
            embed_dim=args.embed_dim * args.decoder_scale,
            encoder_dim=encoder.n_features,
            decoder_dim=args.decoder_dim * args.decoder_scale,
            dropout=args.dropout,
            n_layer=ckpt['decoder_layer'],
            tokenizer=tokenizer)
        states = load_states(args, ckpt['ckpt'])
        safe_load(encoder, states['encoder'])
        safe_load(decoder, states['decoder'])
        print_rank_0(f"Model loaded from {ckpt['ckpt']}")

        encoders.append(encoder)
        decoders.append(decoder)

    ensemble_encoder = EnsembleEncoder(encoders)
    ensemble_decoder = EnsembleMultiTaskDecoder(decoders, args.formats)
    ensemble_encoder.to(device)
    ensemble_decoder.to(device)

    if args.local_rank != -1:
        ensemble_encoder = DDP(ensemble_encoder, device_ids=[args.local_rank], output_device=args.local_rank)
        ensemble_decoder = DDP(ensemble_decoder, device_ids=[args.local_rank], output_device=args.local_rank)
        print_rank_0("DDP setup finished")

    return ensemble_encoder, ensemble_decoder
