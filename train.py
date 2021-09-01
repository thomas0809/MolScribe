import os
import sys
import re
import time
import json
import math
import random
import argparse
import datetime
import collections
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.optim import Adam, AdamW, SGD
from torch.utils.data import DataLoader, Dataset, RandomSampler
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from transformers import get_scheduler

from bms.dataset import TrainDataset, TestDataset, bms_collate
from bms.model import Encoder, DecoderWithAttention, MultiTaskDecoder
from bms.ensemble import EnsembleEncoder, EnsembleMultiTaskDecoder
from bms.loss import LabelSmoothingLoss
from bms.utils import init_summary_writer, seed_torch, get_score, AverageMeter, asMinutes, timeSince, is_valid, Tokenizer, \
                      batch_convert_smiles_to_inchi, batch_convert_selfies_to_inchi, merge_inchi, FORMAT_INFO, PAD_ID, print_rank_0


import warnings 
warnings.filterwarnings('ignore')


class CFG:
    num_workers=8
    weight_decay=1e-6
    max_grad_norm=5
    attention_dim=256
    embed_dim=256
    decoder_dim=512
    dropout=0.5


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank.')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--print_freq', type=int, default=200)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--dataset', type=str, default='bms', choices=['bms','chemdraw'])
    # Model
    parser.add_argument('--encoder', type=str, default='resnet34')
    parser.add_argument('--decoder', type=str, default='lstm')
    parser.add_argument('--decoder_scale', type=int, default=1)
    parser.add_argument('--decoder_layer', type=int, default=1)
    parser.add_argument('--trunc_encoder', action='store_true')  # use the hidden states before downsample
    parser.add_argument('--encoder_ape', action='store_true')
    parser.add_argument('--no_pretrained', action='store_true')
    parser.add_argument('--use_checkpoint', action='store_true')
    # Data
    parser.add_argument('--data_path', type=str, default=None)
    parser.add_argument('--train_file', type=str, default=None)
    parser.add_argument('--valid_file', type=str, default=None)
    parser.add_argument('--test_file', type=str, default=None)
    parser.add_argument('--dynamic_indigo', action='store_true')
    parser.add_argument('--format', type=str, choices=['inchi','atomtok','spe'], default='atomtok')
    parser.add_argument('--formats', type=str, default=None)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--resize_pad', action='store_true')
    parser.add_argument('--no_crop_white', action='store_true')
    # Training
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--encoder_lr', type=float, default=1e-4)
    parser.add_argument('--decoder_lr', type=float, default=4e-4)
    parser.add_argument('--scheduler', type=str, choices=['cosine','constant'], default='cosine')
    parser.add_argument('--warmup_ratio', type=float, default=0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--load_encoder_only', action='store_true')
    parser.add_argument('--save_path', type=str, default='output/')
    parser.add_argument('--save_mode', type=str, default='best', choices=['best','all'])
    parser.add_argument('--load_ckpt', type=str, default='best')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--all_data', action='store_true', help='Use both train and valid data for training.')
    parser.add_argument('--init_scheduler', action='store_true')
    parser.add_argument('--trunc_train', action='store_true')
    parser.add_argument('--label_smoothing', type=float, default=0.0)
    parser.add_argument('--selftrain', type=str, default=None)
    parser.add_argument('--cycada', action='store_true')
    # Inference
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--n_best', type=int, default=1)
    parser.add_argument('--check_validity', action='store_true')
    parser.add_argument('--ensemble_cfg', type=str, default=None)
    args = parser.parse_args()
    return args


def load_states(args, load_path):
    if load_path.endswith('.pth'):
        path = load_path
    elif args.load_ckpt == 'best':
        path = os.path.join(load_path, f'{args.encoder}_{args.decoder}_best.pth')
    else:
        path = os.path.join(load_path, f'{args.encoder}_{args.decoder}_{args.load_ckpt}.pth')
    states = torch.load(path, map_location=torch.device('cpu'))
    return states


def safe_load(module, module_states):
    def remove_prefix(state_dict):
        return {k.replace('module.', ''): v for k,v in state_dict.items()}
    missing_keys, unexpected_keys = module.load_state_dict(remove_prefix(module_states), strict=False)
    if missing_keys:
        print_rank_0('Missing keys: ' + str(missing_keys))
    if unexpected_keys:
        print_rank_0('Unexpected keys: ' + str(unexpected_keys))
    return


def get_model(args, tokenizer, device, load_path=None, ddp=True):
    encoder = Encoder(
        args.encoder, 
        pretrained=(not args.no_pretrained and load_path is None), 
        use_checkpoint=args.use_checkpoint, 
        ape=args.encoder_ape,
        trunc_encoder=args.trunc_encoder)
    encoder_dim = encoder.n_features
    print_rank_0(f'encoder_dim: {encoder_dim}')
    decoder = MultiTaskDecoder(
        formats=args.formats,
        attention_dim=CFG.attention_dim * args.decoder_scale,
        embed_dim=CFG.embed_dim * args.decoder_scale,
        encoder_dim=encoder_dim,
        decoder_dim=CFG.decoder_dim * args.decoder_scale,
        dropout=CFG.dropout,
        n_layer=args.decoder_layer,
        tokenizer=tokenizer)
    
    if load_path:
        states = load_states(args, load_path)
        print_rank_0('Loading encoder')
        safe_load(encoder, states['encoder'])
#         if not args.load_encoder_only:
        print_rank_0('Loading decoder')
        safe_load(decoder, states['decoder'])
        print_rank_0(f"Model loaded from {load_path}")
    
    encoder.to(device)
    decoder.to(device)
    
    if args.local_rank != -1:
        encoder = DDP(encoder, device_ids=[args.local_rank], output_device=args.local_rank)
        decoder = DDP(decoder, device_ids=[args.local_rank], output_device=args.local_rank)
        print_rank_0("DDP setup finished")

    return encoder, decoder


def get_ensemble_model(args, tokenizer, device, ckpts, ddp=True):
    encoders = []
    decoders = []
    
    for ckpt in ckpts:
        encoder = Encoder(
            ckpt['encoder'],
            pretrained=True,
            use_checkpoint=args.use_checkpoint)
        decoder = MultiTaskDecoder(
            formats=ckpt['formats'],
            attention_dim=CFG.attention_dim * args.decoder_scale,
            embed_dim=CFG.embed_dim * args.decoder_scale,
            encoder_dim=encoder.n_features,
            decoder_dim=CFG.decoder_dim * args.decoder_scale,
            dropout=CFG.dropout,
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


def get_optimizer_and_scheduler(args, encoder, decoder, load_path=None):
    
    encoder_optimizer = AdamW(encoder.parameters(), lr=args.encoder_lr, weight_decay=CFG.weight_decay, amsgrad=False)
    encoder_scheduler= get_scheduler(args.scheduler, encoder_optimizer, args.num_warmup_steps, args.num_training_steps)

    decoder_optimizer = AdamW(decoder.parameters(), lr=args.decoder_lr, weight_decay=CFG.weight_decay, amsgrad=False)
    decoder_scheduler= get_scheduler(args.scheduler, decoder_optimizer, args.num_warmup_steps, args.num_training_steps)
    
    if load_path and args.resume:
        states = load_states(args, load_path)
        encoder_optimizer.load_state_dict(states['encoder_optimizer'])
        decoder_optimizer.load_state_dict(states['decoder_optimizer'])
        if args.init_scheduler:
            for group in encoder_optimizer.param_groups:
                group['lr'] = args.encoder_lr
            for group in decoder_optimizer.param_groups:
                group['lr'] = args.decoder_lr
        else:
            encoder_scheduler.load_state_dict(states['encoder_scheduler'])
            decoder_scheduler.load_state_dict(states['decoder_scheduler'])
        print_rank_0(f"Optimizer loaded from {load_path}")
        
    return encoder_optimizer, encoder_scheduler, decoder_optimizer, decoder_scheduler


def train_fn(train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch,
             encoder_scheduler, decoder_scheduler, scaler, device, global_step, SUMMARY, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    epoch_losses = AverageMeter()
    # switch to train mode
    encoder.train()
    decoder.train()
    
    start = end = time.time()
    encoder_grad_norm = decoder_grad_norm = 0

    for step, (images, refs) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        batch_size = images.size(0)
        with torch.cuda.amp.autocast(enabled=args.fp16):
            loss = 0
            features = encoder(images)
            results = decoder(features, refs)
            for format_ in args.formats:
                predictions, caps_sorted, decode_lengths = results[format_]
                targets = caps_sorted[:, 1:]
                predictions = pack_padded_sequence(predictions, decode_lengths, batch_first=True).data
                targets = pack_padded_sequence(targets, decode_lengths, batch_first=True).data
                format_loss = criterion[format_](predictions, targets)
                loss = loss + format_loss
        # record loss
        losses.update(loss.item(), batch_size)
        epoch_losses.update(loss.item(), batch_size)
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        scaler.scale(loss).backward()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            scaler.unscale_(encoder_optimizer)
            scaler.unscale_(decoder_optimizer)
            encoder_grad_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), CFG.max_grad_norm)
            decoder_grad_norm = torch.nn.utils.clip_grad_norm_(decoder.parameters(), CFG.max_grad_norm)
            scaler.step(encoder_optimizer)
            scaler.step(decoder_optimizer)
            scaler.update()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            encoder_scheduler.step()
            decoder_scheduler.step()
            global_step += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % args.print_freq == 0 or step == (len(train_loader)-1):
            print_rank_0('Epoch: [{0}][{1}/{2}] '
                  'Data {data_time.avg:.3f}s ({sum_data_time}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Enc Grad: {encoder_grad_norm:.4f}  '
                  'Dec Grad: {decoder_grad_norm:.4f}  '
                  'LR: {encoder_lr:.6f} {decoder_lr:.6f}  '
                  .format(
                   epoch+1, step, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   sum_data_time=asMinutes(data_time.sum),
                   remain=timeSince(start, float(step+1)/len(train_loader)),
                   encoder_grad_norm=encoder_grad_norm,
                   decoder_grad_norm=decoder_grad_norm,
                   encoder_lr=encoder_scheduler.get_lr()[0],
                   decoder_lr=decoder_scheduler.get_lr()[0],
                   ))
            losses.reset()
        
    return epoch_losses.avg, global_step


def valid_fn(valid_loader, encoder, decoder, tokenizer, device, args):
    
    def _pick_valid(preds, format_):
        """Pick the top valid prediction from n_best outputs
        """
        best = preds[0] # default
        if args.check_validity:
            for i, p in enumerate(preds):
                if is_valid(p, format_):
                    best = p
                    break
        return best
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # switch to evaluation mode
    if hasattr(decoder, 'module'):
        encoder = encoder.module
        decoder = decoder.module
    encoder.eval()
    decoder.eval()
    all_preds = {format_:{} for format_ in args.formats}
    final_preds = {format_:{} for format_ in args.formats}
    start = end = time.time()
    for step, (indices, images) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        indices = indices.tolist()
        images = images.to(device)
        batch_size = images.size(0)
        with torch.cuda.amp.autocast(enabled=args.fp16):
            with torch.no_grad():
                features = encoder(images)
                # predictions = decoder.predict(features)
                # replace predict -> decode, output predicted sequence directly
                predictions = decoder.decode(
                    features, beam_size=args.beam_size, n_best=args.n_best)
        for format_ in args.formats:
            # predicted_sequence = torch.argmax(predictions[format_].detach().cpu(), -1).numpy()
            # text_preds = tokenizer[format_].predict_captions(predicted_sequence)
            text_preds, scores = predictions[format_]
            text_preds = [
                [tokenizer[format_].predict_caption(x.cpu().numpy()) for x in preds] 
                for preds in text_preds
            ]
            for idx, preds, sc in zip(indices, text_preds, scores):
                all_preds[format_][idx] = (preds, sc)
            valid_preds = [_pick_valid(preds, format_) for preds in text_preds]
            for idx, preds in zip(indices, valid_preds):
                final_preds[format_][idx] = preds
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % args.print_freq == 0 or step == (len(valid_loader)-1):
            print_rank_0('EVAL: [{0}/{1}] '
                  'Data {data_time.avg:.3f}s ({sum_data_time}) '
                  'Elapsed {remain:s} '
                  .format(
                   step, len(valid_loader), batch_time=batch_time,
                   data_time=data_time,
                   sum_data_time=asMinutes(data_time.sum),
                   remain=timeSince(start, float(step+1)/len(valid_loader))))
    return all_preds, final_preds


def train_loop(args, train_df, valid_df, tokenizer, save_path):
    
    SUMMARY = None
    
    if args.local_rank == 0 and not args.debug:
        os.makedirs(save_path, exist_ok=True)
        SUMMARY = init_summary_writer(save_path)
        
    print_rank_0("========== training ==========")
        
    device = args.device

    # ====================================================
    # loader
    # ====================================================

    train_dataset = TrainDataset(args, train_df, tokenizer, split='train')
    if args.local_rank != -1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, 
                              sampler=train_sampler,
                              num_workers=CFG.num_workers, 
                              pin_memory=True,
                              drop_last=True, 
                              collate_fn=bms_collate)
    
    args.num_training_steps = args.epochs * (len(train_loader) // args.gradient_accumulation_steps)
    args.num_warmup_steps = int(args.num_training_steps * args.warmup_ratio)

    # ====================================================
    # model & optimizer
    # ====================================================
    encoder, decoder = get_model(args, tokenizer, device, load_path=args.load_path)
    
    encoder_optimizer, encoder_scheduler, decoder_optimizer, decoder_scheduler = \
        get_optimizer_and_scheduler(args, encoder, decoder, load_path=args.load_path)
    
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        
    # ====================================================
    # loop
    # ====================================================
    criterion = {}
    for format_ in args.formats:
        if args.label_smoothing > 0:
            criterion[format_] = LabelSmoothingLoss(
                args.label_smoothing, len(tokenizer[format_]), ignore_index=PAD_ID)
        else:
            criterion[format_] = nn.CrossEntropyLoss(ignore_index=PAD_ID)
        criterion[format_].to(device)

    best_score = np.inf
    best_loss = np.inf
    
    global_step = encoder_scheduler.last_epoch
    start_epoch = global_step // (len(train_loader) // args.gradient_accumulation_steps)

    for epoch in range(start_epoch, args.epochs):
        
        if args.local_rank != -1:
            dist.barrier()
            train_sampler.set_epoch(epoch)
        
        start_time = time.time()
        encoder_lr = encoder_scheduler.get_lr()[0]
        decoder_lr = decoder_scheduler.get_lr()[0]
        
        # train
        avg_loss, global_step = train_fn(
            train_loader, encoder, decoder, criterion, encoder_optimizer, decoder_optimizer, epoch, 
            encoder_scheduler, decoder_scheduler, scaler, device, global_step, SUMMARY, args)
        
        # eval
        scores = inference(args, valid_df, tokenizer, encoder, decoder, save_path, split='valid')
        
        if args.local_rank != 0:
            continue
    
        elapsed = time.time() - start_time

        print_rank_0(f'Epoch {epoch+1} - Time: {elapsed:.0f}s')
        print_rank_0(f'Epoch {epoch+1} - Score: ' + json.dumps(scores))
        
        if SUMMARY:
            SUMMARY.add_scalar('train/loss', avg_loss, epoch)
            SUMMARY.add_scalar('train/encoder_lr', encoder_lr, epoch)
            SUMMARY.add_scalar('train/decoder_lr', decoder_lr, epoch)
            for key in scores:
                SUMMARY.add_scalar(f'valid/{key}', scores[key], epoch)

        save_obj = {'encoder': encoder.state_dict(), 
                    'encoder_optimizer': encoder_optimizer.state_dict(), 
                    'encoder_scheduler': encoder_scheduler.state_dict(), 
                    'decoder': decoder.state_dict(),
                    'decoder_optimizer': decoder_optimizer.state_dict(), 
                    'decoder_scheduler': decoder_scheduler.state_dict(),
                    'global_step': global_step
                   }
        if args.save_mode == 'all':
            torch.save(save_obj, os.path.join(save_path, f'{args.encoder}_{args.decoder}_ep{epoch}.pth'))

        if 'selfies' in args.formats:
            score = scores['selfies']
        elif 'atomtok' in args.formats:
            score = scores['smiles']
        else:
            score = scores['inchi']

        if score < best_score:
            best_score = score
            print_rank_0(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save(save_obj, os.path.join(save_path, f'{args.encoder}_{args.decoder}_best.pth'))
            with open(os.path.join(save_path, 'best_valid.json'), 'w') as f:
                json.dump(scores, f)
    
    if args.local_rank != -1:
        dist.barrier()


def inference(args, data_df, tokenizer, encoder=None, decoder=None, save_path=None, split='test'):
    
    print_rank_0("========== inference ==========")
    
    if args.local_rank == 0 and not args.debug:
        os.makedirs(save_path, exist_ok=True)
    
    device = args.device

    dataset = TrainDataset(args, data_df, tokenizer, split=split)
    if args.local_rank != -1:
        sampler = DistributedSampler(dataset, shuffle=False)
    else:
        sampler = SequentialSampler(dataset)
    dataloader = DataLoader(dataset, 
                            batch_size=args.batch_size * 2, 
                            sampler=sampler, 
                            num_workers=CFG.num_workers, 
                            pin_memory=True, 
                            drop_last=False)
    
    if encoder is None or decoder is None:
        # valid/test mode
        if args.ensemble_cfg is not None:
            with open(args.ensemble_cfg, "r") as f:
                ensemble_ckpts = json.load(f)
            encoder, decoder = get_ensemble_model(args, tokenizer, device, ensemble_ckpts, ddp=True)
        else:
            encoder, decoder = get_model(args, tokenizer, device, save_path, ddp=True)
    
    local_preds = valid_fn(dataloader, encoder, decoder, tokenizer, device, args)
    gathered_preds = [None for i in range(dist.get_world_size())]
    dist.all_gather_object(gathered_preds, local_preds)
    
    if args.local_rank != 0:
        return
    
    predictions = {format_:[None for i in range(len(dataset))] for format_ in args.formats}
    all_predictions = {format_:[None for i in range(len(dataset))] for format_ in args.formats}
    for preds in gathered_preds:
        beam_preds, final_preds = preds
        for format_ in args.formats:
            for idx, pred in final_preds[format_].items():
                predictions[format_][idx] = pred
            for idx, pred in beam_preds[format_].items():
                all_predictions[format_][idx] = pred

    if args.beam_size > 1:
        for format_ in args.formats:
            with open(os.path.join(save_path, f'{split}_{format_}_beam.jsonl'), 'w') as f:
                for idx, pred in enumerate(all_predictions[format_]):
                    text, score = pred
                    f.write(json.dumps({'id': idx, 'text': text, 'score': score}) + '\n')

    pred_df = data_df[['image_id']].copy()
    scores = {}
    
    for format_ in args.formats:
        text_preds = predictions[format_]
        if format_ == 'inchi':
            # InChI
            pred_df['InChI'] = [f"InChI=1S/{text}" for text in text_preds]
        elif format_ in ['atomtok', 'spe']:
            # SMILES
            pred_df['SMILES'] = text_preds
            print('Converting SMILES to InChI ...')
            inchi_list, r_success = batch_convert_smiles_to_inchi(text_preds)
            pred_df['SMILES_InChI'] = inchi_list
            print(f'{split} SMILES to InChI success ratio: {r_success:.4f}')
            scores['smiles_inchi_success'] = r_success
        elif format_ == 'selfies':
            # SELFIES
            pred_df['SELFIES'] = text_preds
            print('Converting SELFIES to InChI ...')
            inchi_list, r_success = batch_convert_selfies_to_inchi(text_preds)
            pred_df['SELFIES_InChI'] = inchi_list
            print(f'{split} SELFIES to InChI success ratio: {r_success:.4f}')
            # scores['selfies_inchi_success'] = r_success
    
    if 'atomtok' in args.formats and 'inchi' in args.formats:
        pred_df['merge_InChI'], _ = merge_inchi(pred_df['SMILES_InChI'].values, pred_df['InChI'].values)
    
    # Compute scores
    if split == 'valid':
        if 'inchi' in args.formats:
            scores['inchi'], scores['inchi_em'] = get_score(data_df['InChI'].values, pred_df['InChI'].values)
        if 'atomtok' in args.formats:
            scores['smiles'], scores['smiles_em'] = get_score(data_df['SMILES'].values, pred_df['SMILES'].values)
            scores['smiles_inchi'], scores['smiles_inchi_em'] = get_score(data_df['InChI'].values, pred_df['SMILES_InChI'].values)
            print('label:')
            print(data_df['SMILES'].values[:4])
            print('pred:')
            print(pred_df['SMILES'].values[:4])
        if 'atomtok' in args.formats and 'inchi' in args.formats:
            scores['merge_inchi'], scores['merge_inchi_em'] = get_score(data_df['InChI'].values, pred_df['merge_InChI'].values)
        if 'selfies' in args.formats:
            scores['selfies'], scores['selfies_em'] = get_score(data_df['SELFIES'].values, pred_df['SELFIES'].values)
            scores['selfies_inchi'], scores['selfies_inchi_em'] = get_score(data_df['InChI'].values, pred_df['SELFIES_InChI'].values)
            print('label:')
            print(data_df['SELFIES'].values[:4])
            print('pred:')
            print(pred_df['SELFIES'].values[:4])
            
    pred_df.to_csv(os.path.join(save_path, f'prediction_{split}.csv'), index=False)
    
    # Save predictions
    if split == 'test':
        if 'atomtok' in args.formats and 'inchi' in args.formats:
            pred_df['InChI'] = pred_df['merge_InChI']
        elif 'atomtok' in args.formats:
            pred_df['InChI'] = pred_df['SMILES_InChI']
        elif 'selfies' in args.formats:
            pred_df['InChI'] = pred_df['SELFIES_InChI']
        pred_df[['image_id', 'InChI']].to_csv(os.path.join(save_path, 'submission.csv'), index=False)
    
    return scores


def get_bms_data(args):
    def get_train_file_path(image_id):
        return "data/train/{}/{}/{}/{}.png".format(image_id[0], image_id[1], image_id[2], image_id)
    def get_test_file_path(image_id):
        return "data/test/{}/{}/{}/{}.png".format(image_id[0], image_id[1], image_id[2], image_id)
    train_df, valid_df, test_df = None, None, None
    if args.do_train:
        train_df = pd.read_csv('data/train_folds.csv')
        train_df['file_path'] = train_df['image_id'].apply(get_train_file_path)
        print_rank_0(f'train.shape: {train_df.shape}')
    if args.do_train or args.do_valid:
        valid_df = pd.read_csv('data/valid_folds.csv')
        valid_df['file_path'] = valid_df['image_id'].apply(get_train_file_path)
        print_rank_0(f'valid.shape: {valid_df.shape}')
    if args.do_test:
        test_df = pd.read_csv('data/sample_submission.csv')
        test_df['file_path'] = test_df['image_id'].apply(get_test_file_path)
        print_rank_0(f'test.shape: {test_df.shape}')
    tokenizer = {}
    for format_ in args.formats:
        tokenizer[format_] = Tokenizer('data/' + FORMAT_INFO[format_]['tokenizer'])
    return train_df, valid_df, test_df, tokenizer


def get_chemdraw_data(args):
    train_df, valid_df, test_df = None, None, None
    if args.do_train:
        train_df = pd.read_csv(os.path.join(args.data_path, args.train_file))
        print_rank_0(f'train.shape: {train_df.shape}')
    if args.do_train or args.do_valid:
        valid_df = pd.read_csv(os.path.join(args.data_path, args.valid_file))
        print_rank_0(f'valid.shape: {valid_df.shape}')
    if args.do_test:
        test_df = pd.read_csv(os.path.join(args.data_path, args.test_file))
        print_rank_0(f'test.shape: {test_df.shape}')
    tokenizer = {}
    for format_ in args.formats:
        tokenizer[format_] = Tokenizer(os.path.join(args.data_path, 'chemdraw-data/tokenizer_smiles_atomtok.json'))
    return train_df, valid_df, test_df, tokenizer


def main():

    args = get_args()
    seed_torch(seed=args.seed)
    
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.local_rank != -1:
        dist.init_process_group(backend='gloo', init_method='env://', timeout=datetime.timedelta(0, 7200))
        torch.cuda.set_device(args.local_rank)
        torch.backends.cudnn.benchmark = True
        
    if args.formats is None: 
        args.formats = [args.format]
    else:
        args.formats = args.formats.split(',')
    print_rank_0('Output formats: ' + ' '.join(args.formats))

    if args.dataset == 'bms':
        train_df, valid_df, test_df, tokenizer = get_bms_data(args)
    elif args.dataset == 'chemdraw':
        train_df, valid_df, test_df, tokenizer = get_chemdraw_data(args)
        
    if args.do_train and args.all_data:
        train_df = pd.concat([train_df, valid_df])
        print_rank_0(f'train.shape: {train_df.shape}')
        
    if args.trunc_train:
        if args.do_train:
            train_df = train_df.sample(n=50000, random_state=42).reset_index(drop=True)
        if args.do_valid:
            valid_df = valid_df.sample(n=10000, random_state=42).reset_index(drop=True)
    
    if args.debug:
        args.epochs = 2
        args.save_path = 'output/debug'
        args.print_freq = 50
        if args.do_train:
            train_df = train_df.sample(n=10000, random_state=42).reset_index(drop=True)
        if args.do_train or args.do_valid:
            valid_df = valid_df.sample(n=1000, random_state=42).reset_index(drop=True)
        if args.do_test:
            test_df = test_df.sample(n=1000, random_state=42).reset_index(drop=True)
    
    if args.selftrain:
        from bms.selftrain import get_self_training_data
        selftrain_df = get_self_training_data(test_df, args.selftrain, tokenizer)
        print_rank_0(f'selftrain.shape: {selftrain_df.shape}')
        train_df = pd.concat([train_df, selftrain_df])
    
    if args.do_train:
        train_loop(args, train_df, valid_df, tokenizer, args.save_path)
        
    if args.do_valid:
        scores = inference(args, valid_df, tokenizer, save_path=args.save_path, split='valid')
        print_rank_0(json.dumps(scores, indent=4))

    if args.do_test:
        inference(args, test_df, tokenizer, save_path=args.save_path, split='test')


if __name__ == "__main__":
    main()
