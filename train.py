import os
import sys
import re
import time
import json
import random
import argparse
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, ReduceLROnPlateau
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from bms.dataset import TrainDataset, TestDataset, bms_collate
from bms.model import Encoder, DecoderWithAttention, MultiTaskDecoder
from bms.utils import init_logger, init_summary_writer, seed_torch, get_score, AverageMeter, asMinutes, timeSince, is_valid, \
                      batch_convert_smiles_to_inchi, batch_convert_selfies_to_inchi, merge_inchi, FORMAT_INFO, PAD_ID, print_rank_0


import warnings 
warnings.filterwarnings('ignore')


class CFG:
    max_len=120
    print_freq=200
    num_workers=4
    scheduler='CosineAnnealingLR' # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
    encoder_lr=1e-4
    decoder_lr=4e-4
    min_lr=1e-6
    weight_decay=1e-6
    max_grad_norm=5
    attention_dim=256
    embed_dim=256
    decoder_dim=512
    dropout=0.5


def get_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--encoder', type=str, default='resnet34')
    parser.add_argument('--decoder', type=str, default='lstm')
    parser.add_argument('--decoder_scale', type=int, default=1)
    parser.add_argument('--decoder_layer', type=int, default=1)
    parser.add_argument('--use_checkpoint', action='store_true')
    # Data
    parser.add_argument('--format', type=str, choices=['inchi','atomtok','spe'], default='atomtok')
    parser.add_argument('--formats', type=str, default=None)
    parser.add_argument('--input_size', type=int, default=224)
    # Training
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='output/')
    parser.add_argument('--load_ckpt', type=str, default='best')
    parser.add_argument('--do_train', action='store_true')
    parser.add_argument('--do_valid', action='store_true')
    parser.add_argument('--do_test', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--all_data', action='store_true', help='Use both train and valid data for training.')
    parser.add_argument('--local_rank', type=int, default=-1, metavar='N', help='Local process rank.')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--init_scheduler', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--debug', action='store_true')
    # Inference
    parser.add_argument('--beam_size', type=int, default=1)
    parser.add_argument('--n_best', type=int, default=1)
    args = parser.parse_args()
    return args


def load_states(args, load_path):
    if args.load_ckpt == 'best':
        path = os.path.join(load_path, f'{args.encoder}_{args.decoder}_best.pth')
    else:
        path = os.path.join(load_path, f'{args.encoder}_{args.decoder}_{args.load_ckpt}.pth')
    states = torch.load(path, map_location=torch.device('cpu'))
    return states


def get_model(args, tokenizer, device, load_path=None, ddp=True):
    encoder = Encoder(args.encoder, pretrained=True, use_checkpoint=args.use_checkpoint)
    decoder = MultiTaskDecoder(
        formats=args.formats,
        attention_dim=CFG.attention_dim * args.decoder_scale,
        embed_dim=CFG.embed_dim * args.decoder_scale,
        encoder_dim=encoder.n_features,
        decoder_dim=CFG.decoder_dim * args.decoder_scale,
        max_len=CFG.max_len,
        dropout=CFG.dropout,
        n_layer=args.decoder_layer,
        tokenizer=tokenizer)
    
    if load_path:
        states = load_states(args, load_path)
        def remove_prefix(state_dict):
            return {k.replace('module.', ''): v for k,v in state_dict.items()}
        encoder.load_state_dict(remove_prefix(states['encoder']))
        decoder.load_state_dict(remove_prefix(states['decoder']))
        print_rank_0(f"Model loaded from {load_path}")
    
    encoder.to(device)
    decoder.to(device)
    
    if args.local_rank != -1:
        encoder = DDP(encoder, device_ids=[args.local_rank], output_device=args.local_rank)
        decoder = DDP(decoder, device_ids=[args.local_rank], output_device=args.local_rank)
        print_rank_0("DDP setup finished")

    return encoder, decoder


def get_optimizer_and_scheduler(args, encoder, decoder, load_path=None):
    
    def get_scheduler(optimizer):
        if CFG.scheduler=='ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=CFG.factor, patience=CFG.patience, verbose=True, eps=CFG.eps)
        elif CFG.scheduler=='CosineAnnealingLR':
            scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=CFG.min_lr, last_epoch=-1)
        elif CFG.scheduler=='CosineAnnealingWarmRestarts':
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=CFG.T_0, T_mult=1, eta_min=CFG.min_lr, last_epoch=-1)
        return scheduler
    
    encoder_optimizer = AdamW(encoder.parameters(), lr=CFG.encoder_lr, weight_decay=CFG.weight_decay, amsgrad=False)
    encoder_scheduler = get_scheduler(encoder_optimizer)

    decoder_optimizer = AdamW(decoder.parameters(), lr=CFG.decoder_lr, weight_decay=CFG.weight_decay, amsgrad=False)
    decoder_scheduler = get_scheduler(decoder_optimizer)
    
    if load_path and args.resume:
        states = load_states(args, load_path)
        encoder_optimizer.load_state_dict(states['encoder_optimizer'])
        decoder_optimizer.load_state_dict(states['decoder_optimizer'])
        if args.init_scheduler:
            for group in encoder_optimizer.param_groups:
                group['lr'] = CFG.encoder_lr
            for group in decoder_optimizer.param_groups:
                group['lr'] = CFG.decoder_lr
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
                format_loss = criterion(predictions, targets)
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
            global_step += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
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
            
    if isinstance(encoder_scheduler, CosineAnnealingLR):
        encoder_scheduler.step()
    elif isinstance(encoder_scheduler, CosineAnnealingWarmRestarts):
        encoder_scheduler.step()

    if isinstance(decoder_scheduler, CosineAnnealingLR):
        decoder_scheduler.step()
    elif isinstance(decoder_scheduler, CosineAnnealingWarmRestarts):
        decoder_scheduler.step()
        
    return epoch_losses.avg, global_step


def valid_fn(valid_loader, encoder, decoder, tokenizer, device, args):
    
    def _pick_valid(preds, format_):
        """Pick the top valid prediction from n_best outputs
        """
        best = preds[0] # default
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
            text_preds = [
                [tokenizer[format_].predict_caption(x.cpu().numpy()) for x in preds]
                for preds in predictions[format_]
            ]
            for idx, preds in zip(indices, text_preds):
                all_preds[format_][idx] = preds
            valid_preds = [_pick_valid(preds, format_) for preds in text_preds]
            for idx, preds in zip(indices, valid_preds):
                final_preds[format_][idx] = preds
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(valid_loader)-1):
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
        # LOGGER = init_logger()
        SUMMARY = init_summary_writer(save_path)
        
    print_rank_0("========== training ==========")
        
    device = args.device
    os.makedirs(save_path, exist_ok=True)

    # ====================================================
    # loader
    # ====================================================

    train_dataset = TrainDataset(args, train_df, tokenizer, labelled=True)
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
    criterion = nn.CrossEntropyLoss(ignore_index=PAD_ID)

    best_score = np.inf
    best_loss = np.inf
    
    global_step = 0
    start_epoch = encoder_scheduler.last_epoch

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
    
    if args.local_rank != -1:
        dist.barrier()


def inference(args, data_df, tokenizer, encoder=None, decoder=None, save_path=None, split='test'):
    
    print_rank_0("========== inference ==========")
    
    device = args.device

    dataset = TrainDataset(args, data_df, tokenizer, labelled=False)
    if args.local_rank != -1:
        sampler = DistributedSampler(dataset, shuffle=False)
    else:
        sampler = RandomSampler(dataset)
    dataloader = DataLoader(dataset, 
                            batch_size=args.batch_size * 2, 
                            sampler=sampler, 
                            num_workers=CFG.num_workers, 
                            pin_memory=True, 
                            drop_last=False)
    
    if encoder is None or decoder is None:
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

    if split == 'valid':
        for format_ in args.formats:
            with open(os.path.join(save_path, f'{split}_{format_}_beam.txt'), 'w') as f:
                for idx, pred in enumerate(all_predictions[format_]):
                    f.write(f'ID {idx}\n')
                    f.write('\n'.join(pred) + '\n')
                
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
            

def get_train_file_path(image_id):
    return "data/train/{}/{}/{}/{}.png".format(image_id[0], image_id[1], image_id[2], image_id)

def get_test_file_path(image_id):
    return "data/test/{}/{}/{}/{}.png".format(image_id[0], image_id[1], image_id[2], image_id)


def main():

    args = get_args()
    seed_torch(seed=args.seed)
    
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if args.local_rank != -1:
        dist.init_process_group(backend='gloo', init_method='env://')
        torch.cuda.set_device(args.local_rank)
        torch.backends.cudnn.benchmark = True

    if args.do_train:
        train_df = pd.read_csv('data/train_folds.csv')
        train_df['file_path'] = train_df['image_id'].apply(get_train_file_path)
        print_rank_0(f'train.shape: {train_df.shape}')
    
    if args.do_train or args.do_valid:
        valid_df = pd.read_csv('data/valid_folds.csv')
        valid_df['file_path'] = valid_df['image_id'].apply(get_train_file_path)
        print_rank_0(f'valid.shape: {valid_df.shape}')
        if args.all_data:
            train_df = pd.concat([train_df, valid_df])
            print_rank_0(f'train.shape: {train_df.shape}')
    
    if args.do_test:
        test_df = pd.read_csv('data/sample_submission.csv')
        test_df['file_path'] = test_df['image_id'].apply(get_test_file_path)
        print_rank_0(f'test.shape: {test_df.shape}')
    
    if args.debug:
        args.epochs = 2
#         args.save_path = 'output/debug'
        CFG.print_freq = 50
        if args.do_train:
            train_df = train_df.sample(n=10000, random_state=42).reset_index(drop=True)
        if args.do_train or args.do_valid:
            valid_df = valid_df.sample(n=1000, random_state=42).reset_index(drop=True)
        if args.do_test:
            test_df = test_df.sample(n=1000, random_state=42).reset_index(drop=True)
    
    if args.formats is None:
        args.formats = [args.format]
    else:
        args.formats = args.formats.split(',')
    print_rank_0('Output formats: ' + ' '.join(args.formats))
    tokenizer = {}
    for format_ in args.formats:
        tokenizer[format_] = torch.load('data/' + FORMAT_INFO[format_]['tokenizer'])
    
    if args.do_train:
        train_loop(args, train_df, valid_df, tokenizer, args.save_path)
        
    if args.do_valid:
        scores = inference(args, valid_df, tokenizer, save_path=args.save_path, split='valid')
        print_rank_0(json.dumps(scores, indent=4))

    if args.do_test:
        inference(args, test_df, tokenizer, save_path=args.save_path, split='test')


if __name__ == "__main__":
    main()
