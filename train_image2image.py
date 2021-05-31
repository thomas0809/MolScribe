import os
import sys
import re
import time
import json
import math
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
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, CosineAnnealingLR, LambdaLR
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from transformers import get_scheduler

from bms.model import Encoder
from image2image.dataset import TrainDataset
from image2image.model import Image2ImageModel
from bms.utils import init_summary_writer, seed_torch, AverageMeter, asMinutes, timeSince, FORMAT_INFO, print_rank_0


import warnings 
warnings.filterwarnings('ignore')


class CFG:
    print_freq=100
    num_workers=4
    scheduler='CosineAnnealingLR' # ['ReduceLROnPlateau', 'CosineAnnealingLR', 'CosineAnnealingWarmRestarts']
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
    parser.add_argument('--use_checkpoint', action='store_true')
    # Data
    parser.add_argument('--format', type=str, choices=['inchi','atomtok','spe'], default='atomtok')
    parser.add_argument('--formats', type=str, default=None)
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--augment', action='store_true')
    parser.add_argument('--resize_pad', action='store_true')
    parser.add_argument('--no_crop_white', action='store_true')
    # Training
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--scheduler', type=str, choices=['cosine','constant'], default='cosine')
    parser.add_argument('--warmup_ratio', type=float, default=0)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--load_path', type=str, default=None)
    parser.add_argument('--save_path', type=str, default='i2i_output/')
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
    parser.add_argument('--trunc_train', action='store_true')
    parser.add_argument('--debug', action='store_true')
    # Inference
    parser.add_argument('--train_beam_size', type=int, default=4)
    parser.add_argument('--beam_size', type=int, default=10)
    parser.add_argument('--train_beam_file', type=str, default=None)
    parser.add_argument('--valid_beam_file', type=str, default=None)
    parser.add_argument('--test_beam_file', type=str, default=None)
    args = parser.parse_args()
    return args


def load_states(args, load_path):
    if args.load_ckpt == 'best':
        path = os.path.join(load_path, f'{args.encoder}_best.pth')
    else:
        path = os.path.join(load_path, f'{args.encoder}_{args.load_ckpt}.pth')
    states = torch.load(path, map_location=torch.device('cpu'))
    return states


def get_model(args, device, load_path=None, ddp=True):
    encoder1 = Encoder(args.encoder, pretrained=True, use_checkpoint=args.use_checkpoint)
    encoder2 = Encoder(args.encoder, pretrained=True, use_checkpoint=args.use_checkpoint)
    model = Image2ImageModel(encoder1, encoder2)
    
    if load_path:
        states = load_states(args, load_path)
        def remove_prefix(state_dict):
            return {k.replace('module.', ''): v for k,v in state_dict.items()}
        model.load_state_dict(remove_prefix(states['model']))
        print_rank_0(f"Model loaded from {load_path}")
    
    model.to(device)
    
    if args.local_rank != -1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
        print_rank_0("DDP setup finished")

    return model


def get_optimizer_and_scheduler(args, model, load_path=None):
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=CFG.weight_decay, amsgrad=False)
    scheduler= get_scheduler(args.scheduler, optimizer, args.num_warmup_steps, args.num_training_steps)
    start_epoch = 0
    global_step = 0

    if load_path and args.resume:
        states = load_states(args, load_path)
        optimizer.load_state_dict(states['optimizer'])
        if args.init_scheduler:
            for group in optimizer.param_groups:
                group['lr'] = args.lr
        else:
            scheduler.load_state_dict(states['scheduler'])
            global_step = states['global_step']
            if 'epoch' in states:
                start_epoch = states['epoch'] + 1
        print_rank_0(f"Optimizer loaded from {load_path}")
        
    return optimizer, scheduler, start_epoch, global_step


def train_fn(train_loader, model, criterion, optimizer, epoch,
             scheduler, scaler, device, global_step, SUMMARY, args):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    epoch_losses = AverageMeter()
    # switch to train mode
    model.train()
    
    start = end = time.time()
    grad_norm = 0

    for step, (indices, images, beam_images) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        images = images.to(device)
        beam_images = beam_images.to(device)
        batch_size = images.size(0)
        with torch.cuda.amp.autocast(enabled=args.fp16):
            logits = model(images, beam_images)
            labels = torch.zeros((batch_size,), dtype=torch.long, device=device)
            loss = criterion(logits, labels)
        # record loss
        losses.update(loss.item(), batch_size)
        epoch_losses.update(loss.item(), batch_size)
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        scaler.scale(loss).backward()
        if (step + 1) % args.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        if step % CFG.print_freq == 0 or step == (len(train_loader)-1):
            print_rank_0('Epoch: [{0}][{1}/{2}] '
                  'Data {data_time.avg:.3f}s ({sum_data_time}) '
                  'Elapsed {remain:s} '
                  'Loss: {loss.val:.4f}({loss.avg:.4f}) '
                  'Grad: {grad_norm:.4f}  '
                  'LR: {lr:.6f}'
                  .format(
                   epoch+1, step, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses,
                   sum_data_time=asMinutes(data_time.sum),
                   remain=timeSince(start, float(step+1)/len(train_loader)),
                   grad_norm=grad_norm,
                   lr=scheduler.get_lr()[0],
                   ))
            losses.reset()
        
    return epoch_losses.avg, global_step


def valid_fn(valid_loader, model, device, args):
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    # switch to evaluation mode
    if hasattr(model, 'module'):
        model = model.module
    model.eval()
    all_preds = {}
    start = end = time.time()
    for step, (indices, images, beam_images) in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        indices = indices.tolist()
        images = images.to(device)
        beam_images = beam_images.to(device)
        with torch.cuda.amp.autocast(enabled=args.fp16):
            with torch.no_grad():
                logits = model(images, beam_images)
        predictions = logits.argmax(-1).tolist()
        for idx, pred in zip(indices, predictions):
            all_preds[idx] = pred
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
    return all_preds


def train_loop(args, train_df, train_beam_file, valid_df, valid_beam_file, tokenizer, save_path):
    
    SUMMARY = None
    
    if args.local_rank == 0 and not args.debug:
        os.makedirs(save_path, exist_ok=True)
        SUMMARY = init_summary_writer(save_path)
        
    print_rank_0("========== training ==========")
        
    device = args.device

    # ====================================================
    # loader
    # ====================================================

    train_dataset = TrainDataset(args, train_df, train_beam_file, tokenizer, split='train')
    
#     n = len(train_df) // 8
# #     for idx in range(n*args.local_rank+65000, n*(args.local_rank+1)):
# #     for start in [888999,1979999,70999,1161999,343999,1434999,616999,1707999]:
#     for start in [617210]:
#         print(start)
#         for idx in range(start, start+1000):
#             _, image, beam_image = train_dataset[idx]
#             if (idx + 1) % 1000 == 0:
#                 print('Rank', args.local_rank, idx)
#     return

    if args.debug and args.local_rank == 0:
        train_dataset.save_images(os.path.join(save_path, 'images'), 5)
    if args.local_rank != -1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True)
    else:
        train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(train_dataset, 
                              batch_size=args.batch_size, 
                              sampler=train_sampler,
                              num_workers=CFG.num_workers, 
                              pin_memory=True,
                              drop_last=True)
    
    args.num_training_steps = args.epochs * (len(train_loader) // args.gradient_accumulation_steps)
    args.num_warmup_steps = int(args.num_training_steps * args.warmup_ratio)

    # ====================================================
    # model & optimizer
    # ====================================================
    model = get_model(args, device, load_path=args.load_path)
    optimizer, scheduler, start_epoch, global_step = get_optimizer_and_scheduler(args, model, load_path=args.load_path)
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)
        
    # ====================================================
    # loop
    # ====================================================
    criterion = nn.CrossEntropyLoss()

    best_score = -np.inf
    best_loss = np.inf

    for epoch in range(start_epoch, args.epochs):
        
        if args.local_rank != -1:
            dist.barrier()
            train_sampler.set_epoch(epoch)
        
        start_time = time.time()
        encoder_lr = scheduler.get_lr()[0]
        
        # train
        avg_loss, global_step = train_fn(train_loader, model, criterion, optimizer, epoch,
                                         scheduler, scaler, device, global_step, SUMMARY, args)
        print_rank_0(f"Failed images: {train_dataset.get_num_image_failed()}")
        
        # eval
        scores = inference(args, valid_df, valid_beam_file, tokenizer, model, save_path, split='valid')
        
        if args.local_rank != 0:
            continue
    
        elapsed = time.time() - start_time

        print_rank_0(f'Epoch {epoch+1} - Time: {elapsed:.0f}s')
        print_rank_0(f'Epoch {epoch+1} - Score: ' + json.dumps(scores))
        
        if SUMMARY:
            SUMMARY.add_scalar('train/loss', avg_loss, epoch)
            SUMMARY.add_scalar('train/lr', encoder_lr, epoch)
            for key in scores:
                SUMMARY.add_scalar(f'valid/{key}', scores[key], epoch)

        save_obj = {'model': model.state_dict(), 
                    'optimizer': optimizer.state_dict(), 
                    'scheduler': scheduler.state_dict(), 
                    'global_step': global_step,
                    'epoch': epoch
                   }
        torch.save(save_obj, os.path.join(save_path, f'{args.encoder}_ep{epoch}.pth'))

        score = scores['acc']

        if score > best_score:
            best_score = score
            print_rank_0(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save(save_obj, os.path.join(save_path, f'{args.encoder}_best.pth'))
            with open(os.path.join(save_path, 'best_valid.json'), 'w') as f:
                json.dump(scores, f)
    
    if args.local_rank != -1:
        dist.barrier()


def inference(args, data_df, beam_file, tokenizer, model=None, save_path=None, split='test'):
    
    print_rank_0("========== inference ==========")
    
    device = args.device

    dataset = TrainDataset(args, data_df, beam_file, tokenizer, split=split)
    if args.debug and args.local_rank == 0:
        dataset.save_images(os.path.join(save_path, f'{split}_images'), 5)
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
    
    if model is None:
        model = get_model(args, device, save_path, ddp=True)
    
    local_preds = valid_fn(dataloader, model, device, args)
    gathered_preds = [None for i in range(dist.get_world_size())]
    dist.all_gather_object(gathered_preds, local_preds)
    
    if args.local_rank != 0:
        return
    
    predictions = [None for i in range(len(dataset))]
    for preds in gathered_preds:
        for idx, pred in preds.items():
            predictions[idx] = pred
                        
    scores = {}
    
    # Compute scores
    if split == 'valid':
        scores['acc'] = sum([pred == 0 for pred in predictions]) / len(predictions)
    
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
    
    if args.do_test:
        test_df = pd.read_csv('data/sample_submission.csv')
        test_df['file_path'] = test_df['image_id'].apply(get_test_file_path)
        print_rank_0(f'test.shape: {test_df.shape}')
        
    if args.trunc_train:
        train_df = train_df.truncate(after=50000)
        valid_df = valid_df.truncate(after=50000)
    
    if args.debug:
        args.epochs = 5
        args.save_path = 'i2i_output/debug'
        args.augment = False
        CFG.print_freq = 50
        if args.do_train:
            train_df = train_df.truncate(after=5000)
        if args.do_train or args.do_valid:
            valid_df = valid_df.truncate(after=1000)
        if args.do_test:
            test_df = test_df.truncate(after=1000)
    
    if args.formats is None:
        args.formats = [args.format]
    else:
        args.formats = args.formats.split(',')
    print_rank_0('Output formats: ' + ' '.join(args.formats))
    tokenizer = {}
    for format_ in args.formats:
        tokenizer[format_] = torch.load('data/' + FORMAT_INFO[format_]['tokenizer'])
    
    if args.do_train:
        assert args.train_beam_file
        train_loop(args, train_df, args.train_beam_file, valid_df, args.valid_beam_file, tokenizer, args.save_path)
        
    if args.do_valid:
        assert args.valid_beam_file
        scores = inference(args, valid_df, args.valid_beam_file, tokenizer, save_path=args.save_path, split='valid')
        print_rank_0(json.dumps(scores, indent=4))

    if args.do_test:
        assert args.test_beam_file
        inference(args, test_df, args.test_beam_file, tokenizer, save_path=args.save_path, split='test')


if __name__ == "__main__":
    main()
