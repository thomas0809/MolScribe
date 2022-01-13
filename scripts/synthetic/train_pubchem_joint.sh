#!/bin/bash

NUM_NODES=1
NUM_GPUS_PER_NODE=4
NODE_RANK=0

BATCH_SIZE=128
ACCUM_STEP=1

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

set -x

torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank $NODE_RANK --master_addr localhost --master_port $MASTER_PORT \
    train.py \
    --dataset chemdraw \
    --data_path data/molbank \
    --train_file pubchem/train_200k.csv \
    --valid_file pubchem/valid.csv \
    --test_file pubchem/test.csv,pubchem/test_chemdraw.csv,indigo-data/test_uspto.csv,chemdraw-data/test_uspto.csv \
    --formats atomtok_coords,edges \
    --input_size 384 \
    --encoder swin_base \
    --decoder transformer \
    --encoder_lr 1e-3 \
    --decoder_lr 1e-3 \
    --dynamic_indigo --augment \
    --coord_bins 64 --sep_xy \
    --save_path output/pubchem/synthetic/swin_base_200k_joint \
    --label_smoothing 0.1 \
    --epochs 50 \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps $ACCUM_STEP \
    --use_checkpoint \
    --warmup 0.05 \
    --print_freq 200 \
    --do_train --do_valid --do_test \
    --trunc_valid 10000 \
    --fp16


#    --test_file Img2Mol/CLEF.csv,Img2Mol/JPO.csv,Img2Mol/UOB.csv,Img2Mol/USPTO.csv,Img2Mol/staker/staker.csv \
#    --test_file pubchem/test.csv,pubchem/test_chemdraw.csv,indigo-data/test_uspto.csv,chemdraw-data/test_uspto.csv,zinc/test.csv \
#    --decoder_dim 1024 --embed_dim 512 --attention_dim 512 \
#    --train_steps_per_epoch 3000 \
#    --valid_file indigo-data/valid.csv \
#    --valid_file real-acs-evaluation/test.csv \
#    --save_path output/indigo/swin_base_20_dynamic_aug \
#    --no_pretrained --scheduler cosine --warmup 0.05 \
#    --load_path output/pubchem/swin_base_10 --resume \