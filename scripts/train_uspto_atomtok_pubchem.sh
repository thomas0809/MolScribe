#!/bin/bash

NUM_NODES=1
NUM_GPUS_PER_NODE=4
NODE_RANK=0

BATCH_SIZE=128
ACCUM_STEP=1

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

DATESTR=$(date +"%m-%d-%H-%M")
SAVE_PATH=output/uspto/swin_base_atomtok_pubchem_200k
mkdir -p ${SAVE_PATH}

set -x

torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank $NODE_RANK --master_addr localhost --master_port $MASTER_PORT \
    train.py \
    --dataset chemdraw \
    --data_path data/molbank \
    --train_file pubchem/train_200k.csv \
    --valid_file Img2Mol/USPTO.csv \
    --test_file Img2Mol/CLEF.csv,Img2Mol/JPO.csv,Img2Mol/UOB.csv,Img2Mol/USPTO.csv,Img2Mol/staker.csv \
    --vocab_file vocab/vocab_uspto.json \
    --formats atomtok \
    --dynamic_indigo --augment --mol_augment \
    --coord_bins 64 --sep_xy \
    --input_size 384 \
    --encoder swin_base \
    --decoder transformer \
    --encoder_lr 4e-4 \
    --decoder_lr 4e-4 \
    --save_path $SAVE_PATH --save_mode last \
    --label_smoothing 0.1 \
    --epochs 20 \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps $ACCUM_STEP \
    --use_checkpoint \
    --warmup 0.05 \
    --print_freq 200 \
    --do_train --do_valid --do_test \
    --fp16 2>&1  # | tee $SAVE_PATH/log_${DATESTR}.txt


#    --decoder_dim 1024 --embed_dim 512 --attention_dim 512 \
#    --train_steps_per_epoch 3000 \
#    --valid_file indigo-data/valid.csv \
#    --valid_file real-acs-evaluation/test.csv \
#    --save_path output/indigo/swin_base_20_dynamic_aug \
#    --no_pretrained --scheduler cosine --warmup 0.05 \
#    --load_path output/pubchem/swin_base_10 --resume \
#    --test_file pubchem/test.csv,pubchem/test_chemdraw.csv,indigo-data/test_uspto.csv,chemdraw-data/test_uspto.csv,zinc/test.csv \
