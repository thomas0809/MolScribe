#!/bin/bash

NUM_NODES=1
NUM_GPUS_PER_NODE=4
NODE_RANK=0

BATCH_SIZE=256
ACCUM_STEP=1

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

DATESTR=$(date +"%m-%d-%H-%M")
SAVE_PATH=output/uspto/swin_base_atomtok_200k_ep50
mkdir -p ${SAVE_PATH}

set -x

torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank $NODE_RANK --master_addr localhost --master_port $MASTER_PORT \
    train.py \
    --dataset chemdraw \
    --data_path data/molbank \
    --train_file pubchem/train_200k.csv \
    --aux_file uspto_mol/train_200k.csv  \
    --valid_file Img2Mol/USPTO.csv \
    --test_file Img2Mol/CLEF.csv,Img2Mol/JPO.csv,Img2Mol/UOB.csv,Img2Mol/USPTO.csv,Img2Mol/staker.csv,acs/acs-331.csv,uspto_test/uspto_indigo.csv,uspto_test/uspto_chemdraw.csv \
    --vocab_file bms/vocab_uspto.json \
    --formats atomtok \
    --dynamic_indigo --augment --mol_augment \
    --input_size 384 \
    --encoder swin_base \
    --decoder transformer \
    --encoder_lr 4e-4 \
    --decoder_lr 4e-4 \
    --save_path $SAVE_PATH --save_mode last --load_ckpt last \
    --label_smoothing 0.1 \
    --epochs 50 \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps $ACCUM_STEP \
    --use_checkpoint \
    --warmup 0.05 \
    --print_freq 200 \
<<<<<<< HEAD
    --do_train --do_valid --do_test \
    --fp16 2>&1   | tee $SAVE_PATH/log_${DATESTR}.txt
=======
    --do_test \
    --fp16 --backend nccl 2>&1  # | tee $SAVE_PATH/log_${DATESTR}.txt
>>>>>>> 15fd3606e3243e2a4f536443b6a6fba09c3c30eb

