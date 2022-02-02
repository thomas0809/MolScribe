#!/bin/bash

NUM_NODES=1
NUM_GPUS_PER_NODE=4
NODE_RANK=0

BATCH_SIZE=128
ACCUM_STEP=1

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

DATESTR=$(date +"%m-%d-%H-%M")
SAVE_PATH=output/pubchem/synthetic/swin_base_200k_joint_large100k
mkdir -p ${SAVE_PATH}

set -x

torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank $NODE_RANK --master_addr localhost --master_port $MASTER_PORT \
    train.py \
    --dataset chemdraw \
    --data_path data/molbank \
    --train_file pubchem/train_200k.csv,pubchem/train_large_100k.csv,pubchem/train_small.csv \
    --valid_file pubchem/valid.csv \
    --test_file pubchem/test.csv \
    --formats atomtok_coords,edges \
    --input_size 384 \
    --encoder swin_base \
    --decoder transformer \
    --encoder_lr 4e-4 \
    --decoder_lr 4e-4 \
    --dynamic_indigo --augment \
    --coord_bins 64 --sep_xy \
    --save_path $SAVE_PATH \
    --label_smoothing 0.1 \
    --epochs 40 \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps $ACCUM_STEP \
    --use_checkpoint \
    --warmup 0.05 \
    --print_freq 200 \
    --do_test \
    --trunc_valid 10000 \
    --fp16  2>&1 # | tee $SAVE_PATH/log_${DATESTR}.txt


#--test_file pubchem/test.csv,pubchem/test_chemdraw.csv,uspto_test/uspto_indigo.csv,uspto_test/uspto_chemdraw.csv \