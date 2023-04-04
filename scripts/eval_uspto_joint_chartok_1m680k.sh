#!/bin/bash

NUM_NODES=1
NUM_GPUS_PER_NODE=1
NODE_RANK=0
MASTER_PORT=$(shuf -n 1 -i 10000-65535)

BATCH_SIZE=64

LOAD_PATH=ckpts/swin_base_char_aux_1m680k.pth
SAVE_PATH=output/uspto/swin_base_char_aux_1m680k/
mkdir -p ${SAVE_PATH}

set -x

torchrun \
    --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank $NODE_RANK --master_addr localhost --master_port $MASTER_PORT \
    train.py \
    --data_path data \
    --test_file real/CLEF.csv,real/UOB.csv,real/USPTO.csv,real/staker.csv,real/acs.csv,synthetic/indigo.csv,synthetic/chemdraw.csv \
    --vocab_file molscribe/vocab/vocab_chars.json \
    --formats chartok_coords,edges \
    --coord_bins 64 --sep_xy \
    --input_size 384 \
    --encoder swin_base \
    --decoder transformer \
    --load_path $LOAD_PATH \
    --save_path $SAVE_PATH \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE)) \
    --use_checkpoint \
    --print_freq 200 \
    --do_test \
    --fp16 --backend gloo 2>&1
