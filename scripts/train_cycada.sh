#!/bin/bash

NUM_NODES=1
NUM_GPUS_PER_NODE=8
NODE_RANK=0

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

set -ex
python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank $NODE_RANK --master_addr localhost --master_port $MASTER_PORT \
    train_cycada.py \
    --dataset_mode chemdraw \
    --dataroot data/molbank \
    --dir_A data/molbank/indigo-data \
    --dir_B data/molbank/chemdraw-data \
    --encoder swin_base_patch4_window12_384 \
    --decoder_scale 2 \
    --load_path output/indigo/swin_base_20_dynamic_aug \
    --name chemdraw_cycada_1 \
    --model cycada \
    --input_size 384 \
    --lr 1e-4 \
    --batch_size 8 \
    --label_smoothing 0.1 \
    --pool_size 50 \
    --no_dropout \
    --display_server http://rosetta5.csail.mit.edu \
    --display_port 10000 \
    --display_ncols -1
