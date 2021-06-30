#!/bin/bash

NUM_NODES=1
NUM_GPUS_PER_NODE=8
NODE_RANK=0

BATCH_SIZE=256
ACCUM_STEP=1

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

set -x

python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank $NODE_RANK --master_addr localhost --master_port $MASTER_PORT \
    train.py \
    --dataset chemdraw \
    --data_path data/molbank \
    --train_file indigo-data/train.csv \
    --valid_file indigo-data/valid.csv \
    --test_file indigo-data/test.csv \
    --formats atomtok \
    --input_size 384 \
    --encoder swin_base_patch4_window12_384 \
    --decoder_scale 2 \
    --encoder_lr 4e-4 \
    --decoder_lr 4e-4 \
    --load_path output/indigo/swin_base_20 \
    --resume \
    --save_path output/indigo/swin_base_20 \
    --label_smoothing 0.1 \
    --epochs 24 \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps $ACCUM_STEP \
    --use_checkpoint \
    --do_valid \
    --fp16


# swin_base
#     --encoder swin_base_patch4_window12_384 \
#     --decoder_scale 2 \
#     --save_path output/swin_base_384_epoch_16 \

#     --encoder swin_base_patch4_window12_384 \
#     --decoder_scale 2 \
#     --save_path output/swin_base_384_epoch_16_cont \
#     --load_path output/swin_base_384_epoch_16 \
#     --init_scheduler \

# swin_large
#     --encoder swin_large_patch4_window12_384 \
#     --decoder_scale 2 \
#     --load_path output/swin_large_384_epoch_16 \
#     --save_path output/swin_large_384_epoch_16 \

# resnet101
#     --encoder resnet101d \
#     --decoder_scale 2 \
#     --save_path output/resnet101_epoch_16 \