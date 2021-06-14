NUM_NODES=1
NUM_GPUS_PER_NODE=8
NODE_RANK=0

BATCH_SIZE=512
ACCUM_STEP=1

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank $NODE_RANK --master_addr localhost --master_port $MASTER_PORT \
    train.py \
    --formats atomtok \
    --input_size 384 \
    --encoder swin_base_patch4_window12_384 \
    --decoder_scale 2 \
    --save_path output/swin_base_384_epoch_24 \
    --epochs 20 \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps $ACCUM_STEP \
    --use_checkpoint \
    --do_valid \
    --beam_size 1 --n_best 1 \
    --fp16

#     --do_test \    --no_crop_white \


