NUM_NODES=1
NUM_GPUS_PER_NODE=8
NODE_RANK=0

BATCH_SIZE=512
ACCUM_STEP=1

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank $NODE_RANK --master_addr localhost --master_port $MASTER_PORT \
    train.py \
    --format atomtok \
    --save_path output/resnet50_input_320_dlayer_2_atomtok_epoch_20 \
    --input_size 320 \
    --epochs 20 \
    --encoder resnet50 \
    --decoder_scale 2 \
    --decoder_layer 2 \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps $ACCUM_STEP \
    --do_train \
    --do_test
#     --debug

#     --do_train \
#     --do_test \
