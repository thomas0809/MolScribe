NUM_NODES=1
NUM_GPUS_PER_NODE=8
NODE_RANK=0

BATCH_SIZE=1024
ACCUM_STEP=1

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank $NODE_RANK --master_addr localhost --master_port $MASTER_PORT \
    train_image2image.py \
    --formats atomtok \
    --input_size 384 \
    --encoder tf_efficientnetv2_s_in21ft1k \
    --lr 1e-4 \
    --temperature 0.1 \
    --save_path i2i_output/efficientnetv2_s_beam2 \
    --train_beam_file data/beam_search/train_beam.txt \
    --valid_beam_file output/ensemble3/valid_atomtok_beam.jsonl \
    --augment \
    --epochs 16 \
    --train_beam_size 2 \
    --beam_size 2 \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps $ACCUM_STEP \
    --use_checkpoint \
    --do_valid \
    --print_freq 50 \
    --fp16


#     --encoder resnet34 \
#     --encoder tf_efficientnetv2_s_in21ft1k \
#     --valid_beam_file data/beam_search/valid_beam.txt \