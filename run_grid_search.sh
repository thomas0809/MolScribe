NUM_NODES=1
NUM_GPUS_PER_NODE=4
NODE_RANK=0

BATCH_SIZE=256
ACCUM_STEP=1

MASTER_PORT=$(shuf -n 1 -i 10000-65535)

BASE_PATH=grid_search_output/atomtok_aug_nocrop
mkdir -p $BASE_PATH
GRID_LOG=$BASE_PATH/summary.txt
echo "" > $GRID_LOG


for decoder_layer in 1
do
for encoder_lr in 4e-4 # 1e-4 4e-4 
do
for decoder_lr in 4e-4 # 1e-4 4e-4 
do
for batch_size in 64
do

HYPERS=decoder_layer_${decoder_layer}-encoder_lr_${encoder_lr}-decoder_lr_${decoder_lr}-batch_size_${batch_size}
SAVE_PATH=$BASE_PATH/$HYPERS
mkdir -p $SAVE_PATH

echo $SAVE_PATH

python -m torch.distributed.launch \
    --nproc_per_node=$NUM_GPUS_PER_NODE --nnodes=$NUM_NODES --node_rank $NODE_RANK --master_addr localhost --master_port $MASTER_PORT \
    train.py \
    --formats atomtok \
    --input_size 384 \
    --encoder swin_base_patch4_window12_384 \
    --decoder_scale 2 \
    --decoder_layer $decoder_layer \
    --encoder_lr $encoder_lr \
    --decoder_lr $decoder_lr \
    --save_path $SAVE_PATH \
    --trunc_train \
    --augment \
    --no_crop_white \
    --scheduler cosine \
    --epochs 32 \
    --batch_size $((batch_size / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps $ACCUM_STEP \
    --use_checkpoint \
    --do_train \
    --fp16  2>&1 | tee $SAVE_PATH/log.txt

echo $HYPERS >> $GRID_LOG
cat $SAVE_PATH/best_valid.json >> $GRID_LOG

done
done
done
done

#     --do_train \
#     --do_test \


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