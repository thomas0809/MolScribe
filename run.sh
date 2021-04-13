
CUDA_VISIBLE_DEVICES=5 python train_transformer.py \
  --format atomtok \
  --save-path output/smiles_atomtok/ \
  --do-train \
  --do-test

