# BMS

## Data
Required data
- `data/train_labels.csv`
- `data/train`
- `data/train_smiles.csv`
- `data/train_smiles_atomtok.csv`
- `data/train_smiles_spe_chembl.csv`

```
mkdir data; cd data
ln -s /data/rsg/chemistry/jiang_guo/chemai/literature-ie/bms-kaggle/data/train ./train
ln -s /data/rsg/chemistry/jiang_guo/chemai/literature-ie/bms-kaggle/data/train_labels.csv ./train_labels.csv
ln -s /data/rsg/chemistry/jiang_guo/chemai/literature-ie/bms-kaggle/data/train_labels.csv ./train_smiles.csv
ln -s /data/rsg/chemistry/jiang_guo/chemai/literature-ie/bms-kaggle/data/train_labels.csv ./train_smiles_atomtok.csv
ln -s /data/rsg/chemistry/jiang_guo/chemai/literature-ie/bms-kaggle/data/train_labels.csv ./train_smiles_spe_chembl.csv
```

## Preprocess
```
python preprocess.py
```
This will generate `train.pkl` and tokenizers.


## Train
```
CUDA_VISIBLE_DEVICES=0 python train.py \
  --format atomtok \
  --save-path output/ \
  --do-train \
  --do-test
```

## TODO:
- [x] Inference

- [ ] Encoder
  - [x] CNN
  - [x] ResNet
  - [ ] Spatial Transformer
  - [ ] Pre-training

- [ ] Decoder
  - [x] Output format: InChI vs. SMILES (atomtok vs. BPE)
  - [x] Attentional LSTM
  - [ ] Transformer (sub-molecule structure)
  - [ ] Pre-trained CLIP as reranker

- [ ] MolVec Baseline
