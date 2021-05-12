# BMS

## Documents
[Experiment results](https://docs.google.com/spreadsheets/d/1mBak3YB7iAUzhaTbrqkybi2RzqYKS07TRsSD0PA_ozE/edit#gid=0)

[Presentation](https://docs.google.com/presentation/d/1nqjjXtA-COamCz2O0bHV9DDuYc2ksbqf/edit)

## Data
Required data
- `data/train_labels.csv`
- `data/train`
- `data/train_smiles.csv`
- `data/train_smiles_atomtok.csv`
- `data/train_smiles_spe_chembl.csv`

```
mkdir data; cd data
export BMS_DATA=/data/rsg/chemistry/jiang_guo/chemai/literature-ie/bms-kaggle/data
ln -s ${BMS_DATA}/train ./train
ln -s ${BMS_DATA}/train_labels.csv ./train_labels.csv
ln -s ${BMS_DATA}/train_smiles_labels.csv ./train_smiles.csv
ln -s ${BMS_DATA}/train_smiles_atomtok.csv ./train_smiles_atomtok.csv
ln -s ${BMS_DATA}/train_smiles_spe_chembl.csv ./train_smiles_spe_chembl.csv
ln -s ${BMS_DATA}/sample_submission.csv ./sample_submission.csv
ln -s ${BMS_DATA}/extra_approved_InChIs.csv ./extra_approved_InChIs.csv
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

## Evaluate on validation set
```
CUDA_VISIBLE_DEVICES=0 python train.py \
  --format atomtok \
  --save-path output/resnet34_input_288_atomtok_epoch_8 \
  --input-size 288 \
  --do-valid
```

## TODO:
- [x] Inference

- [ ] Encoder
  - [x] CNN
  - [x] ResNet
  - [ ] Image Transformers (ViT, TNT, Deit, etc.)
  - [ ] Spatial Transformer
  - [ ] Pre-training

- [ ] Decoder
  - [x] Output format: InChI vs. SMILES (atomtok vs. BPE)
  - [x] Attentional LSTM
  - [x] Transformer (sub-molecule structure)
  - [ ] Pre-trained CLIP as reranker
  - [ ] Extra set of InChIs (newly provided by organizers)

- [ ] MolVec Baseline
