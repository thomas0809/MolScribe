# BMS

## Data
Required data
- `data/train_labels.csv`
- `data/train`
- `data/train_smiles.csv`
- `data/train_smiles_atomtok.csv`
- `data/train_smiles_spe_chembl.csv`

## Preprocess
```
python preprocess.py
```
This will generate `train.pkl` and tokenizers.


## Train
```
CUDA_VISIBLE_DEVICES=0 python train.py --format atomtok --save-path output/ --do-train --do-test
```
