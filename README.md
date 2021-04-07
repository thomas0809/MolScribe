# BMS

## Data
Required data
- `data/train_labels.csv`
- `data/train`

```
mkdir data; cd data
ln -s /data/rsg/chemistry/jiang_guo/chemai/literature-ie/bms-kaggle/data/train ./train
ln -s /data/rsg/chemistry/jiang_guo/chemai/literature-ie/bms-kaggle/data/train_labels.csv ./train_labels.csv
```

## Preprocess
`python preprocess.py`

## Train
`CUDA_VISIBLE_DEVICES=0 python train.py`

