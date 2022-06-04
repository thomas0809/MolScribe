# BMS

## Documents

### BMS

[Experiment results](https://docs.google.com/spreadsheets/d/1mBak3YB7iAUzhaTbrqkybi2RzqYKS07TRsSD0PA_ozE/edit#gid=0)

[Presentation](https://docs.google.com/presentation/d/1nqjjXtA-COamCz2O0bHV9DDuYc2ksbqf/edit)

## Data
```
export DATA=/Mounts/rbg-storage1/users/yujieq/bms/data/molbank
mkdir data
ln -s $DATA data/molbank
```

## Example script
```
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/train_uspto_1msmall.sh
```
