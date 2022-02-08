# Robust Molecular Image Recognition
This is the codebase for robust molecular image recognition. (under review)

## Data
Training data are collected from [PubChem](https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/) and 
[USPTO](https://bulkdata.uspto.gov/). Our processed datasets can be found here.

## Train
To train a SMILES generation model
```
bash scripts/synthetic/train_pubchem.sh
```

To train a graph generation model
```
bash scripts/synthetic/train_pubchem_joint.sh
```

## Evaluate
