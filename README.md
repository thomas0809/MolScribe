# Robust Molecular Image Recognition
This is the codebase for robust molecular image recognition (under review).

## Requirements
Install the required packages
```
pip install -r requirements.txt
```

Please use the modified Indigo toolkit is included in ``indigo/``.

## Data
Training data are collected from [PubChem](https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/) and 
[USPTO](https://bulkdata.uspto.gov/).

Test datasets:
- CLEF, UOB, and USPTO are downloaded from https://github.com/Kohulan/OCSR_Revie.
- Staker is downloaded from https://drive.google.com/drive/folders/16OjPwQ7bQ486VhdX4DWpfYzRsTGgJkSu 
- Perturbed datasets are downloaded from https://github.com/bayer-science-for-a-better-life/Img2Mol/

## Model
Our trained models will be released after the anonymous period.

## Train
Train a SMILES generation model on PubChem data
```
bash scripts/synthetic/train_pubchem.sh
```

Train a graph generation model on PubChem data
```
bash scripts/synthetic/train_pubchem_joint.sh
```

Train a graph generation model on PubChem + USPTO
```
bash scripts/train_uspto_joint.sh
```
