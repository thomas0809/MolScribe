# MolScribe: Robust Molecular Structure Recognition with Image-To-Graph Generation

---
This is the repository for MolScribe, an image-to-graph model that translates a molecular image to its chemical
structure. Try our [demo](https://huggingface.co/spaces/yujieq/MolScribe) on HuggingFace first!

![MolScribe](assets/model.png)

[Paper](https://arxiv.org/abs/2205.14311):
```
@article{qian2022robust,
  title={Robust Molecular Image Recognition: A Graph Generation Approach},
  author={Qian, Yujie and Tu, Zhengkai and Guo, Jiang and Coley, Connor W and Barzilay, Regina},
  journal={arXiv preprint arXiv:2205.14311},
  year={2022}
}
```

## Quick Start
Run the following command to install the package and its dependencies:
```
git clone git@github.com:thomas0809/MolScribe.git
cd MolScribe
python setup.py install
```

Download the MolScribe checkpoint from [HuggingFace Hub](https://huggingface.co/yujieq/MolScribe/tree/main) 
and predict molecular structures:
```python
import torch
from molscribe import MolScribe
from huggingface_hub import hf_hub_download

ckpt_path = hf_hub_download("yujieq/MolScribe", "swin_base_char_aux_1m.pth")

model = MolScribe(ckpt_path, device=torch.device('cpu'))
smiles, molblock = model.predict_image_file('assets/example.png')
```
Alternatively, manually download the checkpoint and instantiate MolScribe with the local path. 

For development or reproducing the experiments, follow the instructions below.

## Requirements
Install the required packages
```
pip install -r requirements.txt
```

## Data
For training or evaluation, please download the corresponding datasets to `data/`.

Training data:

| Datasets                                                                            | Description                                                                                                                                   |
|-------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| USPTO <br> [Download](https://www.dropbox.com/s/3podz99nuwagudy/uspto_mol.zip?dl=0) | Downloaded from [USPTO, Grant Red Book](https://bulkdata.uspto.gov/).                                                                         |
| PubChem <br> [Download](https://www.dropbox.com/s/mxvm5i8139y5cvk/pubchem.zip?dl=0) | Molecules are downloaded from [PubChem](https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/), and images are dynamically rendered during training. |

Benchmarks:

| Category                                                                                | Datasets                                      | Description                                                                                                                                                                                                                                |
|-----------------------------------------------------------------------------------------|-----------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Synthetic <br> [Download](https://www.dropbox.com/s/kihxlv4mx7qplc9/synthetic.zip?dl=0) | Indigo <br> ChemDraw                          | Images are rendered by Indigo and ChemDraw.                                                                                                                                                                                                |
| Realistic <br> [Download](https://www.dropbox.com/s/4v8pktjcdsjsou8/real.zip?dl=0)      | CLEF <br> UOB <br> USPTO <br> Staker <br> ACS | CLEF, UOB, and USPTO are downloaded from https://github.com/Kohulan/OCSR_Review. <br/> Staker is downloaded from https://drive.google.com/drive/folders/16OjPwQ7bQ486VhdX4DWpfYzRsTGgJkSu. <br> ACS is a new dataset collected by ourself. |
| Perturbed <br> [Download](https://www.dropbox.com/s/a6fje4vc0iowwgr/perturb.zip?dl=0)   | CLEF <br> UOB <br> USPTO <br> Staker          | Downloaded from https://github.com/bayer-science-for-a-better-life/Img2Mol/                                                                                                                                                                |


## Model
Our model checkpoints can be downloaded from [Dropbox](https://www.dropbox.com/sh/91u508kf48cotv4/AACQden2waMXIqLwYSi8zO37a?dl=0) 
or [HuggingFace Hub](https://huggingface.co/yujieq/MolScribe/tree/main).

Model architecture:
- Encoder: [Swin Transformer](https://github.com/microsoft/Swin-Transformer), Swin-B.
- Decoder: Transformer, 6 layers, hidden_size=256, attn_heads=8.
- Input size: 384x384

Download the model checkpoint to reproduce our experiments:
```
mkdir -p ckpts
wget -P ckpts https://huggingface.co/yujieq/MolScribe/resolve/main/swin_base_char_aux_200k.pth
```

## Usage

### Prediction
```
python predict.py --model_path ckpts/swin_base_char_aux_200k.pth --image_path assets/example.png
```
MolScribe prediction interface is in [`molscribe/interface.py`](molscribe/interface.py).
See python script [`predict.py`](predict.py) or jupyter notebook [`notebook/predict.ipynb`](notebook/predict.ipynb)
for example usage.

### Evaluate MolScribe
```
bash scripts/eval_uspto_joint_chartok.sh
```
The script uses one GPU and batch size of 64 by default. If more GPUs are available, update `NUM_GPUS_PER_NODE` and 
`BATCH_SIZE` for faster evaluation.

### Train MolScribe
```
bash scripts/train_uspto_joint_chartok.sh
```
The script uses four GPUs and batch size of 256 by default. It takes about one day to train the model with four A100 GPUs.
We modified the code of [Indigo](https://github.com/epam/Indigo), which is included in `molscribe/indigo/`.


## Evaluation Script
We implement a standalone evaluation script [`evaluate.py`](evaluate.py). Example usage:
```
python evaluate.py \
    --gold_file data/real/acs.csv \
    --pred_file output/uspto/swin_base_char_aux_200k/prediction_acs.csv \
    --pred_field post_SMILES
```
The prediction should be saved in a csv file, with columns `image_id` for the index (must match the gold file),
and `SMILES` for predicted SMILES. If prediction has a different column name, specify it with `--pred_field`.

The result contains three scores:
- canon_smiles: our main metric, exact matching accuracy.
- graph: graph exact matching accuracy, ignoring tetrahedral chirality.
- chiral: exact matching accuracy on chiral molecules.
