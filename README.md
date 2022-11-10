# MolScribe: Robust Molecular Structure Recognition with Image-To-Graph Generation

---
This is the repository for MolScribe, an image-to-graph model that translates a molecular image to its chemical
structure.

![MolScribe](assets/model.pdf)

[Paper](https://arxiv.org/abs/2205.14311):
```
@article{qian2022robust,
  title={Robust Molecular Image Recognition: A Graph Generation Approach},
  author={Qian, Yujie and Tu, Zhengkai and Guo, Jiang and Coley, Connor W and Barzilay, Regina},
  journal={arXiv preprint arXiv:2205.14311},
  year={2022}
}
```

## Requirements
Install the required packages
```
pip install -r requirements.txt
```
Please use the modified [Indigo](https://github.com/epam/Indigo) toolkit is included in ``indigo/``.

## Data
Training data:

| Dataset                                                                | Description                                                                                                                                   |
|------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| [USPTO](https://www.dropbox.com/s/3podz99nuwagudy/uspto_mol.zip?dl=0)  | Downloaded from [USPTO, Grant Red Book](https://bulkdata.uspto.gov/).                                                                         |
| [PubChem](https://www.dropbox.com/s/mxvm5i8139y5cvk/pubchem.zip?dl=0)  | Molecules are downloaded from [PubChem](https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/), and images are dynamically rendered during training. |

Benchmarks:

| Dataset                                                                                              | Description                                                                                                                                                                                |
|------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Synthetic](https://www.dropbox.com/s/kihxlv4mx7qplc9/synthetic.zip?dl=0) (Indigo, ChemDraw)         | Images are rendered by Indigo and ChemDraw.                                                                                                                                                |
| [Realistic](https://www.dropbox.com/s/4v8pktjcdsjsou8/real.zip?dl=0) (CLEF, UOB, USPTO, Staker, ACS) | CLEF, UOB, and USPTO are downloaded from https://github.com/Kohulan/OCSR_Review. <br/> Staker is downloaded from https://drive.google.com/drive/folders/16OjPwQ7bQ486VhdX4DWpfYzRsTGgJkSu. |
| [Perturbed](https://www.dropbox.com/s/a6fje4vc0iowwgr/perturb.zip?dl=0) (CLEF, UOB, USPTO, Staker)   | Downloaded from https://github.com/bayer-science-for-a-better-life/Img2Mol/                                                                                                                |

For training or evaluation, please download the corresponding datasets to `data/`.

## Model
Please download the [checkpoint](https://www.dropbox.com/s/suoa8kb72u23psj/swin_base_char_aux_200k.zip?dl=0) to `output/uspto/`.

- Encoder: [Swin Transformer](https://github.com/microsoft/Swin-Transformer), Swin-B.
- Decoder: Transformer, 6 layers, hidden_size=256, attn_heads=8.
- Input size: 384x384

## Usage

### Prediction
```
python predict.py \
    --model_path output/uspto/swin_base_char_aux_200k/swin_base_transformer_last.pth \
    --image_path assets/example.png
```
See python script [`predict.py`](predict.py) or jupyter notebook [`notebook/predict.ipynb`](notebook/predict.ipynb)
for more details.

### Evaluate MolScribe on benchmarks
```
bash scripts/eval_uspto_joint_chartok.sh
```
The script uses one GPU and batch size of 64 by default. If more GPUs are available, update `NUM_GPUS_PER_NODE` and 
`BATCH_SIZE` for faster evaluation.

### Evaluation Script
We have developed a standalone evaluation script [`evaluate.py`](evaluate.py). Example usage:
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

### Train MolScribe
```
bash scripts/train_uspto_joint_chartok.sh
```
The script uses four GPUs and batch size of 256 by default. It takes about one day to train the model with four A100 GPUs.

