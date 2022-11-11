import argparse
import torch
from molscribe import MolScribe

import warnings 
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, required=True)
    parser.add_argument('--image_path', type=str, default=None, required=True)
    args = parser.parse_args()

    device = torch.device('cuda')
    model = MolScribe(args.model_path, device)
    smiles, molblock = model.predict_image_file(args.image_path)
    print(smiles)
    print(molblock)
