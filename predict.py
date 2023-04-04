import argparse
import json
import torch
from molscribe import MolScribe

import warnings 
warnings.filterwarnings('ignore')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default=None, required=True)
    parser.add_argument('--image_path', type=str, default=None, required=True)
    parser.add_argument('--return_confidence', action='store_true')
    parser.add_argument('--return_atoms_bonds', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda')
    model = MolScribe(args.model_path, device)
    output = model.predict_image_file(
        args.image_path, return_atoms_bonds=args.return_atoms_bonds, return_confidence=args.return_confidence)
    for key, value in output.items():
        print(f"{key}:")
        print(value + '\n' if isinstance(value, str) else json.dumps(value) + '\n')
