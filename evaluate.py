import os
import json
import argparse
import pandas as pd
from bms.chemistry import SmilesEvaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_file', type=str, required=True)
    parser.add_argument('--pred_file', type=str, required=True)
    parser.add_argument('--pred_field', type=str, default='SMILES')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    gold_df = pd.read_csv(args.gold_file)
    pred_df = pd.read_csv(args.pred_file)
    if len(pred_df) != len(gold_df):
        print(f"Pred ({len(pred_df)}) and Gold ({len(gold_df)}) have different lengths!")
        # exit()
    image2goldidx = {image_id: idx for idx, image_id in enumerate(gold_df['image_id'])}
    image2predidx = {image_id: idx for idx, image_id in enumerate(pred_df['image_id'])}
    for image_id in gold_df['image_id']:
        if image_id not in image2predidx:
            pred_df = pred_df.append({'image_id': image_id, args.pred_field: ""}, ignore_index=True)
    image2predidx = {image_id: idx for idx, image_id in enumerate(pred_df['image_id'])}
    pred_df = pred_df.reindex([image2predidx[image_id] for image_id in gold_df['image_id']])
    evaluator = SmilesEvaluator(gold_df['SMILES'])
    scores = evaluator.evaluate(pred_df[args.pred_field])
    print(json.dumps(scores, indent=4))
