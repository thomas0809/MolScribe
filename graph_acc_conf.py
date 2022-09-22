import os
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from bms.chemistry import SmilesEvaluator


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gold_file', type=str, required=True)
    parser.add_argument('--pred_file', type=str, required=True)
    parser.add_argument('--pred_field', type=str, default='SMILES')
    parser.add_argument('--acc_metric', type=str, default='canon_smiles')
    parser.add_argument('--score_field', type=str, default='SMILES_score')
    parser.add_argument('--low', type=float, default=0.)
    parser.add_argument('--high', type=float, default=1.)
    parser.add_argument('--step', type=float, default=.01)
    parser.add_argument('--save_fig', type=str)
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
    conf_thres, acc = [], []
    for c in np.arange(args.low, args.high, args.step):
        print("Current confidence threshold: {:.4f}".format(c))
        passing_idx = pred_df[args.score_field] >= c
        if not passing_idx.any():
            print("    Predictions above this threshold don't exist")
            break
        gold_df, pred_df = gold_df[passing_idx], pred_df[passing_idx]
        evaluator = SmilesEvaluator(gold_df['SMILES'])
        scores = evaluator.evaluate(pred_df[args.pred_field])
        conf_thres.append(c)
        acc.append(scores[args.acc_metric])
        print("    Model accuracy: {:.4f}".format(acc[-1]))
    plt.plot(conf_thres, acc)
    plt.title("Model accuracy vs. confidence threshold")
    plt.ylabel(f"Model accuracy ({args.acc_metric})")
    plt.xlabel(f"Confidence threshold ({args.score_field})")
    if args.save_fig is None:
        plt.show()
    else:
        plt.savefig(args.save_fig)
