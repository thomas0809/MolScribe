import math
import numpy as np
import multiprocessing
from indigo import Indigo


def normalize_nodes(nodes):
    x, y = nodes[:, 0], nodes[:, 1]
    minx, maxx = min(x), max(x)
    miny, maxy = min(y), max(y)
    x = (x - minx) / max(maxx - minx, 1e-6)
    y = (maxy - y) / max(maxy - miny, 1e-6)
    return np.stack([x, y], axis=1)


def convert_smiles_to_nodes(smiles):
    indigo = Indigo()
    mol = indigo.loadMolecule(smiles)
    mol.layout()
    nodes = []
    for atom in mol.iterateAtoms():
        x, y, z = atom.xyz()
        nodes.append([x, y])
    nodes = normalize_nodes(np.array(nodes))
    return nodes


def _evaluate(arguments):
    smiles, pred = arguments
    gold = convert_smiles_to_nodes(smiles)
    if len(pred) == 0:
        pred.append([0, 0])
    n = len(gold)
    m = len(pred)
    num_node_correct = (n == m)
    pred = np.array(pred)
    dist = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            dist[i, j] = np.linalg.norm(gold[i] - pred[j])
    score = (dist.min(axis=1).mean() + dist.min(axis=0).mean()) / 2
    return score, num_node_correct


def evaluate_nodes(smiles_list, pred_nodes):
    with multiprocessing.Pool(16) as p:
        results = p.map(_evaluate, zip(smiles_list, pred_nodes))
    results = np.array(results)
    score, num_node_acc = results.mean(axis=0)
    return score, num_node_acc
