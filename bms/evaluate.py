import numpy as np
import multiprocessing

import rdkit
import rdkit.Chem as Chem
rdkit.RDLogger.DisableLog('rdApp.*')
import Levenshtein
from SmilesPE.pretokenizer import atomwise_tokenizer


def get_score(y_true, y_pred):
    scores = []
    exact_match = 0
    for true, pred in zip(y_true, y_pred):
        if type(true) is not str:
            true = ""
        if type(pred) is not str:
            pred = ""
        score = Levenshtein.distance(true, pred)
        scores.append(score)
        exact_match += int(true == pred)
    avg_score = np.mean(scores)
    exact_match = exact_match / len(y_true)
    return avg_score, exact_match


def canonicalize_smiles(smiles, ignore_chiral=False, ignore_cistrans=False, replace_rgroup=True):
    if type(smiles) is not str or smiles == '':
        return '', False
    if ignore_cistrans:
        smiles = smiles.replace('/', '').replace('\\', '')
    if replace_rgroup:
        tokens = atomwise_tokenizer(smiles)
        for j, token in enumerate(tokens):
            if token[0] == '[' and token[-1] == ']':
                symbol = token[1:-1]
                if symbol[0] == 'R' and symbol[1:].isdigit():
                    tokens[j] = f'[{symbol[1:]}*]'
                elif Chem.AtomFromSmiles(token) is None:
                    tokens[j] = '*'
        smiles = ''.join(tokens)
    try:
        canon_smiles = Chem.CanonSmiles(smiles, useChiral=(not ignore_chiral))
        success = True
    except:
        canon_smiles = smiles
        success = False
    return canon_smiles, success


def convert_smiles_to_canonsmiles(
        smiles_list, ignore_chiral=False, ignore_cistrans=False, replace_rgroup=True, num_workers=16):
    with multiprocessing.Pool(num_workers) as p:
        results = p.starmap(canonicalize_smiles,
                            [(smiles, ignore_chiral, ignore_cistrans, replace_rgroup) for smiles in smiles_list],
                            chunksize=128)
    canon_smiles, success = zip(*results)
    return list(canon_smiles), np.mean(success)


class SmilesEvaluator(object):

    def __init__(self, gold_smiles):
        self.gold_smiles = gold_smiles
        self.gold_canon_smiles, self.gold_valid = convert_smiles_to_canonsmiles(gold_smiles)
        self.gold_smiles_chiral, _ = convert_smiles_to_canonsmiles(gold_smiles, ignore_chiral=True)
        self.gold_smiles_cistrans, _ = convert_smiles_to_canonsmiles(gold_smiles, ignore_cistrans=True)
        self.gold_canon_smiles = self._replace_empty(self.gold_canon_smiles)
        self.gold_smiles_chiral = self._replace_empty(self.gold_smiles_chiral)
        self.gold_smiles_cistrans = self._replace_empty(self.gold_smiles_cistrans)

    def _replace_empty(self, smiles_list):
        return [smiles if smiles is not None and type(smiles) is str and smiles != "" else '<empty>'
                for smiles in smiles_list]

    def evaluate(self, pred_smiles, include_details=False):
        results = {}
        results['gold_valid'] = self.gold_valid
        # Exact match
        results['smiles'], results['smiles_em'] = get_score(self.gold_smiles, pred_smiles)
        # Canon SMILES
        pred_canon_smiles, pred_valid = convert_smiles_to_canonsmiles(pred_smiles)
        results['canon_smiles_em'] = (np.array(self.gold_canon_smiles) == np.array(pred_canon_smiles)).mean()
        results['pred_valid'] = pred_valid
        # Ignore chirality (Graph exact match)
        pred_smiles_chiral, _ = convert_smiles_to_canonsmiles(pred_smiles, ignore_chiral=True)
        results['graph'] = (np.array(self.gold_smiles_chiral) == np.array(pred_smiles_chiral)).mean()
        # Ignore double bond cis/trans
        pred_smiles_cistrans, _ = convert_smiles_to_canonsmiles(pred_smiles, ignore_cistrans=True)
        canon_smiles_details = np.array(self.gold_smiles_cistrans) == np.array(pred_smiles_cistrans)
        if include_details:
            results['canon_smiles_details'] = canon_smiles_details
        results['canon_smiles'] = canon_smiles_details.mean()
        # Evaluate on molecules with chiral centers
        chiral = np.array([[g, p] for g, p in zip(self.gold_smiles_cistrans, pred_smiles_cistrans) if '@' in g])
        results['chiral_ratio'] = len(chiral) / len(self.gold_smiles)
        results['chiral'] = (chiral[:, 0] == chiral[:, 1]).mean() if len(chiral) > 0 else -1
        return results
