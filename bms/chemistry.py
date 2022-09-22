# import sys
# sys.path[0] = '/Mounts/rbg-storage1/users/urop/zli11010/bms'

import cv2
import copy
import numpy as np
import multiprocessing

from indigo import Indigo
import rdkit
import rdkit.Chem as Chem
import rdkit.Chem.AllChem as AllChem

rdkit.RDLogger.DisableLog('rdApp.*')
import Levenshtein
from SmilesPE.pretokenizer import atomwise_tokenizer

from bms.constants import RGROUP_SYMBOLS, PLACEHOLDER_ATOMS, SUBSTITUTIONS, ABBREVIATIONS, VALENCES, FORMULA_REGEX, \
    FORMULA_REGEX_BACKUP


def is_valid_mol(s, format_='atomtok'):
    if format_ == 'atomtok':
        mol = Chem.MolFromSmiles(s)
    elif format_ == 'inchi':
        if not s.startswith('InChI=1S'):
            s = f"InChI=1S/{s}"
        mol = Chem.MolFromInchi(s)
    else:
        raise NotImplemented
    return mol is not None


def _convert_smiles_to_inchi(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        inchi = Chem.MolToInchi(mol)
    except:
        inchi = None
    return inchi


def convert_smiles_to_inchi(smiles_list, num_workers=16):
    with multiprocessing.Pool(num_workers) as p:
        inchi_list = p.map(_convert_smiles_to_inchi, smiles_list, chunksize=128)
    n_success = sum([x is not None for x in inchi_list])
    r_success = n_success / len(inchi_list)
    inchi_list = [x if x else 'InChI=1S/H2O/h1H2' for x in inchi_list]
    return inchi_list, r_success


def canonicalize_smiles(smiles, ignore_chiral=False, ignore_cistrans=False):
    if type(smiles) is not str or smiles == '':
        return '', False
    if ignore_cistrans:
        smiles = smiles.replace('/', '').replace('\\', '')
    # rlist = RGROUP_SYMBOLS
    # rdict = {}
    # for i, symbol in enumerate(rlist):
    #     rdict[f'[{symbol}]'] = f'[*:{i}]'
    # for a, b in rdict.items():
    #     smiles = smiles.replace(a, b)
    success = False
    try:
        canon_smiles = Chem.CanonSmiles(smiles, useChiral=(not ignore_chiral))
        success = True
    except:
        canon_smiles = smiles
    return canon_smiles, success


def convert_smiles_to_canonsmiles(smiles_list, ignore_chiral=False, ignore_cistrans=False, num_workers=16):
    with multiprocessing.Pool(num_workers) as p:
        results = p.starmap(canonicalize_smiles,
                            [(smiles, ignore_chiral, ignore_cistrans) for smiles in smiles_list],
                            chunksize=128)
    canon_smiles, success = zip(*results)
    return list(canon_smiles), np.mean(success)


def get_canon_smiles_score(gold_smiles, pred_smiles, ignore_chiral=False, num_workers=16):
    gold_canon_smiles, gold_success = convert_smiles_to_canonsmiles(gold_smiles, ignore_chiral)
    pred_canon_smiles, pred_success = convert_smiles_to_canonsmiles(pred_smiles, ignore_chiral)
    score = (np.array(gold_canon_smiles) == np.array(pred_canon_smiles)).mean()
    if ignore_chiral:
        return score
    # ignore double bond cis/trans
    gold_canon_smiles, _ = convert_smiles_to_canonsmiles(gold_canon_smiles, ignore_cistrans=True)
    pred_canon_smiles, _ = convert_smiles_to_canonsmiles(pred_canon_smiles, ignore_cistrans=True)
    score_corrected = (np.array(gold_canon_smiles) == np.array(pred_canon_smiles)).mean()
    chiral = np.array([[g, p] for g, p in zip(gold_canon_smiles, pred_canon_smiles) if '@' in g])
    score_chiral = (chiral[:, 0] == chiral[:, 1]).mean() if len(chiral) > 0 else -1
    return score, score_corrected, score_chiral


def merge_inchi(inchi1, inchi2):
    replaced = 0
    inchi1 = copy.deepcopy(inchi1)
    for i in range(len(inchi1)):
        if inchi1[i] == 'InChI=1S/H2O/h1H2':
            inchi1[i] = inchi2[i]
            replaced += 1
    return inchi1, replaced


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


def _get_num_atoms(smiles):
    try:
        return Chem.MolFromSmiles(smiles).GetNumAtoms()
    except:
        return 0


def get_num_atoms(smiles, num_workers=16):
    if type(smiles) is str:
        return _get_num_atoms(smiles)
    with multiprocessing.Pool(num_workers) as p:
        num_atoms = p.map(_get_num_atoms, smiles)
    return num_atoms


def get_edge_prediction(edge_prob):
    n = len(edge_prob)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(5):
                edge_prob[i][j][k] = (edge_prob[i][j][k] + edge_prob[j][i][k]) / 2
                edge_prob[j][i][k] = edge_prob[i][j][k]
            edge_prob[i][j][5] = (edge_prob[i][j][5] + edge_prob[j][i][6]) / 2
            edge_prob[i][j][6] = (edge_prob[i][j][6] + edge_prob[j][i][5]) / 2
            edge_prob[j][i][5] = edge_prob[i][j][6]
            edge_prob[j][i][6] = edge_prob[i][j][5]
    try:
        return np.argmax(edge_prob, axis=2).tolist()
    except Exception as e:
        import pickle as pkl
        with open("debug.pkl", "wb") as f:
            pkl.dump(edge_prob, f)
        raise Exception from e


def get_edge_scores(edge_prob):
    n = len(edge_prob)
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(5):
                assert edge_prob[j][i][k] == edge_prob[i][j][k]
            assert edge_prob[j][i][5] == edge_prob[i][j][6] and edge_prob[j][i][6] == edge_prob[i][j][5]
    max_prob = np.max(edge_prob, axis=2)
    return np.prod(max_prob).item(), max_prob.tolist()


def normalize_nodes(nodes, flip_y=True):
    x, y = nodes[:, 0], nodes[:, 1]
    minx, maxx = min(x), max(x)
    miny, maxy = min(y), max(y)
    x = (x - minx) / max(maxx - minx, 1e-6)
    if flip_y:
        y = (maxy - y) / max(maxy - miny, 1e-6)
    else:
        y = (y - miny) / max(maxy - miny, 1e-6)
    return np.stack([x, y], axis=1)


def convert_smiles_to_nodes(smiles):
    try:
        indigo = Indigo()
        mol = indigo.loadMolecule(smiles)
        coords, symbols = [], []
        for atom in mol.iterateAtoms():
            symbols.append(atom.symbol())
    except:
        return [], []
    return coords, symbols


def _evaluate_nodes(smiles, coords, symbols):
    gold_coords, gold_symbols = convert_smiles_to_nodes(smiles)
    n = len(gold_symbols)
    m = len(symbols)
    num_node_correct = (n == m)
    # coords = np.array(coords)
    # dist = np.zeros((n, m))
    # for i in range(n):
    #     for j in range(m):
    #         dist[i, j] = np.linalg.norm(gold_coords[i] - coords[j])
    # score = (dist.min(axis=1).mean() + dist.min(axis=0).mean()) / 2 if n * m > 0 else 0
    score = 0
    symbols_em = (sorted(symbols) == sorted(gold_symbols))
    return score, num_node_correct, symbols_em


def evaluate_nodes(smiles_list, node_coords, node_symbols, num_workers=16):
    with multiprocessing.Pool(num_workers) as p:
        results = p.starmap(_evaluate_nodes,
                            zip(smiles_list, node_coords, node_symbols),
                            chunksize=128)
    results = np.array(results)
    score, num_node_acc, symbols_em = results.mean(axis=0)
    return score, num_node_acc, symbols_em


def _sub_condensed(smiles):
    tokens = atomwise_tokenizer(smiles)
    for j, token in enumerate(tokens):
        if token[0] == '[' and token[-1] == ']':
            symbol = token[1:-1]
            if symbol[0] == 'R' and symbol[1:].isdigit():
                tokens[j] = f'[{symbol[1:]}*]'
            elif symbol in RGROUP_SYMBOLS:
                tokens[j] = '*'
            elif Chem.AtomFromSmiles(token) is None:
                sub_smiles, _, _, _, success = _condensed_formula_to_smiles(token[1:-1], _get_num_bonds(tokens, j))
                if success:
                    # print(tokens[j], sub_smiles)
                    tokens[j] = sub_smiles
    return ''.join(tokens)


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


def _convert_graph_to_smiles_simple(coords, symbols, edges):
    mol = Chem.RWMol()
    n = len(symbols)
    ids = []
    for i in range(n):
        # TODO: R-group, functional group
        try:
            idx = mol.AddAtom(Chem.Atom(symbols[i]))
        except:
            idx = mol.AddAtom(Chem.Atom('C'))
        ids.append(idx)
    for i in range(n):
        for j in range(n):
            if i < j and edges[i][j] != 0:
                if edges[i][j] in [1, 5, 6]:
                    mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
                elif edges[i][j] == 2:
                    mol.AddBond(ids[i], ids[j], Chem.BondType.DOUBLE)
                elif edges[i][j] == 3:
                    mol.AddBond(ids[i], ids[j], Chem.BondType.TRIPLE)
                elif edges[i][j] == 4:
                    mol.AddBond(ids[i], ids[j], Chem.BondType.AROMATIC)
    try:
        mol = mol.GetMol()
        pred_smiles = Chem.MolToSmiles(mol)
    except:
        pred_smiles = ''
    return pred_smiles


def _verify_chirality(mol, coords, symbols, edges, debug=False):
    try:
        n = mol.GetNumAtoms()
        # Make a temp mol to find chiral centers
        mol_tmp = mol.GetMol()
        Chem.SanitizeMol(mol_tmp)

        chiral_centers = Chem.FindMolChiralCenters(
            mol_tmp, includeUnassigned=True, includeCIP=False, useLegacyImplementation=False)
        chiral_center_ids = [idx for idx, _ in chiral_centers]  # List[Tuple[int, any]] -> List[int]
        # print(chiral_center_ids)
        # [print(e) for e in edges]

        # correction to clear pre-condition violation (for some corner cases)
        for bond in mol.GetBonds():
            if bond.GetBondType() == Chem.BondType.SINGLE:
                bond.SetBondDir(Chem.BondDir.NONE)

        # Create conformer from 2D coordinate
        conf = Chem.Conformer(n)
        conf.Set3D(True)
        for i, (x, y) in enumerate(coords):
            conf.SetAtomPosition(i, (x, 1 - y, 0))
        mol.AddConformer(conf)
        Chem.SanitizeMol(mol)
        Chem.AssignStereochemistryFrom3D(mol)
        # NOTE: seems that only AssignStereochemistryFrom3D can handle double bond E/Z
        # So we do this first, remove the conformer and add back the 2D conformer for chiral correction

        mol.RemoveAllConformers()
        conf = Chem.Conformer(n)
        conf.Set3D(False)
        for i, (x, y) in enumerate(coords):
            conf.SetAtomPosition(i, (x, 1 - y, 0))
        mol.AddConformer(conf)

        # Magic, infering chirality from coordinates and BondDir. DO NOT CHANGE.
        Chem.SanitizeMol(mol)
        Chem.AssignChiralTypesFromBondDirs(mol)
        Chem.AssignStereochemistry(mol, force=True)

        # Second loop to reset any wedge/dash bond to be starting from the chiral center)
        for i in chiral_center_ids:
            for j in range(n):
                if edges[i][j] == 5:
                    # print(f"5, i: {i}, j: {j}")
                    # assert edges[j][i] == 6
                    mol.RemoveBond(i, j)
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                    mol.GetBondBetweenAtoms(i, j).SetBondDir(Chem.BondDir.BEGINWEDGE)
                elif edges[i][j] == 6:
                    # print(f"6, i: {i}, j: {j}")
                    # assert edges[j][i] == 5
                    mol.RemoveBond(i, j)
                    mol.AddBond(i, j, Chem.BondType.SINGLE)
                    mol.GetBondBetweenAtoms(i, j).SetBondDir(Chem.BondDir.BEGINDASH)
            Chem.AssignChiralTypesFromBondDirs(mol)
            Chem.AssignStereochemistry(mol, force=True)

        # reset chiral tags for nitrogen
        for atom in mol.GetAtoms():
            if atom.GetSymbol() == "N":
                atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
        mol = mol.GetMol()

    except Exception as e:
        if debug:
            raise e
        pass
    return mol


def _parse_tokens(tokens: list):
    """
    Parse tokens of condensed formula into list of pairs `(elt, num)`
    where `num` is the multiplicity of the atom (or nested condensed formula) `elt`
    Used by `_parse_formula`, which does the same thing but takes a formula in string form as input
    """
    elements = []
    i = j = 0
    while i < len(tokens):
        if tokens[i] == '(':
            while j < len(tokens) and tokens[j] != ')':
                j += 1
            elt = _parse_tokens(tokens[i + 1:j])
        else:
            elt = tokens[i]
        j += 1
        if j < len(tokens) and tokens[j].isnumeric():
            num = int(tokens[j])
            j += 1
        else:
            num = 1
        elements.append((elt, num))
        i = j
    return elements


def _parse_formula(formula: str):
    """
    Parse condensed formula into list of pairs `(elt, num)`
    where `num` is the subscript to the atom (or nested condensed formula) `elt`
    Example: "C2H4O" -> [('C', 2), ('H', 4), ('O', 1)]
    """
    tokens = FORMULA_REGEX.findall(formula)
    if ''.join(tokens) != formula:
        tokens = FORMULA_REGEX_BACKUP.findall(formula)
    return _parse_tokens(tokens)


def _expand_carbon(elements: list):
    """
    Given list of pairs `(elt, num)`, output single list of all atoms in order,
    expanding carbon sequences (CaXb where a > 1 and X is halogen) if necessary
    Example: [('C', 2), ('H', 4), ('O', 1)] -> ['C', 'H', 'H', 'C', 'H', 'H', 'O'])
    """
    expanded = []
    i = 0
    while i < len(elements):
        elt, num = elements[i]
        # expand carbon sequence
        if elt == 'C' and num > 1 and i + 1 < len(elements):
            next_elt, next_num = elements[i + 1]
            quotient, remainder = next_num // num, next_num % num
            for _ in range(num):
                expanded.append('C')
                for _ in range(quotient):
                    expanded.append(next_elt)
            for _ in range(remainder):
                expanded.append(next_elt)
            i += 2
        # recurse if `elt` itself is a list (nested formula)
        elif isinstance(elt, list):
            new_elt = _expand_carbon(elt)
            for _ in range(num):
                expanded.append(new_elt)
            i += 1
        # simplest case: simply append `elt` `num` times
        else:
            for _ in range(num):
                expanded.append(elt)
            i += 1
    return expanded


def _expand_abbreviation(abbrev):
    """
    Expand abbreviation into its SMILES; also converts [Rn] to [n*]
    Used in `_condensed_formula_list_to_smiles` when encountering abbrev. in condensed formula
    """
    if abbrev in ABBREVIATIONS:
        return ABBREVIATIONS[abbrev].smiles
    if abbrev in RGROUP_SYMBOLS or (abbrev[0] == 'R' and abbrev[1:].isdigit()):
        if abbrev[1:].isdigit():
            return f'[{abbrev[1:]}*]'
        return '*'
    return f'[{abbrev}]'


def _get_bond_symb(bond_num):
    """
    Get SMILES symbol for a bond given bond order
    Used in `_condensed_formula_list_to_smiles` while writing the SMILES string
    """
    if bond_num == 0:
        return '.'
    if bond_num == 1:
        return ''
    if bond_num == 2:
        return '='
    if bond_num == 3:
        return '#'
    return ''


def _count_non_H(atoms):
    """
    Count non-H atoms among `atoms` (symbol or list of symbols/lists)
    Used in `_condensed_formula_list_to_smiles` to determine index of last atom token in SMILES
    """
    if isinstance(atoms, str):
        return int(atoms != 'H')
    return sum(_count_non_H(atom) for atom in atoms)


def _condensed_formula_list_to_smiles(formula: list, start_bond, end_bond=None, direction=None, R_valence=[1]):
    """
    Converts condensed formula (in the form of a list of symbols) to smiles
    Input:
    `formula`: e.g. ['C', 'H', 'H', 'N', ['C', 'H', 'H', 'H'], ['C', 'H', 'H', 'H']] for CH2N(CH3)2
    `start_bond`: # bonds attached to beginning of formula
    `end_bond`: # bonds attached to end of formula (deduce automatically if None)
    `direction` (1, -1, or None): direction in which to process the list (1: left to right; -1: right to left; None: deduce automatically)
    `R_valence`: The list of possible valences of an R-group (default: [1])
    Returns:
    `smiles`: smiles corresponding to input condensed formula
    `bonds_left`: bonds remaining at the end of the formula (for connecting back to main molecule); should equal `end_bond` if specified
    `last_flat_idx`: index of last atom in list of atom tokens of output SMILES
    `direction` (1 or -1): direction in which processing the formula was successful; should equal input `direction` if specified
    `success` (bool): whether conversion was successful
    """
    # `direction` not specified: try left to right; if fails, try right to left
    if direction is None:
        for R_val_choice in [[1], [1, 2], [1, 2, 3]]:
            for dir_choice in [1, -1]:
                result = _condensed_formula_list_to_smiles(formula, start_bond, end_bond, dir_choice, R_val_choice)
                if result[4]:
                    return result
        return result
    assert direction == 1 or direction == -1

    def dfs(smiles, cur_idx, cur_flat_idx, bonds_left, add_idx, add_flat_idx):
        """
        `smiles`: SMILES string so far
        `cur_idx`: index (in list `formula`) of current atom (i.e. atom to which subsequent atoms are being attached)
        `cur_flat_idx`: index of current atom in list of atom tokens of SMILES so far
        `bonds_left`: bonds remaining on current atom for subsequent atoms to be attached to
        `add_idx`: index (in list `formula`) of atom to be attached to current atom
        `add_flat_idx`: index of atom to be added in list of atom tokens of SMILES so far
        Note: "atom" could refer to nested condensed formula (e.g. CH3 in CH2N(CH3)2)
        """

        # end of formula: return result
        if (direction == 1 and add_idx == len(formula)) or (direction == -1 and add_idx == -1):
            if end_bond is not None and end_bond != bonds_left:
                return smiles, bonds_left, cur_flat_idx, direction, False
            return smiles, bonds_left, cur_flat_idx, direction, True

        # no more bonds but there are atoms remaining: conversion failed
        if bonds_left <= 0:
            return smiles, bonds_left, cur_flat_idx, direction, False
        to_add = formula[add_idx]  # atom to be added to current atom

        if isinstance(to_add, list):  # "atom" added is a list (i.e. nested condensed formula): assume valence of 1
            if bonds_left > 1:  # "atom" added does not use up remaining bonds of current atom
                # get smiles of "atom" (which is itself a condensed formula)
                add_str, _, _, _, success = _condensed_formula_list_to_smiles(to_add, 1, 0, direction)
                if not success:
                    return smiles, bonds_left, cur_flat_idx, direction, False
                # put smiles of "atom" in parentheses and append to smiles so far; go to next atom to add to current atom
                return dfs(smiles + f'({add_str})', cur_idx, cur_flat_idx, bonds_left - 1, add_idx + direction,
                           add_flat_idx + _count_non_H(to_add))
            # "atom" added uses up remaining bonds of current atom
            add_str, bonds_left, _, _, success = _condensed_formula_list_to_smiles(to_add, 1, None,
                                                                                   direction)  # get smiles of "atom" and bonds left on it
            if not success:
                return smiles, bonds_left, cur_flat_idx, direction, False
            # append smiles of "atom" (without parentheses) to smiles; it becomes new current atom
            return dfs(smiles + add_str, add_idx, add_flat_idx, bonds_left, add_idx + direction,
                       add_flat_idx + _count_non_H(to_add))

        # atom added is a single symbol (as opposed to nested condensed formula)
        for val in VALENCES.get(to_add, R_valence):  # try all possible valences of atom added
            add_str = _expand_abbreviation(to_add)  # expand to smiles if symbol is abbreviation
            if bonds_left > val:  # atom added does not use up remaining bonds of current atom; go to next atom to add to current atom
                result = dfs(smiles + f'({_get_bond_symb(val)}{add_str})', cur_idx, cur_flat_idx,
                             bonds_left - val, add_idx + direction, add_flat_idx + _count_non_H(to_add))
            else:  # atom added uses up remaining bonds of current atom; it becomes new current atom
                result = dfs(smiles + _get_bond_symb(bonds_left) + add_str, add_idx, add_flat_idx,
                             val - bonds_left, add_idx + direction, add_flat_idx + _count_non_H(to_add))
            if result[4]:
                return result
        return smiles, bonds_left, cur_flat_idx, direction, False

    cur_idx = -1 if direction == 1 else len(formula)
    add_idx = 0 if direction == 1 else len(formula) - 1
    return dfs('', cur_idx, -1, start_bond, add_idx, 0)


def _condensed_formula_to_smiles(formula: str, total_bonds: int):
    """
    Converts condensed formula (in the form of a string) to smiles
    Input:
    `formula`: condensed formula (e.g. "CH2N(CH3)2")
    `total_bonds`: total number of bonds used to attach to main body of molecule
    Returns:
    `smiles`: smiles corresponding to input condensed formula
    `bonds_left`: bonds remaining at the end of the formula (for connecting back to main molecule)
    `cur_flat_idx`:
    `direction` (1 or -1): direction in which processing the formula was successful
    `success` (bool): whether conversion was successful
    """
    formula_list = _expand_carbon(_parse_formula(formula))  # convert condensed formula to a list of atoms
    smiles, success = '', False
    for start_bond, end_bond in zip(range(1, total_bonds + 1),
                                    range(total_bonds - 1, -1, -1)):  # try all pairs of (start_bonds, end_bond)
        smiles, _, _, _, success = _condensed_formula_list_to_smiles(formula_list, start_bond, end_bond)
        if success:
            return smiles, success
    return smiles, success


def get_smiles_from_symbol(symbol, mol, atom, bonds):

    if symbol in ABBREVIATIONS:
        return ABBREVIATIONS[symbol].smiles

    # TODO (zhening): change this function; mol, atom, bonds should have all required information.
    #  try to make sure that the smiles is (1) valid, (2) the first atom is to be bonded to the molecule,
    #  (3) specify other missing bonds, i.e. [CH2][CH2]
    smiles, success = _condensed_formula_to_smiles(symbol, len(bonds))
    if success:
        return smiles
    return None


def _replace_functional_group(smiles):
    smiles = smiles.replace('<unk>', 'C')
    for i, r in enumerate(RGROUP_SYMBOLS):
        symbol = f'[{r}]'
        if symbol in smiles:
            if r[0] == 'R' and r[1:].isdigit():
                smiles = smiles.replace(symbol, f'[{int(r[1:])}*]')
            else:
                smiles = smiles.replace(symbol, '*')
    # For unknown tokens (i.e. rdkit cannot parse), replace them with [{isotope}*], where isotope is an identifier.
    tokens = atomwise_tokenizer(smiles)
    new_tokens = []
    mappings = {}  # isotope : symbol
    isotope = 50
    for token in tokens:
        if token[0] == '[':
            if Chem.AtomFromSmiles(token) is None:
                while f'[{isotope}*]' in smiles or f'[{isotope}*]' in new_tokens:
                    isotope += 1
                placeholder = f'[{isotope}*]'
                mappings[isotope] = token[1:-1]
                new_tokens.append(placeholder)
                continue
        new_tokens.append(token)
    smiles = ''.join(new_tokens)
    return smiles, mappings


BOND_TYPES = {1: Chem.rdchem.BondType.SINGLE, 2: Chem.rdchem.BondType.DOUBLE, 3: Chem.rdchem.BondType.TRIPLE}


def convert_smiles_to_mol(smiles):
    if smiles is None:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
    except:
        return None
    return mol


def _expand_functional_group(mol, mappings):

    def _need_expand(mol, mappings):
        return any([len(Chem.GetAtomAlias(atom)) > 0 for atom in mol.GetAtoms()]) or len(mappings) > 0

    if _need_expand(mol, mappings):
        mol_w = Chem.RWMol(mol)
        for i, atom in enumerate(mol_w.GetAtoms()):  # reset radical electrons
            atom.SetNumRadicalElectrons(0)

        flag = True
        cnt = 0
        while flag:
            flag = False
            cnt += 1
            if cnt == 20:
                print(Chem.MolToSmiles(mol))
                print(mappings)
            for i, atom in enumerate(mol_w.GetAtoms()):
                if atom.GetSymbol() == '*':
                    symbol = Chem.GetAtomAlias(atom)
                    isotope = atom.GetIsotope()
                    if isotope > 0 and isotope in mappings:
                        symbol = mappings[isotope]
                    if not(isinstance(symbol, str) and len(symbol) > 0):
                        continue
                    # rgroups do not need to be expanded
                    if symbol in RGROUP_SYMBOLS:
                        continue
    
                    bonds = atom.GetBonds()
    
                    sub_smiles = get_smiles_from_symbol(symbol, mol, atom, bonds)

                    # create mol object for abbreviation/condensed formula from its SMILES
                    mol_r = convert_smiles_to_mol(sub_smiles)

                    if mol_r is None:
                        # TODO: set this atom to C
                        atom.SetAtomicNum(6)
                        atom.SetIsotope(0)
                        continue
    
                    # remove bonds connected to abbreviation/condensed formula
                    adjacent_indices = [bond.GetOtherAtomIdx(i) for bond in bonds]
                    for adjacent_idx in adjacent_indices:
                        mol_w.RemoveBond(i, adjacent_idx)
    
                    adjacent_atoms = [mol_w.GetAtomWithIdx(adjacent_idx) for adjacent_idx in adjacent_indices]
                    for adjacent_atom, bond in zip(adjacent_atoms, bonds):
                        adjacent_atom.SetNumRadicalElectrons(int(bond.GetBondTypeAsDouble()))

                    # get indices of atoms of main body that connect to substituent
                    bonding_atoms_w = adjacent_indices
                    # assume indices are concated after combine mol_w and mol_r
                    bonding_atoms_r = [mol_w.GetNumAtoms()]
                    for atm in mol_r.GetAtoms():
                        if atm.GetNumRadicalElectrons() and atm.GetIdx() > 0:
                            bonding_atoms_r.append(mol_w.GetNumAtoms() + atm.GetIdx())
    
                    # combine main body and substituent into a single molecule object
                    combo = Chem.CombineMols(mol_w, mol_r)
    
                    # connect substituent to main body with bonds
                    mol_w = Chem.RWMol(combo)
                    if len(bonding_atoms_r) == 1:  # substituent uses one atom to bond to main body
                        for atm in bonding_atoms_w:
                            bond_order = mol_w.GetAtomWithIdx(atm).GetNumRadicalElectrons()
                            mol_w.AddBond(atm, bonding_atoms_r[0], order=BOND_TYPES[bond_order])
                    else:
                        # TODO (zhening)
                        pass
    
                    # reset radical electrons
                    for atm in bonding_atoms_w:
                        mol_w.GetAtomWithIdx(atm).SetNumRadicalElectrons(0)
                    for atm in bonding_atoms_r:
                        mol_w.GetAtomWithIdx(atm).SetNumRadicalElectrons(0)
    
                    mol_w.RemoveAtom(i)
                    flag = True
                    break

        smiles = Chem.MolToSmiles(mol_w)
        mol = mol_w.GetMol()
    else:
        smiles = Chem.MolToSmiles(mol)
    return smiles, mol


def _convert_graph_to_smiles(coords, symbols, edges, image=None, debug=False):
    mol = Chem.RWMol()
    n = len(symbols)
    ids = []
    for i in range(n):
        symbol = symbols[i]
        if symbol[0] == '[':
            symbol = symbol[1:-1]
        if symbol in RGROUP_SYMBOLS:
            atom = Chem.Atom("*")
            if symbol[0] == 'R' and symbol[1:].isdigit():
                atom.SetIsotope(int(symbol[1:]))
            Chem.SetAtomAlias(atom, symbol)
        else:
            try:  # try to get SMILES of atom
                atom = Chem.AtomFromSmiles(symbols[i])
                atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
            except:  # otherwise, abbreviation or condensed formula
                atom = Chem.Atom("*")
                Chem.SetAtomAlias(atom, symbol)

        idx = mol.AddAtom(atom)
        assert idx == i
        ids.append(idx)

    for i in range(n):
        for j in range(i + 1, n):
            if edges[i][j] == 1:
                mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
            elif edges[i][j] == 2:
                mol.AddBond(ids[i], ids[j], Chem.BondType.DOUBLE)
            elif edges[i][j] == 3:
                mol.AddBond(ids[i], ids[j], Chem.BondType.TRIPLE)
            elif edges[i][j] == 4:
                mol.AddBond(ids[i], ids[j], Chem.BondType.AROMATIC)
            elif edges[i][j] == 5:
                mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
                mol.GetBondBetweenAtoms(ids[i], ids[j]).SetBondDir(Chem.BondDir.BEGINWEDGE)
            elif edges[i][j] == 6:
                mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
                mol.GetBondBetweenAtoms(ids[i], ids[j]).SetBondDir(Chem.BondDir.BEGINDASH)

    pred_smiles = '<invalid>'

    try:
        # TODO: move to an util function
        if image is not None:
            height, width, _ = image.shape
            ratio = width / height
            coords = [[x * ratio * 10, y * 10] for x, y in coords]
        mol = _verify_chirality(mol, coords, symbols, edges, debug)
        # molblock is obtained before expanding func groups, otherwise the expanded group won't have coordinates.
        # TODO: make sure molblock has the abbreviation information
        pred_molblock = Chem.MolToMolBlock(mol)
        pred_smiles, mol = _expand_functional_group(mol, {})
        success = True
    except Exception as e:
        if debug:
            raise e
        pred_molblock = ''
        success = False

    if debug:
        return pred_smiles, pred_molblock, mol
    return pred_smiles, pred_molblock, success


def convert_graph_to_smiles(coords, symbols, edges, images=None, num_workers=16, simple=False):
    fn = _convert_graph_to_smiles_simple if simple else _convert_graph_to_smiles
    with multiprocessing.Pool(num_workers) as p:
        if images is None:
            results = p.starmap(fn, zip(coords, symbols, edges), chunksize=128)
        else:
            results = p.starmap(fn, zip(coords, symbols, edges, images), chunksize=128)
    smiles_list, molblock_list, success = zip(*results)
    r_success = np.mean(success)
    return smiles_list, molblock_list, r_success


def _postprocess_smiles(smiles, coords=None, symbols=None, edges=None, molblock=False, debug=False):
    if type(smiles) is not str or smiles == '':
        return '', False
    mol = None
    pred_molblock = ''
    try:
        pred_smiles = smiles
        pred_smiles, mappings = _replace_functional_group(pred_smiles)
        if coords is not None and symbols is not None and edges is not None:
            pred_smiles = pred_smiles.replace('@', '').replace('/', '').replace('\\', '')
            mol = Chem.RWMol(Chem.MolFromSmiles(pred_smiles, sanitize=False))
            mol = _verify_chirality(mol, coords, symbols, edges, debug)
        else:
            mol = Chem.MolFromSmiles(pred_smiles, sanitize=False)
        # pred_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        if molblock:
            pred_molblock = Chem.MolToMolBlock(mol)
        pred_smiles, mol = _expand_functional_group(mol, mappings)
        success = True
    except Exception as e:
        if debug:
            print(e)
        pred_smiles = smiles
        pred_molblock = ''
        success = False
    if debug:
        return pred_smiles, pred_molblock, mol
    return pred_smiles, pred_molblock, success


def postprocess_smiles(smiles, coords=None, symbols=None, edges=None, molblock=False, num_workers=16):
    with multiprocessing.Pool(num_workers) as p:
        if coords is not None and symbols is not None and edges is not None:
            results = p.starmap(_postprocess_smiles, zip(smiles, coords, symbols, edges, [molblock] * len(smiles)),
                                chunksize=128)
        else:
            results = p.map(_postprocess_smiles, smiles, chunksize=128)
    smiles_list, molblock_list, success = zip(*results)
    r_success = np.mean(success)
    return smiles_list, molblock_list, r_success


if __name__ == "__main__":
    test_convert_condensed = True
    if test_convert_condensed:
        formulas = ["NHOH", "CONH2", "F3CH2CO", "SO3H", "(CH2)2", "SO2NH2", "HNOC", "CONH", "CH2CO2CH3", "CO2CH3",
                    "HOOC", "CO2CH3", "OC2H5", "SO2Me", "PMBN", "CO2iPr", "PPh2", "OMe", "(C2H4O)4CH3", "Si(OEt)3",
                    "CO2tBu", "i-Pr2P", "OiPr"]
        total_bonds = [1, 1, 1, 1, 2, 1, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 3, 1]
        # formulas = ["OiPr"]
        # total_bonds = [1]
        for formula, tb in zip(formulas, total_bonds):
            print(formula)
            # parsed = _parse_formula(formula)
            # print(parsed)
            # expanded = _expand_carbon(parsed)
            # print(expanded)
            # smiles = _condensed_formula_list_to_smiles(expanded, 1)
            # print(canonicalize_smiles(smiles[0]), smiles[1], smiles[2], smiles[3])
            smiles, bonds_left, last_idx, direction, success = _condensed_formula_to_smiles(formula, tb)
            print(smiles)
            print("BL:", bonds_left, "\tLI:", last_idx, "\tDIR:", direction, "\tS:", success)
    else:
        import pandas as pd

        gold_files = ["../data/molbank/Img2Mol/staker.csv"]
        pred_files = ["../output/uspto/swin_base_aux_200k_new1/prediction_staker.csv"]
        indices = [18500]
        for gold_file, pred_file, idx in zip(gold_files, pred_files, indices):
            gold_df = pd.read_csv(gold_file)
            pred_df = pd.read_csv(pred_file)
            gold_row = gold_df.iloc[idx]
            pred_row = pred_df.iloc[idx]
            res = _convert_graph_to_smiles(eval(pred_row['node_coords']), eval(pred_row['node_symbols']),
                                           eval(pred_row['edges']), True)
            print(res)

