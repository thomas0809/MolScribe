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

from bms.constants import RGROUP_SYMBOLS, PLACEHOLDER_ATOMS, SUBSTITUTIONS, ABBREVIATIONS


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
        for j in range(i+1, n):
            for k in range(5):
                edge_prob[i][j][k] = (edge_prob[i][j][k] + edge_prob[j][i][k]) / 2
                edge_prob[j][i][k] = edge_prob[i][j][k]
            edge_prob[i][j][5] = (edge_prob[i][j][5] + edge_prob[j][i][6]) / 2
            edge_prob[i][j][6] = (edge_prob[i][j][6] + edge_prob[j][i][5]) / 2
            edge_prob[j][i][5] = edge_prob[i][j][6]
            edge_prob[j][i][6] = edge_prob[i][j][5]
    return np.argmax(edge_prob, axis=2).tolist()


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

    def evaluate(self, pred_smiles):
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
        results['canon_smiles'] = (np.array(self.gold_smiles_cistrans) == np.array(pred_smiles_cistrans)).mean()
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

        # Create conformer from 2D coordinate
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


def _replace_functional_group(smiles):
    smiles = smiles.replace('<unk>', 'C')
    for i, r in enumerate(RGROUP_SYMBOLS):
        symbol = f'[{r}]'
        if symbol in smiles:
            if r[0] == 'R' and r[1:].isdigit():
                smiles = smiles.replace(symbol, f'[{int(r[1:])}*]')
            else:
                smiles = smiles.replace(symbol, '*')
    mappings = []
    i = 1
    for sub in SUBSTITUTIONS:
        for abbrv in sub.abbrvs:
            symbol = f'[{abbrv}]'
            if symbol in smiles:
                # assert i < len(PLACEHOLDER_ATOMS), "Not enough placeholders"
                while f'[{i}*]' in smiles:
                    i += 1
                # placeholder = PLACEHOLDER_ATOMS[i]
                placeholder = f'[{i}*]'
                while symbol in smiles:
                    smiles = smiles.replace(symbol, placeholder, 1)
                    mappings.append((placeholder, sub.smiles))
    tokens = atomwise_tokenizer(smiles)
    new_tokens = []
    for t in tokens:
        if t[0] == '[':
            try:
                atom = Chem.AtomFromSmiles(t)
            except:
                atom = None
            new_tokens.append('C' if atom is None else t)
        else:
            new_tokens.append(t)
    smiles = ''.join(new_tokens)
    return smiles, mappings


def _expand_functional_group(mol, mappings, molblock=False):
    if len(mappings) > 0:
        # m = Chem.MolFromSmiles(smiles)
        if molblock:
            Chem.SanitizeMol(mol)
            AllChem.EmbedMolecule(mol)
        mw = Chem.RWMol(mol)
        for i, atom in enumerate(mw.GetAtoms()):  # reset radical electrons
            atom.SetNumRadicalElectrons(0)
        for placeholder_atom, sub_smiles in mappings:
            isotope = int(placeholder_atom[1:-2])
            for i, atom in enumerate(mw.GetAtoms()):
                symbol = atom.GetSymbol()
                if symbol == '*' and atom.GetIsotope() == isotope:
                    bond = atom.GetBonds()[0]  # assuming R is singly bonded to the other atom
                    # TODO: is it true to assume singly bonded?
                    adjacent_idx = bond.GetOtherAtomIdx(i)  # getting the idx of the other atom
                    mw.RemoveBond(i, adjacent_idx)

                    adjacent_atom = mw.GetAtomWithIdx(adjacent_idx)
                    adjacent_atom.SetNumRadicalElectrons(1)

                    mR = Chem.MolFromSmiles(sub_smiles)
                    if molblock:
                        conf = Chem.Conformer(mR.GetNumAtoms())
                        conf.Set3D(False)
                        atom_pos = mw.GetConformer().GetAtomPosition(i)
                        for j in range(mR.GetNumAtoms()):
                            conf.SetAtomPosition(j, atom_pos)
                        mR.AddConformer(conf)
                    combo = Chem.CombineMols(mw, mR)  # combine two subgraphs into a single graph

                    bonding_atoms = []
                    for j, new_atom in enumerate(combo.GetAtoms()):
                        if new_atom.GetNumRadicalElectrons() == 1:
                            new_atom.SetNumRadicalElectrons(0)  # reset radical electrons
                            bonding_atoms.append(j)
                    assert len(bonding_atoms) == 2, bonding_atoms

                    mw = Chem.RWMol(combo)
                    mw.AddBond(bonding_atoms[0], bonding_atoms[1], order=Chem.rdchem.BondType.SINGLE)
                    mw.RemoveAtom(i)
                    break
        smiles = Chem.MolToSmiles(mw)
        mol = mw.GetMol()
    else:
        smiles = Chem.MolToSmiles(mol)
    return smiles, mol


def _convert_graph_to_smiles(coords, symbols, edges, debug=False):
    mol = Chem.RWMol()
    n = len(symbols)
    ids = []
    symbol_to_placeholder = {}
    mappings = []
    for i in range(n):
        symbol = symbols[i]
        if symbol[0] == '[':
            symbol = symbol[1:-1]
        if symbol in RGROUP_SYMBOLS:
            atom = Chem.Atom("*")
            atom.SetIsotope(RGROUP_SYMBOLS.index(symbol))
            idx = mol.AddAtom(atom)
        elif symbol in ABBREVIATIONS:
            if symbol not in symbol_to_placeholder:
                j = len(symbol_to_placeholder) + 1
                # assert j < len(PLACEHOLDER_ATOMS), "Not enough placeholders"
                # placeholder = PLACEHOLDER_ATOMS[j]
                placeholder = f"[{j}*]"
                symbol_to_placeholder[symbol] = placeholder
            else:
                placeholder = symbol_to_placeholder[symbol]
            sub = ABBREVIATIONS[symbol]
            mappings.append((placeholder, sub.smiles))
            atom = Chem.Atom("*")
            atom.SetIsotope(int(placeholder[1:-2]))
            idx = mol.AddAtom(atom)
        else:
            symbol = symbols[i]
            try:
                atom = Chem.AtomFromSmiles(symbol)
                atom.SetChiralTag(Chem.rdchem.ChiralType.CHI_UNSPECIFIED)
                idx = mol.AddAtom(atom)
            except:
                atom = Chem.Atom("*")
                idx = mol.AddAtom(atom)
        assert idx == i
        ids.append(idx)

    has_chirality = False

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
                has_chirality = True
            elif edges[i][j] == 6:
                mol.AddBond(ids[i], ids[j], Chem.BondType.SINGLE)
                mol.GetBondBetweenAtoms(ids[i], ids[j]).SetBondDir(Chem.BondDir.BEGINDASH)
                has_chirality = True

    pred_smiles = '<invalid>'

    try:
        if has_chirality:
            mol = _verify_chirality(mol, coords, symbols, edges, debug)
        # pred_smiles = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
        pred_smiles, mol = _expand_functional_group(mol, mappings)
        success = True
    except Exception as e:
        if debug:
            raise e
        success = False

    return pred_smiles, success


def convert_graph_to_smiles(coords, symbols, edges, num_workers=16, simple=False):
    fn = _convert_graph_to_smiles_simple if simple else _convert_graph_to_smiles
    with multiprocessing.Pool(num_workers) as p:
        results = p.starmap(fn, zip(coords, symbols, edges), chunksize=128)
    smiles_list, success = zip(*results)
    r_success = np.mean(success)
    return smiles_list, r_success


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
        if molblock and len(mappings) == 0:
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
        return pred_smiles, mol
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

