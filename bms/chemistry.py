import cv2
import copy
import numpy as np
import multiprocessing

from indigo import Indigo
from indigo.renderer import IndigoRenderer
import rdkit
import rdkit.Chem as Chem
rdkit.RDLogger.DisableLog('rdApp.*')
import Levenshtein


RGROUP_SYMBOLS = ['R', 'R1', 'R2', 'R3', 'R4', 'R5', 'X', 'Ar']


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
        inchi_list = p.map(_convert_smiles_to_inchi, smiles_list)
    n_success = sum([x is not None for x in inchi_list])
    r_success = n_success / len(inchi_list)
    inchi_list = [x if x else 'InChI=1S/H2O/h1H2' for x in inchi_list]
    return inchi_list, r_success


def canonicalize_smiles(smiles, ignore_chiral=False, ignore_charge=False):
    rlist = RGROUP_SYMBOLS
    rdict = {}
    for i, symbol in enumerate(rlist):
        rdict[f'[{symbol}]'] = f'[*:{i}]'
    for a, b in rdict.items():
        smiles = smiles.replace(a, b)
    try:
        # mol = Chem.MolFromSmiles(smiles, sanitize=False)
        # if ignore_charge:
        #     for atom in mol.GetAtoms():
        #         atom.SetFormalCharge(0)
        # canon_smiles = Chem.MolToSmiles(mol, isomericSmiles=(not ignore_chiral))
        canon_smiles = Chem.CanonSmiles(smiles, useChiral=(not ignore_chiral))
    except:
        canon_smiles = smiles
    return canon_smiles


def get_canon_smiles_score(gold_smiles, pred_smiles, ignore_chiral=False, ignore_charge=False):
    gold_canon_smiles = np.array([canonicalize_smiles(smiles, ignore_chiral, ignore_charge) for smiles in gold_smiles])
    pred_canon_smiles = np.array([canonicalize_smiles(smiles, ignore_chiral, ignore_charge) for smiles in pred_smiles])
    return (gold_canon_smiles == pred_canon_smiles).mean()


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
        score = Levenshtein.distance(true, pred)
        scores.append(score)
        exact_match += int(true == pred)
    avg_score = np.mean(scores)
    exact_match = exact_match / len(y_true)
    return avg_score, exact_match


def normalize_nodes(nodes):
    x, y = nodes[:, 0], nodes[:, 1]
    minx, maxx = min(x), max(x)
    miny, maxy = min(y), max(y)
    x = (x - minx) / max(maxx - minx, 1e-6)
    y = (maxy - y) / max(maxy - miny, 1e-6)
    return np.stack([x, y], axis=1)


def convert_smiles_to_nodes(smiles):
    indigo = Indigo()
    renderer = IndigoRenderer(indigo)
    indigo.setOption('render-output-format', 'png')
    indigo.setOption('render-background-color', '1,1,1')
    indigo.setOption('render-stereo-style', 'none')
    mol = indigo.loadMolecule(smiles)
    mol.layout()
    buf = renderer.renderToBuffer(mol)
    img = cv2.imdecode(np.asarray(bytearray(buf), dtype=np.uint8), 1)
    height, width, _ = img.shape
    coords, symbols = [], []
    for atom in mol.iterateAtoms():
        # x, y, z = atom.xyz()
        # coords.append([x, y])
        x, y = atom.coords()
        coords.append([y / height, x / width])
        symbols.append(atom.symbol())
    # coords = normalize_nodes(np.array(coords))
    return coords, symbols


def _evaluate_nodes(arguments):
    smiles, coords, symbols = arguments
    gold_coords, gold_symbols = convert_smiles_to_nodes(smiles)
    n = len(gold_coords)
    m = len(coords)
    num_node_correct = (n == m)
    coords = np.array(coords)
    dist = np.zeros((n, m))
    for i in range(n):
        for j in range(m):
            dist[i, j] = np.linalg.norm(gold_coords[i] - coords[j])
    score = (dist.min(axis=1).mean() + dist.min(axis=0).mean()) / 2 if n * m > 0 else 0
    symbols_em = (sorted(symbols) == sorted(gold_symbols))
    return score, num_node_correct, symbols_em


def evaluate_nodes(smiles_list, node_coords, node_symbols, num_workers=16):
    with multiprocessing.Pool(num_workers) as p:
        results = p.map(_evaluate_nodes, zip(smiles_list, node_coords, node_symbols))
    results = np.array(results)
    score, num_node_acc, symbols_em = results.mean(axis=0)
    return score, num_node_acc, symbols_em


def _convert_graph_to_smiles(arguments):
    coords, symbols, edges = arguments
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


def convert_graph_to_smiles(node_coords, node_symbols, edges, num_workers=16):
    with multiprocessing.Pool(num_workers) as p:
        smiles_list = p.map(_convert_graph_to_smiles, zip(node_coords, node_symbols, edges))
    return smiles_list


'''
Define common substitutions for chemical shorthand
Note: does not include R groups or halogens as X

Each tuple records the
    string_tO_substitute, smarts_to_match, probability
'''
substitutions = [
    # added by ztu 210917
    (['NO2', 'O2N'], '[N+](=O)[O-]', 0.5),
    (['CHO', 'OHC'], '[CH1](=O)', 0.5),
    (['CO2Et', 'COOEt'], 'C(=O)[OH0;D2][CH2;D2][CH3]', 0.5),

    (['OAc'], '[OH0;X2]C(=O)[CH3]', 0.8),
    (['NHAc'], '[NH1;D2]C(=O)[CH3]', 0.8),
    (['Ac'], 'C(=O)[CH3]', 0.1),

    (['OBz'], '[OH0;D2]C(=O)[cH0]1[cH][cH][cH][cH][cH]1', 0.7),  # Benzoyl
    (['Bz'], 'C(=O)[cH0]1[cH][cH][cH][cH][cH]1', 0.2),  # Benzoyl

    (['OBn'], '[OH0;D2][CH2;D2][cH0]1[cH][cH][cH][cH][cH]1', 0.7),  # Benzyl
    (['Bn'], '[CH2;D2][cH0]1[cH][cH][cH][cH][cH]1', 0.2),  # Benzyl

    (['NHBoc'], '[NH1;D2]C(=O)OC([CH3])([CH3])[CH3]', 0.9),
    (['NBoc'], '[NH0;D3]C(=O)OC([CH3])([CH3])[CH3]', 0.9),
    (['Boc'], 'C(=O)OC([CH3])([CH3])[CH3]', 0.2),

    (['Cbm'], 'C(=O)[NH2;D1]', 0.2),
    (['Cbz'], 'C(=O)OC[cH]1[cH][cH][cH1][cH][cH]1', 0.4),
    (['Cy'], '[CH1;X3]1[CH2][CH2][CH2][CH2][CH2]1', 0.3),
    (['Fmoc'], 'C(=O)O[CH2][CH1]1c([cH1][cH1][cH1][cH1]2)c2c3c1[cH1][cH1][cH1][cH1]3', 0.6),
    (['Mes'], '[cH0]1c([CH3])cc([CH3])cc([CH3])1', 0.5),
    (['OMs'], '[OH0;D2]S(=O)(=O)[CH3]', 0.8),
    (['Ms'], 'S(=O)(=O)[CH3]', 0.2),
    (['Ph'], '[cH0]1[cH][cH][cH1][cH][cH]1', 0.7),
    (['Py'], '[cH0]1[n;+0][cH1][cH1][cH1][cH1]1', 0.1),
    (['Suc'], 'C(=O)[CH2][CH2]C(=O)[OH]', 0.2),
    (['TBS'], '[Si]([CH3])([CH3])C([CH3])([CH3])[CH3]', 0.5),
    (['TBZ'], 'C(=S)[cH]1[cH][cH][cH1][cH][cH]1', 0.2),
    (['OTf'], '[OH0;D2]S(=O)(=O)C(F)(F)F', 0.8),
    (['Tf'], 'S(=O)(=O)C(F)(F)F', 0.2),
    (['TFA'], 'C(=O)C(F)(F)F', 0.3),
    (['TMS'], '[Si]([CH3])([CH3])[CH3]', 0.5),
    (['Ts'], 'S(=O)(=O)c1[cH1][cH1][cH0]([CH3])[cH1][cH1]1', 0.6),  # Tos

    # Alkyl chains
    (['OMe', 'MeO'], '[OH0;D2][CH3;D1]', 0.3),
    (['SMe', 'MeS'], '[SH0;D2][CH3;D1]', 0.3),
    (['NMe', 'MeN'], '[N;X3][CH3;D1]', 0.3),
    (['Me'], '[CH3;D1]', 0.1),
    (['OEt', 'EtO'], '[OH0;D2][CH2;D2][CH3]', 0.5),
    (['Et'], '[CH2;D2][CH3]', 0.2),
    (['Pr', 'nPr'], '[CH2;D2][CH2;D2][CH3]', 0.1),
    (['Bu', 'nBu'], '[CH2;D2][CH2;D2][CH2;D2][CH3]', 0.1),

    # Branched
    (['iPr'], '[CH1;D3]([CH3])[CH3]', 0.1),
    (['iBu'], '[CH2;D2][CH1;D3]([CH3])[CH3]', 0.1),
    (['OiBu'], '[OH0;D2][CH2;D2][CH1;D3]([CH3])[CH3]', 0.1),
    (['OtBu'], '[OH0;D2][CH0]([CH3])([CH3])[CH3]', 0.7),
    (['tBu'], '[CH0]([CH3])([CH3])[CH3]', 0.3),

    # Other shorthands (MIGHT NOT WANT ALL OF THESE)
    (['CF3', 'F3C'], '[CH0;D4](F)(F)F', 0.5),
    (['NCF3'], '[N;X3][CH0;D4](F)(F)F', 0.5),
    (['CCl3'], '[CH0;D4](Cl)(Cl)Cl', 0.5),
    (['CO2H', 'COOH'], 'C(=O)[OH]', 0.2),  # COOH
    (['CN'], 'C#[ND1]', 0.1),
    (['OCH3'], '[OH0;D2][CH3]', 0.2),
    (['SO3H'], 'S(=O)(=O)[OH]', 0.4),
]


def get_substitutions():
    for abbrvs, smarts, p in substitutions:
        assert type(abbrvs) is list
    return substitutions