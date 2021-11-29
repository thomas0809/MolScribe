import ast
import pandas as pd
from bms.chemistry import _convert_graph_to_smiles
from rdkit import Chem
from tqdm import tqdm


def test_main():
    valid_df = pd.read_csv("indigo-data/valid.csv")
    pred_df = pd.read_csv("indigo-data/prediction_valid.csv")

    valid_df = valid_df.join(pred_df)

    valid_df = valid_df[valid_df.smiles_atomtok.str.contains("@")]
    valid_df = valid_df[valid_df.smiles_atomtok.str.contains("Si")]
    valid_df["canon_smiles"] = valid_df["smiles_atomtok"].apply(
        lambda x: Chem.CanonSmiles("".join(x.split()), useChiral=True))
    valid_df.drop(columns=["smiles_mittok", "image_id"], inplace=True)
    valid_df.reset_index(drop=True, inplace=True)

    valid_df["chiral_SMILES"] = valid_df.progress_apply(
        lambda row: _convert_graph_to_smiles(
            ast.literal_eval(row["node_coords"]),
            ast.literal_eval(row["node_symbols"]),
            ast.literal_eval(row["edges"])
        ), axis=1)

    print(valid_df[["image_path", "canon_smiles", "chiral_SMILES"]])


def test_single():
    coords = [[0.047619047619047616, 0.2698412698412698], [0.23809523809523808, 0.19047619047619047],
              [0.23809523809523808, 0.0], [0.047619047619047616, 0.09523809523809523],
              [0.23809523809523808, 0.36507936507936506], [0.4444444444444444, 0.4603174603174603],
              [0.4444444444444444, 0.6349206349206349], [0.6349206349206349, 0.7301587301587301],
              [0.6349206349206349, 0.9047619047619048], [0.4444444444444444, 1.0],
              [0.23809523809523808, 0.9047619047619048], [0.23809523809523808, 0.7301587301587301],
              [0.047619047619047616, 0.6349206349206349], [0.6349206349206349, 0.36507936507936506],
              [0.6507936507936508, 0.19047619047619047], [0.873015873015873, 0.14285714285714285],
              [0.9841269841269841, 0.30158730158730157], [0.8412698412698413, 0.4444444444444444]]
    symbols = ['C', '[Si]', 'C', 'C', 'O', '[C@@H]', '[C@H]', 'C', 'C', 'C', 'C', 'C', 'O', 'C', 'C', 'C', 'C', 'O']
    edges = [[0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 6, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 0, 6, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 5, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 2, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 1],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0]]

    print(_convert_graph_to_smiles(coords, symbols, edges))

    # Orig: C[Si](C)(C)O[C@H]([C@H]1CCCCC1=O)C2=CC=CO2
    # Gold: C[Si](C)(C)O[C@@H](c1ccco1)[C@H]1CCCCC1=O
    # Pred: C[Si](C)(C)O[C@@H]([C@H]1CCCCC1=O)C2=CC=CO2
    # Post: C[Si](C)(C)O[C@H](c1ccco1)[C@H]1CCCCC1=O

    # I got CC(C)(C)O[C@@H](c1ccco1)[C@H]1CCCCC1=O
    # which is the same as gold (ignoring Si)


if __name__ == "__main__":
    tqdm.pandas()
    # test_main()
    test_single()
