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
    valid_df["canon_smiles"] = valid_df["smiles_atomtok"].apply(
        lambda x: Chem.CanonSmiles("".join(x.split()), useChiral=True))
    valid_df.drop(columns=["smiles_mittok", "image_id"], inplace=True)
    valid_df.reset_index(drop=True, inplace=True)

    valid_df["chiral_SMILES"] = valid_df.progress_apply(
        lambda row: _convert_graph_to_smiles(
            arguments=(ast.literal_eval(row["node_coords"]),
                       ast.literal_eval(row["node_symbols"]),
                       ast.literal_eval(row["edges"]))
        ), axis=1)

    print(valid_df[["image_path", "canon_smiles", "chiral_SMILES"]])


if __name__ == "__main__":
    tqdm.pandas()
    test_main()
