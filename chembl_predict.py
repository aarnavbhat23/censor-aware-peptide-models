"""Inference for the independent hERG assay models, not the peptide models."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from likelihood import CensoredHead
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator


def predict_features(x, checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    projected = (np.asarray(x, dtype="float32") - state["pca_mean"].numpy()) @ state[
        "pca"
    ].numpy().T
    projected = (projected - state["location"].numpy()) / state["scale"].numpy()
    model = CensoredHead(state["feature_width"])
    model.load_state_dict(state["state"])
    model.eval()
    with torch.inference_mode():
        mean, scale = model(torch.tensor(projected, dtype=torch.float32))
    return mean.numpy(), scale.numpy()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("smiles")
    parser.add_argument("--seed", choices=[901, 902, 903], type=int, default=901)
    parser.add_argument(
        "--objective",
        choices=["endpoint_gaussian", "observation_likelihood"],
        default="observation_likelihood",
    )
    args = parser.parse_args()
    molecule = Chem.MolFromSmiles(args.smiles)
    if molecule is None or not molecule.GetNumAtoms():
        parser.error("Valid nonempty molecule required")
    fp = rdFingerprintGenerator.GetMorganGenerator(
        radius=2, fpSize=2048
    ).GetFingerprintAsNumPy(molecule)
    mean, scale = predict_features(
        fp[None, :], Path(f"results/chembl-herg/{args.seed}-{args.objective}.pt")
    )
    print(
        json.dumps(
            {
                "target": "CHEMBL240",
                "assay": "IC50",
                "mean_log10_micromolar": float(mean[0]),
                "gaussian_scale": float(scale[0]),
                "scope": "Assay-table prediction; fitted scale is not independently calibrated uncertainty.",
            },
            indent=2,
        )
    )
