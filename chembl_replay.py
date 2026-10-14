"""Recover assay metrics, strict holdouts and actual checkpoint predictions."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chembl_benchmark import constant_observation, masks, prepare, summary
from chembl_predict import predict_features
from likelihood import observed_nll


def main():
    torch.set_num_threads(4)
    out = Path("results/chembl-herg")
    rows = pd.read_csv(out / "predictions.csv")
    report = json.loads((out / "metrics.json").read_text())
    frame, x, _ = prepare(Path("data/chembl-herg-v37.jsonl.gz"))
    for (seed, model), group in rows.groupby(["split_seed", "model"]):
        values = observed_nll(
            *[
                torch.tensor(group[k].to_numpy())
                for k in ["mean", "scale", "lower", "upper"]
            ]
        ).numpy()
        np.testing.assert_allclose(values, group.nll, atol=3e-6, rtol=1e-5)
        actual = summary(group)
        expected = report["split_results"][f"{seed}/{model}"]
        for k in ["nll", "exact_only_rmse"]:
            np.testing.assert_allclose(actual[k], expected[k])
        train, val, test = masks(frame, seed)
        ledger = pd.read_csv(out / f"split-{seed}.csv")
        np.testing.assert_array_equal(
            ledger.partition,
            np.where(
                train,
                "train",
                np.where(val, "validation", np.where(test, "test", "purged")),
            ),
        )
        assert set(group.activity_id) == set(frame.loc[test, "activity_id"])
        if model in ["endpoint_gaussian", "observation_likelihood"]:
            mu, scale = predict_features(x[test], out / f"{seed}-{model}.pt")
            np.testing.assert_allclose(mu, group["mean"], atol=1e-5, rtol=1e-5)
            np.testing.assert_allclose(scale, group["scale"], atol=1e-5, rtol=1e-5)
        elif model == "constant_observation":
            mean, scale = constant_observation(
                frame.lower.to_numpy()[train], frame.upper.to_numpy()[train]
            )
            np.testing.assert_allclose(group["mean"], mean)
            np.testing.assert_allclose(group["scale"], scale)
    print(
        "Replayed three molecule-purged document splits, all observation types, six fitted heads and constant comparators."
    )


if __name__ == "__main__":
    main()
