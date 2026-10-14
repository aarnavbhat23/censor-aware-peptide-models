"""Add a stronger train-only constant comparator without refitting molecular heads."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from chembl_benchmark import constant_observation, masks, prepare, summary
from likelihood import observed_nll


def main():
    torch.set_num_threads(4)
    out = Path("results/chembl-herg")
    frame, _, _ = prepare(Path("data/chembl-herg-v37.jsonl.gz"))
    rows = pd.read_csv(out / "predictions.csv")
    rows = rows[rows.model != "constant_observation"]
    extra = []
    for seed in [901, 902, 903]:
        train, _, test = masks(frame, seed)
        lo, hi = frame.lower.to_numpy(), frame.upper.to_numpy()
        mean, scale = constant_observation(lo[train], hi[train])
        nll = observed_nll(
            torch.full((test.sum(),), mean, dtype=torch.float64),
            torch.full((test.sum(),), scale, dtype=torch.float64),
            torch.tensor(lo[test]),
            torch.tensor(hi[test]),
        ).numpy()
        for i, value in zip(np.flatnonzero(test), nll):
            extra.append(
                {
                    "split_seed": seed,
                    "model": "constant_observation",
                    "activity_id": int(frame.activity_id[i]),
                    "document": frame.document[i],
                    "kind": frame.kind[i],
                    "lower": lo[i],
                    "upper": hi[i],
                    "mean": mean,
                    "scale": scale,
                    "nll": float(value),
                }
            )
    rows = pd.concat([rows, pd.DataFrame(extra)], ignore_index=True)
    rows.to_csv(out / "predictions.csv", index=False)
    report = json.loads((out / "metrics.json").read_text())
    report["split_results"] = {
        f"{seed}/{name}": summary(g)
        for (seed, name), g in rows.groupby(["split_seed", "model"])
    }
    report["adaptive_addendum"] = (
        "Constant-observation baseline was added after seeing the first external head results; fitted on training rows only. Not preregistered."
    )
    by_doc = (
        rows.groupby(["split_seed", "model", "document"]).nll.mean().unstack("model")
    )
    by_doc.to_csv(out / "per-document.csv")
    report["proper_head_minus_proper_constant_document_nll"] = {
        str(s): float((g.observation_likelihood - g.constant_observation).mean())
        for s, g in by_doc.groupby(level=0)
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                k: {"nll": v["nll"], "exact_rmse": v["exact_only_rmse"]}
                for k, v in report["split_results"].items()
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
