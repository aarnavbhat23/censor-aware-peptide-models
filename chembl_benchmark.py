"""Independent assay-domain comparison with document holdouts and molecule purge."""

import argparse
import copy
import gzip
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from likelihood import CensoredHead, observed_nll
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.optimize import minimize
from sklearn.decomposition import PCA


def open_rows(path):
    return gzip.open(path, "rt") if path.suffix == ".gz" else path.open()


def constant_observation(lower, upper):
    """Train-only two-parameter Gaussian MLE; no molecular features."""
    lo, hi = (
        torch.tensor(lower, dtype=torch.float64),
        torch.tensor(upper, dtype=torch.float64),
    )
    endpoint = np.where(np.isfinite(lower), lower, upper)

    def objective(values):
        params = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        loss = observed_nll(
            params[0].expand_as(lo), params[1].exp().expand_as(lo), lo, hi
        ).mean()
        loss.backward()
        return float(loss.detach()), params.grad.numpy()

    result = minimize(
        objective,
        [endpoint.mean(), np.log(max(0.1, endpoint.std()))],
        jac=True,
        method="L-BFGS-B",
        bounds=[(-20, 20), (-6, 5)],
    )
    if not result.success:
        raise RuntimeError(result.message)
    return float(result.x[0]), float(np.exp(result.x[1]))


def observation(row):
    if row.get("standard_units") != "nM" or row.get("standard_type") != "IC50":
        raise ValueError("outside_standard_IC50_nM_scope")
    value = row["standard_value"]
    upper = row.get("standard_upper_value")
    relation = row["standard_relation"]
    if row.get("data_validity_comment") or row.get("potential_duplicate"):
        raise ValueError("validity_flag_or_potential_duplicate")
    if value is None or relation not in ["=", ">", ">=", "<", "<="]:
        raise ValueError("missing_or_unsupported_observation")
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("nonpositive_or_nonfinite")
    threshold = math.log10(value / 1000.0)  # nM to micromolar, retaining the relation.
    if upper is not None:
        upper = float(upper)
        if relation != "=" or not math.isfinite(upper) or upper <= value:
            raise ValueError("invalid_finite_interval")
        return threshold, math.log10(upper / 1000.0), "interval"
    if relation in [">", ">="]:
        return threshold, math.inf, "right"
    if relation in ["<", "<="]:
        return -math.inf, threshold, "left"
    return threshold, threshold, "exact"


def prepare(path):
    records = []
    excluded = []
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    features = {}
    seen = set()
    for line in open_rows(path):
        row = json.loads(line)
        try:
            lo, hi, kind = observation(row)
            if not row.get("document_chembl_id"):
                raise ValueError("missing_document")
            molecule = Chem.MolFromSmiles(row.get("canonical_smiles") or "")
            if molecule is None or not molecule.GetNumAtoms():
                raise ValueError("invalid_molecule")
            smiles = Chem.MolToSmiles(molecule, isomericSmiles=True)
            identity = (smiles, row["assay_chembl_id"], lo, hi)
            if identity in seen:
                raise ValueError("duplicate_molecule_assay_observation")
            seen.add(identity)
            if smiles not in features:
                features[smiles] = generator.GetFingerprintAsNumPy(molecule).astype(
                    "float32"
                )
            records.append(
                {
                    "activity_id": row["activity_id"],
                    "molecule": smiles,
                    "document": row["document_chembl_id"],
                    "assay": row["assay_chembl_id"],
                    "lower": lo,
                    "upper": hi,
                    "kind": kind,
                }
            )
        except (ValueError, TypeError) as error:
            excluded.append(
                {"activity_id": row.get("activity_id"), "reason": str(error)}
            )
    frame = pd.DataFrame(records)
    x = np.array([features[s] for s in frame.molecule])
    return frame, x, excluded


def masks(frame, seed):
    documents = np.random.default_rng(seed).permutation(sorted(frame.document.unique()))
    n = len(documents)
    validation = frame.document.isin(documents[int(0.7 * n) : int(0.8 * n)]).to_numpy(
        copy=True
    )
    test = frame.document.isin(documents[int(0.8 * n) :]).to_numpy()
    blocked = set(frame.molecule[validation | test])
    train = (
        frame.document.isin(documents[: int(0.7 * n)]).to_numpy()
        & ~frame.molecule.isin(blocked).to_numpy()
    )
    validation &= ~frame.molecule.isin(set(frame.molecule[test])).to_numpy()
    assert not set(frame.molecule[train]) & set(frame.molecule[test])
    assert not set(frame.document[train]) & set(frame.document[test])
    return train, validation, test


def fit(x, lo, hi, train, val, name):
    torch.manual_seed(417)
    features = torch.tensor(x)
    lower = torch.tensor(lo, dtype=torch.float32)
    upper = torch.tensor(hi, dtype=torch.float32)
    endpoint = torch.where(torch.isfinite(lower), lower, upper)
    model = CensoredHead(x.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=0.01)
    best, stale, state, trace = float("inf"), 0, None, []
    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        mean, scale = model(features[train])
        a, b = (
            (lower[train], upper[train])
            if name == "observation_likelihood"
            else (endpoint[train], endpoint[train])
        )
        loss = observed_nll(mean, scale, a, b).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()
        model.eval()
        with torch.no_grad():
            mu, sigma = model(features[val])
            score = float(observed_nll(mu, sigma, lower[val], upper[val]).mean())
        trace.append(
            {
                "epoch": epoch + 1,
                "training_objective": float(loss.detach()),
                "validation_observation_nll": score,
            }
        )
        if score < best:
            best, stale, state = score, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale == 25:
            break
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        mean, scale = model(features)
    return mean.numpy(), scale.numpy(), state, trace


def summary(rows):
    exact = rows.kind == "exact"
    return {
        "rows": len(rows),
        "nll": float(rows.nll.mean()),
        "exact_only_rmse": float(
            np.sqrt(np.mean((rows.loc[exact, "mean"] - rows.loc[exact, "lower"]) ** 2))
        )
        if exact.any()
        else None,
        "by_observation": {
            k: {"rows": len(g), "nll": float(g.nll.mean())}
            for k, g in rows.groupby("kind")
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=Path("data/chembl-herg-v37.jsonl.gz")
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    RDLogger.DisableLog("rdApp.warning")
    frame, x, excluded = prepare(args.data)
    out = Path("results/chembl-herg")
    out.mkdir(parents=True, exist_ok=True)
    records = []
    histories = {}
    splits = []
    for seed in [901, 902, 903]:
        train, val, test = masks(frame, seed)
        pca = PCA(n_components=64, svd_solver="randomized", random_state=417).fit(
            x[train]
        )
        projected = pca.transform(x).astype("float32")
        loc = projected[train].mean(0)
        scale = np.maximum(projected[train].std(0), 0.05)
        projected = (projected - loc) / scale
        endpoint = np.where(np.isfinite(frame.lower), frame.lower, frame.upper)
        constants = {
            "constant_endpoint": (
                float(endpoint[train].mean()),
                max(0.1, float(endpoint[train].std())),
            ),
            "constant_observation": constant_observation(
                frame.lower.to_numpy()[train], frame.upper.to_numpy()[train]
            ),
        }
        for name in ["endpoint_gaussian", "observation_likelihood"]:
            mean, sigma, state, trace = fit(
                projected,
                frame.lower.to_numpy(),
                frame.upper.to_numpy(),
                train,
                val,
                name,
            )
            torch.save(
                {
                    "state": state,
                    "feature_width": 64,
                    "pca": torch.tensor(pca.components_),
                    "pca_mean": torch.tensor(pca.mean_),
                    "location": torch.tensor(loc),
                    "scale": torch.tensor(scale),
                },
                out / f"{seed}-{name}.pt",
            )
            histories[f"{seed}/{name}"] = trace
            nll = observed_nll(
                torch.tensor(mean[test]),
                torch.tensor(sigma[test]),
                torch.tensor(frame.lower.to_numpy()[test], dtype=torch.float32),
                torch.tensor(frame.upper.to_numpy()[test], dtype=torch.float32),
            ).numpy()
            for i, value in zip(np.flatnonzero(test), nll):
                records.append(
                    {
                        "split_seed": seed,
                        "model": name,
                        "activity_id": int(frame.activity_id[i]),
                        "document": frame.document[i],
                        "kind": frame.kind[i],
                        "lower": float(frame.lower[i]),
                        "upper": float(frame.upper[i]),
                        "mean": float(mean[i]),
                        "scale": float(sigma[i]),
                        "nll": float(value),
                    }
                )
        for name, (mean, sigma) in constants.items():
            indexes = np.flatnonzero(test)
            nll = observed_nll(
                torch.full((len(indexes),), mean),
                torch.full((len(indexes),), sigma),
                torch.tensor(frame.lower.to_numpy()[test], dtype=torch.float32),
                torch.tensor(frame.upper.to_numpy()[test], dtype=torch.float32),
            ).numpy()
            for i, value in zip(indexes, nll):
                records.append(
                    {
                        "split_seed": seed,
                        "model": name,
                        "activity_id": int(frame.activity_id[i]),
                        "document": frame.document[i],
                        "kind": frame.kind[i],
                        "lower": float(frame.lower[i]),
                        "upper": float(frame.upper[i]),
                        "mean": mean,
                        "scale": sigma,
                        "nll": float(value),
                    }
                )
        splits.append(
            {
                "seed": seed,
                "train_rows": int(train.sum()),
                "validation_rows": int(val.sum()),
                "test_rows": int(test.sum()),
                "training_test_molecule_overlap": 0,
                "training_test_document_overlap": 0,
            }
        )
        pd.DataFrame(
            {
                "activity_id": frame.activity_id,
                "document": frame.document,
                "molecule_sha256": [
                    __import__("hashlib").sha256(s.encode()).hexdigest()
                    for s in frame.molecule
                ],
                "partition": np.where(
                    train,
                    "train",
                    np.where(val, "validation", np.where(test, "test", "purged")),
                ),
            }
        ).to_csv(out / f"split-{seed}.csv", index=False)
        print("completed split", seed, flush=True)
    predictions = pd.DataFrame(records)
    predictions.to_csv(out / "predictions.csv", index=False)
    report = {
        "source_rows": sum(1 for _ in open_rows(args.data)),
        "retained_rows": len(frame),
        "molecules": frame.molecule.nunique(),
        "documents": frame.document.nunique(),
        "observation_counts": frame.kind.value_counts().to_dict(),
        "excluded_rows": len(excluded),
        "splits": splits,
        "split_results": {
            f"{seed}/{name}": summary(g)
            for (seed, name), g in predictions.groupby(["split_seed", "model"])
        },
        "scope": "Independent hERG IC50 table; not a replication of peptide MIC biology. Repeated partitions are dependent.",
    }
    by_document = (
        predictions.groupby(["split_seed", "model", "document"])
        .nll.mean()
        .unstack("model")
    )
    report["paired_document_nll_differences"] = {
        str(seed): float((g.observation_likelihood - g.endpoint_gaussian).mean())
        for seed, g in by_document.groupby(level=0)
    }
    by_document.to_csv(out / "per-document.csv")
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "histories.json").write_text(json.dumps(histories, indent=2) + "\n")
    (out / "exclusions.json").write_text(json.dumps(excluded, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
