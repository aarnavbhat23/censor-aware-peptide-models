from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 14,
        "font.size": 11,
        "figure.facecolor": "#fafbfd",
        "axes.facecolor": "#fafbfd",
    }
)
colors = {
    "additive": "#95a5ad",
    "matched_ridge": "#2d7790",
    "neural_ensemble": "#ba5b39",
    "endpoint_gaussian": "#95a5ad",
    "observation_likelihood": "#ba5b39",
    "constant_observation": "#2d7790",
}
p = Path("results/chembl-herg")
df = pd.read_csv(p / "predictions.csv")
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
for method in ["endpoint_gaussian", "observation_likelihood", "constant_observation"]:
    g = df[df.model == method]
    nll = g.groupby("split_seed").nll.mean()
    exact = (
        g[g.kind == "exact"]
        .groupby("split_seed")
        .apply(lambda x: np.sqrt(np.mean((x["mean"] - x.lower) ** 2)))
    )
    axes[0].plot(
        nll.index.astype(str),
        nll,
        marker="o",
        label=method.replace("_", " "),
        color=colors[method],
    )
    axes[1].plot(exact.index.astype(str), exact, marker="o", color=colors[method])
axes[0].set(
    title="All retained observation types",
    ylabel="Observation NLL (lower better)",
    xlabel="Document split seed",
)
axes[1].set(
    title="Exact observations only",
    ylabel="log10 IC50 RMSE (lower better)",
    xlabel="Document split seed",
)
axes[0].legend(frameon=False, fontsize=9)
fig.suptitle(
    "Likelihood gain does not imply better exact-value prediction", fontsize=16
)
fig.savefig(p / "comparison.png", dpi=150)
fig.savefig(p / "comparison.svg")
plt.close(fig)
