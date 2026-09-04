#!/usr/bin/env python
"""The two pre-finalization checks for the Branch-C static attribution figure:

1. For each of {mean_annual_precip, elevation, mean_annual_vpd} x each
   aridity class: is the attribution-vs-predictor trend distinguishable from
   zero once BOTH seed variability and grid-cell sampling are resampled?
   Point estimate = Pearson r between the predictor and the cross-seed-mean
   per-cell attribution; 95% CI via a hierarchical bootstrap (resample 5
   seeds with replacement, then resample cells with replacement, 2000 reps).

2. Branch C x Branch A correlation (redundancy) and Branch C x Branch C
   self-correlation -- already computed by run_eg_attribution_branchC.py;
   this script just loads and reports them plainly, and flags whether
   grouping is warranted (any |r| >= 0.5 among Branch C variables).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DATA_DIR = "results/eg_attribution_branchC"
CLASSES = ["Semi-Arid", "Sub-Humid", "Humid"]
CONTINUOUS_VARS = {
    "map_mean_annual_precip": "Mean annual precip",
    "elevation": "Elevation",
    "mavpd_mean_annual_vpd": "Mean annual VPD",
}
N_BOOT = 2000


def hierarchical_bootstrap_corr(df_cell_seed: pd.DataFrame, xcol: str, ycol_prefix: str, seeds, rng):
    """df_cell_seed: rows = (seed, cell_id), with a static x column (same for
    all seeds of a cell) and an attribution y column. Returns (point, lo, hi)."""
    piv = df_cell_seed.pivot(index="cell_id", columns="seed", values=ycol_prefix)  # (n_cells, 5)
    x = df_cell_seed.groupby("cell_id")[xcol].first().reindex(piv.index).to_numpy()
    cell_ids = piv.index.to_numpy()
    seed_cols = piv.columns.to_numpy()

    point = np.corrcoef(x, piv.mean(axis=1).to_numpy())[0, 1]

    boots = np.empty(N_BOOT)
    n_cells = len(cell_ids)
    for b in range(N_BOOT):
        seed_pick = rng.choice(len(seed_cols), size=len(seed_cols), replace=True)
        y_seed_resampled = piv.to_numpy()[:, seed_pick].mean(axis=1)  # (n_cells,)
        cell_pick = rng.integers(0, n_cells, size=n_cells)
        boots[b] = np.corrcoef(x[cell_pick], y_seed_resampled[cell_pick])[0, 1]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return point, lo, hi


def main():
    rng = np.random.default_rng(0)
    percell = pd.read_csv(f"{DATA_DIR}/eg_branchC_percell_perseed.csv")

    print("=" * 100)
    print("CHECK 1: running-mean trend vs. zero, hierarchical bootstrap (5 seeds x cells, 2000 reps)")
    print("=" * 100)
    rows = []
    for var, label in CONTINUOUS_VARS.items():
        for cls in CLASSES:
            sub = percell[percell["aridity_tier"] == cls][["seed", "cell_id", var, f"eg_fwi_{var}"]]
            point, lo, hi = hierarchical_bootstrap_corr(sub, var, f"eg_fwi_{var}", sub["seed"].unique(), rng)
            distinguishable = not (lo <= 0 <= hi)
            rows.append({"variable": label, "class": cls, "r": point, "ci_lo": lo, "ci_hi": hi, "distinguishable_from_zero": distinguishable})
            flag = "YES" if distinguishable else "no"
            print(f"{label:20s} {cls:12s}  r={point:+.3f}  95% CI=[{lo:+.3f}, {hi:+.3f}]  distinguishable from zero: {flag}")
    pd.DataFrame(rows).to_csv(f"{DATA_DIR}/check1_trend_significance.csv", index=False)

    print()
    print("=" * 100)
    print("CHECK 2a: Branch C vs Branch A correlation (redundancy)")
    print("=" * 100)
    cross = pd.read_csv(f"{DATA_DIR}/branchC_vs_branchA_correlation.csv", index_col=0)
    print(cross.round(3))
    max_abs = cross.abs().max(axis=1).sort_values(ascending=False)
    print("\nmax |r| with any Branch-A variable, per Branch-C variable:")
    print(max_abs.round(3))

    print()
    print("=" * 100)
    print("CHECK 2b: Branch C self-correlation (per-cell grain)")
    print("=" * 100)
    self_corr = pd.read_csv(f"{DATA_DIR}/branchC_self_correlation.csv", index_col=0)
    print(self_corr.round(3))
    offdiag = self_corr.where(~np.eye(len(self_corr), dtype=bool))
    strong = offdiag.abs().stack()
    strong = strong[strong >= 0.5].sort_values(ascending=False)
    print(f"\nBranch-C variable pairs with |r| >= 0.5 ({len(strong) // 2} pairs, symmetric):")
    print(strong)
    print(f"\n=> grouping warranted: {'YES' if len(strong) > 0 else 'NO'}")


if __name__ == "__main__":
    main()
