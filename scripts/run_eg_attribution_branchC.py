#!/usr/bin/env python
"""Expected Gradients for Branch C (static), pooled model, ALL 5 SEEDS,
joint with Branch A/B (same completeness argument as the main Figure 4
script). Background=500 training samples, K=200 MC draws, FWI units via
each sample's cell target std.

Explained samples are chosen PER CELL (not per class pooled), so every one
of the ~1412 cells in the three aridity classes gets its own attribution
estimate: up to 20 random test weeks per cell, attributed under each of the
5 seed models. This is what panels (b)-(d)'s per-cell scatter/running-mean
and the seed+cell bootstrap need.

Also computes (no model needed, pure data):
  - Branch C x Branch A correlation (for panel e's redundancy test): each
    Branch C variable's value broadcast across weeks, correlated against
    each Branch A variable's weekly value, over the full pooled sample table.
  - Branch C x Branch C correlation matrix, at the per-cell grain (Branch C
    doesn't vary weekly, so per-cell is the right grain -- not weekly-pooled).

Outputs (results/eg_attribution_branchC/):
  eg_branchC_percell_perseed.csv  seed, cell_id, aridity_tier, dominant_nlcd,
                                   static value + eg_fwi_<var> per Branch-C var
  branchC_vs_branchA_correlation.csv
  branchC_self_correlation.csv
  summary.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from fwi_attn.config import load_config
from fwi_attn.data.loaders import build_sample_table, load_raw_datasets
from fwi_attn.data.schema import BRANCH_A_FEATURES, BRANCH_B_FEATURES, BRANCH_B_N_WEEKS, BRANCH_C_FEATURES, NLCD_CODES
from fwi_attn.data.splits import assign_split
from fwi_attn.data.standardization import Standardizer
from fwi_attn.models.lstm_attention import FWIAttnModel

MODEL_DIRS = {
    0: "results/cluster-ALL-seed0_20260904T171251Z_fd2892fb",
    1: "results/cluster-ALL-seed1_20260904T171251Z_9c7594f7",
    2: "results/cluster-ALL-seed2_20260904T171245Z_a17997e6",
    3: "results/cluster-ALL-seed3_20260904T171245Z_2bce5905",
    4: "results/cluster-ALL-seed4_20260904T171245Z_06acdb35",
}
OUT_DIR = Path("results/eg_attribution_branchC")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOOKBACK = BRANCH_B_N_WEEKS
CLASSES = ["Semi-Arid", "Sub-Humid", "Humid"]
K_DRAWS = 200
BACKGROUND_N = 500
N_WEEKS_PER_CELL = 20
NLCD_NAMES = {41: "Deciduous forest", 42: "Evergreen forest", 43: "Mixed forest", 52: "Shrub/scrub", 71: "Grassland/herb.", 81: "Pasture/hay"}


def fit_standardizer(mode, eps, X, group_key, feature_names, extra_lead_dim=None):
    if extra_lead_dim is not None:
        n, t, f = X.shape
        return Standardizer(mode, eps).fit(X.reshape(n * t, f), np.repeat(group_key, t), feature_names)
    return Standardizer(mode, eps).fit(X, group_key, feature_names)


def apply_standardizer(std, X, group_key, extra_lead_dim=None):
    if extra_lead_dim is not None:
        n, t, f = X.shape
        return std.transform(X.reshape(n * t, f), np.repeat(group_key, t)).reshape(n, t, f)
    return std.transform(X, group_key)


def main():
    config = load_config("configs/default.yaml")
    print("loading raw datasets + sample table (ALL)...")
    ds = load_raw_datasets(config)
    table, _ = build_sample_table(ds, config, lookback=LOOKBACK, cluster=None)

    splits = assign_split(table.year)
    train_mask, test_mask = splits == "train", splits == "test"
    mode, eps = config.standardization.mode, config.standardization.eps
    group_key = table.cell_id if mode == "per_grid_cell" else table.cluster_id
    # Branch C is static per cell -> per_grid_cell standardization gives zero
    # within-cell variance and collapses it to 0 everywhere. Standardize it
    # globally (one group) instead, both here and wherever it's applied.
    global_key = np.zeros(table.n, dtype=np.int64)

    xa_std = fit_standardizer(mode, eps, table.xa[train_mask], group_key[train_mask], BRANCH_A_FEATURES)
    xb_std = fit_standardizer(mode, eps, table.xb[train_mask], group_key[train_mask], BRANCH_B_FEATURES, extra_lead_dim=LOOKBACK)
    xc_std = fit_standardizer(mode, eps, table.xc[train_mask], global_key[train_mask], BRANCH_C_FEATURES)
    y_std = fit_standardizer(mode, eps, table.y[train_mask].reshape(-1, 1), group_key[train_mask], ["fwi"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    gk_test = group_key[test_mask]
    xa_te = apply_standardizer(xa_std, table.xa[test_mask], gk_test)
    xb_te = apply_standardizer(xb_std, table.xb[test_mask], gk_test, extra_lead_dim=LOOKBACK)
    xc_te = apply_standardizer(xc_std, table.xc[test_mask], global_key[test_mask])
    cell_id_test = table.cell_id[test_mask]
    aridity_tier_test = table.aridity_tier[test_mask]

    # --- explained samples: up to N_WEEKS_PER_CELL random test weeks, for every cell in the 3 classes ---
    rng = np.random.default_rng(0)
    in_scope = np.isin(aridity_tier_test, CLASSES)
    df_idx = pd.DataFrame({"row": np.arange(len(cell_id_test))[in_scope], "cell_id": cell_id_test[in_scope]})
    expl_idx = (
        df_idx.groupby("cell_id")["row"]
        .apply(lambda s: rng.choice(s.to_numpy(), size=min(N_WEEKS_PER_CELL, len(s)), replace=False))
        .explode()
        .to_numpy()
        .astype(int)
    )
    n_cells_covered = df_idx["cell_id"].nunique()
    print(f"explained samples: {len(expl_idx)} across {n_cells_covered} cells (up to {N_WEEKS_PER_CELL} weeks/cell)")

    train_idx_local = np.where(train_mask)[0]
    bg_pick = rng.choice(len(train_idx_local), size=BACKGROUND_N, replace=False)
    gk_bg = group_key[train_mask][bg_pick]
    xa_bg = apply_standardizer(xa_std, table.xa[train_mask][bg_pick], gk_bg)
    xb_bg = apply_standardizer(xb_std, table.xb[train_mask][bg_pick], gk_bg, extra_lead_dim=LOOKBACK)
    xc_bg = apply_standardizer(xc_std, table.xc[train_mask][bg_pick], global_key[train_mask][bg_pick])

    xa_expl = torch.as_tensor(xa_te[expl_idx], dtype=torch.float32, device=device)
    xb_expl = torch.as_tensor(xb_te[expl_idx], dtype=torch.float32, device=device)
    xc_expl = torch.as_tensor(xc_te[expl_idx], dtype=torch.float32, device=device)
    xa_bg_t = torch.as_tensor(xa_bg, dtype=torch.float32, device=device)
    xb_bg_t = torch.as_tensor(xb_bg, dtype=torch.float32, device=device)
    xc_bg_t = torch.as_tensor(xc_bg, dtype=torch.float32, device=device)
    n_expl = xa_expl.shape[0]

    gk_expl = gk_test[expl_idx]
    std_y_per_sample = np.array([y_std.std_[g][0] for g in gk_expl])
    cell_id_expl = cell_id_test[expl_idx]
    aridity_tier_expl = aridity_tier_test[expl_idx]

    # 60-step LSTM backward retains the full unrolled graph for every explained
    # sample; 28k samples in one batch OOMs a 15GB A2. Chunk the explained set.
    CHUNK = 3000
    n_chunks = int(np.ceil(n_expl / CHUNK))

    all_seed_frames = []
    for seed, model_dir in MODEL_DIRS.items():
        print(f"seed {seed}: loading {model_dir}/model.pt")
        model = FWIAttnModel(config.model, branch_a_in=7, branch_b_in=3, branch_c_in=9, branches=("A", "B", "C")).to(device)
        model.load_state_dict(torch.load(Path(model_dir) / "model.pt", map_location=device))
        model.eval()

        eg_c_chunks = []
        torch.manual_seed(seed)
        with torch.backends.cudnn.flags(enabled=False):
            for ci in range(n_chunks):
                sl = slice(ci * CHUNK, (ci + 1) * CHUNK)
                a_c, b_c, c_c = xa_expl[sl], xb_expl[sl], xc_expl[sl]
                n_c = a_c.shape[0]
                eg_c_sum = torch.zeros_like(c_c)
                for k in range(K_DRAWS):
                    bidx = torch.randint(0, BACKGROUND_N, (n_c,), device=device)
                    baseline_a, baseline_b, baseline_c = xa_bg_t[bidx], xb_bg_t[bidx], xc_bg_t[bidx]
                    alpha = torch.rand(n_c, 1, device=device)

                    xa_interp = (baseline_a + alpha * (a_c - baseline_a)).clone().requires_grad_(True)
                    xb_interp = (baseline_b + alpha.unsqueeze(-1) * (b_c - baseline_b)).clone().requires_grad_(True)
                    xc_interp = (baseline_c + alpha * (c_c - baseline_c)).clone().requires_grad_(True)

                    pred, _ = model(xa_interp, xb_interp, xc_interp)
                    grad_c = torch.autograd.grad(pred.sum(), xc_interp)[0]
                    eg_c_sum += grad_c.detach() * (c_c - baseline_c)
                eg_c_chunks.append((eg_c_sum / K_DRAWS).cpu().numpy())
                print(f"  seed {seed} chunk {ci + 1}/{n_chunks} done")

        eg_c = np.concatenate(eg_c_chunks, axis=0)
        eg_c_fwi = eg_c * std_y_per_sample[:, None]

        cols = {f"eg_fwi_{v}": eg_c_fwi[:, i] for i, v in enumerate(BRANCH_C_FEATURES)}
        frame = pd.DataFrame({"seed": seed, "cell_id": cell_id_expl, "aridity_tier": aridity_tier_expl, **cols})
        all_seed_frames.append(frame)
        del model
        torch.cuda.empty_cache()

    raw = pd.concat(all_seed_frames, ignore_index=True)
    # aggregate to one row per (seed, cell): mean over that cell's explained weeks
    eg_cols = [f"eg_fwi_{v}" for v in BRANCH_C_FEATURES]
    percell = raw.groupby(["seed", "cell_id", "aridity_tier"])[eg_cols].mean().reset_index()

    # attach static (physical-unit) values and dominant NLCD class, once per cell
    static_df = pd.DataFrame(ds["static"], columns=BRANCH_C_FEATURES)
    static_df["cell_id"] = ds["cell_id"]
    static_df["dominant_nlcd"] = np.array(NLCD_CODES)[np.argmax(ds["static"][:, :6], axis=1)]
    static_df["lon"], static_df["lat"] = ds["lon"], ds["lat"]

    out = percell.merge(static_df, on="cell_id", how="left", suffixes=("", "_staticval"))
    out.to_csv(OUT_DIR / "eg_branchC_percell_perseed.csv", index=False)
    print(f"saved eg_branchC_percell_perseed.csv ({out.shape[0]} rows = {n_cells_covered} cells x 5 seeds)")

    # --- Branch C vs Branch A correlation (redundancy test), full pooled sample table ---
    print("computing Branch C vs Branch A correlation (full table)...")
    xa_full = pd.DataFrame(table.xa, columns=BRANCH_A_FEATURES)
    xc_full = pd.DataFrame(table.xc, columns=BRANCH_C_FEATURES)
    cross_corr = pd.DataFrame(index=BRANCH_C_FEATURES, columns=BRANCH_A_FEATURES, dtype=float)
    for cvar in BRANCH_C_FEATURES:
        for avar in BRANCH_A_FEATURES:
            cross_corr.loc[cvar, avar] = np.corrcoef(xc_full[cvar], xa_full[avar])[0, 1]
    cross_corr.to_csv(OUT_DIR / "branchC_vs_branchA_correlation.csv")
    print(cross_corr.round(3))

    # --- Branch C self-correlation, per-cell grain ---
    self_corr = static_df[BRANCH_C_FEATURES].corr(method="pearson")
    self_corr.to_csv(OUT_DIR / "branchC_self_correlation.csv")
    print(self_corr.round(3))

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({
            "n_cells_covered": int(n_cells_covered), "n_weeks_per_cell": N_WEEKS_PER_CELL, "n_explained_total": int(n_expl),
            "k_draws": K_DRAWS, "background_n": BACKGROUND_N, "seeds": list(MODEL_DIRS.keys()),
        }, f, indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
