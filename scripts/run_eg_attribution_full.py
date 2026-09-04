#!/usr/bin/env python
"""Expected Gradients attribution for Figure 4: Branch A (concurrent met) and
Branch B (antecedent SM/NDVI/VOD) attributed JOINTLY (shared baseline sample,
shared interpolation alpha) so the two branches' attributions sit on one
additive scale -- required for panel (h)'s branch-level comparison. Branch C
(static) is held fixed at its actual value, not attributed.

Background = 500 training samples, 200 Monte Carlo (baseline, alpha) draws
per explained sample, pooled model (seed 0), matching the user's spec.

CRITICAL UNITS STEP: the model predicts in per-cell-standardized target
space, so a raw gradient-based attribution is in standardized-target units,
not FWI. Because inverse-transform is affine (fwi = std_val * sigma_cell +
mu_cell), attribution rescales by the SAME per-cell sigma: EG_fwi = EG_std *
sigma_cell. Applied per explained sample using that sample's own cell's
fitted target std (per_grid_cell mode) before any aggregation.

Outputs (results/eg_attribution/):
  eg_percell_wide.csv   one row per explained sample: cell_id, aridity_tier,
                         eg_fwi_<branchA var> (7 cols), eg_fwi_lag<1-60>_<sm|ndvi|vod> (180 cols)
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
from fwi_attn.data.schema import BRANCH_A_FEATURES, BRANCH_B_FEATURES, BRANCH_B_N_WEEKS, BRANCH_C_FEATURES
from fwi_attn.data.splits import assign_split
from fwi_attn.data.standardization import Standardizer
from fwi_attn.models.lstm_attention import FWIAttnModel

MODEL_PATH = Path("results/cluster-ALL-seed0_20260904T171251Z_fd2892fb/model.pt")
OUT_DIR = Path("results/eg_attribution")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOOKBACK = BRANCH_B_N_WEEKS
CLASSES = ["Semi-Arid", "Sub-Humid", "Humid"]
N_PER_CLASS = 2000
K_DRAWS = 200
BACKGROUND_N = 500


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
    # globally (one group) instead.
    global_key = np.zeros(table.n, dtype=np.int64)

    xa_std = fit_standardizer(mode, eps, table.xa[train_mask], group_key[train_mask], BRANCH_A_FEATURES)
    xb_std = fit_standardizer(mode, eps, table.xb[train_mask], group_key[train_mask], BRANCH_B_FEATURES, extra_lead_dim=LOOKBACK)
    xc_std = fit_standardizer(mode, eps, table.xc[train_mask], global_key[train_mask], BRANCH_C_FEATURES)
    y_std = fit_standardizer(mode, eps, table.y[train_mask].reshape(-1, 1), group_key[train_mask], ["fwi"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    model = FWIAttnModel(config.model, branch_a_in=7, branch_b_in=3, branch_c_in=9, branches=("A", "B", "C")).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    gk_test = group_key[test_mask]
    xa_te = apply_standardizer(xa_std, table.xa[test_mask], gk_test)
    xb_te = apply_standardizer(xb_std, table.xb[test_mask], gk_test, extra_lead_dim=LOOKBACK)
    xc_te = apply_standardizer(xc_std, table.xc[test_mask], global_key[test_mask])
    cell_id_test = table.cell_id[test_mask]
    aridity_tier_test = table.aridity_tier[test_mask]

    rng = np.random.default_rng(0)
    expl_idx = np.concatenate([
        rng.choice(np.where(aridity_tier_test == c)[0], size=N_PER_CLASS, replace=False) for c in CLASSES
    ])
    print(f"explained samples: {len(expl_idx)} ({N_PER_CLASS}/class)")

    train_idx_local = np.where(train_mask)[0]
    bg_pick = rng.choice(len(train_idx_local), size=BACKGROUND_N, replace=False)
    gk_bg = group_key[train_mask][bg_pick]
    xa_bg = apply_standardizer(xa_std, table.xa[train_mask][bg_pick], gk_bg)
    xb_bg = apply_standardizer(xb_std, table.xb[train_mask][bg_pick], gk_bg, extra_lead_dim=LOOKBACK)

    xa_expl = torch.as_tensor(xa_te[expl_idx], dtype=torch.float32, device=device)
    xb_expl = torch.as_tensor(xb_te[expl_idx], dtype=torch.float32, device=device)
    xc_expl = torch.as_tensor(xc_te[expl_idx], dtype=torch.float32, device=device)
    xa_bg_t = torch.as_tensor(xa_bg, dtype=torch.float32, device=device)
    xb_bg_t = torch.as_tensor(xb_bg, dtype=torch.float32, device=device)

    n_expl = xa_expl.shape[0]
    eg_a_sum = torch.zeros_like(xa_expl)
    eg_b_sum = torch.zeros_like(xb_expl)

    torch.manual_seed(0)
    print(f"computing joint Expected Gradients: K={K_DRAWS} draws, background={BACKGROUND_N}...")
    with torch.backends.cudnn.flags(enabled=False):  # cuDNN RNN backward requires train() mode; model stays eval()
        for k in range(K_DRAWS):
            bidx = torch.randint(0, BACKGROUND_N, (n_expl,), device=device)  # same background sample -> coherent (xa,xb) pair
            baseline_a = xa_bg_t[bidx]
            baseline_b = xb_bg_t[bidx]
            alpha = torch.rand(n_expl, 1, device=device)  # shared interpolation depth across both branches

            xa_interp = (baseline_a + alpha * (xa_expl - baseline_a)).clone().requires_grad_(True)
            xb_interp = (baseline_b + alpha.unsqueeze(-1) * (xb_expl - baseline_b)).clone().requires_grad_(True)

            pred, _ = model(xa_interp, xb_interp, xc_expl)
            grad_a, grad_b = torch.autograd.grad(pred.sum(), [xa_interp, xb_interp])

            eg_a_sum += grad_a.detach() * (xa_expl - baseline_a)
            eg_b_sum += grad_b.detach() * (xb_expl - baseline_b)
            if (k + 1) % 20 == 0:
                print(f"  draw {k + 1}/{K_DRAWS}")

    eg_a = (eg_a_sum / K_DRAWS).cpu().numpy()  # (n_expl, 7), standardized-output units
    eg_b = (eg_b_sum / K_DRAWS).cpu().numpy()  # (n_expl, 60, 3), sequence order (oldest..recent), standardized-output units

    # --- rescale to FWI units via each sample's own cell target std (affine inverse-transform slope) ---
    gk_expl = gk_test[expl_idx]
    std_y_per_sample = np.array([y_std.std_[g][0] for g in gk_expl])
    eg_a_fwi = eg_a * std_y_per_sample[:, None]
    eg_b_fwi = eg_b * std_y_per_sample[:, None, None]

    # sequence index 0 = t-60 (oldest) .. 59 = t-1 (most recent); reverse -> lag 1..60, recent to old
    eg_b_fwi_lag = eg_b_fwi[:, ::-1, :]

    cols = {}
    for i, name in enumerate(BRANCH_A_FEATURES):
        cols[f"eg_fwi_{name}"] = eg_a_fwi[:, i]
    for lag in range(1, LOOKBACK + 1):
        for j, name in enumerate(BRANCH_B_FEATURES):
            cols[f"eg_fwi_lag{lag}_{name}"] = eg_b_fwi_lag[:, lag - 1, j]

    out = pd.DataFrame({"cell_id": cell_id_test[expl_idx], "aridity_tier": aridity_tier_test[expl_idx], **cols})
    out.to_csv(OUT_DIR / "eg_percell_wide.csv", index=False)
    print(f"saved eg_percell_wide.csv ({out.shape[0]} rows, {out.shape[1]} cols)")

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({
            "n_per_class": N_PER_CLASS, "n_total": int(n_expl), "k_draws": K_DRAWS, "background_n": BACKGROUND_N,
            "units": "FWI (rescaled per-sample by that cell's fitted target std, per_grid_cell mode)",
        }, f, indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
