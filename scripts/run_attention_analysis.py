#!/usr/bin/env python
"""Extracts and aggregates Branch-B attention weights from the trained pooled
A+B+C model (seed 0), plus an Expected Gradients attribution baseline, for
Figure 5. Attention weights are never persisted by the training/eval scripts
-- this is the first place they're pulled out of the model.

Lag convention used throughout: lag=1 is the most recent antecedent week
(t-1), lag=60 is the oldest (t-60). Branch B's sequence is fed oldest-first,
so lag k corresponds to sequence index (60-k); reversing the raw attention
array along its time axis converts it to lag order directly.

Outputs (results/attention_analysis/):
  attn_by_lag_class.csv     class, lag, mean, q25, q75
  attn_by_lag_month.csv     lag, month, mean          (domain: 3 classes pooled)
  attn_centroid_percell.csv cell_id, lon, lat, centroid_mean
  eg_vs_attention_by_lag.csv lag, mean_attention, mean_abs_eg
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
OUT_DIR = Path("results/attention_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOOKBACK = BRANCH_B_N_WEEKS
CLASSES = ["Semi-Arid", "Sub-Humid", "Humid"]
EG_N_PER_CLASS = 2000
EG_K_DRAWS = 50
EG_BACKGROUND_N = 300
BATCH = 4096


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
    table, exclusions = build_sample_table(ds, config, lookback=LOOKBACK, cluster=None)
    print(f"n={table.n}")

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
    month_test = table.month[test_mask]

    n_test = xa_te.shape[0]
    attn_all = np.empty((n_test, LOOKBACK), dtype=np.float32)
    print("forward pass to extract attention weights...")
    with torch.no_grad():
        for i in range(0, n_test, BATCH):
            sl = slice(i, i + BATCH)
            a = torch.as_tensor(xa_te[sl], dtype=torch.float32, device=device)
            b = torch.as_tensor(xb_te[sl], dtype=torch.float32, device=device)
            c = torch.as_tensor(xc_te[sl], dtype=torch.float32, device=device)
            _, attn = model(a, b, c)
            attn_all[sl] = attn.cpu().numpy()

    # sequence index 0 = t-60 (oldest) ... 59 = t-1 (most recent); reverse -> lag 1..60, recent to old
    attn_by_lag = attn_all[:, ::-1]
    lags = np.arange(1, LOOKBACK + 1)

    meta = pd.DataFrame({
        "cell_id": cell_id_test, "aridity_tier": aridity_tier_test, "month": month_test,
    })
    meta_lookup = pd.read_csv("results/aridity_and_percell_analysis/per_grid_cell_kge_components_mean_std_5seeds.csv")[["cell_id", "lon", "lat"]]

    # --- (a) mean attention by lag, per class, with IQR ---
    rows_a = []
    for c in CLASSES:
        cmask = (aridity_tier_test == c)
        vals = attn_by_lag[cmask]  # (n_c, 60)
        rows_a.append(pd.DataFrame({
            "aridity_tier": c, "lag": lags,
            "mean": vals.mean(axis=0), "q25": np.percentile(vals, 25, axis=0), "q75": np.percentile(vals, 75, axis=0),
        }))
    pd.concat(rows_a, ignore_index=True).to_csv(OUT_DIR / "attn_by_lag_class.csv", index=False)
    print("saved attn_by_lag_class.csv")

    # --- (b) mean attention by (lag, month), domain = 3 classes pooled ---
    domain_mask = np.isin(aridity_tier_test, CLASSES)
    df_lm = pd.DataFrame(attn_by_lag[domain_mask], columns=lags)
    df_lm["month"] = month_test[domain_mask]
    long = df_lm.melt(id_vars="month", var_name="lag", value_name="attn")
    by_lag_month = long.groupby(["lag", "month"])["attn"].mean().reset_index()
    by_lag_month.to_csv(OUT_DIR / "attn_by_lag_month.csv", index=False)
    print("saved attn_by_lag_month.csv")

    # --- (c) per-cell attention centroid = sum(lag * attn) ---
    centroid = (attn_by_lag * lags[None, :]).sum(axis=1)  # (n_test,) since attn sums to 1 per sample
    df_centroid = pd.DataFrame({"cell_id": cell_id_test, "centroid": centroid})
    percell_centroid = df_centroid.groupby("cell_id")["centroid"].mean().reset_index().rename(columns={"centroid": "centroid_mean"})
    percell_centroid = percell_centroid.merge(meta_lookup, on="cell_id", how="left")
    percell_centroid.to_csv(OUT_DIR / "attn_centroid_percell.csv", index=False)
    print(f"saved attn_centroid_percell.csv ({len(percell_centroid)} cells); "
          f"centroid range [{centroid.min():.1f}, {centroid.max():.1f}], mean={centroid.mean():.2f}")

    # --- (d) Expected Gradients for Branch B, vs. mean attention, by lag ---
    print("computing Expected Gradients...")
    rng = np.random.default_rng(0)
    expl_idx = []
    for c in CLASSES:
        idx_c = np.where(aridity_tier_test == c)[0]
        expl_idx.append(rng.choice(idx_c, size=min(EG_N_PER_CLASS, len(idx_c)), replace=False))
    expl_idx = np.concatenate(expl_idx)

    train_idx_pool = np.where(train_mask)[0]
    bg_idx = rng.choice(len(train_idx_pool), size=EG_BACKGROUND_N, replace=False)
    gk_bg = group_key[train_mask][bg_idx]
    xb_bg = apply_standardizer(xb_std, table.xb[train_mask][bg_idx], gk_bg, extra_lead_dim=LOOKBACK)
    xb_bg_t = torch.as_tensor(xb_bg, dtype=torch.float32, device=device)

    xa_expl = torch.as_tensor(xa_te[expl_idx], dtype=torch.float32, device=device)
    xb_expl = torch.as_tensor(xb_te[expl_idx], dtype=torch.float32, device=device)
    xc_expl = torch.as_tensor(xc_te[expl_idx], dtype=torch.float32, device=device)

    eg_sum = torch.zeros_like(xb_expl)
    torch.manual_seed(0)
    # cuDNN refuses to backprop through an RNN while the module is in eval()
    # mode ("cudnn RNN backward can only be called in training mode"). Disable
    # cuDNN for just this block so PyTorch falls back to its native LSTM
    # backward, which works in eval mode -- model stays in eval() throughout,
    # so dropout stays off and this doesn't add gradient noise.
    with torch.backends.cudnn.flags(enabled=False):
        for k in range(EG_K_DRAWS):
            bidx = torch.randint(0, xb_bg_t.shape[0], (xb_expl.shape[0],), device=device)
            baseline = xb_bg_t[bidx]
            alpha = torch.rand(xb_expl.shape[0], 1, 1, device=device)
            x_interp = (baseline + alpha * (xb_expl - baseline)).clone().requires_grad_(True)
            pred, _ = model(xa_expl, x_interp, xc_expl)
            grad = torch.autograd.grad(pred.sum(), x_interp)[0]
            eg_sum += grad.detach() * (xb_expl - baseline)
            if (k + 1) % 10 == 0:
                print(f"  EG draw {k + 1}/{EG_K_DRAWS}")
    eg = (eg_sum / EG_K_DRAWS).cpu().numpy()  # (n_expl, 60, 3), sequence order (oldest..recent)

    eg_by_lag = np.abs(eg[:, ::-1, :]).mean(axis=(0, 2))  # -> lag order, mean |attribution| over samples & features
    attn_by_lag_expl = attn_by_lag[expl_idx].mean(axis=0)

    eg_df = pd.DataFrame({"lag": lags, "mean_attention": attn_by_lag_expl, "mean_abs_eg": eg_by_lag})
    eg_df.to_csv(OUT_DIR / "eg_vs_attention_by_lag.csv", index=False)
    r = np.corrcoef(eg_df["mean_attention"], eg_df["mean_abs_eg"])[0, 1]
    print(f"saved eg_vs_attention_by_lag.csv; Pearson r (attention vs |EG|, by lag) = {r:.3f}")

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump({
            "n_test": int(n_test), "centroid_mean": float(centroid.mean()), "centroid_std": float(centroid.std()),
            "eg_n_explained": int(len(expl_idx)), "eg_k_draws": EG_K_DRAWS, "eg_background_n": EG_BACKGROUND_N,
            "attention_vs_eg_pearson_r": float(r),
        }, f, indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
