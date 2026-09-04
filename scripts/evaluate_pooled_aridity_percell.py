#!/usr/bin/env python
"""Post-hoc evaluation of the 5 already-trained pooled (ALL, no-cluster) models:
  1. Pooled skill (KGE + r/alpha/beta, R2, RMSE, MAE, NRMSE) aggregated by
     aridity tier (Arid/Semi-Arid/Sub-Humid/Humid), mean +- std across seeds.
  2. Per-grid-cell KGE (+ r/alpha/beta) computed separately for each of the 5
     seeds' test predictions, then averaged (mean +- std) across seeds --
     one row per grid cell.

Re-fits the input standardizers (xa/xb/xc) fresh rather than loading saved
ones, because run_cluster_gpu.py only persisted the target standardizer;
this is safe since standardization is fit on the training split only, which
is identical across seeds (seeds affect model init/shuffling, not the fit).
"""
from __future__ import annotations

import glob
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
from fwi_attn.evaluation.metrics import per_cell_metrics, pooled_metrics
from fwi_attn.models.lstm_attention import FWIAttnModel

SEEDS = [0, 1, 2, 3, 4]
RESULTS_DIR = Path("results")
OUT_DIR = Path("results/aridity_and_percell_analysis")


def fit_standardizer(mode, eps, X, group_key, feature_names, extra_lead_dim=None):
    if extra_lead_dim is not None:
        n, t, f = X.shape
        X_flat = X.reshape(n * t, f)
        group_flat = np.repeat(group_key, t)
    else:
        X_flat = X
        group_flat = group_key
    return Standardizer(mode, eps).fit(X_flat, group_flat, feature_names)


def apply_standardizer(std, X, group_key, extra_lead_dim=None):
    if extra_lead_dim is not None:
        n, t, f = X.shape
        X_flat = X.reshape(n * t, f)
        group_flat = np.repeat(group_key, t)
        return std.transform(X_flat, group_flat).reshape(n, t, f)
    return std.transform(X, group_key)


def find_run_dir(seed: int) -> Path:
    matches = sorted(glob.glob(str(RESULTS_DIR / f"cluster-ALL-seed{seed}_*")))
    if not matches:
        raise FileNotFoundError(f"No results dir found for seed={seed} under {RESULTS_DIR}")
    return Path(matches[-1])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    config = load_config("configs/default.yaml")
    lookback = BRANCH_B_N_WEEKS

    print("loading raw datasets + sample table (ALL, no cluster filter)...")
    ds = load_raw_datasets(config)
    table, exclusions = build_sample_table(ds, config, lookback=lookback, cluster=None)
    print(f"sample table n={table.n}, exclusions={exclusions}")

    splits = assign_split(table.year)
    train_mask, test_mask = splits == "train", splits == "test"

    mode, eps = config.standardization.mode, config.standardization.eps
    group_key = table.cell_id if mode == "per_grid_cell" else table.cluster_id
    # Branch C is static per cell -> per_grid_cell standardization gives zero
    # within-cell variance and collapses it to 0 everywhere. Standardize it
    # globally (one group) instead.
    global_key = np.zeros(table.n, dtype=np.int64)

    print("fitting input standardizers on train split (deterministic, seed-independent)...")
    xa_std = fit_standardizer(mode, eps, table.xa[train_mask], group_key[train_mask], BRANCH_A_FEATURES)
    xb_std = fit_standardizer(mode, eps, table.xb[train_mask], group_key[train_mask], BRANCH_B_FEATURES, extra_lead_dim=lookback)
    xc_std = fit_standardizer(mode, eps, table.xc[train_mask], global_key[train_mask], BRANCH_C_FEATURES)
    y_std = fit_standardizer(mode, eps, table.y[train_mask].reshape(-1, 1), group_key[train_mask], ["fwi"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}")

    gk_test = group_key[test_mask]
    xa_te = torch.as_tensor(apply_standardizer(xa_std, table.xa[test_mask], gk_test), dtype=torch.float32, device=device)
    xb_te = torch.as_tensor(apply_standardizer(xb_std, table.xb[test_mask], gk_test, extra_lead_dim=lookback), dtype=torch.float32, device=device)
    xc_te = torch.as_tensor(apply_standardizer(xc_std, table.xc[test_mask], global_key[test_mask]), dtype=torch.float32, device=device)

    obs_fwi = table.y[test_mask]
    cell_id_test = table.cell_id[test_mask]
    aridity_tier_test = table.aridity_tier[test_mask]
    month_test = table.month[test_mask]

    aridity_rows = []
    percell_frames = []
    monthly_rows = []
    decile_rows = []
    raw_frames = []
    N_DECILES = 10
    batch_size = 4096

    for seed in SEEDS:
        run_dir = find_run_dir(seed)
        print(f"seed={seed}: loading {run_dir}/model.pt")
        model = FWIAttnModel(config.model, branch_a_in=7, branch_b_in=3, branch_c_in=9, branches=("A", "B", "C")).to(device)
        model.load_state_dict(torch.load(run_dir / "model.pt", map_location=device))
        model.eval()

        preds_std = []
        with torch.no_grad():
            for i in range(0, xa_te.shape[0], batch_size):
                sl = slice(i, i + batch_size)
                p, _ = model(xa_te[sl], xb_te[sl], xc_te[sl])
                preds_std.append(p.cpu().numpy())
        preds_std = np.concatenate(preds_std)
        preds_fwi = y_std.inverse_transform(preds_std.reshape(-1, 1), gk_test).ravel()

        pooled_by_tier = pooled_metrics(obs_fwi, preds_fwi, aridity_tier_test, units="fwi", nrmse_norm=config.evaluation.nrmse_norm)
        pooled_by_tier["seed"] = seed
        aridity_rows.append(pooled_by_tier)

        pc = per_cell_metrics(obs_fwi, preds_fwi, cell_id_test, units="fwi", nrmse_norm=config.evaluation.nrmse_norm)
        pc["seed"] = seed
        percell_frames.append(pc)

        # monthly skill per aridity tier (pooled within each tier x month cell)
        tier_month_key = np.array([f"{t}|{m}" for t, m in zip(aridity_tier_test, month_test)])
        monthly = pooled_metrics(obs_fwi, preds_fwi, tier_month_key, units="fwi", nrmse_norm=config.evaluation.nrmse_norm)
        monthly[["aridity_tier", "month"]] = monthly["group"].str.split("|", expand=True)
        monthly["month"] = monthly["month"].astype(int)
        monthly["seed"] = seed
        monthly_rows.append(monthly)

        # conditional bias by observed-FWI decile, computed within each tier's own obs distribution
        df_seed = pd.DataFrame({"tier": aridity_tier_test, "obs": obs_fwi, "pred": preds_fwi})
        for tier, sub in df_seed.groupby("tier"):
            sub = sub.assign(decile=pd.qcut(sub["obs"], N_DECILES, labels=False, duplicates="drop") + 1)
            sub = sub.assign(bias=sub["pred"] - sub["obs"])
            g = sub.groupby("decile").agg(bias=("bias", "mean"), obs_mean=("obs", "mean"), n=("obs", "size"))
            g["aridity_tier"] = tier
            g["seed"] = seed
            decile_rows.append(g.reset_index())

        raw_frames.append(
            pd.DataFrame(
                {
                    "seed": seed,
                    "cell_id": cell_id_test,
                    "aridity_tier": aridity_tier_test,
                    "month": month_test,
                    "obs_fwi": obs_fwi,
                    "pred_fwi": preds_fwi,
                }
            )
        )

        print(f"seed={seed} done")

    aridity_all = pd.concat(aridity_rows, ignore_index=True)
    aridity_all.to_csv(OUT_DIR / "aridity_tier_pooled_per_seed.csv", index=False)

    metric_cols = ["kge", "r", "alpha", "beta", "r2", "rmse", "mae", "nrmse"]
    aridity_summary = aridity_all.groupby("group")[metric_cols].agg(["mean", "std"])
    aridity_summary.columns = [f"{c}_{s}" for c, s in aridity_summary.columns]
    aridity_summary = aridity_summary.reset_index().rename(columns={"group": "aridity_tier"})
    aridity_summary.to_csv(OUT_DIR / "aridity_tier_pooled_mean_std.csv", index=False)
    print("\n=== Aridity tier pooled skill (mean +- std across 5 seeds) ===")
    print(aridity_summary.to_string(index=False))

    percell_all = pd.concat(percell_frames, ignore_index=True)
    percell_summary = percell_all.groupby("cell_id")[metric_cols].agg(["mean", "std"])
    percell_summary.columns = [f"{c}_{s}" for c, s in percell_summary.columns]
    percell_summary = percell_summary.reset_index()

    cell_meta = pd.DataFrame(
        {
            "cell_id": ds["cell_id"],
            "lon": ds["lon"],
            "lat": ds["lat"],
            "cluster_region": ds["cluster_id"],
            "aridity_index": ds["aridity_index"],
            "aridity_tier": ds["aridity_tier"],
        }
    )
    percell_summary = cell_meta.merge(percell_summary, on="cell_id", how="right")
    percell_summary.to_csv(OUT_DIR / "per_grid_cell_kge_components_mean_std_5seeds.csv", index=False)
    print(f"\nPer-grid-cell KGE(+r/alpha/beta) across 5 seeds saved: {OUT_DIR / 'per_grid_cell_kge_components_mean_std_5seeds.csv'} ({len(percell_summary)} cells)")

    monthly_all = pd.concat(monthly_rows, ignore_index=True)
    monthly_all.to_csv(OUT_DIR / "monthly_skill_per_seed.csv", index=False)
    monthly_summary = monthly_all.groupby(["aridity_tier", "month"])[metric_cols].agg(["mean", "std"])
    monthly_summary.columns = [f"{c}_{s}" for c, s in monthly_summary.columns]
    monthly_summary = monthly_summary.reset_index()
    monthly_summary.to_csv(OUT_DIR / "monthly_skill_mean_std.csv", index=False)
    print(f"Monthly skill by tier saved: {OUT_DIR / 'monthly_skill_mean_std.csv'}")

    decile_all = pd.concat(decile_rows, ignore_index=True)
    decile_all.to_csv(OUT_DIR / "decile_bias_per_seed.csv", index=False)
    decile_summary = decile_all.groupby(["aridity_tier", "decile"]).agg(
        bias_mean=("bias", "mean"), bias_std=("bias", "std"), obs_mean=("obs_mean", "mean"), n_mean=("n", "mean")
    ).reset_index()
    decile_summary.to_csv(OUT_DIR / "decile_bias_mean_std.csv", index=False)
    print(f"Decile conditional bias saved: {OUT_DIR / 'decile_bias_mean_std.csv'}")

    raw_all = pd.concat(raw_frames, ignore_index=True)
    raw_all.to_parquet(OUT_DIR / "raw_test_predictions_5seeds.parquet", index=False)
    print(f"Raw predictions saved: {OUT_DIR / 'raw_test_predictions_5seeds.parquet'} ({len(raw_all)} rows)")

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(
            {
                "n_test": int(test_mask.sum()),
                "seeds": SEEDS,
                "aridity_tier_summary": aridity_summary.to_dict("records"),
            },
            f,
            indent=2,
            default=str,
        )


if __name__ == "__main__":
    main()
