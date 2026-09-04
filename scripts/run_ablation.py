#!/usr/bin/env python
"""One (branch-combination, seed) training + evaluation run for the pooled
(domain-wide, no cluster split) ablation study, used to populate Figure 3.

Saves, per run, under results/ablation/<tag>/:
  - per_cell_metrics_test.csv   (cell_id, lon, lat, aridity_tier, r2, rmse, mae, n)
  - pooled_by_tier_test.csv     (aridity_tier, r2, rmse, mae, n)  -- domain "All" too
  - summary.json

--extra-precip appends a 60-week gridMET precipitation history (same lookback
as Branch B) onto Branch A -- the control experiment: does Branch B still
help once antecedent precipitation is directly available to the model.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from fwi_attn.config import load_config
from fwi_attn.data.loaders import build_sample_table, load_raw_datasets
from fwi_attn.data.schema import BRANCH_A_FEATURES, BRANCH_B_FEATURES, BRANCH_B_N_WEEKS, BRANCH_C_FEATURES
from fwi_attn.data.splits import assign_split, split_counts
from fwi_attn.data.standardization import Standardizer
from fwi_attn.evaluation.metrics import per_cell_metrics, pooled_metrics
from fwi_attn.models.lstm_attention import FWIAttnModel
from fwi_attn.training.run_manager import init_run
from fwi_attn.training.seed import set_all_seeds

RESULTS_ROOT = Path("results/ablation")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--branches", nargs="+", required=True, choices=["A", "B", "C"])
    p.add_argument("--extra-precip", action="store_true", help="append 60-wk gridMET precip history to Branch A")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--lookback", type=int, default=BRANCH_B_N_WEEKS)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


def fit_standardizer(mode, eps, X, group_key, feature_names, extra_lead_dim=None):
    if extra_lead_dim is not None:
        n, t, f = X.shape
        X_flat = X.reshape(n * t, f)
        group_flat = np.repeat(group_key, t)
    else:
        X_flat, group_flat = X, group_key
    return Standardizer(mode, eps).fit(X_flat, group_flat, feature_names)


def apply_standardizer(std, X, group_key, extra_lead_dim=None):
    if extra_lead_dim is not None:
        n, t, f = X.shape
        X_flat = X.reshape(n * t, f)
        group_flat = np.repeat(group_key, t)
        return std.transform(X_flat, group_flat).reshape(n, t, f)
    return std.transform(X, group_key)


def main():
    args = parse_args()
    config = load_config(args.config)
    branches = tuple(sorted(args.branches))
    tag = "".join(branches) + ("_precipctrl" if args.extra_precip else "")
    set_all_seeds(args.seed)
    run_dir, logger, run_id = init_run(config, run_name=f"ablation-{tag}-seed{args.seed}")

    t0 = time.time()
    logger.info(f"branches={branches} extra_precip={args.extra_precip} seed={args.seed} tag={tag}")

    ds = load_raw_datasets(config)
    table, exclusions = build_sample_table(ds, config, lookback=args.lookback, cluster=None)
    logger.info(f"sample table n={table.n} exclusions={exclusions}")

    splits = assign_split(table.year)
    logger.info(f"split counts: {split_counts(splits)}")
    train_mask, val_mask, test_mask = splits == "train", splits == "val", splits == "test"

    mode, eps = config.standardization.mode, config.standardization.eps
    group_key = table.cell_id if mode == "per_grid_cell" else table.cluster_id
    # Branch C is static per cell -> per_grid_cell standardization gives zero
    # within-cell variance and collapses it to 0 everywhere. Standardize it
    # globally (one group) instead.
    global_key = np.zeros(table.n, dtype=np.int64)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    use_a, use_b, use_c = "A" in branches, "B" in branches, "C" in branches

    xa_full = table.xa
    branch_a_in = 7
    if args.extra_precip:
        xa_full = np.concatenate([table.xa, table.xa_precip_hist], axis=-1)
        branch_a_in = 7 + args.lookback
    a_feature_names = list(BRANCH_A_FEATURES) + [f"precip_tm{k}" for k in range(args.lookback, 0, -1)] if args.extra_precip else BRANCH_A_FEATURES

    std = {}
    if use_a:
        std["a"] = fit_standardizer(mode, eps, xa_full[train_mask], group_key[train_mask], a_feature_names)
    if use_b:
        std["b"] = fit_standardizer(mode, eps, table.xb[train_mask], group_key[train_mask], BRANCH_B_FEATURES, extra_lead_dim=args.lookback)
    if use_c:
        std["c"] = fit_standardizer(mode, eps, table.xc[train_mask], global_key[train_mask], BRANCH_C_FEATURES)
    y_std = fit_standardizer(mode, eps, table.y[train_mask].reshape(-1, 1), group_key[train_mask], ["fwi"])
    y_std.save(run_dir / "target_standardizer.joblib")

    def make_tensors(mask):
        gk = group_key[mask]
        xa_t = torch.as_tensor(apply_standardizer(std["a"], xa_full[mask], gk), dtype=torch.float32, device=device) if use_a else None
        xb_t = torch.as_tensor(apply_standardizer(std["b"], table.xb[mask], gk, extra_lead_dim=args.lookback), dtype=torch.float32, device=device) if use_b else None
        xc_t = torch.as_tensor(apply_standardizer(std["c"], table.xc[mask], global_key[mask]), dtype=torch.float32, device=device) if use_c else None
        y_t = torch.as_tensor(apply_standardizer(y_std, table.y[mask].reshape(-1, 1), gk).ravel(), dtype=torch.float32, device=device)
        return xa_t, xb_t, xc_t, y_t, gk

    n_train, n_val, n_test = int(train_mask.sum()), int(val_mask.sum()), int(test_mask.sum())
    xa_tr, xb_tr, xc_tr, y_tr, gk_tr = make_tensors(train_mask)
    xa_va, xb_va, xc_va, y_va, gk_va = make_tensors(val_mask)
    logger.info(f"device={device} train n={n_train} val n={n_val} test n={n_test}")

    model = FWIAttnModel(config.model, branch_a_in=branch_a_in, branch_b_in=3, branch_c_in=9, branches=branches).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    def batched_forward(xa_t, xb_t, xc_t, n, train=False):
        preds = []
        idx_iter = range(0, n, args.batch_size)
        for i in idx_iter:
            sl = slice(i, i + args.batch_size)
            a = xa_t[sl] if xa_t is not None else None
            b = xb_t[sl] if xb_t is not None else None
            c = xc_t[sl] if xc_t is not None else None
            p, _ = model(a, b, c)
            preds.append(p)
        return preds

    best_val_rmse, best_state, epochs_no_improve = float("inf"), None, 0
    history = []
    from fwi_attn.evaluation.metrics import rmse as rmse_fn

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss, n_seen = 0.0, 0
        for i in range(0, n_train, args.batch_size):
            idx = perm[i:i + args.batch_size]
            a = xa_tr[idx] if xa_tr is not None else None
            b = xb_tr[idx] if xb_tr is not None else None
            c = xc_tr[idx] if xc_tr is not None else None
            optimizer.zero_grad()
            pred, _ = model(a, b, c)
            loss = loss_fn(pred, y_tr[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
            n_seen += len(idx)
        train_loss = epoch_loss / n_seen

        model.eval()
        with torch.no_grad():
            preds_va = torch.cat(batched_forward(xa_va, xb_va, xc_va, n_val)).cpu().numpy()
        preds_va_fwi = y_std.inverse_transform(preds_va.reshape(-1, 1), gk_va).ravel()
        obs_va_fwi = y_std.inverse_transform(y_va.cpu().numpy().reshape(-1, 1), gk_va).ravel()
        val_rmse_fwi = rmse_fn(obs_va_fwi, preds_va_fwi)

        history.append({"epoch": epoch, "train_loss_std": train_loss, "val_rmse_fwi": val_rmse_fwi})
        logger.info(f"tag={tag} seed={args.seed} epoch={epoch} train_loss_std={train_loss:.4f} val_rmse_fwi={val_rmse_fwi:.4f}")

        if val_rmse_fwi < best_val_rmse:
            best_val_rmse = val_rmse_fwi
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                logger.info(f"early stopping at epoch={epoch} (best val_rmse_fwi={best_val_rmse:.4f})")
                break

    model.load_state_dict(best_state)
    xa_te, xb_te, xc_te, y_te, gk_te = make_tensors(test_mask)
    model.eval()
    with torch.no_grad():
        preds_te = torch.cat(batched_forward(xa_te, xb_te, xc_te, n_test)).cpu().numpy()
    preds_te_fwi = y_std.inverse_transform(preds_te.reshape(-1, 1), gk_te).ravel()
    obs_te_fwi = y_std.inverse_transform(y_te.cpu().numpy().reshape(-1, 1), gk_te).ravel()

    cell_id_test = table.cell_id[test_mask]
    aridity_tier_test = table.aridity_tier[test_mask]

    per_cell = per_cell_metrics(obs_te_fwi, preds_te_fwi, cell_id_test, units="fwi")
    cell_tier_map = pd.Series(aridity_tier_test, index=cell_id_test).groupby(level=0).first()
    per_cell["aridity_tier"] = per_cell["cell_id"].map(cell_tier_map)
    out_dir = RESULTS_ROOT / tag / f"seed{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    per_cell.to_csv(out_dir / "per_cell_metrics_test.csv", index=False)

    pooled_tier = pooled_metrics(obs_te_fwi, preds_te_fwi, aridity_tier_test, units="fwi")
    pooled_all = pooled_metrics(obs_te_fwi, preds_te_fwi, np.full(len(obs_te_fwi), "All"), units="fwi")
    pooled = pd.concat([pooled_tier, pooled_all], ignore_index=True)
    pooled.to_csv(out_dir / "pooled_by_tier_test.csv", index=False)

    summary = {
        "tag": tag, "branches": list(branches), "extra_precip": args.extra_precip, "seed": args.seed,
        "branch_a_in": branch_a_in, "n_train": int(n_train), "n_test": int(n_test),
        "epochs_trained": len(history), "best_val_rmse_fwi": best_val_rmse,
        "pooled_all_test": pooled_all.to_dict("records")[0], "wall_time_sec": time.time() - t0,
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"DONE tag={tag} seed={args.seed} pooled_all={pooled_all.to_dict('records')[0]} wall_time={summary['wall_time_sec']:.0f}s")
    print(json.dumps(summary, default=str))


if __name__ == "__main__":
    main()
