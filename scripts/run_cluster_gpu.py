#!/usr/bin/env python
"""Fast, single-cluster, single-seed training run for the full A+B+C
LSTM+attention model, meant to run as one SLURM task per climate cluster
(5 clusters -> 5 parallel GPU jobs, see scripts/submit_all_clusters.sh).

Bypasses the generic DataLoader path in training/train.py for speed: with a
60-week lookback, a whole cluster's standardized train split still fits
comfortably in GPU memory, so batches are sliced directly from GPU-resident
tensors instead of going through a Dataset/DataLoader per epoch.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from fwi_attn.config import load_config
from fwi_attn.data.loaders import build_sample_table, load_raw_datasets
from fwi_attn.data.schema import BRANCH_A_FEATURES, BRANCH_B_FEATURES, BRANCH_B_N_WEEKS, BRANCH_C_FEATURES
from fwi_attn.data.splits import assign_split, split_counts
from fwi_attn.data.standardization import Standardizer
from fwi_attn.evaluation.evaluate import evaluate_predictions
from fwi_attn.evaluation.metrics import rmse
from fwi_attn.models.lstm_attention import FWIAttnModel
from fwi_attn.training.run_manager import init_run
from fwi_attn.training.seed import set_all_seeds


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--cluster", required=True, help="region name (e.g. Hot-Dry), or ALL for one pooled domain-wide model")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lookback", type=int, default=BRANCH_B_N_WEEKS)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    return p.parse_args()


def fit_standardizer(mode, eps, X, group_key, feature_names, extra_lead_dim=None):
    """extra_lead_dim: if X is (N, T, F), pass T to flatten the T axis into rows
    (each timestep of a sample shares that sample's group id)."""
    if extra_lead_dim is not None:
        n, t, f = X.shape
        X_flat = X.reshape(n * t, f)
        group_flat = np.repeat(group_key, t)
    else:
        X_flat = X
        group_flat = group_key
    return Standardizer(mode, eps).fit(X_flat, group_flat, feature_names)


def apply_standardizer(std: Standardizer, X, group_key, extra_lead_dim=None):
    if extra_lead_dim is not None:
        n, t, f = X.shape
        X_flat = X.reshape(n * t, f)
        group_flat = np.repeat(group_key, t)
        return std.transform(X_flat, group_flat).reshape(n, t, f)
    return std.transform(X, group_key)


def main():
    args = parse_args()
    config = load_config(args.config)
    set_all_seeds(args.seed)
    run_dir, logger, run_id = init_run(config, run_name=f"cluster-{args.cluster}-seed{args.seed}")

    t0 = time.time()
    logger.info(f"cluster={args.cluster} seed={args.seed} lookback={args.lookback} run_id={run_id}")

    ds = load_raw_datasets(config)
    logger.info(f"raw grid loaded: n_cells={ds['n_cells']} n_weeks={len(ds['week_dates'])}")

    cluster_filter = None if args.cluster.upper() == "ALL" else args.cluster
    table, exclusions = build_sample_table(ds, config, lookback=args.lookback, cluster=cluster_filter)
    logger.info(f"sample table: n={table.n} n_cells_in_cluster={len(np.unique(table.cell_id))}")
    logger.info(f"exclusions: {exclusions}")
    if table.n == 0:
        raise RuntimeError(f"No samples for cluster={args.cluster!r} -- check the cluster name matches the region field")

    splits = assign_split(table.year)
    counts = split_counts(splits)
    logger.info(f"split counts: {counts}")
    train_mask, val_mask, test_mask = splits == "train", splits == "val", splits == "test"

    mode, eps = config.standardization.mode, config.standardization.eps
    group_key = table.cell_id if mode == "per_grid_cell" else table.cluster_id
    # Branch C is static (constant within a cell across all its weekly rows),
    # so a per_grid_cell standardizer sees zero within-group variance and
    # collapses every cell's Branch C input to exactly 0 -- the model never
    # sees any cross-cell difference in land cover/elevation/climatology.
    # Branch C needs *cross-cell* (global) standardization, not within-cell.
    global_key = np.zeros(table.n, dtype=np.int64)

    xa_std = fit_standardizer(mode, eps, table.xa[train_mask], group_key[train_mask], BRANCH_A_FEATURES)
    xb_std = fit_standardizer(mode, eps, table.xb[train_mask], group_key[train_mask], BRANCH_B_FEATURES, extra_lead_dim=args.lookback)
    xc_std = fit_standardizer(mode, eps, table.xc[train_mask], global_key[train_mask], BRANCH_C_FEATURES)
    y_std = fit_standardizer(mode, eps, table.y[train_mask].reshape(-1, 1), group_key[train_mask], ["fwi"])
    y_std.save(run_dir / "target_standardizer.joblib")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"device={device} ({torch.cuda.get_device_name(0) if device == 'cuda' else 'CPU'})")

    def to_tensor_split(mask):
        gk = group_key[mask]
        xa = apply_standardizer(xa_std, table.xa[mask], gk)
        xb = apply_standardizer(xb_std, table.xb[mask], gk, extra_lead_dim=args.lookback)
        xc = apply_standardizer(xc_std, table.xc[mask], global_key[mask])
        y = apply_standardizer(y_std, table.y[mask].reshape(-1, 1), gk).ravel()
        return (
            torch.as_tensor(xa, dtype=torch.float32, device=device),
            torch.as_tensor(xb, dtype=torch.float32, device=device),
            torch.as_tensor(xc, dtype=torch.float32, device=device),
            torch.as_tensor(y, dtype=torch.float32, device=device),
            gk,
        )

    xa_tr, xb_tr, xc_tr, y_tr, gk_tr = to_tensor_split(train_mask)
    xa_va, xb_va, xc_va, y_va, gk_va = to_tensor_split(val_mask)
    logger.info(f"train n={xa_tr.shape[0]} val n={xa_va.shape[0]} test n={int(test_mask.sum())}")

    model = FWIAttnModel(config.model, branch_a_in=7, branch_b_in=3, branch_c_in=9, branches=("A", "B", "C")).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    n_train = xa_tr.shape[0]
    best_val_rmse, best_state, epochs_no_improve = float("inf"), None, 0
    history = []

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(n_train, device=device)
        epoch_loss, n_seen = 0.0, 0
        for i in range(0, n_train, args.batch_size):
            idx = perm[i:i + args.batch_size]
            optimizer.zero_grad()
            pred, _ = model(xa_tr[idx], xb_tr[idx], xc_tr[idx])
            loss = loss_fn(pred, y_tr[idx])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)
            n_seen += len(idx)
        train_loss = epoch_loss / n_seen

        model.eval()
        with torch.no_grad():
            preds_va = []
            for i in range(0, xa_va.shape[0], args.batch_size):
                sl = slice(i, i + args.batch_size)
                p, _ = model(xa_va[sl], xb_va[sl], xc_va[sl])
                preds_va.append(p.cpu().numpy())
            preds_va = np.concatenate(preds_va)
        preds_va_fwi = y_std.inverse_transform(preds_va.reshape(-1, 1), gk_va).ravel()
        obs_va_fwi = y_std.inverse_transform(y_va.cpu().numpy().reshape(-1, 1), gk_va).ravel()
        val_rmse_fwi = rmse(obs_va_fwi, preds_va_fwi)

        history.append({"epoch": epoch, "train_loss_std": train_loss, "val_rmse_fwi": val_rmse_fwi})
        logger.info(f"epoch={epoch} train_loss_std={train_loss:.4f} val_rmse_fwi={val_rmse_fwi:.4f}")

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
    torch.save(best_state, run_dir / "model.pt")
    import pandas as pd

    pd.DataFrame(history).to_csv(run_dir / "history.csv", index=False)

    xa_te, xb_te, xc_te, y_te, gk_te = to_tensor_split(test_mask)
    model.eval()
    with torch.no_grad():
        preds_te = []
        for i in range(0, xa_te.shape[0], args.batch_size):
            sl = slice(i, i + args.batch_size)
            p, _ = model(xa_te[sl], xb_te[sl], xc_te[sl])
            preds_te.append(p.cpu().numpy())
        preds_te = np.concatenate(preds_te)
    preds_te_fwi = y_std.inverse_transform(preds_te.reshape(-1, 1), gk_te).ravel()
    obs_te_fwi = y_std.inverse_transform(y_te.cpu().numpy().reshape(-1, 1), gk_te).ravel()

    result = evaluate_predictions(
        obs_te_fwi,
        preds_te_fwi,
        table.cell_id[test_mask],
        table.cluster_id[test_mask],
        run_dir,
        tag="test",
        nrmse_norm=config.evaluation.nrmse_norm,
    )

    headline = result["pooled_domain_fwi"].to_dict("records")[0]
    logger.info(f"TEST (pooled, FWI units): {headline}")

    summary = {
        "run_id": run_id,
        "cluster": args.cluster,
        "seed": args.seed,
        "lookback": args.lookback,
        "n_train": int(xa_tr.shape[0]),
        "n_val": int(xa_va.shape[0]),
        "n_test": int(xa_te.shape[0]),
        "exclusions": exclusions,
        "split_counts": counts,
        "best_val_rmse_fwi": best_val_rmse,
        "epochs_trained": len(history),
        "test_pooled_metrics_fwi": headline,
        "wall_time_sec": time.time() - t0,
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"DONE cluster={args.cluster} seed={args.seed} wall_time={summary['wall_time_sec']:.0f}s")
    print(json.dumps(summary, default=str))


if __name__ == "__main__":
    main()
