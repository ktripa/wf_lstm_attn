"""Ties prediction -> back-transformation -> metrics -> CSV together, so no
call site can compute a metric on standardized values by accident.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from fwi_attn.data.standardization import Standardizer
from fwi_attn.evaluation.metrics import aggregate_by_group, per_cell_metrics, pooled_metrics


@torch.no_grad()
def predict_fwi(model, loader: DataLoader, device: str, target_standardizer: Standardizer, group_ids: np.ndarray) -> np.ndarray:
    """Returns model predictions back-transformed to FWI physical units.
    group_ids must align with `loader`'s iteration order (shuffle=False)."""
    model.eval()
    preds_std = []
    for batch in loader:
        xa = batch["xa"].to(device) if model.branch_a is not None else None
        xb = batch["xb"].to(device) if model.branch_b is not None else None
        xc = batch["xc"].to(device) if model.branch_c is not None else None
        pred, _ = model(xa, xb, xc)
        preds_std.append(pred.cpu().numpy())
    preds_std = np.concatenate(preds_std)
    return target_standardizer.inverse_transform(preds_std.reshape(-1, 1), group_ids).ravel()


def evaluate_predictions(
    obs_fwi: np.ndarray,
    pred_fwi: np.ndarray,
    cell_ids: np.ndarray,
    cluster_ids: np.ndarray,
    run_dir: Path,
    tag: str,
    nrmse_norm: str = "std",
    obs_normalized: np.ndarray | None = None,
    pred_normalized: np.ndarray | None = None,
) -> dict:
    """Computes and saves: per-cell metrics CSV (FWI units), cluster- and
    domain-aggregated metrics (mean-of-per-cell AND pooled-over-grid-weeks,
    both labeled), and -- if normalized arrays are also passed -- the
    equivalent normalized-space metrics saved separately and clearly tagged.
    """
    run_dir = Path(run_dir)
    per_cell = per_cell_metrics(obs_fwi, pred_fwi, cell_ids, units="fwi", nrmse_norm=nrmse_norm)
    per_cell.to_csv(run_dir / f"per_cell_metrics_{tag}_fwi.csv", index=False)

    cluster_lookup = pd.Series(cluster_ids, index=cell_ids).groupby(level=0).first()
    per_cell_cluster = per_cell["cell_id"].map(cluster_lookup)

    by_cluster = aggregate_by_group(per_cell, per_cell_cluster)
    by_cluster.to_csv(run_dir / f"cluster_agg_metrics_{tag}_fwi.csv", index=False)

    domain = aggregate_by_group(per_cell, pd.Series(["domain"] * len(per_cell)))
    domain.to_csv(run_dir / f"domain_agg_metrics_{tag}_fwi.csv", index=False)

    pooled_cluster = pooled_metrics(obs_fwi, pred_fwi, cluster_ids, units="fwi", nrmse_norm=nrmse_norm)
    pooled_cluster.to_csv(run_dir / f"pooled_metrics_by_cluster_{tag}_fwi.csv", index=False)
    pooled_domain = pooled_metrics(obs_fwi, pred_fwi, np.full(len(obs_fwi), "domain"), units="fwi", nrmse_norm=nrmse_norm)

    result = {
        "per_cell_fwi": per_cell,
        "by_cluster_fwi": by_cluster,
        "domain_fwi": domain,
        "pooled_by_cluster_fwi": pooled_cluster,
        "pooled_domain_fwi": pooled_domain,
    }

    if obs_normalized is not None and pred_normalized is not None:
        per_cell_norm = per_cell_metrics(obs_normalized, pred_normalized, cell_ids, units="normalized", nrmse_norm=nrmse_norm)
        per_cell_norm.to_csv(run_dir / f"per_cell_metrics_{tag}_normalized.csv", index=False)
        result["per_cell_normalized"] = per_cell_norm

    return result
