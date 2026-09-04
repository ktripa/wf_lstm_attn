"""Metrics. Callers must pass obs/sim already back-transformed to FWI physical
units for the "physical" metrics; pass standardized (normalized) arrays
separately and label results as normalized when that's wanted. This module
never assumes which space it was handed -- mislabeling is a caller error, so
`compute_all_metrics` takes an explicit `units` tag that gets stamped into
its output to make the two paths hard to confuse downstream.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _mask_valid(obs: np.ndarray, sim: np.ndarray):
    obs = np.asarray(obs, dtype=np.float64)
    sim = np.asarray(sim, dtype=np.float64)
    mask = np.isfinite(obs) & np.isfinite(sim)
    return obs[mask], sim[mask]


def kge(obs: np.ndarray, sim: np.ndarray) -> dict:
    """Gupta et al. (2009) KGE and its three components.
    KGE = 1 - sqrt((r-1)^2 + (alpha-1)^2 + (beta-1)^2)
      r     = Pearson correlation(obs, sim)
      alpha = std(sim) / std(obs)      -- variability ratio
      beta  = mean(sim) / mean(obs)    -- bias ratio
    """
    obs, sim = _mask_valid(obs, sim)
    if len(obs) < 2:
        return {"kge": np.nan, "r": np.nan, "alpha": np.nan, "beta": np.nan}
    obs_std, sim_std = np.std(obs), np.std(sim)
    obs_mean, sim_mean = np.mean(obs), np.mean(sim)
    if obs_std == 0 or obs_mean == 0:
        return {"kge": np.nan, "r": np.nan, "alpha": np.nan, "beta": np.nan}
    r = np.corrcoef(obs, sim)[0, 1]
    alpha = sim_std / obs_std
    beta = sim_mean / obs_mean
    val = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return {"kge": val, "r": r, "alpha": alpha, "beta": beta}


def r2(obs: np.ndarray, sim: np.ndarray) -> float:
    obs, sim = _mask_valid(obs, sim)
    if len(obs) < 2:
        return np.nan
    ss_res = np.sum((obs - sim) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    if ss_tot == 0:
        return np.nan
    return 1 - ss_res / ss_tot


def rmse(obs: np.ndarray, sim: np.ndarray) -> float:
    obs, sim = _mask_valid(obs, sim)
    if len(obs) == 0:
        return np.nan
    return float(np.sqrt(np.mean((obs - sim) ** 2)))


def mae(obs: np.ndarray, sim: np.ndarray) -> float:
    obs, sim = _mask_valid(obs, sim)
    if len(obs) == 0:
        return np.nan
    return float(np.mean(np.abs(obs - sim)))


def nrmse(obs: np.ndarray, sim: np.ndarray, norm: str = "std") -> float:
    """RMSE normalized by a scale of `obs`: "std", "mean", or "range"."""
    obs, sim = _mask_valid(obs, sim)
    if len(obs) == 0:
        return np.nan
    r = rmse(obs, sim)
    if norm == "std":
        scale = np.std(obs)
    elif norm == "mean":
        scale = np.mean(obs)
    elif norm == "range":
        scale = np.ptp(obs)
    else:
        raise ValueError(f"Unknown norm: {norm!r}")
    return float(r / scale) if scale != 0 else np.nan


def compute_all_metrics(obs: np.ndarray, sim: np.ndarray, units: str, nrmse_norm: str = "std") -> dict:
    """units: free-text tag, e.g. "fwi" or "normalized" -- stamped into the output
    so a metrics table can never silently mix the two spaces."""
    out = {"units": units, "n": int(np.isfinite(np.asarray(obs, dtype=float)).sum())}
    out.update(kge(obs, sim))
    out["r2"] = r2(obs, sim)
    out["rmse"] = rmse(obs, sim)
    out["mae"] = mae(obs, sim)
    out["nrmse"] = nrmse(obs, sim, norm=nrmse_norm)
    return out


def per_cell_metrics(obs: np.ndarray, sim: np.ndarray, cell_ids: np.ndarray, units: str, nrmse_norm: str = "std") -> pd.DataFrame:
    """obs/sim/cell_ids: (N,) flat arrays over all grid-weeks. One row per cell."""
    rows = []
    cell_ids = np.asarray(cell_ids)
    for cid in np.unique(cell_ids):
        mask = cell_ids == cid
        m = compute_all_metrics(obs[mask], sim[mask], units=units, nrmse_norm=nrmse_norm)
        m["cell_id"] = cid
        rows.append(m)
    return pd.DataFrame(rows)


def aggregate_by_group(per_cell_df: pd.DataFrame, group_ids: pd.Series, metric_cols: list[str] | None = None) -> pd.DataFrame:
    """Mean +/- std of per-cell metrics, grouped (e.g. by cluster, or a
    constant column for domain-wide). This is distinct from `pooled_metrics`,
    which recomputes the metric directly on pooled grid-weeks -- report both,
    labeled, since they answer different questions."""
    df = per_cell_df.copy()
    df["_group"] = np.asarray(group_ids)
    metric_cols = metric_cols or ["kge", "r", "alpha", "beta", "r2", "rmse", "mae", "nrmse"]
    agg = df.groupby("_group")[metric_cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{stat}" for c, stat in agg.columns]
    return agg.reset_index().rename(columns={"_group": "group"})


def pooled_metrics(obs: np.ndarray, sim: np.ndarray, group_ids: np.ndarray, units: str, nrmse_norm: str = "std") -> pd.DataFrame:
    """Metrics computed directly on all pooled grid-weeks within each group
    (not an average of per-cell metrics)."""
    rows = []
    group_ids = np.asarray(group_ids)
    for g in np.unique(group_ids):
        mask = group_ids == g
        m = compute_all_metrics(obs[mask], sim[mask], units=units, nrmse_norm=nrmse_norm)
        m["group"] = g
        rows.append(m)
    return pd.DataFrame(rows)
