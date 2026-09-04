"""Bootstrap CIs over grid cells, for the ablation study's skill-gain reporting."""
from __future__ import annotations

import numpy as np


def bootstrap_ci(values: np.ndarray, n_boot: int = 2000, ci: float = 0.95, seed: int | None = None) -> dict:
    """Percentile bootstrap CI for the mean of a per-cell metric array."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(seed)
    n = len(values)
    if n == 0:
        return {"mean": np.nan, "lower": np.nan, "upper": np.nan, "n_cells": 0}
    boot_means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[i] = values[idx].mean()
    alpha = (1 - ci) / 2
    lower, upper = np.quantile(boot_means, [alpha, 1 - alpha])
    return {"mean": float(values.mean()), "lower": float(lower), "upper": float(upper), "n_cells": n}


def bootstrap_paired_diff_ci(
    values_a: np.ndarray, values_b: np.ndarray, n_boot: int = 2000, ci: float = 0.95, seed: int | None = None
) -> dict:
    """Paired bootstrap CI for mean(values_a - values_b) over grid cells, e.g.
    delta-R2 or delta-RMSE between two branch combinations for the same cells."""
    values_a = np.asarray(values_a, dtype=np.float64)
    values_b = np.asarray(values_b, dtype=np.float64)
    mask = np.isfinite(values_a) & np.isfinite(values_b)
    diff = values_a[mask] - values_b[mask]
    return bootstrap_ci(diff, n_boot=n_boot, ci=ci, seed=seed)
