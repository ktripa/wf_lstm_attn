"""Diagnostic plots. Stubs for now -- fill in once end-to-end results exist
on the first cluster (see repo README priority order).

Planned, all faceted/colored by cluster:
  - skill_by_decile(obs, pred, cluster_ids): skill metric conditioned on
    observed-FWI decile, to check whether the model damps extremes.
  - skill_by_season(obs, pred, cluster_ids, week_of_year)
  - conditional_bias(obs, pred, cluster_ids): (pred - obs) vs obs.
  - obs_vs_pred_scatter(obs, pred, cluster_ids)
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def obs_vs_pred_scatter(obs: np.ndarray, pred: np.ndarray, title: str = ""):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(obs, pred, s=4, alpha=0.3)
    lims = [min(obs.min(), pred.min()), max(obs.max(), pred.max())]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlabel("Observed FWI")
    ax.set_ylabel("Predicted FWI")
    ax.set_title(title)
    return fig
