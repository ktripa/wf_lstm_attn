#!/usr/bin/env python
"""Figure 5: attention weights -- what the architecture's namesake mechanism
actually does, not yet used anywhere else in the paper.

Standalone: reads results/attention_analysis/*.csv written by
scripts/run_attention_analysis.py (pooled A+B+C model, seed 0), and writes
figures/fig5_attention_weights.{pdf,png} (300 dpi).

Lag convention: lag=1 is the most recent antecedent week (t-1), lag=60 the
oldest (t-60) -- "further back" means larger lag throughout.

  (a) Mean attention weight by lag, one line per aridity class, IQR shaded.
  (b) Heatmap: mean attention weight by lag (x) and calendar month (y),
      domain (three classes pooled) -- does the model look further back
      during fire season?
  (c) Spatial map of the attention centroid, sum(lag * attn) per grid cell --
      where the model draws on longer vs. shorter memory.
  (d) Mean attention weight vs. mean |Expected Gradients| attribution, by
      lag -- do the two interpretability methods agree on which weeks
      matter?
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

for _f in [
    "/usr/share/fonts/urw-base35/NimbusRoman-Regular.otf",
    "/usr/share/fonts/urw-base35/NimbusRoman-Bold.otf",
    "/usr/share/fonts/urw-base35/NimbusRoman-Italic.otf",
    "/usr/share/fonts/urw-base35/NimbusRoman-BoldItalic.otf",
]:
    if Path(_f).exists():
        fm.fontManager.addfont(_f)

BIG = 22
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
        "font.size": BIG,
        "axes.titlesize": BIG + 1,
        "axes.labelsize": BIG,
        "xtick.labelsize": BIG - 4,
        "ytick.labelsize": BIG - 4,
        "legend.fontsize": BIG - 6,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.linewidth": 1.2,
    }
)
PANEL_LABEL_SIZE = BIG + 6

DATA_DIR = Path("results/attention_analysis")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

CLASSES = ["Semi-Arid", "Sub-Humid", "Humid"]
CLASS_COLORS = {"Semi-Arid": "#E69F00", "Sub-Humid": "#009E73", "Humid": "#0072B2"}
MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def label_panel(ax, letter, dx=-0.17, dy=1.10):
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE, fontweight="bold", va="top", ha="left")


def panel_a(ax, df):
    for c in CLASSES:
        sub = df[df["aridity_tier"] == c].sort_values("lag")
        ax.plot(sub["lag"], sub["mean"], color=CLASS_COLORS[c], linewidth=2.6, label=c)
        ax.fill_between(sub["lag"], sub["q25"], sub["q75"], color=CLASS_COLORS[c], alpha=0.18, linewidth=0)
    ax.set_xlabel("Lag (weeks before present)")
    ax.set_ylabel("Attention weight")
    ax.set_xlim(1, 60)
    ax.legend(frameon=False, loc="upper right")
    label_panel(ax, "a")


def panel_b(ax, df):
    piv = df.pivot(index="month", columns="lag", values="attn").sort_index()
    lags = piv.columns.values
    months = piv.index.values
    pcm = ax.pcolormesh(lags, months, piv.values, cmap="magma", shading="nearest", rasterized=True)
    cbar = plt.colorbar(pcm, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Mean attention weight")
    cbar.ax.tick_params(labelsize=BIG - 5)
    ax.set_yticks(range(1, 13))
    ax.set_yticklabels(MONTH_ABBR)
    ax.set_xlabel("Lag (weeks before present)")
    ax.set_ylabel("Calendar month")
    label_panel(ax, "b")


def panel_c(ax, df):
    piv = df.pivot(index="lat", columns="lon", values="centroid_mean").sort_index().sort_index(axis=1)
    lons, lats, grid = piv.columns.values.astype(float), piv.index.values.astype(float), piv.values
    step = np.median(np.diff(lons))
    lon_e = np.concatenate([lons - step / 2, [lons[-1] + step / 2]])
    lat_e = np.concatenate([lats - step / 2, [lats[-1] + step / 2]])
    # Robust color limits: per-cell centroid is genuinely almost spatially
    # uniform (~29.5-31.3 weeks over the 10th-90th percentile) except for one
    # anomalous cell pair near (-94, 31.5-31.75) -- the same location flagged
    # as an outlier in Figure 3 -- which would otherwise stretch the whole
    # scale and wash out the real (subtle) spatial pattern.
    vmin, vmax = np.nanpercentile(grid, [2, 98])
    pcm = ax.pcolormesh(lon_e, lat_e, grid, cmap="cividis", vmin=vmin, vmax=vmax, shading="flat", rasterized=True)
    cbar = plt.colorbar(pcm, ax=ax, orientation="horizontal", location="bottom", fraction=0.046, pad=0.16, extend="both")
    cbar.set_label("Attention centroid (weeks before present)")
    cbar.ax.tick_params(labelsize=BIG - 5)
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_aspect(1 / np.cos(np.radians(np.nanmean(lats))))
    label_panel(ax, "c")


def panel_d(ax, df):
    from scipy.stats import spearmanr

    sc = ax.scatter(df["mean_attention"], df["mean_abs_eg"], c=df["lag"], cmap="viridis", s=90, edgecolor="black", linewidth=0.5, zorder=3)
    cbar = plt.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Lag (weeks before present)")
    cbar.ax.tick_params(labelsize=BIG - 5)

    # A handful of lags near the sequence boundaries (very recent / very old)
    # behave distinctly from the rest and visually separate from the main
    # cluster -- label them directly rather than let them read as unexplained
    # scatter, and report both a linear and a rank correlation since a few
    # extreme points can dominate Pearson's r.
    r_pearson = np.corrcoef(df["mean_attention"], df["mean_abs_eg"])[0, 1]
    r_spearman = spearmanr(df["mean_attention"], df["mean_abs_eg"]).correlation
    ax.text(
        0.04, 0.96, f"Pearson $r$ = {r_pearson:.2f}\nSpearman $\\rho$ = {r_spearman:.2f}",
        transform=ax.transAxes, va="top", ha="left", fontsize=BIG - 3, fontweight="bold",
    )
    for _, row in df.nlargest(2, "mean_abs_eg").iterrows():
        ax.annotate(f"lag {int(row['lag'])}", (row["mean_attention"], row["mean_abs_eg"]), textcoords="offset points", xytext=(10, 4), fontsize=BIG - 7)
    for _, row in df.nlargest(2, "mean_attention").iterrows():
        ax.annotate(f"lag {int(row['lag'])}", (row["mean_attention"], row["mean_abs_eg"]), textcoords="offset points", xytext=(-10, 8), ha="right", fontsize=BIG - 7)

    ax.set_xlabel("Mean attention weight")
    ax.set_ylabel("Mean |Expected Gradients| attribution")
    label_panel(ax, "d")


def main():
    a = pd.read_csv(DATA_DIR / "attn_by_lag_class.csv")
    b = pd.read_csv(DATA_DIR / "attn_by_lag_month.csv")
    c = pd.read_csv(DATA_DIR / "attn_centroid_percell.csv")
    d = pd.read_csv(DATA_DIR / "eg_vs_attention_by_lag.csv")

    fig, axes = plt.subplots(2, 2, figsize=(16, 14), constrained_layout=True)
    panel_a(axes[0, 0], a)
    panel_b(axes[0, 1], b)
    panel_c(axes[1, 0], c)
    panel_d(axes[1, 1], d)

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig5_attention_weights.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved {OUT_DIR / 'fig5_attention_weights.pdf'} and .png")


if __name__ == "__main__":
    main()
