#!/usr/bin/env python
"""Figure 4: Expected Gradients attribution, pooled model (seed 0), joint
Branch-A + Branch-B attribution (background=500 training samples, K=200 MC
draws per attribution), rescaled to FWI units via each sample's own cell's
target standard deviation.

2x3 grid:
  (a-c) Concurrent meteorology (Branch A), one panel per aridity class:
        horizontal bars, mean signed attribution (FWI units), 95% CI,
        sorted by |magnitude|.
  (d-f) Antecedent branch (Branch B), one panel per class: mean attribution
        by lag week, one line per variable (SM/NDVI/VOD) with a shaded 95%
        CI band, shared y-axis, zero reference. x-axis runs lag 12 (left,
        furthest back) -> lag 1 (right, most recent), so time reads
        left-to-right toward the present.

(Grouped Branch-A attribution and the branch-level 60-week comparison,
formerly panels d/h, are computed in scripts/run_eg_attribution_full.py's
output but dropped from this figure per request.)
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

BIG = 30  # was 19; "add 10 points minimum" to every font in the figure
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

DATA_DIR = Path("results/eg_attribution")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)

CLASSES = ["Semi-Arid", "Sub-Humid", "Humid"]
CLASS_COLORS = {"Semi-Arid": "#E69F00", "Sub-Humid": "#009E73", "Humid": "#0072B2"}

BRANCH_A_FEATURES = ["tmax", "tmin", "rhmax", "rhmin", "precip", "vpd", "wind_speed"]
A_LABELS = {"tmax": "Tmax", "tmin": "Tmin", "rhmax": "RHmax", "rhmin": "RHmin", "precip": "Precip", "vpd": "VPD", "wind_speed": "Wind speed"}

BRANCH_B_FEATURES = ["soil_moisture", "ndvi", "vod"]
B_LABELS = {"soil_moisture": "SM", "ndvi": "NDVI", "vod": "VOD"}
B_COLORS = {"soil_moisture": "#8B5A2B", "ndvi": "#3A9D45", "vod": "#2166AC"}

LOOKBACK_SHOWN = 12


def label_panel(ax, letter, dx=-0.22, dy=1.10):
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE, fontweight="bold", va="top", ha="left")


def mean_ci95(x: np.ndarray):
    x = x[np.isfinite(x)]
    m, sem = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    return m, 1.96 * sem


def panel_abc(ax, df, cls, letter):
    sub = df[df["aridity_tier"] == cls]
    rows = []
    for v in BRANCH_A_FEATURES:
        m, ci = mean_ci95(sub[f"eg_fwi_{v}"].to_numpy())
        rows.append((v, m, ci))
    rows.sort(key=lambda r: -abs(r[1]))
    names = [A_LABELS[r[0]] for r in rows]
    means = [r[1] for r in rows]
    cis = [r[2] for r in rows]

    y = np.arange(len(names))[::-1]
    ax.barh(y, means, xerr=cis, color=CLASS_COLORS[cls], edgecolor="black", linewidth=0.8, capsize=4, error_kw={"linewidth": 1.4})
    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Mean attribution (FWI)")
    ax.set_title(cls, color=CLASS_COLORS[cls], fontweight="bold", fontsize=BIG)
    label_panel(ax, letter)


def panel_def(ax, df, cls, letter, ylim):
    sub = df[df["aridity_tier"] == cls]
    lags = np.arange(1, LOOKBACK_SHOWN + 1)
    for v in BRANCH_B_FEATURES:
        means, los, his = [], [], []
        for lag in lags:
            m, ci = mean_ci95(sub[f"eg_fwi_lag{lag}_{v}"].to_numpy())
            means.append(m)
            los.append(m - ci)
            his.append(m + ci)
        ax.plot(lags, means, color=B_COLORS[v], marker="o", markersize=7, linewidth=3, label=B_LABELS[v])
        ax.fill_between(lags, los, his, color=B_COLORS[v], alpha=0.2, linewidth=0)
    ax.axhline(0, color="black", linewidth=1.2, linestyle="--")
    ax.set_xticks(lags)
    ax.set_xlim(LOOKBACK_SHOWN + 0.5, 0.5)  # reverse: lag 1 (most recent) on the right
    ax.set_xlabel("Lag (weeks before present)")
    ax.set_ylabel("Mean attribution (FWI)")
    ax.set_ylim(*ylim)
    ax.set_title(cls, color=CLASS_COLORS[cls], fontweight="bold", fontsize=BIG)
    ax.legend(frameon=False, loc="upper left")
    label_panel(ax, letter)


def main():
    df = pd.read_csv(DATA_DIR / "eg_percell_wide.csv")

    fig, axes = plt.subplots(2, 3, figsize=(24, 15), constrained_layout=True)

    panel_abc(axes[0, 0], df, "Semi-Arid", "a")
    panel_abc(axes[0, 1], df, "Sub-Humid", "b")
    panel_abc(axes[0, 2], df, "Humid", "c")

    # shared y-axis across d/e/f, including the CI band -> use the CI bounds, not just the means
    all_bounds = []
    for cls in CLASSES:
        sub = df[df["aridity_tier"] == cls]
        for v in BRANCH_B_FEATURES:
            for lag in range(1, LOOKBACK_SHOWN + 1):
                m, ci = mean_ci95(sub[f"eg_fwi_lag{lag}_{v}"].to_numpy())
                all_bounds += [m - ci, m + ci]
    pad = 0.08 * (max(all_bounds) - min(all_bounds))
    ylim = (min(all_bounds) - pad, max(all_bounds) + pad)

    panel_def(axes[1, 0], df, "Semi-Arid", "d", ylim)
    panel_def(axes[1, 1], df, "Sub-Humid", "e", ylim)
    panel_def(axes[1, 2], df, "Humid", "f", ylim)

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig4_eg_attribution.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved {OUT_DIR / 'fig4_eg_attribution.pdf'} and .png")


if __name__ == "__main__":
    main()
