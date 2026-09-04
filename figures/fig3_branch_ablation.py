#!/usr/bin/env python
"""Figure 3: what information the branches carry (7-way branch ablation),
pooled domain-wide model, 5 seeds. No antecedent-precipitation control here.

Standalone: reads only results/ablation/<tag>/seed<n>/{pooled_by_tier_test.csv,
per_cell_metrics_test.csv} written by scripts/run_ablation.py, and writes
figures/fig3_branch_ablation.png (300 dpi). PNG only -- no PDF.

3x3 layout:
  Row 1 -- spatial KGE maps, single-branch and A+B:
    (a) A  (concurrent meteorological drivers)
    (b) B  (antecedent fuel-moisture drivers)
    (c) A+B
  Row 2 -- spatial KGE maps, Branch C added:
    (d) A+C
    (e) B+C
    (f) A+B+C (full model)
  All six maps share one color scale (percentile-robust across all six).
  Row 3 -- skill-gain dot plots (pooled test R2, mean over 5 seeds), one row
  per class (All / Semi-Arid / Sub-Humid / Humid), delta annotated:
    (g) A -> A+B   (h) A -> A+C   (i) B -> B+C
"""
from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------------------------------------------------------------- style ----
for _f in [
    "/usr/share/fonts/urw-base35/NimbusRoman-Regular.otf",
    "/usr/share/fonts/urw-base35/NimbusRoman-Bold.otf",
    "/usr/share/fonts/urw-base35/NimbusRoman-Italic.otf",
    "/usr/share/fonts/urw-base35/NimbusRoman-BoldItalic.otf",
]:
    if Path(_f).exists():
        fm.fontManager.addfont(_f)

BIG = 44  # pushed up again per "still fontsize low"
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
        "font.size": BIG,
        "axes.titlesize": BIG,
        "axes.labelsize": BIG,
        "xtick.labelsize": BIG - 5,
        "ytick.labelsize": BIG - 5,
        "legend.fontsize": BIG - 6,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.linewidth": 1.4,
    }
)
PANEL_LABEL_SIZE = BIG + 6

ABLATION_DIR = Path("results/ablation")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)
SHAPEFILE = Path("results/shapefiles/tx_ok_nm_az_states.shp")
SEEDS = [0, 1, 2, 3, 4]

CLASSES = ["Semi-Arid", "Sub-Humid", "Humid"]
CLASS_COLORS = {"Semi-Arid": "#E69F00", "Sub-Humid": "#009E73", "Humid": "#0072B2"}
CATEGORIES = ["All", "Semi-Arid", "Sub-Humid", "Humid"]
CATEGORY_COLORS = {"All": "#333333", **CLASS_COLORS}

COLOR_A = "#F2B134"
COLOR_B = "#8B2635"
COLOR_AB = "#C1272D"
COLOR_AC = "#6A4C93"
COLOR_BC = "#E8785C"

MAP_SPEC = [
    ("A", "A: concurrent meteorological drivers"),
    ("B", "B: antecedent fuel-moisture drivers"),
    ("AB", "A+B"),
    ("AC", "A+C"),
    ("BC", "B+C"),
    ("ABC", "A+B+C (full model)"),
]


def label_panel(ax, letter, dx=-0.06, dy=1.32):
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE, fontweight="bold", va="bottom", ha="left")


def load_pooled(tag: str) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        f = ABLATION_DIR / tag / f"seed{seed}" / "pooled_by_tier_test.csv"
        df = pd.read_csv(f)
        df["seed"] = seed
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    return out.rename(columns={"group": "aridity_tier"})


def load_percell(tag: str) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        f = ABLATION_DIR / tag / f"seed{seed}" / "per_cell_metrics_test.csv"
        df = pd.read_csv(f)
        df["seed"] = seed
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def cellwise_mean(percell: pd.DataFrame, metric: str) -> pd.DataFrame:
    return percell.groupby("cell_id")[metric].mean().reset_index()


def pooled_r2_by_category(pooled: pd.DataFrame) -> dict:
    stat = pooled.groupby("aridity_tier")["r2"].mean()
    return {c: float(stat.get(c, np.nan)) for c in CATEGORIES}


# ------------------------------------------------------------- panel map ----
def panel_map(fig, gs_cell, kge_cell: pd.DataFrame, cell_meta: pd.DataFrame, states: gpd.GeoDataFrame, vmin, vmax, cmap, title, letter):
    ax = fig.add_subplot(gs_cell)

    df = cell_meta.merge(kge_cell, on="cell_id", how="right")
    piv = df.pivot(index="lat", columns="lon", values="kge").sort_index().sort_index(axis=1)
    lons, lats, grid = piv.columns.values.astype(float), piv.index.values.astype(float), piv.values
    step = np.median(np.diff(lons))
    lon_e = np.concatenate([lons - step / 2, [lons[-1] + step / 2]])
    lat_e = np.concatenate([lats - step / 2, [lats[-1] + step / 2]])

    pcm = ax.pcolormesh(lon_e, lat_e, grid, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat", rasterized=True)
    states.boundary.plot(ax=ax, color="#222222", linewidth=1.8, zorder=5)

    ax.set_xlim(lon_e.min(), lon_e.max())
    ax.set_ylim(lat_e.min(), lat_e.max())
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_aspect(1 / np.cos(np.radians(np.nanmean(lats))))
    ax.set_title(title, fontsize=BIG - 2, pad=12)
    label_panel(ax, letter)
    return pcm


# ------------------------------------------------------------- panel dot ----
def panel_dot(ax, base_vals: dict, combo_vals: dict, base_color, combo_color, letter, xlim):
    y = np.arange(len(CATEGORIES))[::-1]
    for yi, cat in zip(y, CATEGORIES):
        b, c = base_vals[cat], combo_vals[cat]
        ax.plot([b, c], [yi, yi], color="#999999", linewidth=2.2, zorder=1)
        ax.scatter(b, yi, s=340, color=base_color, edgecolor="black", linewidth=1.4, zorder=3)
        ax.scatter(c, yi, s=340, color=combo_color, edgecolor="black", linewidth=1.4, zorder=3)
        ax.annotate(f"+{c - b:.3f}", xy=(max(b, c), yi), xytext=(14, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=BIG - 7, fontweight="bold", color=combo_color)

    ax.set_yticks(y)
    ax.set_yticklabels(CATEGORIES, fontweight="bold")
    for tick, cat in zip(ax.get_yticklabels(), CATEGORIES):
        tick.set_color(CATEGORY_COLORS[cat])
    ax.set_xlim(xlim)
    ax.set_ylim(y.min() - 0.6, y.max() + 0.6)
    ax.set_xlabel(r"Test $R^2$")
    label_panel(ax, letter, dx=-0.20)


def main():
    percell = {tag: load_percell(tag) for tag, _ in MAP_SPEC}
    pooled = {tag: load_pooled(tag) for tag in ["A", "B", "AB", "AC", "BC"]}

    meta_src = pd.read_csv("results/aridity_and_percell_analysis/per_grid_cell_kge_components_mean_std_5seeds.csv")
    cell_meta = meta_src[["cell_id", "lon", "lat"]].drop_duplicates()
    states = gpd.read_file(SHAPEFILE)

    kge_cell = {tag: cellwise_mean(percell[tag], "kge") for tag, _ in MAP_SPEC}
    all_kge = np.concatenate([df["kge"].to_numpy() for df in kge_cell.values()])
    vmin, vmax = np.nanpercentile(all_kge, [1, 99])

    r2_by_cat = {tag: pooled_r2_by_category(pooled[tag]) for tag in pooled}

    fig = plt.figure(figsize=(28, 29), constrained_layout=True)
    gs = fig.add_gridspec(4, 3, height_ratios=[1, 1, 0.16, 0.85])

    letters = "abcdef"
    pcm_ref = None
    for i, (tag, title) in enumerate(MAP_SPEC):
        row, col = divmod(i, 3)
        pcm_ref = panel_map(fig, gs[row, col], kge_cell[tag], cell_meta, states, vmin, vmax, "viridis", title, letters[i])

    cax = fig.add_subplot(gs[2, :])
    cbar = plt.colorbar(pcm_ref, cax=cax, orientation="horizontal", extend="both")
    cbar.set_label("KGE", fontsize=BIG + 2)
    cbar.ax.tick_params(labelsize=BIG - 2, length=10, width=1.6)

    all_r2_vals = [v for d in r2_by_cat.values() for v in d.values() if np.isfinite(v)]
    xlim = (min(all_r2_vals) - 0.04, max(all_r2_vals) + 0.14)

    ax_g = fig.add_subplot(gs[3, 0])
    panel_dot(ax_g, r2_by_cat["A"], r2_by_cat["AB"], COLOR_A, COLOR_AB, "g", xlim)
    ax_g.set_title("Skill gain: B added to A", fontsize=BIG - 2)

    ax_h = fig.add_subplot(gs[3, 1])
    panel_dot(ax_h, r2_by_cat["A"], r2_by_cat["AC"], COLOR_A, COLOR_AC, "h", xlim)
    ax_h.set_title("Skill gain: C added to A", fontsize=BIG - 2)

    ax_i = fig.add_subplot(gs[3, 2])
    panel_dot(ax_i, r2_by_cat["B"], r2_by_cat["BC"], COLOR_B, COLOR_BC, "i", xlim)
    ax_i.set_title("Skill gain: C added to B", fontsize=BIG - 2)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=22, markeredgecolor="black",
                   markerfacecolor=color, label=label)
        for color, label in [
            (COLOR_A, "A: concurrent meteorological drivers"),
            (COLOR_B, "B: antecedent fuel-moisture drivers"),
            (COLOR_AB, "A+B"),
            (COLOR_AC, "A+C"),
            (COLOR_BC, "B+C"),
        ]
    ]
    fig.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.06),
               ncol=3, frameon=False, fontsize=BIG - 6, handletextpad=0.6, columnspacing=2.0)

    fig.savefig(OUT_DIR / "fig3_branch_ablation.png", dpi=300, bbox_inches="tight")
    print(f"Saved {OUT_DIR / 'fig3_branch_ablation.png'}")


if __name__ == "__main__":
    main()
