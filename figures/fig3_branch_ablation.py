#!/usr/bin/env python
"""Figure 3: what information the branches carry (7-way branch ablation +
gridMET-precipitation control), pooled domain-wide model, 5 seeds.

Standalone: reads only results/ablation/<tag>/seed<n>/{pooled_by_tier_test.csv,
per_cell_metrics_test.csv} written by scripts/run_ablation.py, and writes
figures/fig3_branch_ablation.png (300 dpi). PNG only this round -- no PDF.

Six panels, all restricted to the three aridity classes used throughout
(Semi-Arid / Sub-Humid / Humid), same fixed palette as Figure 2:
  (a) Test R2 for all 7 branch combinations, by class, seed error bars.
  (b) Same, test RMSE (FWI units) -- required alongside (a): R2 alone
      overstates the Humid result because Humid cells have the lowest
      observed FWI variance.
  (c) Skill gain from adding Branch B to Branch A: delta-R2 and delta-RMSE
      (twin axes), by class, bootstrapped 95% CI over grid cells (10,000
      resamples). Carries the paper's main claim.
  (d) Same, Branch C added to Branch A.
  (e) Spatial map of delta-R2 (A -> A+B), diverging colormap centered at 0.
  (f) Control: delta-R2 from adding Branch B, with vs. without a 60-week
      gridMET antecedent-precipitation history already in Branch A.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fwi_attn.evaluation.bootstrap import bootstrap_ci, bootstrap_paired_diff_ci

# ---------------------------------------------------------------- style ----
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
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.linewidth": 1.2,
    }
)
PANEL_LABEL_SIZE = BIG + 6

ABLATION_DIR = Path("results/ablation")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)
SEEDS = [0, 1, 2, 3, 4]
N_BOOT = 10000

CLASSES = ["Semi-Arid", "Sub-Humid", "Humid"]
CLASS_COLORS = {"Semi-Arid": "#E69F00", "Sub-Humid": "#009E73", "Humid": "#0072B2"}

COMBOS = ["A", "B", "C", "AB", "AC", "BC", "ABC"]
COMBO_LABELS = {"A": "A", "B": "B", "C": "C", "AB": "A+B", "AC": "A+C", "BC": "B+C", "ABC": "A+B+C"}
_combo_cmap = plt.get_cmap("Blues")
COMBO_COLORS = {tag: _combo_cmap(0.30 + 0.62 * i / (len(COMBOS) - 1)) for i, tag in enumerate(COMBOS)}


def label_panel(ax, letter, dx=-0.16, dy=1.10):
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE, fontweight="bold", va="top", ha="left")


def load_pooled(tag: str) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        f = ABLATION_DIR / tag / f"seed{seed}" / "pooled_by_tier_test.csv"
        df = pd.read_csv(f)
        df["seed"] = seed
        rows.append(df)
    out = pd.concat(rows, ignore_index=True)
    out = out.rename(columns={"group": "aridity_tier"})
    return out


def load_percell(tag: str) -> pd.DataFrame:
    rows = []
    for seed in SEEDS:
        f = ABLATION_DIR / tag / f"seed{seed}" / "per_cell_metrics_test.csv"
        df = pd.read_csv(f)
        df["seed"] = seed
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def panel_bar(ax, pooled_by_combo: dict[str, pd.DataFrame], metric: str, ylabel: str, letter: str):
    spacing = 1.9
    n_bars = len(COMBOS)
    width = 0.11
    x = np.arange(len(CLASSES)) * spacing
    for i, tag in enumerate(COMBOS):
        df = pooled_by_combo[tag]
        df = df[df["aridity_tier"].isin(CLASSES)]
        stat = df.groupby("aridity_tier")[metric].agg(["mean", "std"]).reindex(CLASSES)
        offset = (i - (n_bars - 1) / 2) * width
        ax.bar(
            x + offset, stat["mean"], width, yerr=stat["std"], capsize=2,
            color=COMBO_COLORS[tag], edgecolor="black", linewidth=0.5,
            label=COMBO_LABELS[tag], error_kw={"linewidth": 0.9},
        )
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontweight="bold")
    for tick, c in zip(ax.get_xticklabels(), CLASSES):
        tick.set_color(CLASS_COLORS[c])
    ax.set_xlim(x[0] - 1.0, x[-1] + 1.0)
    ax.set_ylabel(ylabel)
    if metric == "r2":
        ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.20), fontsize=BIG - 8)
    label_panel(ax, letter)


def cell_deltas(percell_from: pd.DataFrame, percell_to: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per-cell, per-seed delta (percell_to - percell_from), then averaged over seeds -> one row per cell."""
    a = percell_from[["cell_id", "seed", "aridity_tier", metric]].rename(columns={metric: "from_val"})
    b = percell_to[["cell_id", "seed", metric]].rename(columns={metric: "to_val"})
    merged = a.merge(b, on=["cell_id", "seed"])
    merged["delta"] = merged["to_val"] - merged["from_val"]
    return merged.groupby(["cell_id", "aridity_tier"])["delta"].mean().reset_index()


def panel_delta(ax, delta_r2: pd.DataFrame, delta_rmse: pd.DataFrame, letter: str, title: str):
    ax2 = ax.twinx()
    ax2.spines["top"].set_visible(False)
    x = np.arange(len(CLASSES))
    width = 0.32

    for i, c in enumerate(CLASSES):
        r2_vals = delta_r2.loc[delta_r2["aridity_tier"] == c, "delta"].to_numpy()
        # Plotted as RMSE *reduction* (-delta), so "up" means "more improvement"
        # on both axes -- a raw deltaRMSE (negative = better) would point the
        # opposite way from deltaR2 (positive = better) and read confusingly.
        rmse_reduction_vals = -delta_rmse.loc[delta_rmse["aridity_tier"] == c, "delta"].to_numpy()
        ci_r2 = bootstrap_ci(r2_vals, n_boot=N_BOOT, seed=0)
        ci_rmse = bootstrap_ci(rmse_reduction_vals, n_boot=N_BOOT, seed=0)

        ax.bar(
            i - width / 2, ci_r2["mean"], width,
            yerr=[[ci_r2["mean"] - ci_r2["lower"]], [ci_r2["upper"] - ci_r2["mean"]]],
            capsize=4, color=CLASS_COLORS[c], edgecolor="black", linewidth=0.7,
        )
        ax2.bar(
            i + width / 2, ci_rmse["mean"], width,
            yerr=[[ci_rmse["mean"] - ci_rmse["lower"]], [ci_rmse["upper"] - ci_rmse["mean"]]],
            capsize=4, color=CLASS_COLORS[c], edgecolor="black", linewidth=0.7, hatch="//", alpha=0.75,
        )

    ax.axhline(0, color="black", linewidth=1, linestyle="--", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontweight="bold")
    for tick, c in zip(ax.get_xticklabels(), CLASSES):
        tick.set_color(CLASS_COLORS[c])
    ax.set_ylabel(r"$\Delta R^2$ (solid)")
    ax2.set_ylabel("RMSE reduction, FWI units (hatched)")
    ax.set_title(title, fontsize=BIG - 1)
    label_panel(ax, letter)


def panel_map(ax, delta_cell: pd.DataFrame, cell_meta: pd.DataFrame, letter: str):
    df = cell_meta.merge(delta_cell[["cell_id", "delta"]], on="cell_id", how="right")
    piv = df.pivot(index="lat", columns="lon", values="delta").sort_index().sort_index(axis=1)
    lons, lats, grid = piv.columns.values.astype(float), piv.index.values.astype(float), piv.values
    step = np.median(np.diff(lons))
    lon_e = np.concatenate([lons - step / 2, [lons[-1] + step / 2]])
    lat_e = np.concatenate([lats - step / 2, [lats[-1] + step / 2]])

    # Robust (percentile-based) color limits: 2 of 1985 cells have pathological
    # deltas (R^2 is unstable when a cell's test-period FWI variance is tiny)
    # that would otherwise wash out the other 1983 cells to near-white. Those
    # cells are not hidden -- pcolormesh clips them to the saturated end color
    # -- only the color *scale* is made robust; no data is excluded.
    vmax = np.nanpercentile(np.abs(grid), 99)
    pcm = ax.pcolormesh(lon_e, lat_e, grid, cmap="RdBu", vmin=-vmax, vmax=vmax, shading="flat", rasterized=True)
    cbar = plt.colorbar(pcm, ax=ax, orientation="horizontal", location="bottom", fraction=0.046, pad=0.16, extend="both")
    cbar.set_label(r"$\Delta R^2$ (A+B $-$ A)")
    cbar.ax.tick_params(labelsize=BIG - 5)

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_aspect(1 / np.cos(np.radians(np.nanmean(lats))))
    label_panel(ax, letter)


def panel_control(ax, percell: dict[str, pd.DataFrame], letter: str):
    delta_noctrl = cell_deltas(percell["A"], percell["AB"], "r2")
    delta_ctrl = cell_deltas(percell["A_precipctrl"], percell["AB_precipctrl"], "r2")

    x = np.arange(len(CLASSES))
    width = 0.32
    for i, c in enumerate(CLASSES):
        v_no = delta_noctrl.loc[delta_noctrl["aridity_tier"] == c, "delta"].to_numpy()
        v_yes = delta_ctrl.loc[delta_ctrl["aridity_tier"] == c, "delta"].to_numpy()
        ci_no = bootstrap_ci(v_no, n_boot=N_BOOT, seed=0)
        ci_yes = bootstrap_ci(v_yes, n_boot=N_BOOT, seed=0)
        ax.bar(
            i - width / 2, ci_no["mean"], width,
            yerr=[[ci_no["mean"] - ci_no["lower"]], [ci_no["upper"] - ci_no["mean"]]],
            capsize=4, color=CLASS_COLORS[c], edgecolor="black", linewidth=0.7, label="without precip hist." if i == 0 else None,
        )
        ax.bar(
            i + width / 2, ci_yes["mean"], width,
            yerr=[[ci_yes["mean"] - ci_yes["lower"]], [ci_yes["upper"] - ci_yes["mean"]]],
            capsize=4, color=CLASS_COLORS[c], edgecolor="black", linewidth=0.7, hatch="//", alpha=0.75,
            label="with 60-wk precip hist. in A" if i == 0 else None,
        )

    ax.axhline(0, color="black", linewidth=1, linestyle="--", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontweight="bold")
    for tick, c in zip(ax.get_xticklabels(), CLASSES):
        tick.set_color(CLASS_COLORS[c])
    ax.set_ylabel(r"$\Delta R^2$ from adding Branch B")
    ax.legend(frameon=False, loc="upper right", fontsize=BIG - 8)
    label_panel(ax, letter)


def main():
    pooled_by_combo = {tag: load_pooled(tag) for tag in COMBOS}
    percell_by_combo = {tag: load_percell(tag) for tag in COMBOS}
    percell_ctrl = {
        "A_precipctrl": load_percell("A_precipctrl"),
        "AB_precipctrl": load_percell("AB_precipctrl"),
    }
    percell_for_control = {**percell_by_combo, **percell_ctrl}

    cell_meta = (
        percell_by_combo["A"][["cell_id"]].drop_duplicates()
    )
    # lon/lat aren't in per_cell_metrics_test.csv -- pull from the aridity/percell analysis output.
    meta_src = pd.read_csv("results/aridity_and_percell_analysis/per_grid_cell_kge_components_mean_std_5seeds.csv")
    cell_meta = cell_meta.merge(meta_src[["cell_id", "lon", "lat"]], on="cell_id", how="left")

    delta_r2_AB = cell_deltas(percell_by_combo["A"], percell_by_combo["AB"], "r2")
    delta_rmse_AB = cell_deltas(percell_by_combo["A"], percell_by_combo["AB"], "rmse")
    delta_r2_AC = cell_deltas(percell_by_combo["A"], percell_by_combo["AC"], "r2")
    delta_rmse_AC = cell_deltas(percell_by_combo["A"], percell_by_combo["AC"], "rmse")

    fig = plt.figure(figsize=(22, 14), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    gs_e = gs[0, 2].subgridspec(2, 1, height_ratios=[18, 1], hspace=0.15)
    ax_e = fig.add_subplot(gs_e[0])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    panel_bar(ax_a, pooled_by_combo, "r2", r"Test $R^2$", "a")
    panel_bar(ax_b, pooled_by_combo, "rmse", "Test RMSE (FWI units)", "b")
    panel_map(ax_e, delta_r2_AB, cell_meta, "e")
    panel_delta(ax_c, delta_r2_AB, delta_rmse_AB, "c", "Skill gain: Branch B added to A")
    panel_delta(ax_d, delta_r2_AC, delta_rmse_AC, "d", "Skill gain: Branch C added to A")
    panel_control(ax_f, percell_for_control, "f")

    fig.savefig(OUT_DIR / "fig3_branch_ablation.png", dpi=300, bbox_inches="tight")
    print(f"Saved {OUT_DIR / 'fig3_branch_ablation.png'}")


if __name__ == "__main__":
    main()
