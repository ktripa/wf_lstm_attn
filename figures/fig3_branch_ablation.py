#!/usr/bin/env python
"""Figure 3: what information the branches carry (7-way branch ablation +
gridMET-precipitation control), pooled domain-wide model, 5 seeds.

Standalone: reads only results/ablation/<tag>/seed<n>/{pooled_by_tier_test.csv,
per_cell_metrics_test.csv} written by scripts/run_ablation.py, and writes
figures/fig3_branch_ablation.png (300 dpi). PNG only -- no PDF.

Five panels. The two things the paper claims -- that Branch B's skill gain
scales with humidity, and that most of that gain collapses once antecedent
precipitation is already available to Branch A -- get equal-sized panels
(b, c). Everything else is support:
  (a) Annotated matrix: 7 branch combinations x 3 aridity classes, cells
      shaded by test R2 with R2 (bold) and RMSE in FWI units (small, below)
      printed in every cell. Replaces 21-bar-group panels a+b.
  (b) Skill gain from adding Branch B to A, plotted against aridity index
      as a continuous x axis: per-cell deltas as a faint scatter, three
      class means connected by a line with a bootstrap 95% CI band. The
      monotonic-with-humidity claim as a line, not an inference exercise.
  (c) The control, as a dumbbell: filled circle = delta-R2 without the
      precip-history control, open circle = with it, arrow pointing down,
      percent-retained annotated. Same y-axis as (b).
  (d) Branch C's skill gain on (b)/(c)'s y-scale -- near-zero for Semi-Arid
      and Sub-Humid, but a real +0.10 R2 for Humid (not negligible there),
      with a small inset at native scale for the actual values.
  (e) Spatial map of delta-R2 from the CONTROLLED experiment (A+precip-hist
      -> A+B+precip-hist), percentile color limits, next to (d).
No twin axes anywhere: delta-RMSE (FWI units) is reported as printed
annotations alongside the delta-R2 values instead of a second axis.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fwi_attn.evaluation.bootstrap import bootstrap_ci

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


def label_panel(ax, letter, dx=-0.14, dy=1.10):
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE, fontweight="bold", va="top", ha="left")


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


def cell_deltas_pair(percell_from: pd.DataFrame, percell_to: pd.DataFrame) -> pd.DataFrame:
    """Per-cell, per-seed delta (to - from) for both r2 and rmse, averaged over
    seeds -> one row per cell, columns cell_id/aridity_tier/delta_r2/delta_rmse."""
    def _delta(metric):
        a = percell_from[["cell_id", "seed", "aridity_tier", metric]].rename(columns={metric: "from_val"})
        b = percell_to[["cell_id", "seed", metric]].rename(columns={metric: "to_val"})
        m = a.merge(b, on=["cell_id", "seed"])
        m["delta"] = m["to_val"] - m["from_val"]
        return m.groupby(["cell_id", "aridity_tier"])["delta"].mean().reset_index()

    r2 = _delta("r2").rename(columns={"delta": "delta_r2"})
    rmse = _delta("rmse").rename(columns={"delta": "delta_rmse"})
    return r2.merge(rmse[["cell_id", "delta_rmse"]], on="cell_id")


# ------------------------------------------------------------- panel (a) ----
def panel_a_matrix(ax, cax, pooled_by_combo, letter):
    r2_mat = np.zeros((len(COMBOS), len(CLASSES)))
    rmse_mat = np.zeros((len(COMBOS), len(CLASSES)))
    for i, tag in enumerate(COMBOS):
        df = pooled_by_combo[tag]
        df = df[df["aridity_tier"].isin(CLASSES)]
        stat = df.groupby("aridity_tier")[["r2", "rmse"]].mean().reindex(CLASSES)
        r2_mat[i] = stat["r2"].to_numpy()
        rmse_mat[i] = stat["rmse"].to_numpy()

    cmap = plt.get_cmap("viridis")
    im = ax.imshow(r2_mat, cmap=cmap, aspect="auto", vmin=r2_mat.min(), vmax=r2_mat.max())
    norm = im.norm
    for i in range(len(COMBOS)):
        for j in range(len(CLASSES)):
            rgba = cmap(norm(r2_mat[i, j]))
            lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
            tc = "white" if lum < 0.55 else "black"
            ax.text(j, i - 0.13, f"{r2_mat[i, j]:.2f}", ha="center", va="center",
                     fontsize=BIG - 2, fontweight="bold", color=tc)
            ax.text(j, i + 0.24, f"{rmse_mat[i, j]:.1f} FWI", ha="center", va="center",
                     fontsize=BIG - 11, color=tc)

    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, fontweight="bold")
    for tick, c in zip(ax.get_xticklabels(), CLASSES):
        tick.set_color(CLASS_COLORS[c])
    ax.set_yticks(range(len(COMBOS)))
    ax.set_yticklabels([COMBO_LABELS[t] for t in COMBOS])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(len(COMBOS) + 1):
        ax.axhline(i - 0.5, color="white", linewidth=2)
    for j in range(len(CLASSES) + 1):
        ax.axvline(j - 0.5, color="white", linewidth=2)
    ax.set_xlim(-0.5, len(CLASSES) - 0.5)
    ax.set_ylim(len(COMBOS) - 0.5, -0.5)

    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(r"Test $R^2$")
    cbar.ax.tick_params(labelsize=BIG - 6)
    ax.set_title("Model skill by combination × class", fontsize=BIG - 1, pad=10)
    label_panel(ax, letter)


# ------------------------------------------------------------- panel (b) ----
def panel_b_gradient(ax, delta_AB, cell_meta, letter):
    d = delta_AB.merge(cell_meta[["cell_id", "aridity_index"]], on="cell_id", how="left")
    d = d[d["aridity_tier"].isin(CLASSES)]

    for c in CLASSES:
        sub = d[d["aridity_tier"] == c]
        ax.scatter(sub["aridity_index"], sub["delta_r2"], s=10, color=CLASS_COLORS[c],
                   alpha=0.15, linewidth=0, zorder=1, rasterized=True)

    xs, ys, lo, hi, rmse_means, classes_sorted = [], [], [], [], [], []
    for c in CLASSES:
        sub = d[d["aridity_tier"] == c]
        ci = bootstrap_ci(sub["delta_r2"].to_numpy(), n_boot=N_BOOT, seed=0)
        xs.append(sub["aridity_index"].mean())
        ys.append(ci["mean"]); lo.append(ci["lower"]); hi.append(ci["upper"])
        rmse_means.append(sub["delta_rmse"].mean())
        classes_sorted.append(c)

    order = np.argsort(xs)
    xs_a, ys_a, lo_a, hi_a = (np.array(v)[order] for v in (xs, ys, lo, hi))
    classes_sorted = [classes_sorted[i] for i in order]
    rmse_sorted = [rmse_means[i] for i in order]

    ax.fill_between(xs_a, lo_a, hi_a, color="#444444", alpha=0.22, zorder=2, label="95% CI")
    ax.plot(xs_a, ys_a, color="#222222", linewidth=2.5, zorder=3)
    for x, y, c, rmse_v in zip(xs_a, ys_a, classes_sorted, rmse_sorted):
        ax.scatter(x, y, s=170, color=CLASS_COLORS[c], edgecolor="black", linewidth=1.4, zorder=4)
        ax.annotate(f"ΔRMSE {rmse_v:+.1f} FWI", xy=(x, y), xytext=(0, 14), textcoords="offset points",
                    ha="center", fontsize=BIG - 9, fontweight="bold", color=CLASS_COLORS[c])

    ax.axhline(0, color="black", linewidth=1, linestyle="--", zorder=0)
    ax.set_xlabel("Aridity index (P / PET)")
    ax.set_ylabel(r"$\Delta R^2$ from adding Branch B")
    ax.set_title("Skill gain scales with humidity", fontsize=BIG - 1)
    label_panel(ax, letter)
    return d["delta_r2"]


# ------------------------------------------------------------- panel (c) ----
def panel_c_control(ax, delta_no, delta_yes, letter, shared_ylim):
    x = np.arange(len(CLASSES))
    for i, c in enumerate(CLASSES):
        v_no = delta_no.loc[delta_no.aridity_tier == c, "delta_r2"].to_numpy()
        v_yes = delta_yes.loc[delta_yes.aridity_tier == c, "delta_r2"].to_numpy()
        rmse_no = delta_no.loc[delta_no.aridity_tier == c, "delta_rmse"].mean()
        rmse_yes = delta_yes.loc[delta_yes.aridity_tier == c, "delta_rmse"].mean()
        ci_no = bootstrap_ci(v_no, n_boot=N_BOOT, seed=0)
        ci_yes = bootstrap_ci(v_yes, n_boot=N_BOOT, seed=0)
        pct = 100 * ci_yes["mean"] / ci_no["mean"]

        ax.annotate(
            "", xy=(i, ci_yes["mean"]), xytext=(i, ci_no["mean"]),
            arrowprops=dict(arrowstyle="-|>", color=CLASS_COLORS[c], lw=2.5, mutation_scale=22), zorder=2,
        )
        ax.errorbar(i, ci_no["mean"], yerr=[[ci_no["mean"] - ci_no["lower"]], [ci_no["upper"] - ci_no["mean"]]],
                     color=CLASS_COLORS[c], capsize=4, linewidth=1.3, zorder=1)
        ax.errorbar(i, ci_yes["mean"], yerr=[[ci_yes["mean"] - ci_yes["lower"]], [ci_yes["upper"] - ci_yes["mean"]]],
                     color=CLASS_COLORS[c], capsize=4, linewidth=1.3, zorder=1)
        ax.scatter(i, ci_no["mean"], s=190, color=CLASS_COLORS[c], edgecolor="black", linewidth=1.4, zorder=3,
                   label="without precip. hist. in A" if i == 0 else None)
        ax.scatter(i, ci_yes["mean"], s=190, facecolor="white", edgecolor=CLASS_COLORS[c], linewidth=2.4, zorder=3,
                   label="with 60-wk precip. hist. in A" if i == 0 else None)

        ax.annotate(f"{pct:.0f}% retained", xy=(i + 0.12, (ci_no["mean"] + ci_yes["mean"]) / 2),
                    fontsize=BIG - 7, fontweight="bold", color=CLASS_COLORS[c], va="center", ha="left")
        ax.text(i - 0.16, ci_no["mean"], f"{rmse_no:+.1f}", fontsize=BIG - 10, color=CLASS_COLORS[c],
                va="center", ha="right")
        ax.text(i - 0.16, ci_yes["mean"], f"{rmse_yes:+.1f}", fontsize=BIG - 10, color=CLASS_COLORS[c],
                va="center", ha="right")

    ax.axhline(0, color="black", linewidth=1, linestyle="--", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontweight="bold")
    for tick, c in zip(ax.get_xticklabels(), CLASSES):
        tick.set_color(CLASS_COLORS[c])
    ax.set_xlim(-0.6, len(CLASSES) - 0.4)
    ax.set_ylim(shared_ylim)
    ax.set_ylabel(r"$\Delta R^2$ from adding Branch B")
    ax.legend(frameon=False, loc="upper right", fontsize=BIG - 9)
    ax.set_title("Most of the gain is antecedent precipitation", fontsize=BIG - 1)
    label_panel(ax, letter)


# ------------------------------------------------------------- panel (d) ----
def panel_d_branchC(ax, delta_AC, letter, shared_ylim):
    x = np.arange(len(CLASSES))
    means, los, his = [], [], []
    for i, c in enumerate(CLASSES):
        v = delta_AC.loc[delta_AC.aridity_tier == c, "delta_r2"].to_numpy()
        ci = bootstrap_ci(v, n_boot=N_BOOT, seed=0)
        means.append(ci["mean"]); los.append(ci["lower"]); his.append(ci["upper"])
        ax.errorbar(x[i], ci["mean"], yerr=[[ci["mean"] - ci["lower"]], [ci["upper"] - ci["mean"]]],
                    fmt="o", color=CLASS_COLORS[c], markersize=10, capsize=5, linewidth=2)

    ax.axhline(0, color="black", linewidth=1, linestyle="--", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(CLASSES, fontweight="bold", fontsize=BIG - 7, rotation=20, ha="right")
    for tick, c in zip(ax.get_xticklabels(), CLASSES):
        tick.set_color(CLASS_COLORS[c])
    ax.set_xlim(-0.6, len(CLASSES) - 0.4)
    ax.set_ylim(shared_ylim)
    ax.set_ylabel(r"$\Delta R^2$ (C)", fontsize=BIG - 4)
    ax.set_title("Branch C: small,\nclass-dependent gain", fontsize=BIG - 3, pad=14)
    label_panel(ax, letter, dx=-0.30, dy=1.28)

    ax_inset = ax.inset_axes([0.12, 0.46, 0.8, 0.30])
    for i, c in enumerate(CLASSES):
        ax_inset.errorbar(i, means[i], yerr=[[means[i] - los[i]], [his[i] - means[i]]],
                           fmt="o", color=CLASS_COLORS[c], markersize=6, capsize=3, linewidth=1.3)
    ax_inset.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax_inset.set_xticks([])
    ax_inset.set_xlim(-0.6, len(CLASSES) - 0.4)
    ax_inset.tick_params(labelsize=BIG - 14)
    ax_inset.set_title("native scale", fontsize=BIG - 13, pad=3)


# ------------------------------------------------------------- panel (e) ----
def panel_e_map(ax, delta_cell: pd.DataFrame, cell_meta: pd.DataFrame, letter: str):
    df = cell_meta.merge(delta_cell[["cell_id", "delta_r2"]], on="cell_id", how="right")
    piv = df.pivot(index="lat", columns="lon", values="delta_r2").sort_index().sort_index(axis=1)
    lons, lats, grid = piv.columns.values.astype(float), piv.index.values.astype(float), piv.values
    step = np.median(np.diff(lons))
    lon_e = np.concatenate([lons - step / 2, [lons[-1] + step / 2]])
    lat_e = np.concatenate([lats - step / 2, [lats[-1] + step / 2]])

    vmax = np.nanpercentile(np.abs(grid), 99)
    pcm = ax.pcolormesh(lon_e, lat_e, grid, cmap="RdBu", vmin=-vmax, vmax=vmax, shading="flat", rasterized=True)
    cbar = plt.colorbar(pcm, ax=ax, orientation="horizontal", location="bottom", fraction=0.046, pad=0.16, extend="both")
    cbar.set_label(r"$\Delta R^2$ (A+precip $\rightarrow$ A+B+precip)")
    cbar.ax.tick_params(labelsize=BIG - 6)

    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_aspect(1 / np.cos(np.radians(np.nanmean(lats))))
    ax.set_title("Controlled skill gain: where B still helps", fontsize=BIG - 3)
    label_panel(ax, letter, dx=-0.20)


def main():
    pooled_by_combo = {tag: load_pooled(tag) for tag in COMBOS}
    percell_by_combo = {tag: load_percell(tag) for tag in COMBOS}
    percell_ctrl = {
        "A_precipctrl": load_percell("A_precipctrl"),
        "AB_precipctrl": load_percell("AB_precipctrl"),
    }

    meta_src = pd.read_csv("results/aridity_and_percell_analysis/per_grid_cell_kge_components_mean_std_5seeds.csv")
    cell_meta = meta_src[["cell_id", "lon", "lat", "aridity_index"]].drop_duplicates()

    delta_AB = cell_deltas_pair(percell_by_combo["A"], percell_by_combo["AB"])
    delta_AC = cell_deltas_pair(percell_by_combo["A"], percell_by_combo["AC"])
    delta_ctrl_no = delta_AB
    delta_ctrl_yes = cell_deltas_pair(percell_ctrl["A_precipctrl"], percell_ctrl["AB_precipctrl"])

    fig = plt.figure(figsize=(23, 16), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.15], height_ratios=[1.15, 1])

    gs_a = gs[0, 0].subgridspec(1, 2, width_ratios=[20, 1], wspace=0.08)
    ax_a = fig.add_subplot(gs_a[0])
    cax_a = fig.add_subplot(gs_a[1])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    gs_de = gs[1, 1].subgridspec(1, 2, width_ratios=[1, 2.3], wspace=0.35)
    ax_d = fig.add_subplot(gs_de[0])
    ax_e = fig.add_subplot(gs_de[1])

    panel_a_matrix(ax_a, cax_a, pooled_by_combo, "a")
    per_cell_all_deltas = panel_b_gradient(ax_b, delta_AB, cell_meta, "b")

    ylim_lo = min(per_cell_all_deltas.quantile(0.01), delta_AC["delta_r2"].quantile(0.01), 0) - 0.02
    ylim_hi = max(per_cell_all_deltas.quantile(0.99), delta_ctrl_no["delta_r2"].quantile(0.99)) + 0.03
    shared_ylim = (ylim_lo, ylim_hi)
    ax_b.set_ylim(shared_ylim)

    panel_c_control(ax_c, delta_ctrl_no, delta_ctrl_yes, "c", shared_ylim)
    panel_d_branchC(ax_d, delta_AC, "d", shared_ylim)
    panel_e_map(ax_e, delta_ctrl_yes, cell_meta, "e")

    fig.savefig(OUT_DIR / "fig3_branch_ablation.png", dpi=300, bbox_inches="tight")
    print(f"Saved {OUT_DIR / 'fig3_branch_ablation.png'}")


if __name__ == "__main__":
    main()
