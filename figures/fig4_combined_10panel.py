#!/usr/bin/env python
"""Combined figure: Figure 4 (unsigned, panels a-f) + Branch C static
attribution (panels g-j), 10 panels in one 2x5 grid, alphabetical reading
order. Pooled model, joint Expected Gradients, FWI units throughout.

  (a-c) Branch A (concurrent weather), one panel per aridity class:
        horizontal bars, mean |attribution|, 95% CI, shared x-axis.
  (d-f) Branch B (antecedent SM/NDVI/VOD), one panel per class: mean
        attribution by lag (t-1..t-12), shaded 95% CI, shared y-axis,
        lag axis reversed (1 = most recent, on the right).
  (g)   Branch C land cover: violin plots, POOLED across the whole region
        (not split by aridity class) -- NLCD categories shown if they cover
        >=5% of all cells region-wide; green gradient, dark = most forested
        (Deciduous) to light = most open (Pasture/hay).
  (h)   Branch C mean annual precipitation (mm): pooled scatter + one
        binned-median trend line + 95% CI band only (no IQR).
  (i)   Branch C elevation (m): same treatment as (h).
  (j)   Branch C grouped attribution: Topography / Climatology / Land cover,
        box plots, split by aridity class.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

for _f in [
    "/usr/share/fonts/urw-base35/NimbusRoman-Regular.otf",
    "/usr/share/fonts/urw-base35/NimbusRoman-Bold.otf",
    "/usr/share/fonts/urw-base35/NimbusRoman-Italic.otf",
    "/usr/share/fonts/urw-base35/NimbusRoman-BoldItalic.otf",
]:
    if Path(_f).exists():
        fm.fontManager.addfont(_f)

BIG = 40  # g-j chain: 24 -> 32 -> 40 ("+8 points" twice); applied figure-wide for consistency
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
        "font.size": BIG,
        "axes.titlesize": BIG + 1,
        "axes.labelsize": BIG,
        "xtick.labelsize": BIG - 6,
        "ytick.labelsize": BIG - 6,
        "legend.fontsize": BIG - 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.linewidth": 1.3,
    }
)
PANEL_LABEL_SIZE = BIG + 6

DIR_AB = Path("results/eg_attribution")
DIR_C = Path("results/eg_attribution_branchC")
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

NLCD_NAMES = {41: "Deciduous\nforest", 42: "Evergreen\nforest", 43: "Mixed\nforest", 52: "Shrub/\nscrub", 71: "Grassland/\nherb.", 81: "Pasture/\nhay"}
NLCD_COL = {c: f"eg_fwi_veg_onehot_nlcd{c}" for c in NLCD_NAMES}
# dark (most forested) -> light (most open); Mixed forest excluded (n/a region-wide anyway)
NLCD_GREEN_ORDER = [41, 42, 52, 71, 81]
PRECIP_COLOR = "#2166AC"
ELEV_COLOR = "#B2622D"

N_BOOT = 2000
N_BINS = 8
COVERAGE_MIN = 0.05


def label_panel(ax, letter, dx=-0.20, dy=1.12):
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE, fontweight="bold", va="top", ha="left")


def mean_ci95(x: np.ndarray):
    x = x[np.isfinite(x)]
    m, sem = x.mean(), x.std(ddof=1) / np.sqrt(len(x))
    return m, 1.96 * sem


# ---------------------------------------------------------------- a-c ----
def panel_abc(ax, df, cls, letter, xlim):
    # Unsigned = mirror the signed mean to positive (multiply by -1 when
    # negative), i.e. |mean(x)|, NOT mean(|x_i|). CI half-width is unaffected
    # by a sign flip and carries over unchanged.
    sub = df[df["aridity_tier"] == cls]
    rows = []
    for v in BRANCH_A_FEATURES:
        m, ci = mean_ci95(sub[f"eg_fwi_{v}"].to_numpy())
        rows.append((v, abs(m), ci))
    rows.sort(key=lambda r: -r[1])
    names = [A_LABELS[r[0]] for r in rows]
    means = [r[1] for r in rows]
    cis = [r[2] for r in rows]

    y = np.arange(len(names))[::-1]
    ax.barh(y, means, xerr=cis, color=CLASS_COLORS[cls], edgecolor="black", linewidth=0.8, capsize=4, error_kw={"linewidth": 1.4})
    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlim(*xlim)
    ax.set_xlabel("|Mean attribution| (FWI)")
    ax.set_title(cls, color=CLASS_COLORS[cls], fontweight="bold", fontsize=BIG)
    label_panel(ax, letter)


# ---------------------------------------------------------------- d-f ----
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
    ax.set_xlim(LOOKBACK_SHOWN + 0.5, 0.5)
    ax.set_xlabel("Lag (weeks before present)")
    ax.set_ylabel("Mean attribution (FWI)")
    ax.set_ylim(*ylim)
    ax.set_title(cls, color=CLASS_COLORS[cls], fontweight="bold", fontsize=BIG)
    ax.legend(frameon=False, loc="upper left")
    label_panel(ax, letter)


# ------------------------------------------------------------------- g ----
def panel_g_violin(ax, df):
    print("\n=== Panel (g): NLCD coverage fractions, REGION-WIDE (pooled) ===")
    seed0 = df[df.seed == 0]
    n = len(seed0)
    vc = seed0["dominant_nlcd"].value_counts()
    kept = []
    for code in NLCD_GREEN_ORDER:
        cnt = int(vc.get(code, 0))
        frac = cnt / n
        keep = frac >= COVERAGE_MIN
        print(f"  {NLCD_NAMES[code].replace(chr(10), ' '):18s} n={cnt:4d} ({frac*100:5.1f}%)  {'KEEP' if keep else 'DROP'}")
        if keep:
            kept.append(code)

    cmap = plt.get_cmap("Greens")
    shades = [cmap(v) for v in np.linspace(0.85, 0.35, len(kept))]

    rows = []
    for code in kept:
        cell_ids = seed0.query("dominant_nlcd == @code")["cell_id"].unique()
        cell_vals = df[df.cell_id.isin(cell_ids)].groupby("cell_id")[NLCD_COL[code]].mean()
        for v in cell_vals:
            rows.append({"nlcd": NLCD_NAMES[code], "value": v})
    long = pd.DataFrame(rows)
    order = [NLCD_NAMES[c] for c in kept]

    sns.violinplot(data=long, x="nlcd", y="value", order=order, palette=shades, ax=ax, cut=0, inner="quart", linewidth=1.4, saturation=1.0)
    ax.axhline(0, color="black", linewidth=1.2, zorder=0)
    ax.set_xlabel("")
    ax.set_ylabel("Attribution (FWI)")
    ax.set_title(f"Land cover (≥{int(COVERAGE_MIN*100)}% coverage)", fontsize=BIG - 3)
    label_panel(ax, "g")


# --------------------------------------------------------------- h, i ----
def hierarchical_bin_trend_pooled(df_cells: pd.DataFrame, xcol: str, ycol: str, rng):
    piv = df_cells.pivot(index="cell_id", columns="seed", values=ycol)
    x = df_cells.groupby("cell_id")[xcol].first().reindex(piv.index).to_numpy()
    y_mean = piv.mean(axis=1).to_numpy()
    order = np.argsort(x)
    x, y_mean, piv_vals = x[order], y_mean[order], piv.to_numpy()[order]

    bin_edges = np.quantile(x, np.linspace(0, 1, N_BINS + 1))
    bin_edges[-1] += 1e-9
    bin_idx = np.digitize(x, bin_edges[1:-1])

    rows = []
    for b in range(N_BINS):
        mask = bin_idx == b
        if mask.sum() < 5:
            continue
        center = x[mask].mean()
        med = np.median(y_mean[mask])
        piv_bin = piv_vals[mask]
        n_in_bin = mask.sum()
        boot_meds = np.empty(N_BOOT)
        for r in range(N_BOOT):
            seed_pick = rng.choice(piv_bin.shape[1], size=piv_bin.shape[1], replace=True)
            y_resampled = piv_bin[:, seed_pick].mean(axis=1)
            cell_pick = rng.integers(0, n_in_bin, size=n_in_bin)
            boot_meds[r] = np.median(y_resampled[cell_pick])
        ci_lo, ci_hi = np.percentile(boot_meds, [2.5, 97.5])
        rows.append({"center": center, "median": med, "ci_lo": ci_lo, "ci_hi": ci_hi, "n": int(n_in_bin)})
    return pd.DataFrame(rows)


def panel_pooled_trend(fig, gs_cell, df, xcol, title, letter, color, rng):
    gs_sub = gs_cell.subgridspec(2, 1, height_ratios=[1, 4], hspace=0.06)
    ax_hist = fig.add_subplot(gs_sub[0])
    ax = fig.add_subplot(gs_sub[1], sharex=ax_hist)

    xvals_all = df.groupby("cell_id")[xcol].first().to_numpy()
    ax_hist.hist(xvals_all, bins=30, color=color, alpha=0.55, edgecolor="none")
    ax_hist.set_yticks([])
    ax_hist.spines["left"].set_visible(False)
    plt.setp(ax_hist.get_xticklabels(), visible=False)
    ax_hist.set_title(title, fontsize=BIG, pad=10)

    ycol = f"eg_fwi_{xcol}"
    sub = df[["cell_id", "seed", xcol, ycol]]
    x_c = sub.groupby("cell_id")[xcol].first().to_numpy()
    y_c = sub.groupby("cell_id")[ycol].mean().to_numpy()
    ax.scatter(x_c, y_c, s=12, color=color, alpha=0.35, linewidth=0)

    trend = hierarchical_bin_trend_pooled(sub, xcol, ycol, rng)
    ax.plot(trend["center"], trend["median"], color=color, linewidth=3.5, marker="o", markersize=8, zorder=5)
    ax.fill_between(trend["center"], trend["ci_lo"], trend["ci_hi"], color=color, alpha=0.30, linewidth=0, zorder=3, label="95% CI")

    ax.axhline(0, color="black", linewidth=1.2, linestyle="--", zorder=1)
    ax.set_xlabel(title)
    ax.set_ylabel("Attribution (FWI)")
    ax.set_ylim(-1, 1)
    ax.legend(frameon=False, loc="best", fontsize=BIG - 9)
    label_panel(ax_hist, letter, dy=1.35)


# ------------------------------------------------------------------- j ----
def panel_j_boxplots(ax, df):
    rows = []
    for cls in CLASSES:
        sub = df[df.aridity_tier == cls].copy()
        sub["Topography"] = sub["eg_fwi_elevation"]
        sub["Climatology"] = sub["eg_fwi_map_mean_annual_precip"] + sub["eg_fwi_mavpd_mean_annual_vpd"]
        sub["Land cover"] = sub[list(NLCD_COL.values())].sum(axis=1)
        percell = sub.groupby("cell_id")[["Topography", "Climatology", "Land cover"]].mean()
        for group_name in ["Topography", "Climatology", "Land cover"]:
            for v in percell[group_name]:
                rows.append({"group": group_name, "aridity_tier": cls, "value": v})
    long = pd.DataFrame(rows)

    sns.boxplot(
        data=long, x="group", y="value", hue="aridity_tier", order=["Topography", "Climatology", "Land cover"],
        hue_order=CLASSES, palette=CLASS_COLORS, ax=ax, linewidth=1.3, fliersize=3, saturation=0.95,
    )
    ax.axhline(0, color="black", linewidth=1.2, zorder=0)
    ax.set_xlabel("")
    ax.set_ylabel("Summed attribution (FWI)")
    ax.set_ylim(-1, 1)
    ax.set_title("Grouped: topography / climatology / land cover", fontsize=BIG - 4)
    ax.legend(frameon=False, loc="best", fontsize=BIG - 9, title=None)
    label_panel(ax, "j")


def main():
    df_ab = pd.read_csv(DIR_AB / "eg_percell_wide.csv")
    df_c = pd.read_csv(DIR_C / "eg_branchC_percell_perseed.csv")
    rng = np.random.default_rng(0)

    # 4 rows total: Figure 4's own 2x3 block (a-f) stacked directly above the
    # Branch-C 2x2 block (g-j). Each block gets its own subfigure (independent
    # constrained-layout region) rather than sharing one GridSpec -- mixing a
    # plain axes (g) with nested histogram+scatter axes (h, i) in one shared
    # grid made constrained_layout misjudge row heights and overlap labels
    # across rows.
    fig = plt.figure(figsize=(24, 34), constrained_layout=True)
    subfig_top, subfig_bottom = fig.subfigures(2, 1, height_ratios=[1, 1.15])
    gs_top = subfig_top.add_gridspec(2, 3)
    gs_bottom = subfig_bottom.add_gridspec(2, 2)

    max_abc = 0.0
    for cls in CLASSES:
        sub = df_ab[df_ab["aridity_tier"] == cls]
        for v in BRANCH_A_FEATURES:
            m, ci = mean_ci95(sub[f"eg_fwi_{v}"].to_numpy())
            max_abc = max(max_abc, abs(m) + ci)
    xlim = (0, max_abc * 1.08)

    panel_abc(subfig_top.add_subplot(gs_top[0, 0]), df_ab, "Semi-Arid", "a", xlim)
    panel_abc(subfig_top.add_subplot(gs_top[0, 1]), df_ab, "Sub-Humid", "b", xlim)
    panel_abc(subfig_top.add_subplot(gs_top[0, 2]), df_ab, "Humid", "c", xlim)

    all_bounds = []
    for cls in CLASSES:
        sub = df_ab[df_ab["aridity_tier"] == cls]
        for v in BRANCH_B_FEATURES:
            for lag in range(1, LOOKBACK_SHOWN + 1):
                m, ci = mean_ci95(sub[f"eg_fwi_lag{lag}_{v}"].to_numpy())
                all_bounds += [m - ci, m + ci]
    pad = 0.08 * (max(all_bounds) - min(all_bounds))
    ylim = (min(all_bounds) - pad, max(all_bounds) + pad)

    panel_def(subfig_top.add_subplot(gs_top[1, 0]), df_ab, "Semi-Arid", "d", ylim)
    panel_def(subfig_top.add_subplot(gs_top[1, 1]), df_ab, "Sub-Humid", "e", ylim)
    panel_def(subfig_top.add_subplot(gs_top[1, 2]), df_ab, "Humid", "f", ylim)

    panel_g_violin(subfig_bottom.add_subplot(gs_bottom[0, 0]), df_c)
    panel_pooled_trend(subfig_bottom, gs_bottom[0, 1], df_c, "map_mean_annual_precip", "Mean annual precipitation (mm)", "h", PRECIP_COLOR, rng)
    panel_pooled_trend(subfig_bottom, gs_bottom[1, 0], df_c, "elevation", "Elevation (m)", "i", ELEV_COLOR, rng)
    panel_j_boxplots(subfig_bottom.add_subplot(gs_bottom[1, 1]), df_c)

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig4_combined_10panel.{ext}", dpi=300, bbox_inches="tight")
    print(f"\nSaved {OUT_DIR / 'fig4_combined_10panel.pdf'} and .png")


if __name__ == "__main__":
    main()
