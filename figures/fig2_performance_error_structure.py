#!/usr/bin/env python
"""Figure 2: model performance and error structure.

Standalone: reads only the CSV/parquet artifacts already written by
scripts/evaluate_pooled_aridity_percell.py under
results/aridity_and_percell_analysis/, and writes
figures/fig2_performance_error_structure.{pdf,png} (300 dpi, vector PDF with
dense elements rasterized to keep file size sane).

Six panels. (a), (d), (e), (f) are restricted to the three aridity classes
(Semi-Arid / Sub-Humid / Humid); (a) also carries a domain-wide "All Regions"
group (all 4 tiers pooled) for reference. (b) and (c) show the full domain
(all grid cells, all tiers).
  (a) KGE decomposed (KGE, r, alpha, beta): All Regions + three classes, seed
      error bars.
  (b) Spatial map of grid-cell KGE, aridity-class boundaries overlaid.
  (c) Histogram of grid-cell KGE (all cells), with the fraction above 0.7.
  (d) Observed vs. modeled FWI hexbin density + per-class KDE contours + 1:1
      line.
  (e) Conditional bias (modeled - observed) by observed-FWI decile, per class.
  (f) KGE by calendar month, per class.

RMSE/MAE spatial and monthly breakdowns are deferred to the supplement per
request; RMSE (FWI units) still appears only in (e)'s bias axis, which is a
difference in FWI units, not an RMSE panel.
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
import seaborn as sns
from matplotlib.lines import Line2D

# ---------------------------------------------------------------- style ----
# "Times New Roman" itself is not installed on this system; Nimbus Roman is
# URW's metric-compatible clone of Times and is available system-wide.
for _f in [
    "/usr/share/fonts/urw-base35/NimbusRoman-Regular.otf",
    "/usr/share/fonts/urw-base35/NimbusRoman-Bold.otf",
    "/usr/share/fonts/urw-base35/NimbusRoman-Italic.otf",
    "/usr/share/fonts/urw-base35/NimbusRoman-BoldItalic.otf",
]:
    if Path(_f).exists():
        fm.fontManager.addfont(_f)

BIG = 36  # +10 over the previous 26, per "font sizes really really big, +10 points minimum"
mpl.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
        "font.size": BIG,
        "axes.titlesize": BIG + 2,
        "axes.labelsize": BIG + 1,
        "xtick.labelsize": BIG - 3,
        "ytick.labelsize": BIG - 3,
        "legend.fontsize": BIG - 4,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,  # embed as real fonts, not Type-3 bitmaps
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "axes.linewidth": 1.3,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
    }
)
PANEL_LABEL_SIZE = BIG + 8

DATA_DIR = Path("results/aridity_and_percell_analysis")
OUT_DIR = Path("figures")
OUT_DIR.mkdir(exist_ok=True)
SHAPEFILE = Path("results/shapefiles/tx_ok_nm_az_states.shp")

# Three classes of interest, and one fixed color per class used identically
# in every panel (Okabe-Ito colorblind-safe qualitative triple).
CLASSES = ["Semi-Arid", "Sub-Humid", "Humid"]
CLASS_COLORS = {"Semi-Arid": "#E69F00", "Sub-Humid": "#009E73", "Humid": "#0072B2"}
ALL_LABEL = "All Regions"
ALL_COLOR = "#404040"

# Panel (a) encodes metric identity, not class, via grayscale + hatch (class
# is already on the x-axis) -- kept out of the class palette on purpose.
METRIC_ORDER = ["kge", "r", "alpha", "beta"]
METRIC_LABELS = {"kge": "KGE", "r": "r", "alpha": r"$\alpha$", "beta": r"$\beta$"}
METRIC_GRAY = {"kge": "#1a1a1a", "r": "#5c5c5c", "alpha": "#a3a3a3", "beta": "#e0e0e0"}
METRIC_HATCH = {"kge": "", "r": "//", "alpha": "xx", "beta": ".."}

MONTH_ABBR = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def label_panel(ax, letter, dx=-0.18, dy=1.10):
    ax.text(dx, dy, f"({letter})", transform=ax.transAxes, fontsize=PANEL_LABEL_SIZE, fontweight="bold", va="top", ha="left")


def to_grid(df: pd.DataFrame, value_col: str):
    piv = df.pivot(index="lat", columns="lon", values=value_col).sort_index().sort_index(axis=1)
    return piv.columns.values.astype(float), piv.index.values.astype(float), piv.values


def grid_edges(centers: np.ndarray) -> np.ndarray:
    step = np.median(np.diff(centers))
    return np.concatenate([centers - step / 2, [centers[-1] + step / 2]])


def domain_wide_metrics(raw_all: pd.DataFrame) -> pd.DataFrame:
    """Per-seed pooled KGE/r/alpha/beta over the WHOLE domain (all aridity tiers),
    then mean+-std across seeds -- the "entire region pooled" row for panel (a)."""
    rows = []
    for seed, sub in raw_all.groupby("seed"):
        obs, sim = sub["obs_fwi"].to_numpy(), sub["pred_fwi"].to_numpy()
        r = np.corrcoef(obs, sim)[0, 1]
        alpha = np.std(sim) / np.std(obs)
        beta = np.mean(sim) / np.mean(obs)
        kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
        rows.append({"seed": seed, "kge": kge, "r": r, "alpha": alpha, "beta": beta})
    df = pd.DataFrame(rows)
    out = {"aridity_tier": ALL_LABEL}
    for m in METRIC_ORDER:
        out[f"{m}_mean"] = df[m].mean()
        out[f"{m}_std"] = df[m].std()
    return pd.DataFrame([out])


def panel_a(ax, tier_summary: pd.DataFrame, domain_row: pd.DataFrame):
    df = pd.concat([domain_row, tier_summary.set_index("aridity_tier").loc[CLASSES].reset_index()], ignore_index=True)
    groups = [ALL_LABEL] + CLASSES
    df = df.set_index("aridity_tier").loc[groups]
    spacing = 2.6  # extra room between groups so the big bold tick labels don't collide
    x = np.arange(len(groups)) * spacing
    width = 0.26
    for i, m in enumerate(METRIC_ORDER):
        ax.bar(
            x + (i - 1.5) * width,
            df[f"{m}_mean"],
            width,
            yerr=df[f"{m}_std"],
            capsize=3,
            color=METRIC_GRAY[m],
            hatch=METRIC_HATCH[m],
            edgecolor="black",
            linewidth=0.8,
            label=METRIC_LABELS[m],
            error_kw={"linewidth": 1.2},
        )
    ax.axhline(1.0, color="firebrick", linestyle="--", linewidth=1.8, zorder=0)
    ax.text(x[-1] + 0.5, 1.02, r"$\alpha=\beta=1$ (ideal)", color="firebrick", fontsize=BIG - 5, ha="right", va="bottom")
    ax.set_xlim(x[0] - 0.9, x[-1] + 0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontweight="bold")
    tick_colors = [ALL_COLOR] + [CLASS_COLORS[c] for c in CLASSES]
    for tick, c in zip(ax.get_xticklabels(), tick_colors):
        tick.set_color(c)
    ax.set_ylabel("Score (dimensionless)")
    ax.set_ylim(0, 1.28)
    ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=BIG + 2)
    label_panel(ax, "a")


def panel_b(ax, cax, df_cell: pd.DataFrame, states: gpd.GeoDataFrame):
    lons, lats, grid = to_grid(df_cell, "kge_mean")
    lon_e, lat_e = grid_edges(lons), grid_edges(lats)
    pcm = ax.pcolormesh(lon_e, lat_e, grid, cmap="viridis", shading="flat", rasterized=True)
    states.boundary.plot(ax=ax, color="#222222", linewidth=1.6, zorder=5)
    # cax is a dedicated sub-grid cell reserved below ax (see main()), so the
    # colorbar can never overflow into the next GridSpec row regardless of
    # how constrained_layout sizes the equal-aspect map above it.
    pos = cax.get_position()
    # Increase the 'bottom' value (e.g., by 0.02) to nudge it up
    cax.set_position([pos.x0, pos.y0 + 0.02, pos.width, pos.height])
    cbar = plt.colorbar(pcm, cax=cax, orientation="horizontal")
    cbar.set_label("KGE (dimensionless)", fontsize=BIG)
    cbar.ax.tick_params(labelsize=BIG - 4)

    ax.set_xlim(lon_e.min(), lon_e.max())
    ax.set_ylim(lat_e.min(), lat_e.max())
    ax.set_xlabel("Longitude (°)")
    ax.set_ylabel("Latitude (°)")
    ax.set_aspect(1 / np.cos(np.radians(np.nanmean(lats))))
    label_panel(ax, "b")


def panel_c(ax, df_cell_all: pd.DataFrame):
    vals = df_cell_all["kge_mean"].dropna().to_numpy()
    n_bins = 30
    counts, bin_edges = np.histogram(vals, bins=n_bins)
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    cmap = plt.get_cmap("RdYlGn")
    norm = mpl.colors.Normalize(vmin=bin_edges[0], vmax=bin_edges[-1])
    colors = cmap(norm(centers))
    ax.bar(centers, counts, width=np.diff(bin_edges), color=colors, edgecolor="black", linewidth=0.7)

    thresh = 0.7
    frac_above = 100 * (vals > thresh).mean()
    ax.axvline(thresh, color="black", linestyle="--", linewidth=3)
    ax.text(
        thresh - 0.03, counts.max() * 0.92,
        f"{frac_above:.0f}% of grid cells\nhave KGE > {thresh:.1f}",
        fontsize=BIG + 2, fontweight="bold", va="top", ha="right",
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.85, pad=5),
    )

    ax.set_xlim(0, 1)
    ax.set_xlabel("Grid-cell KGE (dimensionless)")
    ax.set_ylabel("Number of grid cells")
    label_panel(ax, "c")


def panel_d(ax, raw: pd.DataFrame):
    obs, pred = raw["obs_fwi"].to_numpy(), raw["pred_fwi"].to_numpy()
    lim = max(obs.max(), pred.max()) * 1.02
    hb = ax.hexbin(obs, pred, gridsize=55, cmap="jet", bins="log", mincnt=1, rasterized=True, extent=(0, lim, 0, lim))
    cbar = plt.colorbar(hb, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("log$_{10}$(count)", fontsize=BIG)
    cbar.ax.tick_params(labelsize=BIG - 4)

    rng = np.random.default_rng(0)
    for c in CLASSES:
        sub = raw[raw["aridity_tier"] == c]
        idx = rng.choice(len(sub), size=min(20000, len(sub)), replace=False)
        so, sp = sub["obs_fwi"].to_numpy()[idx], sub["pred_fwi"].to_numpy()[idx]
        sns.kdeplot(x=so, y=sp, ax=ax, levels=4, color=CLASS_COLORS[c], linewidths=2.6, rasterized=True)

    class_handles = [Line2D([0], [0], color=CLASS_COLORS[c], linewidth=3.5, label=c) for c in CLASSES]
    ax.legend(handles=class_handles, frameon=False, loc="upper left", fontsize=BIG - 4)

    ax.plot([0, lim], [0, lim], "k--", linewidth=2, zorder=5)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Observed FWI")
    ax.set_ylabel("Modeled FWI")
    label_panel(ax, "d")


def panel_e(ax, decile: pd.DataFrame):
    for c in CLASSES:
        sub = decile[decile["aridity_tier"] == c].sort_values("decile")
        ax.errorbar(
            sub["decile"], sub["bias_mean"], yerr=sub["bias_std"], color=CLASS_COLORS[c],
            marker="o", markersize=8, linewidth=3.2, capsize=4, capthick=2, elinewidth=2, label=c,
        )
    ax.axhline(0, color="black", linewidth=1.8, linestyle="--")
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Observed FWI decile (1 = lowest, 10 = highest)")
    ax.set_ylabel("Conditional bias\n" + r"[FWI$_{\mathrm{model}}$ $-$ FWI$_{\mathrm{observed}}$]")
    ax.legend(frameon=False, loc="upper right")
    label_panel(ax, "e")


def panel_f(ax, monthly: pd.DataFrame, raw: pd.DataFrame):
    peak_month = int(raw.loc[raw["aridity_tier"] == "Semi-Arid"].groupby("month")["obs_fwi"].mean().idxmax())
    ax.axvspan(peak_month - 0.5, peak_month + 0.5, color="lightgray", alpha=0.4, zorder=0)
    ax.text(
        peak_month, 0.99, "FWI-peak season", ha="center", va="top", fontsize=BIG - 6, color="dimgray",
        transform=ax.get_xaxis_transform(),
    )

    for c in CLASSES:
        sub = monthly[monthly["aridity_tier"] == c].sort_values("month")
        ax.plot(sub["month"], sub["kge_mean"], color=CLASS_COLORS[c], linestyle="-", marker="o", markersize=8, linewidth=3.2, label=c)
        ax.fill_between(sub["month"], sub["kge_mean"] - sub["kge_std"], sub["kge_mean"] + sub["kge_std"], color=CLASS_COLORS[c], alpha=0.18, linewidth=0)

    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_ABBR, rotation=45, ha="right")
    ax.set_ylabel("KGE (dimensionless)")
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=False, loc="lower right")
    label_panel(ax, "f")


def main():
    tier_summary = pd.read_csv(DATA_DIR / "aridity_tier_pooled_mean_std.csv")
    if "aridity_tier" not in tier_summary.columns:
        tier_summary = tier_summary.rename(columns={"group": "aridity_tier"})
    percell_all = pd.read_csv(DATA_DIR / "per_grid_cell_kge_components_mean_std_5seeds.csv")
    raw_all = pd.read_parquet(DATA_DIR / "raw_test_predictions_5seeds.parquet")
    raw = raw_all[raw_all["aridity_tier"].isin(CLASSES)].copy()
    decile = pd.read_csv(DATA_DIR / "decile_bias_mean_std.csv")
    decile = decile[decile["aridity_tier"].isin(CLASSES)].copy()
    monthly = pd.read_csv(DATA_DIR / "monthly_skill_mean_std.csv")
    monthly = monthly[monthly["aridity_tier"].isin(CLASSES)].copy()
    domain_row = domain_wide_metrics(raw_all)
    states = gpd.read_file(SHAPEFILE)

    fig = plt.figure(figsize=(30, 19), constrained_layout=True)
    gs = fig.add_gridspec(2, 3)
    ax_a = fig.add_subplot(gs[0, 0])
    gs_b = gs[0, 1].subgridspec(2, 1, height_ratios=[18, 1], hspace=0.15)
    ax_b = fig.add_subplot(gs_b[0])
    cax_b = fig.add_subplot(gs_b[1])
    ax_c = fig.add_subplot(gs[0, 2])
    ax_d = fig.add_subplot(gs[1, 0])
    ax_e = fig.add_subplot(gs[1, 1])
    ax_f = fig.add_subplot(gs[1, 2])

    panel_a(ax_a, tier_summary, domain_row)
    panel_b(ax_b, cax_b, percell_all, states)
    panel_c(ax_c, percell_all)
    panel_d(ax_d, raw)
    panel_e(ax_e, decile)
    panel_f(ax_f, monthly, raw)

    for ext in ("pdf", "png"):
        fig.savefig(OUT_DIR / f"fig2_performance_error_structure.{ext}", dpi=300, bbox_inches="tight")
    print(f"Saved {OUT_DIR / 'fig2_performance_error_structure.pdf'} and .png")
    print(f"Domain-wide (All Regions) KGE = {domain_row['kge_mean'].iloc[0]:.3f} +- {domain_row['kge_std'].iloc[0]:.3f}")


if __name__ == "__main__":
    main()
