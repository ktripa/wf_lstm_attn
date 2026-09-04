#!/usr/bin/env python
"""Supplementary Figure S1: pairwise Pearson correlation matrix of the 7
Branch-A concurrent-weather predictors -- motivates Figure 4d's grouping
(RHmin, RHmax, VPD, and Tmax are strongly collinear).
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
]:
    if Path(_f).exists():
        fm.fontManager.addfont(_f)

BIG = 18
mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Nimbus Roman", "Times New Roman", "DejaVu Serif"],
    "font.size": BIG, "axes.labelsize": BIG, "xtick.labelsize": BIG - 3, "ytick.labelsize": BIG - 3,
    "figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42, "ps.fonttype": 42,
})

LABELS = {"tmax": "Tmax", "tmin": "Tmin", "rhmax": "RHmax", "rhmin": "RHmin", "precip": "Precip", "vpd": "VPD", "wind_speed": "Wind speed"}


def main():
    corr = pd.read_csv("results/eg_attribution/branchA_predictor_correlation.csv", index_col=0)
    labels = [LABELS[c] for c in corr.columns]

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Pearson r")

    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)

    for i in range(len(labels)):
        for j in range(len(labels)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=BIG - 5, color="white" if abs(v) > 0.6 else "black")

    ax.set_title("Branch A predictor correlation (Pearson r)", fontsize=BIG + 1)
    fig.savefig("figures/figS1_predictor_correlation.pdf", bbox_inches="tight")
    fig.savefig("figures/figS1_predictor_correlation.png", bbox_inches="tight")
    print("Saved figures/figS1_predictor_correlation.{pdf,png}")


if __name__ == "__main__":
    main()
