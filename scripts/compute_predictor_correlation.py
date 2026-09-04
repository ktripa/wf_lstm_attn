#!/usr/bin/env python
"""Supplementary: pairwise Pearson correlation matrix of the 7 Branch-A
concurrent-weather predictors, computed on the full pooled sample table
(physical units; correlation is scale-invariant so standardization doesn't
matter). Motivates Figure 4d's grouping (RHmin/RHmax/VPD highly correlated).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fwi_attn.config import load_config
from fwi_attn.data.loaders import build_sample_table, load_raw_datasets
from fwi_attn.data.schema import BRANCH_A_FEATURES

OUT_DIR = Path("results/eg_attribution")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    config = load_config("configs/default.yaml")
    ds = load_raw_datasets(config)
    table, _ = build_sample_table(ds, config, cluster=None)
    df = pd.DataFrame(table.xa, columns=BRANCH_A_FEATURES)
    corr = df.corr(method="pearson")
    corr.to_csv(OUT_DIR / "branchA_predictor_correlation.csv")
    print(corr.round(3))
    with open(OUT_DIR / "branchA_predictor_correlation_meta.json", "w") as f:
        json.dump({"n_samples": int(len(df)), "variables": BRANCH_A_FEATURES}, f, indent=2)
    print(f"saved branchA_predictor_correlation.csv (n={len(df)})")


if __name__ == "__main__":
    main()
