#!/usr/bin/env python
"""CLI entry point: train one (cluster, branch-set, seed) configuration.

Usage (once loaders.py is implemented):
    python scripts/train.py --config configs/default.yaml --cluster 0 --branches A B C --seed 0
"""
from __future__ import annotations

import argparse

import numpy as np

from fwi_attn.config import load_config
from fwi_attn.data.loaders import build_sample_table, load_raw_datasets
from fwi_attn.data.dataset import FWIWeeklyDataset
from fwi_attn.data.splits import assign_split, split_counts
from fwi_attn.data.standardization import Standardizer
from fwi_attn.data.schema import BRANCH_A_FEATURES, BRANCH_B_FEATURES, BRANCH_C_FEATURES
from fwi_attn.models.lstm_attention import FWIAttnModel
from fwi_attn.training.run_manager import init_run
from fwi_attn.training.seed import set_all_seeds
from fwi_attn.training.train import train_model
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--cluster", type=int, required=True)
    p.add_argument("--branches", nargs="+", default=["A", "B", "C"], choices=["A", "B", "C"])
    p.add_argument("--seed", type=int, required=True)
    return p.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    run_dir, logger, run_id = init_run(config, run_name=f"cluster{args.cluster}_{''.join(args.branches)}_seed{args.seed}")
    set_all_seeds(args.seed)

    logger.info(f"Loading raw datasets for cluster {args.cluster}, branches={args.branches}, seed={args.seed}")
    ds = load_raw_datasets(config)
    table, exclusions = build_sample_table(ds, config)
    logger.info(f"Sample table: n={table.n}; exclusions={exclusions}")

    splits = assign_split(table.year)
    logger.info(f"Split counts: {split_counts(splits)}")

    group_key = table.cell_id if config.standardization.mode == "per_grid_cell" else table.cluster_id
    train_mask = splits == "train"

    xa_std = Standardizer(config.standardization.mode, config.standardization.eps).fit(
        table.xa[train_mask], group_key[train_mask], BRANCH_A_FEATURES
    )
    xc_std = Standardizer(config.standardization.mode, config.standardization.eps).fit(
        table.xc[train_mask], group_key[train_mask], BRANCH_C_FEATURES
    )
    n, t, f = table.xb.shape
    xb_flat = table.xb.reshape(n * t, f)
    group_key_b = np.repeat(group_key, t)
    xb_std = Standardizer(config.standardization.mode, config.standardization.eps).fit(
        xb_flat[np.repeat(train_mask, t)], group_key_b[np.repeat(train_mask, t)], BRANCH_B_FEATURES
    )
    y_std = Standardizer(config.standardization.mode, config.standardization.eps).fit(
        table.y[train_mask].reshape(-1, 1), group_key[train_mask], ["fwi"]
    )
    y_std.save(run_dir / "target_standardizer.joblib")

    logger.warning("TODO: apply standardizers to build model-ready tensors, wire DataLoaders, and call train_model().")
    raise NotImplementedError("scripts/train.py needs loaders.py implemented before this can run end to end.")


if __name__ == "__main__":
    main()
