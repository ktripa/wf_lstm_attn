"""Training loop.

Inputs (xa/xb/xc) and target (y) are expected to already be standardized
(see data/standardization.py) when they come out of the DataLoader; the loss
is MSE in standardized space. For model selection / early stopping and for
every number written to `history.csv`, validation predictions are
back-transformed to FWI physical units first (`target_standardizer.
inverse_transform`) -- the standardized-space loss is kept too, but labeled
as such, so the two are never confused.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from fwi_attn.data.standardization import Standardizer
from fwi_attn.evaluation.metrics import rmse


def train_one_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, device: str) -> float:
    model.train()
    total_loss, n = 0.0, 0
    loss_fn = nn.MSELoss(reduction="sum")
    for batch in loader:
        xa = batch["xa"].to(device) if model.branch_a is not None else None
        xb = batch["xb"].to(device) if model.branch_b is not None else None
        xc = batch["xc"].to(device) if model.branch_c is not None else None
        y = batch["y"].to(device)

        optimizer.zero_grad()
        pred, _ = model(xa, xb, xc)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        n += y.numel()
    return total_loss / n


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    device: str,
    target_standardizer: Standardizer,
    group_ids: np.ndarray,
) -> dict:
    """group_ids: (N,) array aligned to `loader`'s dataset, in the order it will
    be iterated (shuffle must be False for `loader`) -- the id (cell or
    cluster, per config.standardization.mode) used to fit the target
    standardizer, needed to back-transform each sample correctly."""
    model.eval()
    loss_fn = nn.MSELoss(reduction="sum")
    total_loss, n = 0.0, 0
    preds_std, obs_std = [], []
    for batch in loader:
        xa = batch["xa"].to(device) if model.branch_a is not None else None
        xb = batch["xb"].to(device) if model.branch_b is not None else None
        xc = batch["xc"].to(device) if model.branch_c is not None else None
        y = batch["y"].to(device)

        pred, _ = model(xa, xb, xc)
        loss = loss_fn(pred, y)
        total_loss += loss.item()
        n += y.numel()
        preds_std.append(pred.cpu().numpy())
        obs_std.append(y.cpu().numpy())

    preds_std = np.concatenate(preds_std)
    obs_std = np.concatenate(obs_std)
    preds_fwi = target_standardizer.inverse_transform(preds_std.reshape(-1, 1), group_ids).ravel()
    obs_fwi = target_standardizer.inverse_transform(obs_std.reshape(-1, 1), group_ids).ravel()

    return {
        "val_loss_standardized": total_loss / n,
        "val_rmse_fwi": rmse(obs_fwi, preds_fwi),
    }


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    val_group_ids: np.ndarray,
    target_standardizer: Standardizer,
    config,
    run_dir: Path,
    logger,
    seed: int,
) -> tuple[nn.Module, pd.DataFrame]:
    device = config.training.device if torch.cuda.is_available() else "cpu"
    if config.training.device == "cuda" and device == "cpu":
        logger.warning("CUDA requested but unavailable; falling back to CPU")
    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.training.learning_rate, weight_decay=config.training.weight_decay
    )

    best_val_rmse = float("inf")
    best_state = None
    epochs_since_improvement = 0
    history = []

    for epoch in range(config.training.epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device)
        val_metrics = validate(model, val_loader, device, target_standardizer, val_group_ids)
        row = {"epoch": epoch, "seed": seed, "train_loss_standardized": train_loss, **val_metrics}
        history.append(row)
        logger.info(
            f"seed={seed} epoch={epoch} train_loss_std={train_loss:.4f} "
            f"val_loss_std={val_metrics['val_loss_standardized']:.4f} val_rmse_fwi={val_metrics['val_rmse_fwi']:.4f}"
        )

        if val_metrics["val_rmse_fwi"] < best_val_rmse:
            best_val_rmse = val_metrics["val_rmse_fwi"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
            if epochs_since_improvement >= config.training.early_stopping_patience:
                logger.info(f"seed={seed} early stopping at epoch {epoch} (best val_rmse_fwi={best_val_rmse:.4f})")
                break

    model.load_state_dict(best_state)
    history_df = pd.DataFrame(history)
    history_df.to_csv(run_dir / f"history_seed{seed}.csv", index=False)
    torch.save(best_state, run_dir / f"model_seed{seed}.pt")
    return model, history_df
