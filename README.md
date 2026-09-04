# FWI Attribution: LSTM + Attention

Weekly Fire Weather Index (FWI) attribution model for TX/OK/NM/AZ at 0.25°,
2001-2021. This is an **attribution** study, not forecasting: FWI at week t is
modeled from same-week weather plus antecedent fuel conditions. FWI is never
a model input (except for the persistence baseline, which is defined that way).

## Setup

```bash
conda env create -f environment.yml
conda activate fwi-attn
pip install -e .        # editable install of the fwi_attn package (src/)
pytest tests/           # sanity-check the parts that don't depend on real data yet
```

`environment.yml` pins `pytorch-cuda=12.1` as a starting point -- verify against
pathfinder's actual CUDA/driver setup and adjust (or drop to CPU-only) before
installing.

## Architecture

Three input branches, concatenated after branch-specific encoders, then an MLP head:

- **Branch A** (concurrent week t): Tmax, Tmin, RHmax, RHmin, precip, VPD, wind speed -- dense + ReLU.
- **Branch B** (antecedent weeks t-12..t-1): soil moisture, NDVI, VOD -- single-layer LSTM,
  pooled by additive (Bahdanau) attention over the 12 hidden states.
- **Branch C** (static per grid cell): 6 vegetation-class fractions, elevation, mean annual
  precip, mean annual VPD -- dense + ReLU.

Hidden sizes, dropout, and layer counts are all set in `configs/default.yaml`. The same
`FWIAttnModel` class serves the full model, every branch-subset ablation (pass `branches=`),
and the plain-LSTM baseline (`branch_b.pooling: final_state` instead of `attention`).

## Repo layout

```
configs/default.yaml          all hyperparameters, paths, split years, standardization mode
src/fwi_attn/
  data/
    schema.py                 branch feature ordering/shapes -- single source of truth
    loaders.py                 raw data -> sample table  [BLOCKED, see "Open questions"]
    dataset.py                 sample-table container + PyTorch Dataset
    splits.py                  temporal train/val/test assignment (implemented)
    standardization.py         per-grid-cell / per-cluster standardization + inverse_transform (implemented)
  models/
    attention.py                Bahdanau pooling attention (implemented)
    lstm_attention.py           FWIAttnModel: configurable branches + pooling mode (implemented)
    baselines.py                RF, persistence, climatology, MLR (implemented)
  training/
    train.py                    training loop; loss in standardized space, model
                                 selection + logging in back-transformed FWI units (implemented)
    seed.py, run_manager.py     reproducibility (implemented)
  evaluation/
    metrics.py                  KGE (+ r/alpha/beta), R2, RMSE, MAE, NRMSE; per-cell,
                                 pooled, and group-aggregated variants (implemented)
    bootstrap.py                 bootstrap CIs over grid cells, incl. paired skill-gain CIs (implemented)
    evaluate.py                  predict -> inverse_transform -> metrics -> CSV, units always labeled (implemented)
  plotting/plots.py             diagnostic plots -- stub, to fill in once results exist
scripts/train.py                thin CLI wiring the above together -- blocked on loaders.py
tests/                          unit tests for everything not blocked on real data
```

## Status (2026-09-03)

Fully implemented and unit-tested: temporal splits, both standardization modes with
correct fit-on-train-only + back-transform, the model (main + all branch subsets +
plain-LSTM variant), all required metrics, bootstrap CIs, and the training/eval
scaffolding around them.

**Blocked on data specifics** -- `data/loaders.py` is a documented stub until these are
answered (asked separately, not guessed):
1. Where the source datasets (GFWED FWI, weather forcing, soil moisture, NDVI, VOD,
   static covariates, cluster assignment) live and in what format.
2. FWI's native grid/temporal resolution and the daily-to-weekly aggregation convention.
3. How gaps in NDVI / soil moisture / VOD are to be handled (interpolate, forward-fill,
   or exclude -- and at what granularity).

Per the agreed priority order: data loader + standardization verified on one cluster,
then full model + baselines on one cluster end-to-end, then all 5 clusters, then the
ablation, then the two control experiments (gridMET precip in Branch A; gridMET-derived
FWI target) last.
