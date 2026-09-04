"""Non-neural baselines.

ASSUMPTION (flagged for confirmation, not blocking): the Random Forest baseline
uses the same information as the full neural model -- Branch A (7) + Branch B
flattened to 36 + Branch C (n_static) -- so it isolates the value of the
sequence/attention architecture rather than the value of the inputs
themselves. If a Branch-A-only or Branch-A+B-only RF is wanted instead, change
`RandomForestBaseline.build_features`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression


class RandomForestBaseline:
    def __init__(self, n_estimators: int = 500, max_depth: int | None = None, n_jobs: int = -1, random_state: int = 0):
        self.model = RandomForestRegressor(
            n_estimators=n_estimators, max_depth=max_depth, n_jobs=n_jobs, random_state=random_state
        )

    @staticmethod
    def build_features(xa: np.ndarray, xb: np.ndarray, xc: np.ndarray) -> np.ndarray:
        """xa: (N,7), xb: (N,12,3) -> flattened (N,36), xc: (N,n_static) -> concatenated (N, 7+36+n_static)."""
        n = xa.shape[0]
        return np.concatenate([xa, xb.reshape(n, -1), xc], axis=1)

    def fit(self, xa, xb, xc, y):
        self.model.fit(self.build_features(xa, xb, xc), y)
        return self

    def predict(self, xa, xb, xc):
        return self.model.predict(self.build_features(xa, xb, xc))


class MLRBaseline:
    """Multiple linear regression on Branch A only."""

    def __init__(self):
        self.model = LinearRegression()

    def fit(self, xa: np.ndarray, y: np.ndarray):
        self.model.fit(xa, y)
        return self

    def predict(self, xa: np.ndarray) -> np.ndarray:
        return self.model.predict(xa)


class PersistenceBaseline:
    """Predicts FWI(t) = FWI(t-1). No fitting; FWI(t-1) must be supplied at predict time
    as a baseline-only input (the attribution model itself never sees FWI)."""

    def predict(self, fwi_tm1: np.ndarray) -> np.ndarray:
        return np.asarray(fwi_tm1)


class ClimatologyBaseline:
    """Per-grid-cell weekly climatology: mean training-period FWI for that
    grid cell and week-of-year, applied to val/test weeks with the same
    (cell, week-of-year) key."""

    def __init__(self):
        self.table_: pd.Series | None = None

    def fit(self, cell_ids: np.ndarray, week_of_year: np.ndarray, y: np.ndarray) -> "ClimatologyBaseline":
        df = pd.DataFrame({"cell_id": cell_ids, "woy": week_of_year, "y": y})
        self.table_ = df.groupby(["cell_id", "woy"])["y"].mean()
        return self

    def predict(self, cell_ids: np.ndarray, week_of_year: np.ndarray) -> np.ndarray:
        if self.table_ is None:
            raise RuntimeError("fit() must be called before predict()")
        keys = list(zip(cell_ids, week_of_year))
        missing = [k for k in set(keys) if k not in self.table_.index]
        if missing:
            raise KeyError(
                f"{len(missing)} (cell, week-of-year) combination(s) absent from training "
                f"climatology, e.g. {missing[:5]}"
            )
        return self.table_.loc[keys].to_numpy()
