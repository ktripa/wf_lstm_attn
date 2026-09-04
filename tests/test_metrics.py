import numpy as np

from fwi_attn.evaluation.metrics import kge, r2, rmse, mae, nrmse, per_cell_metrics, pooled_metrics


def test_kge_perfect():
    obs = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    m = kge(obs, obs.copy())
    assert np.isclose(m["kge"], 1.0)
    assert np.isclose(m["r"], 1.0)
    assert np.isclose(m["alpha"], 1.0)
    assert np.isclose(m["beta"], 1.0)


def test_r2_rmse_mae_perfect():
    obs = np.array([1.0, 2.0, 3.0])
    assert np.isclose(r2(obs, obs.copy()), 1.0)
    assert np.isclose(rmse(obs, obs.copy()), 0.0)
    assert np.isclose(mae(obs, obs.copy()), 0.0)
    assert np.isclose(nrmse(obs, obs.copy()), 0.0)


def test_nan_handling():
    # only the (1.0, 1.0) pair is jointly finite -> 1 valid sample -> must return nan, not crash
    obs = np.array([1.0, np.nan, 3.0])
    sim = np.array([1.0, 2.0, np.nan])
    assert np.isnan(kge(obs, sim)["kge"])


def test_per_cell_and_pooled_agree_on_single_cell():
    rng = np.random.default_rng(0)
    obs = rng.normal(10, 2, size=50)
    sim = obs + rng.normal(0, 0.5, size=50)
    cell_ids = np.zeros(50, dtype=int)
    pc = per_cell_metrics(obs, sim, cell_ids, units="fwi")
    pooled = pooled_metrics(obs, sim, cell_ids, units="fwi")
    assert np.isclose(pc.loc[0, "rmse"], pooled.loc[0, "rmse"])
