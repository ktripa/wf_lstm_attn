import numpy as np
import pytest

from fwi_attn.data.standardization import Standardizer


def test_fit_transform_inverse_roundtrip_per_grid_cell():
    rng = np.random.default_rng(0)
    n_cells, n_per_cell, n_feat = 3, 100, 2
    cell_id = np.repeat(np.arange(n_cells), n_per_cell)
    X = np.stack(
        [rng.normal(loc=c * 10, scale=2, size=n_cells * n_per_cell) for c in range(n_feat)], axis=1
    )
    # give each cell a distinct mean so per-cell stats actually differ
    for c in range(n_cells):
        X[cell_id == c] += c * 5

    std = Standardizer(mode="per_grid_cell").fit(X, cell_id, feature_names=["f0", "f1"])
    Z = std.transform(X, cell_id)
    for c in range(n_cells):
        mask = cell_id == c
        assert np.allclose(Z[mask].mean(axis=0), 0, atol=1e-6)
        assert np.allclose(Z[mask].std(axis=0), 1, atol=1e-6)

    X_back = std.inverse_transform(Z, cell_id)
    assert np.allclose(X, X_back, atol=1e-8)


def test_stats_computed_on_train_only_not_leaked_from_other_data():
    train_X = np.array([[1.0], [2.0], [3.0]])
    train_ids = np.array([0, 0, 0])
    std = Standardizer(mode="per_grid_cell").fit(train_X, train_ids, feature_names=["f0"])

    test_X = np.array([[100.0]])  # wildly different distribution
    test_back = std.inverse_transform(std.transform(test_X, np.array([0])), np.array([0]))
    assert np.allclose(test_back, test_X)
    # stats must be exactly the train-set stats, unaffected by test_X
    assert np.isclose(std.mean_[0][0], 2.0)


def test_unseen_group_raises():
    std = Standardizer(mode="per_cluster").fit(np.array([[1.0]]), np.array([0]), feature_names=["f0"])
    with pytest.raises(KeyError):
        std.transform(np.array([[1.0]]), np.array([1]))


def test_zero_variance_cell_does_not_divide_by_zero():
    X = np.array([[5.0], [5.0], [5.0]])
    ids = np.array([0, 0, 0])
    std = Standardizer(mode="per_grid_cell", eps=1e-6).fit(X, ids, feature_names=["f0"])
    Z = std.transform(X, ids)
    assert np.all(np.isfinite(Z))
