import numpy as np

from fwi_attn.data.splits import assign_split, split_counts


def test_boundaries():
    years = np.array([2000, 2001, 2015, 2016, 2017, 2018, 2021, 2022])
    labels = assign_split(years)
    expected = ["excluded", "train", "train", "val", "val", "test", "test", "excluded"]
    assert list(labels) == expected


def test_split_counts_reports_excluded():
    years = np.array([2000, 2001, 2001])
    counts = split_counts(assign_split(years))
    assert counts == {"excluded": 1, "train": 2}
