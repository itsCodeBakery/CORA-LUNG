import numpy as np

from cora_lung.data.scribbles import allocate_budget_exact


def test_coronacases003_25pct_capacity_regression():
    capacities = {
        5: 19,
        8: 3,
        9: 13,
        10: 9,
        13: 14,
        14: 6,
        24: 256,
        27: 38,
        30: 14,
        39: 4,
    }

    target = 376

    allocation, feasible = allocate_budget_exact(
        target,
        capacities,
        np.random.default_rng(20260912),
    )

    assert feasible
    assert sum(allocation.values()) == target

    # Since target == total capacity, every group must be saturated.
    assert allocation == capacities


def test_budget_above_capacity_rejected():
    capacities = {
        1: 3,
        2: 4,
    }

    allocation, feasible = allocate_budget_exact(
        8,
        capacities,
        np.random.default_rng(1),
    )

    assert not feasible
    assert allocation == {}


def test_budget_below_one_per_group_rejected():
    capacities = {
        1: 10,
        2: 10,
        3: 10,
    }

    allocation, feasible = allocate_budget_exact(
        2,
        capacities,
        np.random.default_rng(1),
    )

    assert not feasible
    assert allocation == {}
