"""Weak-label budget utilities for CORA-Lung."""

from __future__ import annotations

import numpy as np


def allocate_budget_exact(total_budget, capacities, rng):
    """
    Allocate an exact integer annotation budget over groups.

    Every group receives at least one labelled voxel.

    The group visitation order is deterministically shuffled once, then
    revisited cyclically until the requested budget is exhausted.

    Unlike the original v1.0 implementation, this function has no arbitrary
    iteration cutoff. A full cycle without progress is treated as infeasible.
    """

    group_ids = sorted(capacities.keys())

    if not group_ids:
        return {}, False

    total_budget = int(total_budget)

    minimum_required = len(group_ids)

    total_capacity = int(
        sum(
            int(capacities[g])
            for g in group_ids
        )
    )

    if total_budget < minimum_required:
        return {}, False

    if total_budget > total_capacity:
        return {}, False

    allocation = {
        g: 1
        for g in group_ids
    }

    remaining = (
        total_budget
        - minimum_required
    )

    order = np.asarray(
        group_ids,
        dtype=np.int32,
    )

    rng.shuffle(order)

    while remaining > 0:
        progressed = False

        for g_raw in order:
            g = int(g_raw)

            if remaining == 0:
                break

            if allocation[g] < capacities[g]:
                allocation[g] += 1
                remaining -= 1
                progressed = True

        if not progressed:
            return {}, False

    return allocation, True
