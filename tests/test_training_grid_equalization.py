import numpy as np

from cora_lung.data.equalization import (
    candidate_pool,
    exact_group_preserving_subset,
)


def test_exact_target_and_group_preservation():

    membership_coords = np.asarray(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 2],
        ],
        dtype=np.int32,
    )

    membership_groups = np.asarray(
        [
            1,
            1,
            2,
            2,
            3,
            3,
        ],
        dtype=np.int32,
    )

    pool, groups = candidate_pool(
        membership_coords,
        membership_groups,
    )

    selected = exact_group_preserving_subset(
        pool,
        groups,
        4,
        17,
    )

    assert len(selected) == 4

    represented = set()

    for coord in selected:

        represented.update(
            pool[
                tuple(coord.tolist())
            ]
        )

    assert represented == {
        1,
        2,
        3,
    }


def test_multi_group_collision_can_cover_two_groups():

    membership_coords = np.asarray(
        [
            [1, 1, 1],
            [1, 1, 1],
            [2, 2, 2],
        ],
        dtype=np.int32,
    )

    membership_groups = np.asarray(
        [
            1,
            2,
            3,
        ],
        dtype=np.int32,
    )

    pool, groups = candidate_pool(
        membership_coords,
        membership_groups,
    )

    selected = exact_group_preserving_subset(
        pool,
        groups,
        2,
        3,
    )

    assert len(selected) == 2
