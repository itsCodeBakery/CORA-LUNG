import numpy as np
import pytest

from cora_lung.data.preprocess import (
    world_xyz_to_full_zyx,
    resolve_sparse_training_rows,
)


def test_world_coordinate_roundtrip():
    affine = np.asarray(
        [
            [1.5, 0.0, 0.0, 10.0],
            [0.0, 1.5, 0.0, -20.0],
            [0.0, 0.0, 3.0, 30.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )

    xyz = np.asarray(
        [
            [4, 7, 3],
            [10, 2, 8],
        ],
        dtype=float,
    )

    world = (
        xyz
        @ affine[:3, :3].T
        + affine[:3, 3]
    )

    zyx = world_xyz_to_full_zyx(
        world,
        affine,
    )

    assert np.array_equal(
        zyx,
        xyz.astype(int)[:, ::-1],
    )


def test_same_class_collision_merges():
    coords = np.asarray(
        [
            [1, 2, 3],
            [1, 2, 3],
            [4, 5, 6],
        ],
        dtype=np.int32,
    )

    labels = np.asarray(
        [1, 1, 0],
        dtype=np.int8,
    )

    groups = np.asarray(
        [2, 2, -1],
        dtype=np.int32,
    )

    sup_c, sup_y, memberships = resolve_sparse_training_rows(
        coords,
        labels,
        groups,
    )

    assert len(sup_y) == 2
    assert len(memberships) == 1


def test_multi_group_fg_membership_survives_collision():
    coords = np.asarray(
        [
            [1, 2, 3],
            [1, 2, 3],
        ],
        dtype=np.int32,
    )

    labels = np.asarray(
        [1, 1],
        dtype=np.int8,
    )

    groups = np.asarray(
        [5, 9],
        dtype=np.int32,
    )

    sup_c, sup_y, memberships = resolve_sparse_training_rows(
        coords,
        labels,
        groups,
    )

    assert len(sup_y) == 1
    assert set(
        memberships[:, 3].tolist()
    ) == {5, 9}


def test_fg_bg_collision_rejected():
    coords = np.asarray(
        [
            [1, 2, 3],
            [1, 2, 3],
        ],
        dtype=np.int32,
    )

    labels = np.asarray(
        [1, 0],
        dtype=np.int8,
    )

    groups = np.asarray(
        [2, -1],
        dtype=np.int32,
    )

    with pytest.raises(RuntimeError):
        resolve_sparse_training_rows(
            coords,
            labels,
            groups,
        )
