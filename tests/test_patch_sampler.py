import numpy as np

from cora_lung.data.pilot_dataset import (
    deterministic_patch_origin,
)


def test_patch_sampler_is_deterministic():
    shape = (
        80,
        200,
        180,
    )

    patch = (
        48,
        128,
        128,
    )

    coords = np.asarray(
        [
            [20, 100, 90],
            [40, 120, 110],
        ],
        dtype=np.int32,
    )

    labels = np.asarray(
        [
            1,
            0,
        ],
        dtype=np.int8,
    )

    a, source_a = deterministic_patch_origin(
        shape,
        patch,
        coords,
        labels,
        seed=17,
        sample_index=5,
    )

    b, source_b = deterministic_patch_origin(
        shape,
        patch,
        coords,
        labels,
        seed=17,
        sample_index=5,
    )

    assert np.array_equal(
        a,
        b,
    )

    assert source_a == source_b


def test_patch_origin_inside_volume():
    shape = np.asarray(
        [60, 150, 140],
    )

    patch = np.asarray(
        [48, 128, 128],
    )

    coords = np.asarray(
        [[59, 149, 139]],
        dtype=np.int32,
    )

    labels = np.asarray(
        [1],
        dtype=np.int8,
    )

    origin, _ = deterministic_patch_origin(
        shape,
        patch,
        coords,
        labels,
        seed=17,
        sample_index=1,
    )

    assert np.all(
        origin >= 0
    )

    assert np.all(
        origin
        <= np.maximum(
            0,
            shape - patch,
        )
    )
