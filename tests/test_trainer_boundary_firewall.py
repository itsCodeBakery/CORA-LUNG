import numpy as np
import pytest
import torch

from cora_lung.engine.trainer_firewall import (
    validate_cache_sample,
    validate_patch_batch,
)


DENSE_HASH = "a" * 64


def safe_sample():

    return {
        "case_id":
            "case_001",

        "condition":
            "component_natural_50",

        "image":
            np.zeros(
                (
                    8,
                    16,
                    16,
                ),
                dtype=np.float32,
            ),

        "supervision_voxel_zyx":
            np.asarray(
                [
                    [1, 2, 3],
                    [4, 5, 6],
                ],
                dtype=np.int32,
            ),

        "supervision_label":
            np.asarray(
                [
                    1,
                    0,
                ],
                dtype=np.int8,
            ),

        "fg_membership_voxel_zyx":
            np.asarray(
                [
                    [1, 2, 3],
                ],
                dtype=np.int32,
            ),

        "fg_membership_group_id":
            np.asarray(
                [
                    1,
                ],
                dtype=np.int32,
            ),
    }


def safe_patch():

    target = torch.full(
        (
            1,
            8,
            16,
            16,
        ),
        -1,
        dtype=torch.int8,
    )

    target[
        0,
        1,
        2,
        3,
    ] = 1

    target[
        0,
        4,
        5,
        6,
    ] = 0

    return {
        "image":
            torch.zeros(
                (
                    1,
                    8,
                    16,
                    16,
                ),
                dtype=torch.float32,
            ),

        "target":
            target,

        "membership_voxel_zyx":
            torch.tensor(
                [
                    [1, 2, 3],
                ],
                dtype=torch.int64,
            ),

        "membership_group_id":
            torch.tensor(
                [
                    1,
                ],
                dtype=torch.int64,
            ),

        "origin_zyx":
            torch.tensor(
                [
                    0,
                    0,
                    0,
                ],
                dtype=torch.int64,
            ),

        "sampling_source":
            "foreground",

        "case_id":
            "case_001",

        "condition":
            "component_natural_50",
    }


def test_safe_cache_sample_passes():

    assert validate_cache_sample(
        safe_sample(),
        {
            DENSE_HASH,
        },
    )


def test_safe_patch_passes():

    assert validate_patch_batch(
        safe_patch(),
        {
            DENSE_HASH,
        },
    )


def test_injected_dense_key_fails():

    sample = safe_sample()

    sample[
        "dense_mask"
    ] = np.zeros(
        (
            8,
            16,
            16,
        ),
        dtype=np.uint8,
    )

    with pytest.raises(
        RuntimeError
    ):

        validate_cache_sample(
            sample,
            {
                DENSE_HASH,
            },
        )


def test_injected_dense_checksum_fails():

    sample = safe_sample()

    sample[
        "case_id"
    ] = DENSE_HASH

    with pytest.raises(
        RuntimeError
    ):

        validate_cache_sample(
            sample,
            {
                DENSE_HASH,
            },
        )


def test_injected_patch_key_fails():

    patch = safe_patch()

    patch[
        "infection_mask"
    ] = torch.zeros(
        (
            1,
            8,
            16,
            16,
        )
    )

    with pytest.raises(
        RuntimeError
    ):

        validate_patch_batch(
            patch,
            {
                DENSE_HASH,
            },
        )


def test_patch_dense_checksum_fails():

    patch = safe_patch()

    patch[
        "condition"
    ] = DENSE_HASH

    with pytest.raises(
        RuntimeError
    ):

        validate_patch_batch(
            patch,
            {
                DENSE_HASH,
            },
        )
