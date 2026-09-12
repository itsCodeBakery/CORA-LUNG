"""Optimizer-boundary dense-label firewall for CORA-Lung."""

from __future__ import annotations

import re

import numpy as np
import torch


ALLOWED_CACHE_SAMPLE_KEYS = {
    "case_id",
    "condition",
    "image",
    "supervision_voxel_zyx",
    "supervision_label",
    "fg_membership_voxel_zyx",
    "fg_membership_group_id",
}


ALLOWED_PATCH_BATCH_KEYS = {
    "image",
    "target",
    "membership_voxel_zyx",
    "membership_group_id",
    "origin_zyx",
    "sampling_source",
    "case_id",
    "condition",
}


FORBIDDEN_TOKENS = {
    "infection_mask",
    "lung_mask",
    "lesion_mask",
    "dense_mask",
    "dense_component",
    "hidden_component",
    "component_volume",
    "component_centroid",
    "ground_truth_mask",
    "groundtruth_mask",
}


_SHA256_RE = re.compile(
    r"^[0-9a-fA-F]{64}$"
)


def _walk_values(value):

    if isinstance(
        value,
        dict,
    ):
        for key, child in value.items():
            yield key
            yield from _walk_values(
                child
            )

    elif isinstance(
        value,
        (
            list,
            tuple,
            set,
        ),
    ):
        for child in value:
            yield from _walk_values(
                child
            )

    elif isinstance(
        value,
        str,
    ):
        yield value


def _reject_dense_tokens(
    obj,
):

    for value in _walk_values(
        obj
    ):

        text = str(
            value
        ).lower()

        for token in FORBIDDEN_TOKENS:

            if token in text:

                raise RuntimeError(
                    f"Forbidden trainer token: {token}"
                )


def _reject_dense_hashes(
    obj,
    dense_sha256_denylist,
):

    deny = {
        str(
            value
        ).lower()
        for value
        in dense_sha256_denylist
    }


    for value in _walk_values(
        obj
    ):

        if not isinstance(
            value,
            str,
        ):
            continue


        if not _SHA256_RE.fullmatch(
            value
        ):
            continue


        if value.lower() in deny:

            raise RuntimeError(
                "Dense-label file checksum reached trainer boundary."
            )


def validate_cache_sample(
    sample,
    dense_sha256_denylist,
):

    if set(
        sample.keys()
    ) != ALLOWED_CACHE_SAMPLE_KEYS:

        unexpected = (
            set(
                sample.keys()
            )
            - ALLOWED_CACHE_SAMPLE_KEYS
        )

        missing = (
            ALLOWED_CACHE_SAMPLE_KEYS
            - set(
                sample.keys()
            )
        )

        raise RuntimeError(
            "Unsafe cache-sample schema. "
            f"Unexpected={sorted(unexpected)}, "
            f"missing={sorted(missing)}"
        )


    _reject_dense_tokens(
        sample
    )


    _reject_dense_hashes(
        sample,
        dense_sha256_denylist,
    )


    image = np.asarray(
        sample[
            "image"
        ]
    )


    coords = np.asarray(
        sample[
            "supervision_voxel_zyx"
        ]
    )


    labels = np.asarray(
        sample[
            "supervision_label"
        ]
    )


    membership_coords = np.asarray(
        sample[
            "fg_membership_voxel_zyx"
        ]
    )


    membership_groups = np.asarray(
        sample[
            "fg_membership_group_id"
        ]
    )


    if image.ndim != 3:

        raise RuntimeError(
            "Trainer CT image must be 3-D."
        )


    if coords.shape != (
        len(
            labels
        ),
        3,
    ):

        raise RuntimeError(
            "Invalid direct-supervision coordinates."
        )


    if membership_coords.shape != (
        len(
            membership_groups
        ),
        3,
    ):

        raise RuntimeError(
            "Invalid replay-membership coordinates."
        )


    if not set(
        np.unique(
            labels
        ).tolist()
    ).issubset(
        {
            0,
            1,
        }
    ):

        raise RuntimeError(
            "Cache sample contains non-sparse label values."
        )


    direct_fg = {
        tuple(
            int(
                x
            )
            for x in coord
        )
        for coord
        in coords[
            labels == 1
        ]
    }


    membership_fg = {
        tuple(
            int(
                x
            )
            for x in coord
        )
        for coord
        in membership_coords
    }


    if direct_fg != membership_fg:

        raise RuntimeError(
            "Direct FG and replay-membership support differ."
        )


    return True


def validate_patch_batch(
    batch,
    dense_sha256_denylist,
):

    if set(
        batch.keys()
    ) != ALLOWED_PATCH_BATCH_KEYS:

        unexpected = (
            set(
                batch.keys()
            )
            - ALLOWED_PATCH_BATCH_KEYS
        )

        missing = (
            ALLOWED_PATCH_BATCH_KEYS
            - set(
                batch.keys()
            )
        )

        raise RuntimeError(
            "Unsafe optimizer-boundary batch schema. "
            f"Unexpected={sorted(unexpected)}, "
            f"missing={sorted(missing)}"
        )


    _reject_dense_tokens(
        batch
    )


    _reject_dense_hashes(
        batch,
        dense_sha256_denylist,
    )


    image = batch[
        "image"
    ]


    target = batch[
        "target"
    ]


    membership_coords = batch[
        "membership_voxel_zyx"
    ]


    membership_groups = batch[
        "membership_group_id"
    ]


    if not isinstance(
        image,
        torch.Tensor,
    ):

        raise RuntimeError(
            "Optimizer-boundary image must be a tensor."
        )


    if not isinstance(
        target,
        torch.Tensor,
    ):

        raise RuntimeError(
            "Optimizer-boundary target must be a tensor."
        )


    if image.ndim != 4:

        raise RuntimeError(
            "Expected patch image geometry [C,Z,Y,X]."
        )


    if target.ndim != 4:

        raise RuntimeError(
            "Expected sparse target geometry [C,Z,Y,X]."
        )


    if image.shape != target.shape:

        raise RuntimeError(
            "Image/target patch geometry differs."
        )


    valid_target = (
        (
            target == -1
        )
        | (
            target == 0
        )
        | (
            target == 1
        )
    )


    if not bool(
        torch.all(
            valid_target
        ).item()
    ):

        raise RuntimeError(
            "Target contains values outside {-1,0,1}."
        )


    if (
        membership_coords.ndim != 2
        or membership_coords.shape[
            1
        ] != 3
    ):

        raise RuntimeError(
            "Membership coordinates must be [N,3]."
        )


    if membership_groups.ndim != 1:

        raise RuntimeError(
            "Membership group IDs must be 1-D."
        )


    if len(
        membership_groups
    ) != len(
        membership_coords
    ):

        raise RuntimeError(
            "Membership coordinates/groups length mismatch."
        )


    patch_shape = np.asarray(
        image.shape[
            1:
        ],
        dtype=np.int64,
    )


    if membership_coords.numel():

        coords_np = (
            membership_coords
            .detach()
            .cpu()
            .numpy()
        )


        if np.any(
            coords_np < 0
        ):

            raise RuntimeError(
                "Negative membership coordinate."
            )


        if np.any(
            coords_np
            >= patch_shape[
                None,
                :
            ]
        ):

            raise RuntimeError(
                "Membership coordinate outside patch."
            )


    direct_fg_coords = (
        torch.nonzero(
            target[
                0
            ]
            == 1,
            as_tuple=False,
        )
        .detach()
        .cpu()
        .numpy()
    )


    direct_fg = {
        tuple(
            int(
                x
            )
            for x in coord
        )
        for coord
        in direct_fg_coords
    }


    membership_fg = {
        tuple(
            int(
                x
            )
            for x in coord
        )
        for coord
        in (
            membership_coords
            .detach()
            .cpu()
            .numpy()
        )
    }


    if direct_fg != membership_fg:

        raise RuntimeError(
            "Patch FG support and membership support differ."
        )


    return True


def guard_optimizer_batch(
    batch,
    dense_sha256_denylist,
):
    """
    This function must be called immediately before the future
    scientific training step consumes a batch.
    """

    return validate_patch_batch(
        batch,
        dense_sha256_denylist,
    )
