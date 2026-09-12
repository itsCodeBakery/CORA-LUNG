"""Geometry utilities used by CORA-Lung preprocessing."""

from __future__ import annotations

from collections import defaultdict
import numpy as np
import nibabel as nib


def world_xyz_to_full_zyx(world_xyz, resampled_affine_xyz):
    world_xyz = np.asarray(
        world_xyz,
        dtype=np.float64,
    )

    inverse = np.linalg.inv(
        np.asarray(
            resampled_affine_xyz,
            dtype=np.float64,
        )
    )

    xyz = nib.affines.apply_affine(
        inverse,
        world_xyz,
    )

    xyz = np.rint(
        xyz
    ).astype(
        np.int32
    )

    return xyz[:, ::-1]


def resolve_sparse_training_rows(coords_zyx, labels, group_ids):
    coords_zyx = np.asarray(
        coords_zyx,
        dtype=np.int32,
    )

    labels = np.asarray(
        labels,
        dtype=np.int8,
    )

    group_ids = np.asarray(
        group_ids,
        dtype=np.int32,
    )

    classes = defaultdict(set)

    for coord, label in zip(
        coords_zyx,
        labels,
    ):
        classes[
            tuple(
                int(x)
                for x in coord
            )
        ].add(
            int(
                label
            )
        )

    conflicts = [
        coord
        for coord, values
        in classes.items()
        if len(
            values
        )
        > 1
    ]

    if conflicts:
        raise RuntimeError(
            "Foreground/background transfer collision."
        )

    ordered = sorted(
        classes
    )

    supervision_coords = np.asarray(
        ordered,
        dtype=np.int32,
    )

    supervision_labels = np.asarray(
        [
            next(
                iter(
                    classes[c]
                )
            )
            for c in ordered
        ],
        dtype=np.int8,
    )

    memberships = sorted(
        {
            (
                int(c[0]),
                int(c[1]),
                int(c[2]),
                int(g),
            )
            for c, y, g in zip(
                coords_zyx,
                labels,
                group_ids,
            )
            if int(y) == 1
        }
    )

    memberships = np.asarray(
        memberships,
        dtype=np.int32,
    )

    return (
        supervision_coords,
        supervision_labels,
        memberships,
    )
