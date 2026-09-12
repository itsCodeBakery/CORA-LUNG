"""Development-only sparse training dataset for CORA-Lung."""

from __future__ import annotations

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import torch


def stable_seed(*parts):
    payload = "|".join(
        str(x)
        for x in parts
    )

    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        "little",
        signed=False,
    ) % (2**32 - 1)


class SparseTrainingCache:

    def __init__(
        self,
        root,
        allowed_cases,
    ):
        self.root = Path(root)

        self.manifest = pd.read_csv(
            self.root / "manifest.csv"
        )

        self.allowed_cases = set(
            str(x)
            for x in allowed_cases
        )

        observed = set(
            self.manifest["case_id"]
            .astype(str)
            .unique()
        )

        missing = (
            self.allowed_cases
            - observed
        )

        if missing:
            raise RuntimeError(
                f"Cases absent from cache: {sorted(missing)}"
            )

    def get(
        self,
        case_id,
        condition,
    ):
        case_id = str(case_id)
        condition = str(condition)

        if case_id not in self.allowed_cases:
            raise RuntimeError(
                "Requested case is outside the allowed "
                "development-only dataset."
            )

        rows = self.manifest[
            (
                self.manifest["case_id"]
                .astype(str)
                == case_id
            )
            & (
                self.manifest["condition"]
                .astype(str)
                == condition
            )
        ]

        if len(rows) != 1:
            raise RuntimeError(
                f"Expected one cache row for "
                f"{case_id}/{condition}; found {len(rows)}."
            )

        row = rows.iloc[0]

        image_path = (
            self.root
            / str(row["image_file"])
        )

        annotation_path = (
            self.root
            / str(row["annotation_file"])
        )

        with np.load(
            image_path,
            allow_pickle=False,
        ) as data:
            image = np.asarray(
                data["ct_zyx"],
                dtype=np.float32,
            )

        with np.load(
            annotation_path,
            allow_pickle=False,
        ) as data:
            coords = np.asarray(
                data["supervision_voxel_zyx"],
                dtype=np.int32,
            )

            labels = np.asarray(
                data["supervision_label"],
                dtype=np.int8,
            )

            membership_coords = np.asarray(
                data["fg_membership_voxel_zyx"],
                dtype=np.int32,
            )

            membership_groups = np.asarray(
                data["fg_membership_group_id"],
                dtype=np.int32,
            )

        return {
            "case_id":
                case_id,

            "condition":
                condition,

            "image":
                image,

            "supervision_voxel_zyx":
                coords,

            "supervision_label":
                labels,

            "fg_membership_voxel_zyx":
                membership_coords,

            "fg_membership_group_id":
                membership_groups,
        }


def deterministic_patch_origin(
    volume_shape,
    patch_shape,
    supervision_coords,
    supervision_labels,
    *,
    seed,
    sample_index,
    foreground_probability=0.60,
    background_probability=0.20,
):
    volume_shape = np.asarray(
        volume_shape,
        dtype=np.int32,
    )

    patch_shape = np.asarray(
        patch_shape,
        dtype=np.int32,
    )

    coords = np.asarray(
        supervision_coords,
        dtype=np.int32,
    )

    labels = np.asarray(
        supervision_labels,
        dtype=np.int8,
    )

    rng = np.random.default_rng(
        stable_seed(
            seed,
            "patch",
            sample_index,
        )
    )

    u = float(
        rng.random()
    )

    fg = coords[
        labels == 1
    ]

    bg = coords[
        labels == 0
    ]

    if (
        u < foreground_probability
        and len(fg)
    ):
        center = fg[
            int(
                rng.integers(
                    0,
                    len(fg),
                )
            )
        ]

        source = "foreground"

    elif (
        u
        < (
            foreground_probability
            + background_probability
        )
        and len(bg)
    ):
        center = bg[
            int(
                rng.integers(
                    0,
                    len(bg),
                )
            )
        ]

        source = "background"

    else:
        center = np.asarray(
            [
                int(
                    rng.integers(
                        0,
                        max(
                            1,
                            volume_shape[a],
                        ),
                    )
                )
                for a in range(3)
            ],
            dtype=np.int32,
        )

        source = "random"

    half = (
        patch_shape
        // 2
    )

    origin = (
        center
        - half
    )

    max_origin = np.maximum(
        0,
        volume_shape
        - patch_shape,
    )

    origin = np.minimum(
        np.maximum(
            origin,
            0,
        ),
        max_origin,
    )

    return (
        origin.astype(np.int32),
        source,
    )


def extract_patch(
    sample,
    patch_shape,
    *,
    seed,
    sample_index,
):
    image = sample["image"]

    patch_shape = np.asarray(
        patch_shape,
        dtype=np.int32,
    )

    origin, source = deterministic_patch_origin(
        image.shape,
        patch_shape,
        sample["supervision_voxel_zyx"],
        sample["supervision_label"],
        seed=seed,
        sample_index=sample_index,
    )

    end = (
        origin
        + patch_shape
    )

    pad_after = np.maximum(
        0,
        end
        - np.asarray(
            image.shape,
            dtype=np.int32,
        ),
    )

    source_end = np.minimum(
        end,
        np.asarray(
            image.shape,
            dtype=np.int32,
        ),
    )

    slices = tuple(
        slice(
            int(origin[a]),
            int(source_end[a]),
        )
        for a in range(3)
    )

    patch = image[
        slices
    ]

    if np.any(
        pad_after > 0
    ):
        patch = np.pad(
            patch,
            [
                (
                    0,
                    int(
                        pad_after[a]
                    ),
                )
                for a in range(3)
            ],
            mode="constant",
            constant_values=-1.0,
        )

    if tuple(
        patch.shape
    ) != tuple(
        patch_shape.tolist()
    ):
        raise RuntimeError(
            f"Patch shape mismatch: "
            f"{patch.shape} != {tuple(patch_shape)}"
        )

    coords = np.asarray(
        sample["supervision_voxel_zyx"],
        dtype=np.int32,
    )

    labels = np.asarray(
        sample["supervision_label"],
        dtype=np.int8,
    )

    local_coords = (
        coords
        - origin[
            None,
            :
        ]
    )

    inside = np.all(
        (
            local_coords >= 0
        )
        & (
            local_coords
            < patch_shape[
                None,
                :
            ]
        ),
        axis=1,
    )

    local_coords = local_coords[
        inside
    ]

    local_labels = labels[
        inside
    ]

    membership_coords = np.asarray(
        sample["fg_membership_voxel_zyx"],
        dtype=np.int32,
    )

    membership_groups = np.asarray(
        sample["fg_membership_group_id"],
        dtype=np.int32,
    )

    local_membership = (
        membership_coords
        - origin[
            None,
            :
        ]
    )

    membership_inside = np.all(
        (
            local_membership >= 0
        )
        & (
            local_membership
            < patch_shape[
                None,
                :
            ]
        ),
        axis=1,
    )

    local_membership = local_membership[
        membership_inside
    ]

    local_groups = membership_groups[
        membership_inside
    ]

    target = np.full(
        tuple(
            patch_shape.tolist()
        ),
        -1,
        dtype=np.int8,
    )

    for coord, label in zip(
        local_coords,
        local_labels,
    ):
        idx = tuple(
            int(x)
            for x in coord
        )

        if (
            target[idx] != -1
            and target[idx] != int(label)
        ):
            raise RuntimeError(
                "FG/BG contradiction inside patch."
            )

        target[idx] = int(label)

    return {
        "image":
            torch.from_numpy(
                patch[
                    None,
                    ...
                ].astype(np.float32)
            ),

        "target":
            torch.from_numpy(
                target[
                    None,
                    ...
                ].astype(np.int8)
            ),

        "membership_voxel_zyx":
            torch.from_numpy(
                local_membership.astype(np.int64)
            ),

        "membership_group_id":
            torch.from_numpy(
                local_groups.astype(np.int64)
            ),

        "origin_zyx":
            torch.from_numpy(
                origin.astype(np.int64)
            ),

        "sampling_source":
            source,

        "case_id":
            sample["case_id"],

        "condition":
            sample["condition"],
    }
