# ==========================================================================================
# CORA-LUNG — CODE BLOCK 07D
# Final Pretraining Safety Closure
#
# PURPOSE
#   Close the two remaining safeguards before ANY scientific optimizer update:
#
#   A. Header-only native voxel-volume consistency audit
#      zoom-product versus abs(det(affine[:3,:3]))
#      for all 20 CTs + all 60 associated dense-mask HEADERS.
#
#   B. Trainer-boundary firewall
#      - strict cache-sample key allowlist
#      - strict patch/batch key allowlist
#      - forbidden dense-label tokens
#      - frozen dense-file SHA-256 denylist
#      - injected dense key must fail
#      - injected dense checksum must fail
#
# IMPORTANT
#   • NO dense mask array is read.
#   • nibabel is used for HEADER/AFFINE access only.
#   • NO model fitting.
#   • NO optimizer.step().
#   • Gate B remains NOT RUN.
# ==========================================================================================

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
import subprocess
import hashlib
import importlib
import json
import os
import sys
import textwrap
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import torch

from kaggle_secrets import UserSecretsClient


# ==========================================================================================
# 0. CONSTANTS
# ==========================================================================================

REPO = Path(
    "/kaggle/working/CORA-LUNG"
)

CACHE_ROOT = Path(
    "/kaggle/working/cora_train_cache_v1_1"
)

CACHE_MANIFEST = (
    CACHE_ROOT
    / "manifest.csv"
)

EXPECTED_START_COMMIT = "f45e105d5850"

PATCH_ZYX = (
    48,
    128,
    128,
)

GLOBAL_SEED = 17

HEADER_REL_TOL = 1e-6

NOW_ISO = datetime.now(
    timezone.utc
).strftime(
    "%Y-%m-%dT%H:%M:%SZ"
)


# ==========================================================================================
# 1. HELPERS
# ==========================================================================================

def heading(text):

    print(
        "\n"
        + "=" * 118
    )

    print(text)

    print(
        "=" * 118
    )


def sh(
    cmd,
    cwd=None,
    env=None,
    check=True,
):

    result = subprocess.run(
        cmd,
        cwd=(
            str(cwd)
            if cwd
            else None
        ),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if (
        check
        and result.returncode != 0
    ):

        raise RuntimeError(
            "COMMAND FAILED\n"
            + " ".join(
                map(
                    str,
                    cmd,
                )
            )
            + "\n\nSTDOUT:\n"
            + (
                result.stdout
                or ""
            )
            + "\nSTDERR:\n"
            + (
                result.stderr
                or ""
            )
        )

    return result


def sha256_file(
    path,
):

    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        for chunk in iter(
            lambda:
                f.read(
                    8 * 1024 * 1024
                ),
            b"",
        ):

            h.update(
                chunk
            )

    return h.hexdigest()


def write_json(
    path,
    obj,
):

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def write_text(
    path,
    text,
):

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        textwrap.dedent(
            text
        ).strip()
        + "\n",
        encoding="utf-8",
    )


# ==========================================================================================
# 2. PRECONDITIONS
# ==========================================================================================

heading(
    "CORA-LUNG BLOCK 07D — FINAL PRETRAINING SAFETY CLOSURE"
)


if not (
    REPO
    / ".git"
).exists():

    raise RuntimeError(
        "CORA-LUNG repository missing."
    )


state_path = (
    REPO
    / "PROJECT_STATE.json"
)


state = json.loads(
    state_path.read_text(
        encoding="utf-8"
    )
)


if (
    state.get(
        "last_completed_block"
    )
    != "07"
):

    raise RuntimeError(
        "Block 07 is not frozen as completed."
    )


if (
    state.get(
        "gate_b_pilot_harness"
    )
    != "PASS"
):

    raise RuntimeError(
        "Gate-B pilot harness is not PASS."
    )


if (
    state.get(
        "training_grid_cache"
    )
    != "PASS"
):

    raise RuntimeError(
        "Training-grid cache is not PASS."
    )


if (
    state.get(
        "model_training_started"
    )
    is not False
):

    raise RuntimeError(
        "Unexpected model-training state."
    )


if int(
    state.get(
        "optimizer_steps_performed",
        0,
    )
) != 0:

    raise RuntimeError(
        "Optimizer steps were already recorded."
    )


dirty = sh(
    [
        "git",
        "status",
        "--porcelain",
    ],
    cwd=REPO,
).stdout.strip()


if dirty:

    raise RuntimeError(
        "Repository must be clean before Block 07D."
    )


starting_commit = sh(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    cwd=REPO,
).stdout.strip()


if not starting_commit.startswith(
    EXPECTED_START_COMMIT
):

    raise RuntimeError(
        "Unexpected starting commit. "
        "Do not continue from an unknown repository state."
    )


if not CACHE_MANIFEST.exists():

    raise RuntimeError(
        "Accepted v1.1 cache manifest missing."
    )


print(
    f"✓ Starting commit           : "
    f"{starting_commit[:12]}"
)

print(
    "✓ Block 07 harness          : PASS"
)

print(
    "✓ Scientific training       : NOT STARTED"
)

print(
    "✓ Optimizer steps           : 0"
)


# ==========================================================================================
# 3. LOCATE PRIMARY DATASET
# ==========================================================================================

heading(
    "STEP 1/8 — LOCATE FROZEN PRIMARY DATASET"
)


split_registry = pd.read_csv(
    REPO
    / "data/splits/primary_volume_split_registry.csv"
)


if len(
    split_registry
) != 20:

    raise RuntimeError(
        "Expected exactly 20 primary volumes."
    )


required_columns = {
    "case_id",
    "ct_scan",
    "infection_mask",
    "lung_mask",
    "lung_and_infection_mask",
}


if not required_columns.issubset(
    set(
        split_registry.columns
    )
):

    raise RuntimeError(
        "Primary split registry lacks required source-file columns."
    )


candidate_roots = [
    Path(
        "/kaggle/input/datasets/andrewmvd/covid19-ct-scans"
    ),
    Path(
        "/kaggle/input/covid19-ct-scans"
    ),
]


primary_root = None


probe_row = split_registry.iloc[
    0
]


for root in candidate_roots:

    if (
        root.exists()
        and (
            root
            / str(
                probe_row[
                    "ct_scan"
                ]
            )
        ).exists()
    ):

        primary_root = root

        break


if primary_root is None:

    raise RuntimeError(
        "Could not locate the frozen primary Kaggle dataset."
    )


print(
    f"✓ Primary dataset root      : "
    f"{primary_root}"
)

print(
    f"✓ Frozen primary volumes    : "
    f"{len(split_registry)}"
)


# ==========================================================================================
# 4. HEADER-ONLY VOXEL-VOLUME AUDIT
# ==========================================================================================

heading(
    "STEP 2/8 — HEADER-ONLY VOXEL-VOLUME / AFFINE AUDIT"
)


modalities = {
    "ct":
        "ct_scan",

    "infection":
        "infection_mask",

    "lung":
        "lung_mask",

    "lung_and_infection":
        "lung_and_infection_mask",
}


header_rows = []


for _, row in split_registry.iterrows():

    case_id = str(
        row[
            "case_id"
        ]
    )


    case_measurements = {}


    for modality, column in modalities.items():

        relative_path = str(
            row[
                column
            ]
        )


        path = (
            primary_root
            / relative_path
        )


        if not path.exists():

            raise RuntimeError(
                f"Missing source file: {path}"
            )


        # HEADER / AFFINE ACCESS ONLY.
        #
        # We deliberately DO NOT call:
        #   get_fdata()
        #   np.asarray(img.dataobj)
        #   img.dataobj[...]
        #
        image = nib.load(
            str(
                path
            )
        )


        zooms_xyz = np.asarray(
            image.header.get_zooms()[
                :3
            ],
            dtype=np.float64,
        )


        affine = np.asarray(
            image.affine,
            dtype=np.float64,
        )


        zoom_volume_mm3 = float(
            np.prod(
                np.abs(
                    zooms_xyz
                )
            )
        )


        affine_volume_mm3 = float(
            abs(
                np.linalg.det(
                    affine[
                        :3,
                        :3,
                    ]
                )
            )
        )


        denominator = max(
            zoom_volume_mm3,
            affine_volume_mm3,
            1e-12,
        )


        relative_error = float(
            abs(
                zoom_volume_mm3
                - affine_volume_mm3
            )
            / denominator
        )


        if not np.isfinite(
            relative_error
        ):

            raise RuntimeError(
                f"Non-finite geometry result: {path}"
            )


        if relative_error > HEADER_REL_TOL:

            raise RuntimeError(
                f"Voxel-volume header mismatch for "
                f"{case_id}/{modality}: "
                f"zoom={zoom_volume_mm3:.12f}, "
                f"affine={affine_volume_mm3:.12f}, "
                f"rel_err={relative_error:.3e}"
            )


        case_measurements[
            modality
        ] = affine_volume_mm3


        header_rows.append(
            {
                "case_id":
                    case_id,

                "modality":
                    modality,

                "relative_path":
                    relative_path,

                "shape_x":
                    int(
                        image.shape[
                            0
                        ]
                    ),

                "shape_y":
                    int(
                        image.shape[
                            1
                        ]
                    ),

                "shape_z":
                    int(
                        image.shape[
                            2
                        ]
                    ),

                "zoom_x_mm":
                    float(
                        zooms_xyz[
                            0
                        ]
                    ),

                "zoom_y_mm":
                    float(
                        zooms_xyz[
                            1
                        ]
                    ),

                "zoom_z_mm":
                    float(
                        zooms_xyz[
                            2
                        ]
                    ),

                "zoom_product_mm3":
                    zoom_volume_mm3,

                "affine_det_abs_mm3":
                    affine_volume_mm3,

                "relative_error":
                    relative_error,

                "sform_code":
                    int(
                        image.header[
                            "sform_code"
                        ]
                    ),

                "qform_code":
                    int(
                        image.header[
                            "qform_code"
                        ]
                    ),
            }
        )


    ct_volume = case_measurements[
        "ct"
    ]


    for modality in [
        "infection",
        "lung",
        "lung_and_infection",
    ]:

        paired_value = case_measurements[
            modality
        ]


        paired_rel_error = abs(
            paired_value
            - ct_volume
        ) / max(
            paired_value,
            ct_volume,
            1e-12,
        )


        if paired_rel_error > HEADER_REL_TOL:

            raise RuntimeError(
                f"Paired voxel-volume mismatch: "
                f"{case_id}/{modality}"
            )


header_df = pd.DataFrame(
    header_rows
)


if len(
    header_df
) != 80:

    raise RuntimeError(
        "Expected exactly 80 header audit rows."
    )


max_header_error = float(
    header_df[
        "relative_error"
    ].max()
)


print(
    f"✓ Headers audited           : "
    f"{len(header_df)}/80"
)

print(
    f"✓ CT headers                : "
    f"{(header_df['modality'] == 'ct').sum()}/20"
)

print(
    f"✓ Dense-mask headers        : "
    f"{(header_df['modality'] != 'ct').sum()}/60"
)

print(
    f"✓ Dense arrays read         : 0"
)

print(
    f"✓ Max zoom-vs-affine error  : "
    f"{max_header_error:.3e}"
)

print(
    "✓ Paired CT/mask voxel volume: PASS"
)


# ==========================================================================================
# 5. BUILD FROZEN DENSE-CHECKSUM DENYLIST
# ==========================================================================================

heading(
    "STEP 3/8 — BUILD DENSE-FILE SHA-256 DENYLIST"
)


source_hash_manifest = pd.read_csv(
    REPO
    / "data/manifests/primary_file_sha256.csv"
)


if set(
    source_hash_manifest.columns
) != {
    "relative_path",
    "bytes",
    "sha256",
}:

    raise RuntimeError(
        "Unexpected primary SHA-256 manifest schema."
    )


dense_prefixes = (
    "infection_mask/",
    "lung_mask/",
    "lung_and_infection_mask/",
)


dense_hash_rows = source_hash_manifest[
    source_hash_manifest[
        "relative_path"
    ]
    .astype(
        str
    )
    .str.startswith(
        dense_prefixes
    )
].copy()


if len(
    dense_hash_rows
) != 60:

    raise RuntimeError(
        f"Expected 60 frozen dense-mask hashes, "
        f"found {len(dense_hash_rows)}."
    )


dense_sha256_denylist = set(
    dense_hash_rows[
        "sha256"
    ].astype(
        str
    ).str.lower()
)


if len(
    dense_sha256_denylist
) != 60:

    raise RuntimeError(
        "Dense-file SHA-256 hashes are not unique."
    )


print(
    "✓ Dense-file hashes frozen  : 60"
)

print(
    "✓ Hashes obtained from committed manifest"
)

print(
    "✓ Dense files re-hashed/read : NO"
)


# ==========================================================================================
# 6. WRITE TRAINER-BOUNDARY FIREWALL
# ==========================================================================================

heading(
    "STEP 4/8 — FREEZE TRAINER-BOUNDARY FIREWALL"
)


firewall_source = r'''
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
'''


write_text(
    REPO
    / "src/cora_lung/engine/trainer_firewall.py",
    firewall_source,
)


print(
    "✓ Trainer-boundary firewall source frozen."
)


# ==========================================================================================
# 7. UNIT TEST TRAINER FIREWALL
# ==========================================================================================

heading(
    "STEP 5/8 — TEST POSITIVE AND NEGATIVE FIREWALL PATHS"
)


firewall_test_source = r'''
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
'''


write_text(
    REPO
    / "tests/test_trainer_boundary_firewall.py",
    firewall_test_source,
)


SRC = (
    REPO
    / "src"
)


pytest_env = os.environ.copy()


pytest_env[
    "PYTHONPATH"
] = (
    str(
        SRC
    )
    + (
        os.pathsep
        + pytest_env[
            "PYTHONPATH"
        ]
        if pytest_env.get(
            "PYTHONPATH"
        )
        else ""
    )
)


pytest_targets = [
    "tests/test_firewall.py",
    "tests/test_training_cache_firewall.py",
    "tests/test_training_grid_equalization.py",
    "tests/test_partial_losses.py",
    "tests/test_patch_sampler.py",
    "tests/test_resunet3d.py",
    "tests/test_checkpoint_resume.py",
    "tests/test_pilot_training_contract.py",
    "tests/test_trainer_boundary_firewall.py",
]


pytest_result = sh(
    [
        sys.executable,
        "-m",
        "pytest",
        *pytest_targets,
        "-q",
    ],
    cwd=REPO,
    env=pytest_env,
    check=False,
)


print(
    pytest_result.stdout.strip()
)


if pytest_result.stderr.strip():

    print(
        pytest_result.stderr.strip()
    )


if pytest_result.returncode != 0:

    raise RuntimeError(
        "Trainer-boundary firewall tests FAILED."
    )


print(
    "✓ Full pretraining firewall/unit tests: PASS"
)


# ==========================================================================================
# 8. REAL RUNTIME TRAINER-BOUNDARY TEST
# ==========================================================================================

heading(
    "STEP 6/8 — VALIDATE REAL DEVELOPMENT SAMPLES / PATCHES"
)


if str(
    SRC
) not in sys.path:

    sys.path.insert(
        0,
        str(
            SRC
        ),
    )


for module_name in [
    "cora_lung.data.pilot_dataset",
    "cora_lung.engine.trainer_firewall",
]:

    sys.modules.pop(
        module_name,
        None,
    )


importlib.invalidate_caches()


from cora_lung.data.pilot_dataset import (
    SparseTrainingCache,
    extract_patch,
)

from cora_lung.engine.trainer_firewall import (
    validate_cache_sample,
    validate_patch_batch,
)


development_cases = sorted(
    split_registry.loc[
        split_registry[
            "role"
        ]
        == "permanent_development",
        "case_id",
    ].astype(
        str
    ).tolist()
)


if len(
    development_cases
) != 4:

    raise RuntimeError(
        "Expected four permanent development volumes."
    )


cache = SparseTrainingCache(
    CACHE_ROOT,
    development_cases,
)


runtime_conditions = [
    "complete",
    "component_natural_50",
    "pixel_dropout_matched_50",
    "component_fixed_50",
    "complete_fixed_50",
]


runtime_rows = []


sample_index = 0


for case_id in development_cases:

    for condition in runtime_conditions:

        sample = cache.get(
            case_id,
            condition,
        )


        validate_cache_sample(
            sample,
            dense_sha256_denylist,
        )


        patch = extract_patch(
            sample,
            PATCH_ZYX,
            seed=GLOBAL_SEED,
            sample_index=sample_index,
        )


        validate_patch_batch(
            patch,
            dense_sha256_denylist,
        )


        runtime_rows.append(
            {
                "case_id":
                    case_id,

                "condition":
                    condition,

                "cache_sample_firewall":
                    "PASS",

                "patch_batch_firewall":
                    "PASS",
            }
        )


        sample_index += 1


runtime_df = pd.DataFrame(
    runtime_rows
)


if len(
    runtime_df
) != 20:

    raise RuntimeError(
        "Expected 20 runtime firewall checks."
    )


# Real negative test: use one actual frozen dense-mask checksum.
real_dense_hash = sorted(
    dense_sha256_denylist
)[
    0
]


probe_sample = cache.get(
    development_cases[
        0
    ],
    "complete",
)


injected_checksum_rejected = False


bad_sample = dict(
    probe_sample
)


bad_sample[
    "case_id"
] = real_dense_hash


try:

    validate_cache_sample(
        bad_sample,
        dense_sha256_denylist,
    )

except RuntimeError:

    injected_checksum_rejected = True


if not injected_checksum_rejected:

    raise RuntimeError(
        "Real frozen dense checksum was NOT rejected."
    )


injected_dense_key_rejected = False


bad_sample = dict(
    probe_sample
)


bad_sample[
    "dense_mask"
] = "FORBIDDEN"
    

try:

    validate_cache_sample(
        bad_sample,
        dense_sha256_denylist,
    )

except RuntimeError:

    injected_dense_key_rejected = True


if not injected_dense_key_rejected:

    raise RuntimeError(
        "Injected dense-mask key was NOT rejected."
    )


print(
    "✓ Real cache samples checked : 20/20"
)

print(
    "✓ Real patch batches checked : 20/20"
)

print(
    "✓ Injected dense key rejected: PASS"
)

print(
    "✓ Real dense SHA rejected     : PASS"
)


# ==========================================================================================
# 9. SAVE AUDIT ARTIFACTS
# ==========================================================================================

heading(
    "STEP 7/8 — FREEZE PRETRAINING SAFETY AUDIT"
)


manifest_dir = (
    REPO
    / "data/manifests"
)

audit_dir = (
    REPO
    / "experiments/audits"
)

figure_dir = (
    REPO
    / "figures/audit"
)

docs_dir = (
    REPO
    / "docs"
)


for directory in [
    manifest_dir,
    audit_dir,
    figure_dir,
    docs_dir,
]:

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


header_df.to_csv(
    manifest_dir
    / "native_header_voxel_volume_consistency.csv",
    index=False,
)


dense_hash_rows[
    [
        "relative_path",
        "bytes",
        "sha256",
    ]
].to_csv(
    manifest_dir
    / "trainer_dense_sha256_denylist.csv",
    index=False,
)


runtime_df.to_csv(
    manifest_dir
    / "trainer_boundary_runtime_audit.csv",
    index=False,
)


# ------------------------------------------------------------------------------------------
# Audit figure
# ------------------------------------------------------------------------------------------

figure_summary = (
    header_df.groupby(
        "modality",
        as_index=False,
    )[
        "relative_error"
    ]
    .max()
    .sort_values(
        "modality"
    )
)


fig, ax = plt.subplots(
    figsize=(
        9.5,
        5.8,
    )
)


x = np.arange(
    len(
        figure_summary
    )
)


ax.bar(
    x,
    figure_summary[
        "relative_error"
    ].values,
)


ax.set_xticks(
    x
)


ax.set_xticklabels(
    figure_summary[
        "modality"
    ].tolist(),
    fontweight="bold",
    rotation=15,
)


ax.axhline(
    HEADER_REL_TOL,
    linestyle="--",
    linewidth=1.5,
)


ax.set_yscale(
    "log"
)


ax.set_ylabel(
    "Maximum Relative Error",
    fontweight="bold",
)


ax.set_title(
    "CORA-Lung Native NIfTI Header Geometry Verification\n"
    "Voxel-Spacing Product vs. Absolute Affine Determinant",
    fontweight="bold",
)


ax.grid(
    axis="y",
    alpha=0.25,
)


fig.tight_layout()


fig.savefig(
    figure_dir
    / "fig10_native_header_geometry_verification.png",
    dpi=600,
    bbox_inches="tight",
)


fig.savefig(
    figure_dir
    / "fig10_native_header_geometry_verification.pdf",
    bbox_inches="tight",
)


plt.close(
    fig
)


audit = {
    "project":
        "CORA-Lung",

    "block":
        "07D",

    "status":
        "PASS",

    "generated_at_utc":
        NOW_ISO,

    "scientific_training_performed":
        False,

    "optimizer_steps_performed":
        0,

    "native_header_geometry": {
        "files_audited":
            int(
                len(
                    header_df
                )
            ),

        "ct_headers":
            20,

        "dense_mask_headers":
            60,

        "dense_mask_arrays_read":
            0,

        "max_zoom_affine_relative_error":
            max_header_error,

        "relative_tolerance":
            HEADER_REL_TOL,

        "paired_ct_mask_voxel_volume":
            "PASS",
    },

    "trainer_boundary_firewall": {
        "status":
            "PASS",

        "dense_file_sha256_denylist_count":
            int(
                len(
                    dense_sha256_denylist
                )
            ),

        "runtime_cache_samples_checked":
            20,

        "runtime_patch_batches_checked":
            20,

        "injected_dense_key_rejected":
            True,

        "real_dense_sha256_rejected":
            True,

        "strict_cache_sample_allowlist":
            True,

        "strict_patch_batch_allowlist":
            True,
    },

    "gate_b":
        "NOT_RUN",

    "pilot_training_authorized":
        False,
}


write_json(
    audit_dir
    / "block07d_pretraining_safety_closure.json",
    audit,
)


protocol = """
# CORA-Lung Pretraining Safety Closure

Block 07D is the final no-training safeguard before Gate B.

## Native voxel-volume audit

All 20 primary CT NIfTI headers and all 60 associated mask headers were
inspected without loading dense voxel arrays.

For every file, native voxel volume computed from the product of the first
three header zooms was compared with the absolute determinant of the spatial
3x3 affine matrix.

The relative tolerance is 1e-6.

Paired CT/mask native voxel volume must also agree within the same tolerance.

## Optimizer-boundary firewall

The future scientific trainer must call:

`guard_optimizer_batch(batch, dense_sha256_denylist)`

before consuming every optimizer-bound batch.

The firewall requires an exact sparse-batch schema and rejects:

- unexpected keys;
- dense-label terminology;
- known dense source-file SHA-256 checksums;
- invalid target values;
- inconsistent foreground replay membership;
- invalid membership coordinates.

The dense SHA-256 denylist is derived from the already committed frozen source
checksum manifest. Dense source files are not reopened to construct it.

## Scientific status

No model fitting and no optimizer update occur in Block 07D.

Gate B remains not run until this audit is manually accepted.
"""


write_text(
    docs_dir
    / "pretraining_safety_closure.md",
    protocol,
)


# ==========================================================================================
# 10. CAPTURE SOURCE / UPDATE STATE
# ==========================================================================================

source_capture = "NOT_AVAILABLE"


try:

    ip = get_ipython()

    cell = (
        ip.history_manager
        .input_hist_raw[
            -1
        ]
    )


    if (
        "CORA-LUNG — CODE BLOCK 07D"
        in cell
    ):

        source_path = (
            REPO
            / "scripts/code_blocks/block07d_pretraining_safety_closure.py"
        )


        source_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        source_path.write_text(
            cell,
            encoding="utf-8",
        )


        source_capture = "PASS"


except Exception:
    pass


state.update(
    {
        "last_attempted_block":
            "07D",

        "last_completed_block":
            "07D",

        "last_completed_block_name":
            "pretraining_safety_closure",

        "current_stage":
            "pre_gate_b_safety_verified",

        "current_gate":
            "PRE_GATE_B_PILOT",

        "native_voxel_volume_header_audit":
            "PASS",

        "trainer_boundary_firewall":
            "PASS",

        "dense_checksum_denylist":
            "PASS",

        "gate_b":
            "NOT_RUN",

        "pilot_training_authorized":
            False,

        "training_authorized":
            False,

        "model_training_started":
            False,

        "optimizer_steps_performed":
            0,

        "next_action":
            (
                "Audit Block 07D. If accepted, authorize and run "
                "the development-only Gate-B omission-deficit pilot."
            ),

        "updated_at_utc":
            NOW_ISO,
    }
)


write_json(
    state_path,
    state,
)


# ==========================================================================================
# 11. REFRESH REPOSITORY MANIFEST
# ==========================================================================================

repo_manifest_path = (
    REPO
    / "REPOSITORY_MANIFEST.json"
)


repo_files = sorted(
    [
        path
        for path in REPO.rglob(
            "*"
        )
        if (
            path.is_file()
            and ".git"
            not in path.parts
            and path
            != repo_manifest_path
        )
    ],
    key=lambda path:
        str(
            path.relative_to(
                REPO
            )
        ),
)


write_json(
    repo_manifest_path,
    {
        "generated_at_utc":
            NOW_ISO,

        "completed_block":
            "07D",

        "self_included":
            False,

        "files": [
            {
                "path":
                    str(
                        path.relative_to(
                            REPO
                        )
                    ),

                "bytes":
                    int(
                        path.stat().st_size
                    ),

                "sha256":
                    sha256_file(
                        path
                    ),
            }
            for path in repo_files
        ],
    },
)


# ==========================================================================================
# 12. COMMIT / PUSH
# ==========================================================================================

heading(
    "STEP 8/8 — COMMIT PRETRAINING SAFETY CLOSURE"
)


token = UserSecretsClient().get_secret(
    "pushCora"
)


if not token:

    raise RuntimeError(
        "Kaggle secret 'pushCora' unavailable."
    )


token = token.strip()


askpass = Path(
    "/tmp/cora_git_askpass_block07d.sh"
)


askpass.write_text(
    '#!/bin/sh\n'
    'case "$1" in\n'
    '  *Username*) echo "x-access-token" ;;\n'
    '  *Password*) echo "$GITHUB_TOKEN" ;;\n'
    '  *) echo "" ;;\n'
    'esac\n',
    encoding="utf-8",
)


askpass.chmod(
    0o700
)


git_env = os.environ.copy()


git_env[
    "GITHUB_TOKEN"
] = token


git_env[
    "GIT_ASKPASS"
] = str(
    askpass
)


git_env[
    "GIT_TERMINAL_PROMPT"
] = "0"


sh(
    [
        "git",
        "add",
        "PROJECT_STATE.json",
        "REPOSITORY_MANIFEST.json",
        "src/cora_lung/engine/trainer_firewall.py",
        "tests/test_trainer_boundary_firewall.py",
        "data/manifests/native_header_voxel_volume_consistency.csv",
        "data/manifests/trainer_dense_sha256_denylist.csv",
        "data/manifests/trainer_boundary_runtime_audit.csv",
        "experiments/audits/block07d_pretraining_safety_closure.json",
        "docs/pretraining_safety_closure.md",
        "figures/audit/fig10_native_header_geometry_verification.png",
        "figures/audit/fig10_native_header_geometry_verification.pdf",
    ],
    cwd=REPO,
)


if (
    REPO
    / "scripts/code_blocks/block07d_pretraining_safety_closure.py"
).exists():

    sh(
        [
            "git",
            "add",
            "scripts/code_blocks/block07d_pretraining_safety_closure.py",
        ],
        cwd=REPO,
    )


status = sh(
    [
        "git",
        "status",
        "--short",
    ],
    cwd=REPO,
).stdout.strip()


if not status:

    raise RuntimeError(
        "No Block-07D changes available to commit."
    )


print(
    status
)


commit_message = (
    "audit: close pretraining geometry and trainer firewall"
)


sh(
    [
        "git",
        "commit",
        "-m",
        commit_message,
    ],
    cwd=REPO,
)


push = sh(
    [
        "git",
        "push",
        "origin",
        "main",
    ],
    cwd=REPO,
    env=git_env,
    check=False,
)


if push.returncode != 0:

    safe_error = (
        push.stderr
        or ""
    ).replace(
        token,
        "***TOKEN_REDACTED***",
    )


    raise RuntimeError(
        "GitHub push failed:\n"
        + safe_error
    )


current_commit = sh(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    cwd=REPO,
).stdout.strip()


try:

    askpass.unlink(
        missing_ok=True
    )

except Exception:
    pass


token = None


# ==========================================================================================
# 13. FINAL REPORT
# ==========================================================================================

print(
    "\n"
    + "=" * 118
)

print(
    "CORA-LUNG CODE BLOCK 07D — FINAL PRETRAINING SAFETY REPORT"
)

print(
    "=" * 118
)


print(
    "Native NIfTI headers audited          :",
    len(
        header_df
    ),
    "/ 80",
)

print(
    "CT headers                            : 20 / 20"
)

print(
    "Dense-mask headers                    : 60 / 60"
)

print(
    "Dense-mask voxel arrays read          : 0"
)

print(
    "Max zoom-vs-affine relative error     :",
    "{:.18e}".format(
        max_header_error
    ),
)

print(
    "Header tolerance                      :",
    "{:.1e}".format(
        HEADER_REL_TOL
    ),
)

print(
    "Paired CT/mask voxel volume           : PASS"
)

print(
    "Dense SHA-256 denylist                :",
    len(
        dense_sha256_denylist
    ),
    "files",
)

print(
    "Real cache samples firewall checked   : 20 / 20"
)

print(
    "Real patch batches firewall checked   : 20 / 20"
)

print(
    "Injected dense key rejection          : PASS"
)

print(
    "Real dense checksum rejection         : PASS"
)

print(
    "Strict cache-sample allowlist          : PASS"
)

print(
    "Strict optimizer-batch allowlist       : PASS"
)

print(
    "Full firewall/unit test suite          : PASS"
)

print(
    "Exact Block-07D source captured        :",
    source_capture,
)

print(
    "Scientific model training              : NO"
)

print(
    "Optimizer steps                        : 0"
)

print(
    "Gate B                                 : NOT RUN"
)

print(
    "Pilot training authorized              : NO"
)

print(
    "Starting commit                        :",
    starting_commit[:12],
)

print(
    "Current commit                         :",
    current_commit[:12],
)

print(
    "GitHub synchronization                 : PASS"
)

print()
print(
    "NEXT: Send me this COMPLETE report."
)

print(
    "=" * 118
)