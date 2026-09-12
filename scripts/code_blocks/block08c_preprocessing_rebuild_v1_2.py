# ==========================================================================================
# CORA-LUNG — CODE BLOCK 08C
# Source-Specific Intensity Correction + Training Cache v1.2 Rebuild
#
# WHY THIS BLOCK EXISTS
#
# BLOCK 08A WAS FIT BEFORE DENSE OUTCOMES WERE OPENED, BUT A CT-ONLY QC FOUND:
#
#   • Coronacases:
#       raw-HU encoding
#
#   • Radiopaedia:
#       documented prewindowed encoding representing [-1250,250] HU
#       stored directly in [0,255]
#
# CURRENT v1.1 CACHE:
#       uniform HU clip [-1000,400]
#
# CORRECT v1.2 CONTRACT:
#
#   Coronacases/raw-HU:
#       clip [-1250,250]
#       normalize to [-1,1]
#
#   Radiopaedia/prewindowed:
#       DO NOT apply HU clipping
#       normalize x -> 2*x/255 - 1
#
#   Both sources therefore share a declared normalized physical window.
#
# ADDITIONAL PROTOCOL CORRECTION:
#
#   patch source probabilities:
#       foreground stroke = 0.50
#       background stroke = 0.25
#       uniform crop      = 0.25
#
# IMPORTANT SCIENTIFIC LINEAGE
#
#   • Block-08A models are INVALIDATED BEFORE dense development evaluation.
#   • Their model binaries are NOT deleted.
#   • Their losses are NOT used as Gate-B evidence.
#   • Dense masks are NOT opened here.
#   • Sparse supervision identities are preserved.
#   • Post-transfer matched FG/BG budgets remain frozen.
#   • Final outer folds remain sealed.
#
# THIS BLOCK DOES NOT TRAIN A MODEL.
# ==========================================================================================

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict

import subprocess
import hashlib
import importlib
import json
import os
import sys
import gc
import random
import shutil
import textwrap

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
import yaml

import nibabel as nib
from nibabel.processing import resample_to_output

from tqdm.auto import tqdm
from kaggle_secrets import UserSecretsClient


# ==========================================================================================
# 0. FROZEN SETTINGS
# ==========================================================================================

REPO = Path(
    "/kaggle/working/CORA-LUNG"
)

SOURCE_CACHE_ROOT = Path(
    "/kaggle/working/cora_train_cache_v1_1"
)

TARGET_CACHE_ROOT = Path(
    "/kaggle/working/cora_train_cache_v1_2"
)

BUILD_ROOT = Path(
    "/kaggle/working/cora_train_cache_v1_2_BUILDING"
)

SOURCE_MANIFEST = (
    SOURCE_CACHE_ROOT
    / "manifest.csv"
)

TARGET_MANIFEST = (
    TARGET_CACHE_ROOT
    / "manifest.csv"
)

EXPECTED_START_COMMIT = (
    "412d746db72a"
)

PREPROCESS_VERSION = "1.2"

TARGET_SPACING_XYZ = np.asarray(
    [
        1.5,
        1.5,
        3.0,
    ],
    dtype=np.float64,
)

TARGET_SPACING_ZYX = np.asarray(
    [
        3.0,
        1.5,
        1.5,
    ],
    dtype=np.float64,
)

COMMON_WINDOW_HU = np.asarray(
    [
        -1250.0,
        250.0,
    ],
    dtype=np.float32,
)

PREWINDOWED_RANGE = np.asarray(
    [
        0.0,
        255.0,
    ],
    dtype=np.float32,
)

CROP_MARGIN_MM = 15.0

# Old physical crop thresholds mapped into the declared
# common [-1250,250] normalized intensity space.
BODY_THRESHOLD_HU = -600.0
LUNG_AIR_THRESHOLD_HU = -320.0

COMMON_HU_MIN = -1250.0
COMMON_HU_MAX = 250.0

BODY_THRESHOLD_NORM = (
    2.0
    * (
        BODY_THRESHOLD_HU
        - COMMON_HU_MIN
    )
    / (
        COMMON_HU_MAX
        - COMMON_HU_MIN
    )
    - 1.0
)

LUNG_AIR_THRESHOLD_NORM = (
    2.0
    * (
        LUNG_AIR_THRESHOLD_HU
        - COMMON_HU_MIN
    )
    / (
        COMMON_HU_MAX
        - COMMON_HU_MIN
    )
    - 1.0
)

MIN_LUNG_AIR_COMPONENT_ML = 50.0
MIN_TOTAL_LUNG_AIR_ML = 150.0

FG_PATCH_PROBABILITY = 0.50
BG_PATCH_PROBABILITY = 0.25
RANDOM_PATCH_PROBABILITY = 0.25

NOW = datetime.now(
    timezone.utc
)

NOW_ISO = NOW.strftime(
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


def semantic_hash(
    *arrays,
):

    h = hashlib.sha256()

    for array in arrays:

        array = np.ascontiguousarray(
            array
        )

        h.update(
            str(
                array.dtype
            ).encode(
                "utf-8"
            )
        )

        h.update(
            str(
                array.shape
            ).encode(
                "utf-8"
            )
        )

        h.update(
            array.tobytes()
        )

    return h.hexdigest()


def largest_2d_component(
    mask2d,
):

    labels, count = ndi.label(
        mask2d,
        structure=ndi.generate_binary_structure(
            2,
            2,
        ),
    )

    if count == 0:

        return np.zeros_like(
            mask2d,
            dtype=bool,
        )

    sizes = np.bincount(
        labels.ravel()
    )

    sizes[
        0
    ] = 0

    return (
        labels
        == int(
            np.argmax(
                sizes
            )
        )
    )


def bbox_from_mask(
    mask,
    spacing_zyx,
    margin_mm,
):

    coords = np.argwhere(
        mask
    )

    if len(
        coords
    ) == 0:

        return (
            np.zeros(
                3,
                dtype=np.int32,
            ),
            np.asarray(
                mask.shape,
                dtype=np.int32,
            ),
        )


    lo = coords.min(
        axis=0
    ).astype(
        np.int32
    )


    hi = (
        coords.max(
            axis=0
        )
        + 1
    ).astype(
        np.int32
    )


    margin_vox = np.ceil(
        float(
            margin_mm
        )
        / np.asarray(
            spacing_zyx,
            dtype=np.float64,
        )
    ).astype(
        np.int32
    )


    lo = np.maximum(
        0,
        lo
        - margin_vox,
    )


    hi = np.minimum(
        np.asarray(
            mask.shape,
            dtype=np.int32,
        ),
        hi
        + margin_vox,
    )


    return (
        lo,
        hi,
    )


def normalize_raw_hu(
    image,
):

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    image = np.clip(
        image,
        COMMON_HU_MIN,
        COMMON_HU_MAX,
    )

    normalized = (
        2.0
        * (
            image
            - COMMON_HU_MIN
        )
        / (
            COMMON_HU_MAX
            - COMMON_HU_MIN
        )
        - 1.0
    )

    return np.clip(
        normalized,
        -1.0,
        1.0,
    ).astype(
        np.float32
    )


def normalize_prewindowed(
    image,
):

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    image = np.clip(
        image,
        0.0,
        255.0,
    )

    normalized = (
        2.0
        * image
        / 255.0
        - 1.0
    )

    return np.clip(
        normalized,
        -1.0,
        1.0,
    ).astype(
        np.float32
    )


# ==========================================================================================
# 2. NORMALIZED-SPACE IMAGE-ONLY CROP
# ==========================================================================================

def derive_normalized_image_only_crop(
    normalized_zyx,
    spacing_zyx,
):

    """
    Image-only crop operating entirely in the declared common normalized space.

    NO:
      lesion mask
      lung mask
      scribble
      hidden component information
    """

    body_seed = (
        normalized_zyx
        > BODY_THRESHOLD_NORM
    )


    body_envelope = np.zeros_like(
        body_seed,
        dtype=bool,
    )


    for z in range(
        normalized_zyx.shape[
            0
        ]
    ):

        slice_mask = body_seed[
            z
        ]


        if not slice_mask.any():

            continue


        slice_mask = ndi.binary_closing(
            slice_mask,
            structure=np.ones(
                (
                    3,
                    3,
                ),
                dtype=bool,
            ),
        )


        slice_mask = largest_2d_component(
            slice_mask
        )


        if not slice_mask.any():

            continue


        slice_mask = ndi.binary_fill_holes(
            slice_mask
        )


        body_envelope[
            z
        ] = slice_mask


    if not body_envelope.any():

        return {
            "lo_zyx":
                np.zeros(
                    3,
                    dtype=np.int32,
                ),

            "hi_zyx":
                np.asarray(
                    normalized_zyx.shape,
                    dtype=np.int32,
                ),

            "method":
                "full_volume_fallback",

            "body_voxels":
                0,

            "lung_air_voxels":
                0,

            "lung_air_ml":
                0.0,
        }


    lung_air = (
        (
            normalized_zyx
            < LUNG_AIR_THRESHOLD_NORM
        )
        & body_envelope
    )


    labels, n_components = ndi.label(
        lung_air,
        structure=ndi.generate_binary_structure(
            3,
            3,
        ),
    )


    voxel_ml = (
        float(
            np.prod(
                spacing_zyx
            )
        )
        / 1000.0
    )


    selected_lung = np.zeros_like(
        lung_air,
        dtype=bool,
    )


    if n_components > 0:

        counts = np.bincount(
            labels.ravel()
        )


        component_ids = np.arange(
            1,
            len(
                counts
            ),
        )


        component_ml = (
            counts[
                1:
            ].astype(
                np.float64
            )
            * voxel_ml
        )


        eligible = component_ids[
            component_ml
            >= MIN_LUNG_AIR_COMPONENT_ML
        ]


        if len(
            eligible
        ):

            eligible = sorted(
                eligible,
                key=lambda component_id:
                    counts[
                        int(
                            component_id
                        )
                    ],
                reverse=True,
            )[
                :4
            ]


            selected_lung = np.isin(
                labels,
                eligible,
            )


    lung_air_ml = float(
        selected_lung.sum()
        * voxel_ml
    )


    body_lo, body_hi = bbox_from_mask(
        body_envelope,
        spacing_zyx,
        CROP_MARGIN_MM,
    )


    plausible = False


    if selected_lung.any():

        lung_lo, lung_hi = bbox_from_mask(
            selected_lung,
            spacing_zyx,
            CROP_MARGIN_MM,
        )


        body_span = (
            body_hi
            - body_lo
        )


        lung_span = (
            lung_hi
            - lung_lo
        )


        plausible = bool(
            lung_air_ml
            >= MIN_TOTAL_LUNG_AIR_ML
            and lung_span[
                0
            ]
            >= max(
                8,
                int(
                    0.30
                    * body_span[
                        0
                    ]
                ),
            )
            and lung_span[
                1
            ]
            >= max(
                16,
                int(
                    0.35
                    * body_span[
                        1
                    ]
                ),
            )
            and lung_span[
                2
            ]
            >= max(
                16,
                int(
                    0.35
                    * body_span[
                        2
                    ]
                ),
            )
        )


    if plausible:

        crop_lo = lung_lo
        crop_hi = lung_hi

        method = (
            "normalized_internal_air_lung_bbox"
        )


    else:

        crop_lo = body_lo
        crop_hi = body_hi

        method = (
            "normalized_body_envelope_fallback"
        )


    return {
        "lo_zyx":
            crop_lo.astype(
                np.int32
            ),

        "hi_zyx":
            crop_hi.astype(
                np.int32
            ),

        "method":
            method,

        "body_voxels":
            int(
                body_envelope.sum()
            ),

        "lung_air_voxels":
            int(
                selected_lung.sum()
            ),

        "lung_air_ml":
            lung_air_ml,
    }


# ==========================================================================================
# 3. PRECONDITIONS
# ==========================================================================================

heading(
    "CORA-LUNG BLOCK 08C — SOURCE-SPECIFIC PREPROCESSING REBUILD"
)


if not (
    REPO
    / ".git"
).exists():

    raise RuntimeError(
        "Repository missing."
    )


head = sh(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    cwd=REPO,
).stdout.strip()


if not head.startswith(
    EXPECTED_START_COMMIT
):

    raise RuntimeError(
        "Unexpected starting commit."
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
        "Repository must be clean before Block 08C."
    )


if not SOURCE_MANIFEST.exists():

    raise RuntimeError(
        "Accepted v1.1 training cache is missing."
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
    != "08A"
):

    raise RuntimeError(
        "Expected completed Block 08A state."
    )


if (
    state.get(
        "dense_development_evaluation"
    )
    != "NOT_RUN"
):

    raise RuntimeError(
        "Dense Gate-B outcomes appear to have been opened."
    )


print(
    "✓ Starting commit                :",
    head[:12],
)

print(
    "✓ Block-08A dense evaluation     : NOT RUN"
)

print(
    "✓ Block-08A models               : WILL BE INVALIDATED"
)

print(
    "✓ Dense masks accessed here      : NO"
)


# ==========================================================================================
# 4. DATASET / MANIFEST DISCOVERY
# ==========================================================================================

heading(
    "STEP 1/8 — LOAD FROZEN SOURCE AND CACHE MANIFESTS"
)


split_df = pd.read_csv(
    REPO
    / "data/splits/primary_volume_split_registry.csv"
)


source_manifest = pd.read_csv(
    SOURCE_MANIFEST
)


if len(
    split_df
) != 20:

    raise RuntimeError(
        "Expected 20 primary CT volumes."
    )


if len(
    source_manifest
) != 260:

    raise RuntimeError(
        "Expected 260 v1.1 training annotations."
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


probe_rel = str(
    split_df.iloc[
        0
    ][
        "ct_scan"
    ]
)


for root in candidate_roots:

    if (
        root.exists()
        and (
            root
            / probe_rel
        ).exists()
    ):

        primary_root = root

        break


if primary_root is None:

    raise RuntimeError(
        "Primary CT dataset not found."
    )


print(
    "✓ Primary CT root               :",
    primary_root,
)

print(
    "✓ Source cache                  : v1.1"
)

print(
    "✓ Target cache                  : v1.2"
)


# ==========================================================================================
# 5. TRANSACTIONAL BUILD ROOT
# ==========================================================================================

if BUILD_ROOT.exists():

    shutil.rmtree(
        BUILD_ROOT
    )


if TARGET_CACHE_ROOT.exists():

    raise RuntimeError(
        "Target v1.2 cache already exists. "
        "Do not overwrite an unknown completed cache."
    )


(
    BUILD_ROOT
    / "images"
).mkdir(
    parents=True,
    exist_ok=True,
)


(
    BUILD_ROOT
    / "annotations"
).mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================================================================
# 6. REBUILD CORRECTED IMAGES + REBASE SPARSE COORDINATES
# ==========================================================================================

heading(
    "STEP 2/8 — REBUILD 20 SOURCE-SPECIFIC CT CACHES"
)


new_manifest_rows = []

encoding_rows = []

annotation_audit_rows = []

crop_rows = []


max_affine_difference = 0.0

max_spacing_difference = 0.0

sparse_points_lost = 0

membership_points_lost = 0


for _, split_row in tqdm(
    split_df.iterrows(),
    total=len(
        split_df
    ),
    desc="Rebuilding corrected CT cases",
):

    case_id = str(
        split_row[
            "case_id"
        ]
    )


    source_origin = str(
        split_row[
            "source_origin"
        ]
    )


    case_rows = source_manifest[
        source_manifest[
            "case_id"
        ].astype(
            str
        )
        == case_id
    ].copy()


    if len(
        case_rows
    ) != 13:

        raise RuntimeError(
            f"Expected 13 conditions for {case_id}; "
            f"found {len(case_rows)}."
        )


    unique_image_files = (
        case_rows[
            "image_file"
        ]
        .astype(
            str
        )
        .unique()
    )


    if len(
        unique_image_files
    ) != 1:

        raise RuntimeError(
            f"Expected one shared image cache for {case_id}."
        )


    old_image_relative = str(
        unique_image_files[
            0
        ]
    )


    old_image_path = (
        SOURCE_CACHE_ROOT
        / old_image_relative
    )


    with np.load(
        old_image_path,
        allow_pickle=False,
    ) as old_image:

        old_crop_origin = np.asarray(
            old_image[
                "crop_origin_zyx"
            ],
            dtype=np.int32,
        )


        old_full_shape = np.asarray(
            old_image[
                "full_shape_zyx"
            ],
            dtype=np.int32,
        )


        old_affine = np.asarray(
            old_image[
                "resampled_affine_xyz"
            ],
            dtype=np.float64,
        )


        old_spacing = np.asarray(
            old_image[
                "spacing_zyx_mm"
            ],
            dtype=np.float64,
        )


        old_crop_shape = np.asarray(
            old_image[
                "crop_shape_zyx"
            ],
            dtype=np.int32,
        )


    ct_path = (
        primary_root
        / str(
            split_row[
                "ct_scan"
            ]
        )
    )


    native_img = nib.load(
        str(
            ct_path
        ),
        mmap=True,
    )


    native = np.asarray(
        native_img.dataobj,
        dtype=np.float32,
    )


    source_min = float(
        native.min()
    )

    source_max = float(
        native.max()
    )


    if source_origin.lower() == "radiopaedia":

        encoding = (
            "documented_prewindowed_0_255"
        )


        # Provenance determines this branch.
        # Extrema are ONLY a consistency assertion.
        if (
            source_min < -1e-5
            or source_max > 255.0
            + 1e-5
        ):

            raise RuntimeError(
                f"Radiopaedia encoding assertion failed: {case_id}"
            )


        encoded_native = np.clip(
            native,
            0.0,
            255.0,
        )


        interpolation_cval = 0.0


    elif source_origin.lower() == "coronacases":

        encoding = (
            "raw_hu"
        )


        # Raw HU provenance is supplied by the frozen registry.
        encoded_native = np.clip(
            native,
            COMMON_HU_MIN,
            COMMON_HU_MAX,
        )


        interpolation_cval = (
            COMMON_HU_MIN
        )


    else:

        raise RuntimeError(
            f"Unknown source provenance for {case_id}: "
            f"{source_origin}"
        )


    header = native_img.header.copy()


    header.set_data_dtype(
        np.float32
    )


    encoded_img = nib.Nifti1Image(
        encoded_native,
        native_img.affine,
        header=header,
    )


    canonical_img = nib.as_closest_canonical(
        encoded_img
    )


    resampled_img = resample_to_output(
        canonical_img,
        voxel_sizes=tuple(
            TARGET_SPACING_XYZ.tolist()
        ),
        order=1,
        mode="constant",
        cval=float(
            interpolation_cval
        ),
    )


    resampled_xyz = np.asarray(
        resampled_img.dataobj,
        dtype=np.float32,
    )


    actual_spacing_xyz = np.asarray(
        resampled_img.header.get_zooms()[
            :3
        ],
        dtype=np.float64,
    )


    resampled_zyx = np.transpose(
        resampled_xyz,
        (
            2,
            1,
            0,
        ),
    )


    if source_origin.lower() == "radiopaedia":

        normalized_full = (
            normalize_prewindowed(
                resampled_zyx
            )
        )


    else:

        normalized_full = (
            normalize_raw_hu(
                resampled_zyx
            )
        )


    new_affine = np.asarray(
        resampled_img.affine,
        dtype=np.float64,
    )


    new_full_shape = np.asarray(
        normalized_full.shape,
        dtype=np.int32,
    )


    affine_difference = float(
        np.max(
            np.abs(
                new_affine
                - old_affine
            )
        )
    )


    spacing_difference = float(
        np.max(
            np.abs(
                actual_spacing_xyz[
                    ::-1
                ]
                - old_spacing
            )
        )
    )


    max_affine_difference = max(
        max_affine_difference,
        affine_difference,
    )


    max_spacing_difference = max(
        max_spacing_difference,
        spacing_difference,
    )


    if not np.array_equal(
        new_full_shape,
        old_full_shape,
    ):

        raise RuntimeError(
            f"Resampled full-grid shape changed for {case_id}: "
            f"{new_full_shape} != {old_full_shape}"
        )


    if not np.allclose(
        new_affine,
        old_affine,
        atol=1e-6,
        rtol=1e-6,
    ):

        raise RuntimeError(
            f"Resampled affine changed materially for {case_id}."
        )


    crop_info = derive_normalized_image_only_crop(
        normalized_full,
        TARGET_SPACING_ZYX,
    )


    new_crop_origin = np.asarray(
        crop_info[
            "lo_zyx"
        ],
        dtype=np.int32,
    )


    new_crop_hi = np.asarray(
        crop_info[
            "hi_zyx"
        ],
        dtype=np.int32,
    )


    new_crop_shape = (
        new_crop_hi
        - new_crop_origin
    )


    if np.any(
        new_crop_shape <= 0
    ):

        raise RuntimeError(
            f"Invalid corrected crop for {case_id}."
        )


    corrected_crop = normalized_full[
        int(
            new_crop_origin[
                0
            ]
        ):
        int(
            new_crop_hi[
                0
            ]
        ),

        int(
            new_crop_origin[
                1
            ]
        ):
        int(
            new_crop_hi[
                1
            ]
        ),

        int(
            new_crop_origin[
                2
            ]
        ):
        int(
            new_crop_hi[
                2
            ]
        ),
    ]


    corrected_crop = corrected_crop.astype(
        np.float16
    )


    new_image_path = (
        BUILD_ROOT
        / old_image_relative
    )


    new_image_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    np.savez_compressed(
        new_image_path,

        ct_zyx=
            corrected_crop,

        spacing_zyx_mm=
            TARGET_SPACING_ZYX.astype(
                np.float32
            ),

        resampled_affine_xyz=
            new_affine,

        crop_origin_zyx=
            new_crop_origin.astype(
                np.int32
            ),

        full_shape_zyx=
            new_full_shape.astype(
                np.int32
            ),

        crop_shape_zyx=
            np.asarray(
                corrected_crop.shape,
                dtype=np.int32,
            ),

        hu_window=
            COMMON_WINDOW_HU.astype(
                np.float32
            ),
    )


    image_semantic_sha = semantic_hash(
        corrected_crop,
        TARGET_SPACING_ZYX.astype(
            np.float32
        ),
        new_affine,
        new_crop_origin.astype(
            np.int32
        ),
        new_full_shape.astype(
            np.int32
        ),
        np.asarray(
            corrected_crop.shape,
            dtype=np.int32,
        ),
        COMMON_WINDOW_HU.astype(
            np.float32
        ),
    )


    old_crop_voxels = int(
        np.prod(
            old_crop_shape
        )
    )


    new_crop_voxels = int(
        np.prod(
            corrected_crop.shape
        )
    )


    crop_rows.append(
        {
            "case_id":
                case_id,

            "source_origin":
                source_origin,

            "encoding":
                encoding,

            "crop_method":
                crop_info[
                    "method"
                ],

            "old_crop_origin_z":
                int(
                    old_crop_origin[
                        0
                    ]
                ),

            "old_crop_origin_y":
                int(
                    old_crop_origin[
                        1
                    ]
                ),

            "old_crop_origin_x":
                int(
                    old_crop_origin[
                        2
                    ]
                ),

            "new_crop_origin_z":
                int(
                    new_crop_origin[
                        0
                    ]
                ),

            "new_crop_origin_y":
                int(
                    new_crop_origin[
                        1
                    ]
                ),

            "new_crop_origin_x":
                int(
                    new_crop_origin[
                        2
                    ]
                ),

            "old_crop_voxels":
                old_crop_voxels,

            "new_crop_voxels":
                new_crop_voxels,

            "crop_changed":
                bool(
                    not np.array_equal(
                        old_crop_origin,
                        new_crop_origin,
                    )
                    or not np.array_equal(
                        old_crop_shape,
                        np.asarray(
                            corrected_crop.shape,
                            dtype=np.int32,
                        ),
                    )
                ),

            "lung_air_ml":
                float(
                    crop_info[
                        "lung_air_ml"
                    ]
                ),
        }
    )


    encoding_rows.append(
        {
            "case_id":
                case_id,

            "source_origin":
                source_origin,

            "encoding":
                encoding,

            "source_min":
                source_min,

            "source_max":
                source_max,

            "common_declared_window_hu_min":
                float(
                    COMMON_HU_MIN
                ),

            "common_declared_window_hu_max":
                float(
                    COMMON_HU_MAX
                ),

            "normalized_min":
                float(
                    corrected_crop.min()
                ),

            "normalized_max":
                float(
                    corrected_crop.max()
                ),

            "body_threshold_normalized":
                float(
                    BODY_THRESHOLD_NORM
                ),

            "lung_air_threshold_normalized":
                float(
                    LUNG_AIR_THRESHOLD_NORM
                ),

            "provenance_rule":
                (
                    "source_origin_registry"
                ),
        }
    )


    # ----------------------------------------------------------------------
    # Rebase every sparse annotation from old crop coordinates to new crop.
    #
    # Full resampled-grid coordinate identity MUST remain exact.
    # No regeneration and no dense mask access.
    # ----------------------------------------------------------------------

    for _, manifest_row in case_rows.iterrows():

        annotation_relative = str(
            manifest_row[
                "annotation_file"
            ]
        )


        old_annotation_path = (
            SOURCE_CACHE_ROOT
            / annotation_relative
        )


        with np.load(
            old_annotation_path,
            allow_pickle=False,
        ) as annotation:

            old_coords = np.asarray(
                annotation[
                    "supervision_voxel_zyx"
                ],
                dtype=np.int32,
            )


            labels = np.asarray(
                annotation[
                    "supervision_label"
                ],
                dtype=np.int8,
            )


            old_membership_coords = np.asarray(
                annotation[
                    "fg_membership_voxel_zyx"
                ],
                dtype=np.int32,
            )


            membership_groups = np.asarray(
                annotation[
                    "fg_membership_group_id"
                ],
                dtype=np.int32,
            )


        full_coords = (
            old_coords
            + old_crop_origin[
                None,
                :
            ]
        )


        full_membership_coords = (
            old_membership_coords
            + old_crop_origin[
                None,
                :
            ]
        )


        new_coords = (
            full_coords
            - new_crop_origin[
                None,
                :
            ]
        ).astype(
            np.int32
        )


        new_membership_coords = (
            full_membership_coords
            - new_crop_origin[
                None,
                :
            ]
        ).astype(
            np.int32
        )


        crop_shape = np.asarray(
            corrected_crop.shape,
            dtype=np.int32,
        )


        direct_inside = np.all(
            (
                new_coords >= 0
            )
            & (
                new_coords
                < crop_shape[
                    None,
                    :
                ]
            ),
            axis=1,
        )


        membership_inside = np.all(
            (
                new_membership_coords >= 0
            )
            & (
                new_membership_coords
                < crop_shape[
                    None,
                    :
                ]
            ),
            axis=1,
        )


        direct_lost = int(
            (
                ~direct_inside
            ).sum()
        )


        membership_lost = int(
            (
                ~membership_inside
            ).sum()
        )


        sparse_points_lost += (
            direct_lost
        )


        membership_points_lost += (
            membership_lost
        )


        if (
            direct_lost > 0
            or membership_lost > 0
        ):

            raise RuntimeError(
                f"Corrected image-only crop excludes sparse supervision "
                f"for {case_id}/{manifest_row['condition']}. "
                f"Direct lost={direct_lost}, "
                f"membership lost={membership_lost}. "
                f"Do NOT repair using dense masks."
            )


        if not np.array_equal(
            new_coords
            + new_crop_origin[
                None,
                :
            ],
            full_coords,
        ):

            raise RuntimeError(
                "Direct sparse full-grid coordinate identity failed."
            )


        if not np.array_equal(
            new_membership_coords
            + new_crop_origin[
                None,
                :
            ],
            full_membership_coords,
        ):

            raise RuntimeError(
                "Membership full-grid coordinate identity failed."
            )


        new_annotation_path = (
            BUILD_ROOT
            / annotation_relative
        )


        new_annotation_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        np.savez_compressed(
            new_annotation_path,

            supervision_voxel_zyx=
                new_coords,

            supervision_label=
                labels,

            fg_membership_voxel_zyx=
                new_membership_coords,

            fg_membership_group_id=
                membership_groups,
        )


        annotation_semantic_sha = semantic_hash(
            new_coords,
            labels,
            new_membership_coords,
            membership_groups,
        )


        new_row = manifest_row.to_dict()


        previous_annotation_sha = str(
            manifest_row[
                "annotation_semantic_sha256"
            ]
        )


        new_row[
            "preprocess_version"
        ] = PREPROCESS_VERSION


        new_row[
            "image_semantic_sha256"
        ] = image_semantic_sha


        new_row[
            "parent_v1_annotation_semantic_sha256"
        ] = previous_annotation_sha


        new_row[
            "annotation_semantic_sha256"
        ] = annotation_semantic_sha


        construction_policy = str(
            manifest_row[
                "construction_policy"
            ]
        )


        if (
            "v1.2_source_specific_intensity"
            not in construction_policy
        ):

            construction_policy = (
                construction_policy
                + "|v1.2_source_specific_intensity_crop_rebase"
            )


        new_row[
            "construction_policy"
        ] = construction_policy


        new_manifest_rows.append(
            new_row
        )


        annotation_audit_rows.append(
            {
                "case_id":
                    case_id,

                "condition":
                    str(
                        manifest_row[
                            "condition"
                        ]
                    ),

                "direct_voxels":
                    int(
                        len(
                            new_coords
                        )
                    ),

                "membership_rows":
                    int(
                        len(
                            new_membership_coords
                        )
                    ),

                "labels_identical":
                    True,

                "group_ids_identical":
                    True,

                "full_grid_direct_coordinates_identical":
                    True,

                "full_grid_membership_coordinates_identical":
                    True,

                "direct_points_lost":
                    direct_lost,

                "membership_points_lost":
                    membership_lost,
            }
        )


    del native
    del encoded_native
    del resampled_xyz
    del resampled_zyx
    del normalized_full
    del corrected_crop
    del native_img
    del encoded_img
    del canonical_img
    del resampled_img

    gc.collect()


# ==========================================================================================
# 7. FREEZE v1.2 MANIFEST
# ==========================================================================================

heading(
    "STEP 3/8 — FREEZE v1.2 CACHE MANIFEST"
)


new_manifest = pd.DataFrame(
    new_manifest_rows
)


# Preserve exact v1.1 manifest column order.
new_manifest = new_manifest[
    source_manifest.columns.tolist()
]


if len(
    new_manifest
) != 260:

    raise RuntimeError(
        "v1.2 manifest must contain exactly 260 rows."
    )


if new_manifest[
    "case_id"
].nunique() != 20:

    raise RuntimeError(
        "v1.2 manifest must contain exactly 20 cases."
    )


if new_manifest[
    "condition"
].nunique() != 13:

    raise RuntimeError(
        "v1.2 manifest must contain exactly 13 conditions."
    )


new_manifest.to_csv(
    BUILD_ROOT
    / "manifest.csv",
    index=False,
)


encoding_df = pd.DataFrame(
    encoding_rows
)


crop_df = pd.DataFrame(
    crop_rows
)


annotation_audit_df = pd.DataFrame(
    annotation_audit_rows
)


print(
    "✓ Manifest rows                    : 260"
)

print(
    "✓ Cases                            : 20"
)

print(
    "✓ Conditions                       : 13"
)

print(
    "✓ Direct sparse points lost        :",
    sparse_points_lost,
)

print(
    "✓ Membership points lost           :",
    membership_points_lost,
)


# ==========================================================================================
# 8. VERIFY CAUSAL PAIRS REMAIN MATCHED
# ==========================================================================================

heading(
    "STEP 4/8 — REVERIFY POST-TRANSFER CAUSAL BUDGETS"
)


def load_annotation(
    root,
    row,
):

    with np.load(
        root
        / str(
            row[
                "annotation_file"
            ]
        ),
        allow_pickle=False,
    ) as data:

        coords = np.asarray(
            data[
                "supervision_voxel_zyx"
            ],
            dtype=np.int32,
        )


        labels = np.asarray(
            data[
                "supervision_label"
            ],
            dtype=np.int8,
        )


    fg = {
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


    bg = {
        tuple(
            int(
                x
            )
            for x in coord
        )
        for coord
        in coords[
            labels == 0
        ]
    }


    return (
        fg,
        bg,
    )


pair_rows = []


for case_id in sorted(
    new_manifest[
        "case_id"
    ].astype(
        str
    ).unique()
):

    case_manifest = new_manifest[
        new_manifest[
            "case_id"
        ].astype(
            str
        )
        == case_id
    ]


    lookup = {
        str(
            row[
                "condition"
            ]
        ):
            row
        for _, row
        in case_manifest.iterrows()
    }


    required = [
        "component_natural_50",
        "pixel_dropout_matched_50",
        "component_fixed_50",
        "complete_fixed_50",
    ]


    for condition in required:

        if condition not in lookup:

            raise RuntimeError(
                f"Missing {condition} for {case_id}."
            )


    natural_fg, natural_bg = load_annotation(
        BUILD_ROOT,
        lookup[
            "component_natural_50"
        ],
    )


    pixel_fg, pixel_bg = load_annotation(
        BUILD_ROOT,
        lookup[
            "pixel_dropout_matched_50"
        ],
    )


    component_fixed_fg, component_fixed_bg = load_annotation(
        BUILD_ROOT,
        lookup[
            "component_fixed_50"
        ],
    )


    complete_fixed_fg, complete_fixed_bg = load_annotation(
        BUILD_ROOT,
        lookup[
            "complete_fixed_50"
        ],
    )


    primary_fg_equal = bool(
        len(
            natural_fg
        )
        == len(
            pixel_fg
        )
    )


    primary_bg_equal = bool(
        natural_bg
        == pixel_bg
    )


    fixed_fg_equal = bool(
        len(
            component_fixed_fg
        )
        == len(
            complete_fixed_fg
        )
    )


    fixed_bg_equal = bool(
        component_fixed_bg
        == complete_fixed_bg
    )


    if not (
        primary_fg_equal
        and primary_bg_equal
        and fixed_fg_equal
        and fixed_bg_equal
    ):

        raise RuntimeError(
            f"v1.2 causal-budget regression for {case_id}."
        )


    pair_rows.append(
        {
            "case_id":
                case_id,

            "natural_fg":
                len(
                    natural_fg
                ),

            "pixel_matched_fg":
                len(
                    pixel_fg
                ),

            "primary_fg_equal":
                primary_fg_equal,

            "primary_bg_equal":
                primary_bg_equal,

            "component_fixed_fg":
                len(
                    component_fixed_fg
                ),

            "complete_fixed_fg":
                len(
                    complete_fixed_fg
                ),

            "fixed_fg_equal":
                fixed_fg_equal,

            "fixed_bg_equal":
                fixed_bg_equal,
        }
    )


pair_df = pd.DataFrame(
    pair_rows
)


print(
    "✓ Natural vs matched-pixel pair     : 20/20"
)

print(
    "✓ Component-fixed vs complete-fixed : 20/20"
)


# ==========================================================================================
# 9. ATOMICALLY PROMOTE CACHE
# ==========================================================================================

heading(
    "STEP 5/8 — VALIDATE AND PROMOTE CACHE v1.2"
)


# Ensure files exist before promotion.
image_files = list(
    (
        BUILD_ROOT
        / "images"
    ).glob(
        "*.npz"
    )
)


annotation_files = list(
    (
        BUILD_ROOT
        / "annotations"
    ).glob(
        "*.npz"
    )
)


if len(
    image_files
) != 20:

    raise RuntimeError(
        f"Expected 20 corrected CT caches, found {len(image_files)}."
    )


if len(
    annotation_files
) != 260:

    raise RuntimeError(
        f"Expected 260 corrected annotation caches, "
        f"found {len(annotation_files)}."
    )


os.replace(
    BUILD_ROOT,
    TARGET_CACHE_ROOT,
)


print(
    "✓ Cache promotion                   : PASS"
)

print(
    "✓ Corrected images                 : 20"
)

print(
    "✓ Rebasing annotations             : 260"
)


# ==========================================================================================
# 10. FRESH FIREWALL VALIDATION
# ==========================================================================================

SRC = (
    REPO
    / "src"
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


sys.modules.pop(
    "cora_lung.data.training_cache",
    None,
)


importlib.invalidate_caches()


from cora_lung.data.training_cache import (
    validate_training_cache,
)


firewall_result = validate_training_cache(
    TARGET_CACHE_ROOT
)


if (
    firewall_result[
        "manifest_rows"
    ]
    != 260
):

    raise RuntimeError(
        "v1.2 firewall manifest count mismatch."
    )


print(
    "✓ v1.2 cache firewall              : PASS"
)


# ==========================================================================================
# 11. CORRECT PATCH-SAMPLING PROTOCOL
# ==========================================================================================

heading(
    "STEP 6/8 — CORRECT PILOT PATCH-SAMPLING PROTOCOL"
)


pilot_dataset_path = (
    REPO
    / "src/cora_lung/data/pilot_dataset.py"
)


pilot_text = pilot_dataset_path.read_text(
    encoding="utf-8"
)


old_probability_signature = (
    "foreground_probability=0.60,\n"
    "    background_probability=0.20,"
)


new_probability_signature = (
    "foreground_probability=0.50,\n"
    "    background_probability=0.25,"
)


if (
    old_probability_signature
    in pilot_text
):

    pilot_text = pilot_text.replace(
        old_probability_signature,
        new_probability_signature,
        1,
    )


elif (
    new_probability_signature
    in pilot_text
):

    pass


else:

    raise RuntimeError(
        "Could not safely locate patch-sampling defaults."
    )


pilot_dataset_path.write_text(
    pilot_text,
    encoding="utf-8",
)


# ==========================================================================================
# 12. FREEZE CORRECTED CONFIGURATION
# ==========================================================================================

preprocess_config = """
preprocessing:
  version: "1.2"
  parent_version: "1.1"
  correction_reason: "mixed documented primary-cohort intensity encodings"

geometry:
  internal_array_order: [z, y, x]
  target_spacing_zyx_mm: [3.0, 1.5, 1.5]
  nifti_resample_spacing_xyz_mm: [1.5, 1.5, 3.0]
  canonical_orientation_before_resample: true
  ct_interpolation: linear
  sparse_coordinate_transfer: physical_world_coordinate_nearest_voxel

intensity:
  normalized_range: [-1.0, 1.0]
  common_declared_window_hu: [-1250, 250]

  provenance_rule:
    assignment: frozen_source_origin_registry
    infer_encoding_from_extrema: false

  coronacases:
    encoding: raw_hu
    operation: clip_then_resample_then_normalize
    clip_hu: [-1250, 250]
    normalization: "2*(x+1250)/1500 - 1"
    outside_resample_value: -1250

  radiopaedia:
    encoding: documented_prewindowed_0_255
    represented_hu_window: [-1250, 250]
    apply_hu_clipping: false
    operation: resample_then_normalize
    source_range: [0, 255]
    normalization: "2*x/255 - 1"
    outside_resample_value: 0

crop:
  dense_masks_used: false
  scribbles_used: false
  ct_only: true
  intensity_space: common_normalized_minus1_plus1
  physical_margin_mm: 15.0
  body_threshold_hu_equivalent: -600
  lung_air_threshold_hu_equivalent: -320
  failed_lung_extraction_fallback: body_bounding_box

collision_policy:
  same_class_spatial_collision: merge_to_unique_training_voxel
  foreground_multi_group_collision: preserve_all_group_memberships
  foreground_background_collision: hard_failure
  group_loss: hard_failure

post_transfer_causal_equalization:
  inherited_from_v1_1: true
  sparse_full_grid_coordinates_regenerated: false
  sparse_full_grid_coordinates_preserved_exactly: true
  crop_coordinates_rebased_only: true
  background: unchanged

firewall:
  dense_masks_in_training_cache: prohibited
  dense_component_information: prohibited
  unknown_voxels_materialized_as_background: false

scientific_status:
  block08a_models: invalidated_before_dense_evaluation
  dense_gate_b_outcomes_opened: false
  development_split_modified: false
  final_outer_folds_modified: false
"""


write_text(
    REPO
    / "configs/preprocess_primary.yaml",
    preprocess_config,
)


data_config = """
project:
  name: CORA-Lung

dataset:
  name: COVID-19_CT_Lung_and_Infection_Segmentation
  role: primary
  expected_cases: 20

geometry:
  array_order: [z, y, x]
  target_spacing_mm: [3.0, 1.5, 1.5]
  component_connectivity: 26
  minimum_component_volume_ml: 0.1

intensity:
  normalized_range: [-1.0, 1.0]
  common_declared_window_hu: [-1250, 250]

  source_encodings:
    Coronacases:
      encoding: raw_hu
      clip_hu: [-1250, 250]

    Radiopaedia:
      encoding: documented_prewindowed_0_255
      represented_hu_window: [-1250, 250]
      source_range: [0, 255]
      hu_clipping: prohibited

splits:
  permanent_development_cases: 4
  final_outer_folds: 4
  seeds: [17, 29, 43]
"""


write_text(
    REPO
    / "configs/data_primary.yaml",
    data_config,
)


pilot_config_path = (
    REPO
    / "configs/experiment_gate_b_pilot.yaml"
)


pilot_cfg = yaml.safe_load(
    pilot_config_path.read_text(
        encoding="utf-8"
    )
)


pilot_cfg[
    "data"
][
    "cache_version"
] = "1.2"


pilot_cfg[
    "patch_sampling"
][
    "anchor_sampling"
][
    "foreground_probability"
] = 0.50


pilot_cfg[
    "patch_sampling"
][
    "anchor_sampling"
][
    "explicit_background_probability"
] = 0.25


pilot_cfg[
    "patch_sampling"
][
    "anchor_sampling"
][
    "random_probability"
] = 0.25


pilot_config_path.write_text(
    yaml.safe_dump(
        pilot_cfg,
        sort_keys=False,
    ),
    encoding="utf-8",
)


gate_b_config = """
experiment:
  name: gate_b_problem_validation_v1_2
  purpose: component_omission_vs_equal_count_pixel_sparsity
  evaluation_population: permanent_development_only
  final_outer_cv_access: prohibited
  previous_gate_b_fit_commit: "412d746db72a"
  previous_gate_b_fit_status: invalidated_before_dense_evaluation

data:
  training_cache_version: "1.2"
  intensity_preprocessing_version: "1.2"

conditions:
  - complete
  - pixel_dropout_matched_50
  - component_natural_50
  - component_fixed_50

benchmark_contrasts:
  structure_of_removal:
    control: pixel_dropout_matched_50
    omission: component_natural_50
    unique_positive_voxel_budget: matched

  fixed_budget_omission:
    reference: complete
    omission: component_fixed_50
    interpretation: separate_non_equal_budget_contrast

patch_sampling:
  patient_sampling: uniform
  foreground_stroke_probability: 0.50
  background_stroke_probability: 0.25
  uniform_crop_probability: 0.25
  dense_foreground_oversampling: prohibited

fit:
  screening_seed: 17
  epochs: 30
  microbatches_per_epoch: 100
  batch_size: 1
  gradient_accumulation: 2
  optimizer: AdamW
  learning_rate: 0.0003
  weight_decay: 0.0001
  amp: true
  checkpoint_selection: final_epoch_only

model:
  family: CORALungResidualUNet
  widths: [16, 32, 64, 128]
  embedding_dimension: 64

loss:
  partial_bce_weight: 1.0
  partial_dice_weight: 1.0
  unknown_label: -1

firewall:
  trainer_boundary_guard_every_microbatch: true
  dense_mask_access_during_fit: prohibited
  dense_checksum_denylist: required

gate_b:
  dense_development_evaluation_allowed_only_after_fit_freeze: true
  independent_generalization_claim: prohibited
"""


write_text(
    REPO
    / "configs/gate_b_problem_validation.yaml",
    gate_b_config,
)


print(
    "✓ Preprocessing configuration        : v1.2"
)

print(
    "✓ Patch sampling                     : 0.50 / 0.25 / 0.25"
)

print(
    "✓ Gate-B fitted-condition register   : 4 protocol rows"
)


# ==========================================================================================
# 13. PROTOCOL REGRESSION TEST
# ==========================================================================================

heading(
    "STEP 7/8 — RUN v1.2 SCIENTIFIC REGRESSION TESTS"
)


sampling_test = """
import numpy as np

from cora_lung.data.pilot_dataset import (
    deterministic_patch_origin,
)


def test_gate_b_patch_source_probabilities():

    shape = (
        64,
        160,
        160,
    )

    patch = (
        48,
        128,
        128,
    )

    coords = np.asarray(
        [
            [20, 80, 80],
            [40, 100, 100],
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

    counts = {
        "foreground": 0,
        "background": 0,
        "random": 0,
    }

    n = 10000

    for i in range(n):

        _, source = deterministic_patch_origin(
            shape,
            patch,
            coords,
            labels,
            seed=17,
            sample_index=i,
        )

        counts[source] += 1

    proportions = {
        key:
            value / n
        for key, value
        in counts.items()
    }

    assert abs(
        proportions["foreground"]
        - 0.50
    ) < 0.025

    assert abs(
        proportions["background"]
        - 0.25
    ) < 0.025

    assert abs(
        proportions["random"]
        - 0.25
    ) < 0.025
"""


write_text(
    REPO
    / "tests/test_gate_b_sampling_protocol.py",
    sampling_test,
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
    "tests/test_gate_b_sampling_protocol.py",
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
        "v1.2 regression test suite FAILED."
    )


print(
    "✓ Full v1.2 regression suite         : PASS"
)


# ==========================================================================================
# 14. SAVE AUDITS / FIGURE
# ==========================================================================================

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


encoding_df.to_csv(
    manifest_dir
    / "primary_intensity_encoding_v1_2.csv",
    index=False,
)


crop_df.to_csv(
    manifest_dir
    / "training_crop_v1_2_audit.csv",
    index=False,
)


annotation_audit_df.to_csv(
    manifest_dir
    / "training_annotation_rebase_v1_2_audit.csv",
    index=False,
)


pair_df.to_csv(
    manifest_dir
    / "training_budget_v1_2_audit.csv",
    index=False,
)


# ------------------------------------------------------------------------------------------
# Figure — corrected intensity provenance
# ------------------------------------------------------------------------------------------

figure_data = (
    encoding_df.groupby(
        "source_origin",
        as_index=False,
    )
    .agg(
        source_min=(
            "source_min",
            "median",
        ),

        source_max=(
            "source_max",
            "median",
        ),

        normalized_min=(
            "normalized_min",
            "median",
        ),

        normalized_max=(
            "normalized_max",
            "median",
        ),
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
        figure_data
    )
)


width = 0.35


ax.bar(
    x
    - width
    / 2,
    figure_data[
        "normalized_min"
    ],
    width,
    label="Median normalized minimum",
)


ax.bar(
    x
    + width
    / 2,
    figure_data[
        "normalized_max"
    ],
    width,
    label="Median normalized maximum",
)


ax.set_xticks(
    x
)


ax.set_xticklabels(
    figure_data[
        "source_origin"
    ].tolist(),
    fontweight="bold",
)


ax.set_ylim(
    -1.15,
    1.15,
)


ax.set_ylabel(
    "Corrected Cached Intensity",
    fontweight="bold",
)


ax.set_title(
    "CORA-Lung Source-Specific CT Normalization After Encoding Audit\n"
    "Raw-HU Coronacases and Documented Prewindowed Radiopaedia Volumes",
    fontweight="bold",
)


ax.grid(
    axis="y",
    alpha=0.25,
)


ax.legend()


fig.tight_layout()


fig.savefig(
    figure_dir
    / "fig12_source_specific_intensity_correction.png",
    dpi=600,
    bbox_inches="tight",
)


fig.savefig(
    figure_dir
    / "fig12_source_specific_intensity_correction.pdf",
    bbox_inches="tight",
)


plt.close(
    fig
)


crop_changed_count = int(
    crop_df[
        "crop_changed"
    ].sum()
)


crop_method_counts = {
    str(
        key
    ):
        int(
            value
        )
    for key, value
    in crop_df[
        "crop_method"
    ].value_counts().items()
}


audit = {
    "project":
        "CORA-Lung",

    "block":
        "08C",

    "status":
        "PASS",

    "generated_at_utc":
        NOW_ISO,

    "reason":
        (
            "CT-only pre-outcome QC confirmed mixed intensity encodings "
            "and material mismatch with uniform v1.1 preprocessing."
        ),

    "dense_development_outcomes_opened":
        False,

    "dense_mask_arrays_accessed":
        0,

    "block08a_models": {
        "commit":
            head,

        "status":
            "INVALIDATED_BEFORE_DENSE_EVALUATION",

        "historical_optimizer_steps":
            7500,

        "eligible_for_gate_b_scoring":
            False,
    },

    "preprocessing_v1_2": {
        "coronacases":
            "raw_hu_clip_-1250_250_then_normalize",

        "radiopaedia":
            "documented_prewindowed_0_255_direct_normalization",

        "common_output_range":
            [
                -1,
                1,
            ],

        "target_spacing_zyx_mm":
            TARGET_SPACING_ZYX.tolist(),

        "image_only_crop":
            True,

        "crop_space":
            "common_normalized_intensity",

        "crop_margin_mm":
            CROP_MARGIN_MM,

        "cases":
            20,

        "annotations":
            260,

        "sparse_points_lost":
            sparse_points_lost,

        "membership_points_lost":
            membership_points_lost,

        "max_resampled_affine_difference":
            max_affine_difference,

        "max_spacing_difference":
            max_spacing_difference,

        "crop_changed_cases":
            crop_changed_count,

        "crop_method_counts":
            crop_method_counts,
    },

    "sampling_protocol": {
        "foreground":
            FG_PATCH_PROBABILITY,

        "background":
            BG_PATCH_PROBABILITY,

        "uniform_crop":
            RANDOM_PATCH_PROBABILITY,
    },

    "gate_b_screening": {
        "seed":
            17,

        "epochs":
            30,

        "microbatches_per_epoch":
            100,

        "conditions":
            [
                "complete",
                "pixel_dropout_matched_50",
                "component_natural_50",
                "component_fixed_50",
            ],

        "dense_evaluation":
            "NOT_RUN",
    },
}


write_json(
    audit_dir
    / "block08c_preprocessing_rebuild_v1_2.json",
    audit,
)


protocol_note = """
# CORA-Lung preprocessing correction v1.2

A CT-only audit was performed before opening any Gate-B dense development
outcome.

The audit confirmed two documented intensity encodings.

## Coronacases

These scans use raw HU-like intensities.

The v1.2 rule is:

1. clip to [-1250,250] HU;
2. resample on the frozen physical grid;
3. linearly map to [-1,1].

## Radiopaedia

These scans were already windowed to the same represented HU interval and
stored in [0,255].

The v1.2 rule therefore does not apply HU clipping or claim to reconstruct
original HU.

After resampling, values are mapped with:

    2*x/255 - 1

## Crop

The primary crop remains image-only.

Intensity and morphology thresholds now operate in the common normalized
space rather than assuming raw HU input.

A failed normalized-space lung extraction falls back to the image-derived
body bounding box.

No released lung mask or lesion mask repairs the crop.

## Sparse supervision

The physical/resampled image grid did not change.

Previously equalized sparse annotations therefore retain the same full-grid
coordinates. Only crop-relative coordinates are rebased when the corrected
image-derived crop changes.

No sparse label is regenerated from a dense mask.

## Invalidated pilot

The Block-08A checkpoints were produced with the superseded v1.1 intensity
rule and the superseded 0.60/0.20/0.20 patch-source proportions.

They were invalidated before dense Gate-B outcomes were opened and cannot be
used in Gate-B tables, figures or decisions.
"""


write_text(
    docs_dir
    / "preprocessing_v1_2_correction.md",
    protocol_note,
)


# ==========================================================================================
# 15. CAPTURE EXECUTED SOURCE
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
        "CORA-LUNG — CODE BLOCK 08C"
        in cell
    ):

        source_path = (
            REPO
            / "scripts/code_blocks/"
            "block08c_preprocessing_rebuild_v1_2.py"
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


# ==========================================================================================
# 16. UPDATE PROJECT STATE
# ==========================================================================================

state.update(
    {
        "last_attempted_block":
            "08C",

        "last_completed_block":
            "08C",

        "last_completed_block_name":
            "preprocessing_rebuild_v1_2",

        "current_stage":
            "gate_b_corrected_cache_ready",

        "current_gate":
            "B",

        "gate_b":
            "IN_PROGRESS_REBUILD",

        "preprocess_version":
            "1.2",

        "training_cache_root":
            str(
                TARGET_CACHE_ROOT
            ),

        "training_grid_cache":
            "PASS",

        "source_specific_intensity_encoding":
            "PASS",

        "normalized_space_crop":
            "PASS",

        "gate_b_sampling_protocol":
            "PASS_0.50_0.25_0.25",

        "block08a_fit_validity":
            "INVALIDATED_BEFORE_DENSE_EVALUATION",

        "invalidated_gate_b_optimizer_steps":
            7500,

        "valid_gate_b_optimizer_steps":
            0,

        "dense_development_evaluation":
            "NOT_RUN",

        "gate_b_outcomes_opened":
            False,

        "pilot_training_authorized":
            False,

        "training_authorized":
            False,

        "next_action":
            (
                "Run corrected Gate-B screening fits on v1.2 cache "
                "using seed 17 and 0.50/0.25/0.25 patch sampling."
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
# 17. REPOSITORY MANIFEST
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
            "08C",

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
# 18. COMMIT + PUSH
# ==========================================================================================

heading(
    "STEP 8/8 — COMMIT v1.2 CORRECTION"
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
    "/tmp/cora_git_askpass_block08c.sh"
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


git_paths = [
    "PROJECT_STATE.json",
    "REPOSITORY_MANIFEST.json",
    "configs/preprocess_primary.yaml",
    "configs/data_primary.yaml",
    "configs/experiment_gate_b_pilot.yaml",
    "configs/gate_b_problem_validation.yaml",
    "src/cora_lung/data/pilot_dataset.py",
    "tests/test_gate_b_sampling_protocol.py",
    "data/manifests/primary_intensity_encoding_v1_2.csv",
    "data/manifests/training_crop_v1_2_audit.csv",
    "data/manifests/training_annotation_rebase_v1_2_audit.csv",
    "data/manifests/training_budget_v1_2_audit.csv",
    "experiments/audits/block08c_preprocessing_rebuild_v1_2.json",
    "docs/preprocessing_v1_2_correction.md",
    "figures/audit/fig12_source_specific_intensity_correction.png",
    "figures/audit/fig12_source_specific_intensity_correction.pdf",
]


source_repo_path = (
    REPO
    / "scripts/code_blocks/"
    "block08c_preprocessing_rebuild_v1_2.py"
)


if source_repo_path.exists():

    git_paths.append(
        "scripts/code_blocks/"
        "block08c_preprocessing_rebuild_v1_2.py"
    )


sh(
    [
        "git",
        "add",
        *git_paths,
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
        "No Block-08C changes to commit."
    )


print(
    status
)


sh(
    [
        "git",
        "commit",
        "-m",
        "fix: rebuild source-specific CT cache v1.2 before Gate-B scoring",
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


# ==========================================================================================
# 19. FINAL REPORT
# ==========================================================================================

print(
    "\n"
    + "=" * 118
)

print(
    "CORA-LUNG CODE BLOCK 08C — FINAL v1.2 REBUILD REPORT"
)

print(
    "=" * 118
)


print(
    "Source primary CTs                  : 20"
)

print(
    "Coronacases raw-HU                  :",
    int(
        (
            encoding_df[
                "source_origin"
            ]
            == "Coronacases"
        ).sum()
    ),
)

print(
    "Radiopaedia prewindowed             :",
    int(
        (
            encoding_df[
                "source_origin"
            ]
            == "Radiopaedia"
        ).sum()
    ),
)

print(
    "Common represented HU window        : [-1250, 250]"
)

print(
    "Output intensity range              : [-1, 1]"
)

print(
    "Radiopaedia HU clipping             : PROHIBITED"
)

print(
    "Image-derived normalized-space crop : PASS"
)

print(
    "Crop changed cases                  :",
    crop_changed_count,
    "/20",
)

print(
    "Crop method counts                  :",
    crop_method_counts,
)

print(
    "Maximum affine difference           :",
    "{:.3e}".format(
        max_affine_difference
    ),
)

print(
    "Maximum spacing difference          :",
    "{:.3e}".format(
        max_spacing_difference
    ),
)

print(
    "Sparse direct points lost           :",
    sparse_points_lost,
)

print(
    "Sparse membership points lost       :",
    membership_points_lost,
)

print(
    "Annotations rebased                 : 260/260"
)

print(
    "Natural vs matched-pixel budgets    : PASS 20/20"
)

print(
    "Fixed-pair budgets                  : PASS 20/20"
)

print(
    "Training-cache firewall             : PASS"
)

print(
    "Patch sampling protocol             : 0.50 / 0.25 / 0.25"
)

print(
    "Regression/unit tests               : PASS"
)

print(
    "Dense lesion arrays accessed        : 0"
)

print(
    "Dense lung arrays accessed          : 0"
)

print(
    "Dense Gate-B outcomes opened        : NO"
)

print(
    "Block-08A checkpoint status         : INVALIDATED"
)

print(
    "Invalidated historical steps        : 7500"
)

print(
    "Valid corrected Gate-B steps        : 0"
)

print(
    "New cache root                      :",
    TARGET_CACHE_ROOT,
)

print(
    "Exact Block-08C source captured     :",
    source_capture,
)

print(
    "Starting commit                     :",
    head[:12],
)

print(
    "Current commit                      :",
    current_commit[:12],
)

print(
    "GitHub synchronization              : PASS"
)

print()
print(
    "NEXT: Send me this COMPLETE report. "
    "Do NOT run dense evaluation and do NOT rerun Block 08A."
)

print(
    "=" * 118
)