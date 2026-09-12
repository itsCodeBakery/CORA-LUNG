# ==========================================================================================
# CORA-LUNG — CODE BLOCK 06
# Firewall-Safe Training-Grid Cache + Physical Sparse-Label Transfer Audit
#
# PURPOSE
#   Build the actual image/sparse-supervision representation that future models will read.
#
# FROZEN TRAINING GEOMETRY
#   Array order        : (z, y, x)
#   Spacing            : (3.0, 1.5, 1.5) mm
#   NIfTI world grid   : corresponding (x, y, z) = (1.5, 1.5, 3.0) mm
#   HU clipping        : [-1000, 400]
#   Intensity scaling  : [-1, 1]
#   Crop               : CT-image-derived only
#   Crop margin        : 15 mm
#
# DENSE-LABEL FIREWALL
#   This block DOES NOT open:
#       infection masks
#       lung masks
#       combined masks
#       component tables
#
#   Inputs:
#       CT
#       native sparse coordinates from Block 05
#       frozen geometry/split metadata
#
# SPARSE TRANSFER
#   world coordinate -> resampled (x,y,z) -> round nearest voxel -> (z,y,x) -> crop
#
# COLLISIONS
#   same-class spatial collisions:
#       merged for direct partial supervision
#
#   foreground collisions from different stroke groups:
#       one supervision voxel, but all group memberships are retained separately
#
#   foreground/background collision:
#       HARD FAILURE
#
# CRITICAL SCIENTIFIC AUDITS
#   * zero sparse points outside the resampled volume
#   * zero sparse points outside the image-only crop
#   * zero FG/BG conflicts
#   * 100% foreground-group survival
#   * physical round-trip error within nearest-voxel theoretical bound
#   * exact paired background realization after transfer
#   * post-transfer natural-vs-pixel budget equivalence
#   * post-transfer component-fixed-vs-complete-fixed budget equivalence
#
# IMPORTANT
#   If native budget equivalence is broken by voxel collisions after resampling,
#   this block will record REVIEW_REQUIRED and training will remain unauthorized.
#
# NO MODEL TRAINING.
# ==========================================================================================

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import subprocess
import hashlib
import json
import os
import sys
import gc
import math
import textwrap

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from kaggle_secrets import UserSecretsClient

try:
    import nibabel as nib
    from nibabel.processing import resample_to_output
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "nibabel"]
    )
    import nibabel as nib
    from nibabel.processing import resample_to_output


# ==========================================================================================
# 0. LOCKED SETTINGS
# ==========================================================================================

PROJECT = "CORA-Lung"

BLOCK_ID = "06"
BLOCK_NAME = "firewall_safe_training_grid_cache"

PREPROCESS_VERSION = "1.0"

GITHUB_OWNER = "itsCodeBakery"
GITHUB_REPO = "CORA-LUNG"
REMOTE_URL = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}.git"

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")

REPO = WORK / GITHUB_REPO

WEAK_ROOT = WORK / "cora_weak_train_v1"
CACHE_ROOT = WORK / "cora_train_cache_v1"

IMAGE_DIR = CACHE_ROOT / "images"
ANNOTATION_DIR = CACHE_ROOT / "annotations"

WEAK_MANIFEST = WEAK_ROOT / "manifest.csv"

TARGET_SPACING_XYZ = np.asarray(
    [1.5, 1.5, 3.0],
    dtype=np.float64,
)

TARGET_SPACING_ZYX = np.asarray(
    [3.0, 1.5, 1.5],
    dtype=np.float64,
)

HU_MIN = -1000.0
HU_MAX = 400.0

CROP_MARGIN_MM = 15.0

BODY_THRESHOLD_HU = -600.0
LUNG_AIR_THRESHOLD_HU = -320.0

MIN_LUNG_AIR_COMPONENT_ML = 50.0
MIN_TOTAL_LUNG_AIR_ML = 150.0

NOW = datetime.now(timezone.utc)
NOW_ISO = NOW.strftime("%Y-%m-%dT%H:%M:%SZ")


# ==========================================================================================
# 1. HELPERS
# ==========================================================================================

def heading(text):
    print("\n" + "=" * 116)
    print(text)
    print("=" * 116)


def sh(cmd, cwd=None, env=None, check=True):
    result = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if check and result.returncode != 0:
        raise RuntimeError(
            "COMMAND FAILED\n"
            + " ".join(map(str, cmd))
            + "\n\nSTDOUT:\n"
            + (result.stdout or "")
            + "\nSTDERR:\n"
            + (result.stderr or "")
        )

    return result


def write_json(path, obj):
    path = Path(path)

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


def write_text(path, text):
    path = Path(path)

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


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda: f.read(chunk_size),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def semantic_hash(*arrays):
    h = hashlib.sha256()

    for arr in arrays:
        arr = np.ascontiguousarray(
            arr
        )

        h.update(
            str(
                arr.dtype
            ).encode()
        )

        h.update(
            str(
                arr.shape
            ).encode()
        )

        h.update(
            arr.tobytes()
        )

    return h.hexdigest()


def human_bytes(n):
    n = float(n)

    for unit in [
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ]:
        if n < 1024.0:
            return f"{n:.2f} {unit}"

        n /= 1024.0

    return f"{n:.2f} PB"


def locate_dataset():
    candidates = [
        INPUT / "covid19-ct-scans",
        INPUT
        / "datasets"
        / "andrewmvd"
        / "covid19-ct-scans",
    ]

    for p in candidates:
        if p.exists():
            return p.resolve()

    for p in INPUT.rglob(
        "metadata.csv"
    ):
        try:
            df = pd.read_csv(
                p,
                nrows=2,
            )

            required = {
                "ct_scan",
                "infection_mask",
                "lung_mask",
                "lung_and_infection_mask",
            }

            if required.issubset(
                df.columns
            ):
                return p.parent.resolve()

        except Exception:
            pass

    raise RuntimeError(
        "Primary dataset could not be located."
    )


def bbox_from_mask(
    mask,
    spacing_zyx,
    margin_mm,
):
    coords = np.argwhere(
        mask
    )

    if len(coords) == 0:
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
            dtype=float,
        )
    ).astype(
        np.int32
    )

    lo = np.maximum(
        0,
        lo - margin_vox,
    )

    hi = np.minimum(
        np.asarray(
            mask.shape,
            dtype=np.int32,
        ),
        hi + margin_vox,
    )

    return lo, hi


def largest_2d_component(
    mask2d,
):
    labels, n = ndi.label(
        mask2d,
        structure=ndi.generate_binary_structure(
            2,
            2,
        ),
    )

    if n == 0:
        return np.zeros_like(
            mask2d,
            dtype=bool,
        )

    counts = np.bincount(
        labels.ravel()
    )

    counts[0] = 0

    return (
        labels
        == int(
            np.argmax(counts)
        )
    )


# ==========================================================================================
# 2. IMAGE-ONLY THORACIC CROP
# ==========================================================================================

def derive_image_only_crop(
    hu_zyx,
    spacing_zyx,
):
    """
    Uses CT intensity only.

    No released lung mask, infection mask, scribble, component map or
    outcome is consulted.
    """

    body_seed = (
        hu_zyx
        > BODY_THRESHOLD_HU
    )

    body_envelope = np.zeros_like(
        body_seed,
        dtype=bool,
    )

    # Per-axial-slice body envelope.
    for z in range(
        hu_zyx.shape[0]
    ):
        sl = body_seed[z]

        if not sl.any():
            continue

        sl = ndi.binary_closing(
            sl,
            structure=np.ones(
                (3, 3),
                dtype=bool,
            ),
        )

        sl = largest_2d_component(
            sl
        )

        if not sl.any():
            continue

        sl = ndi.binary_fill_holes(
            sl
        )

        body_envelope[
            z
        ] = sl

    if not body_envelope.any():
        full_lo = np.zeros(
            3,
            dtype=np.int32,
        )

        full_hi = np.asarray(
            hu_zyx.shape,
            dtype=np.int32,
        )

        return {
            "lo_zyx":
                full_lo,

            "hi_zyx":
                full_hi,

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
            hu_zyx
            < LUNG_AIR_THRESHOLD_HU
        )
        & body_envelope
    )

    labels, n = ndi.label(
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

    if n > 0:
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
                float
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
                key=lambda c:
                    counts[
                        int(c)
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
            and lung_span[0]
            >= max(
                8,
                int(
                    0.30
                    * body_span[0]
                ),
            )
            and lung_span[1]
            >= max(
                16,
                int(
                    0.35
                    * body_span[1]
                ),
            )
            and lung_span[2]
            >= max(
                16,
                int(
                    0.35
                    * body_span[2]
                ),
            )
        )

    else:
        plausible = False

    if plausible:
        lo = lung_lo
        hi = lung_hi
        method = (
            "internal_air_lung_bbox"
        )

    else:
        lo = body_lo
        hi = body_hi
        method = (
            "body_envelope_fallback"
        )

    return {
        "lo_zyx":
            lo.astype(
                np.int32
            ),

        "hi_zyx":
            hi.astype(
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
            float(
                lung_air_ml
            ),
    }


# ==========================================================================================
# 3. SPARSE COLLISION RESOLUTION
# ==========================================================================================

def resolve_sparse_training_rows(
    coords_zyx,
    labels,
    group_ids,
):
    """
    Direct supervision:
        one binary target per spatial voxel.

    Replay membership:
        foreground group memberships are preserved separately, allowing
        multiple FG groups to map to the same training voxel without losing
        group identity.
    """

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

    class_map = defaultdict(
        set
    )

    for coord, label in zip(
        coords_zyx,
        labels,
    ):
        class_map[
            tuple(
                int(x)
                for x in coord
            )
        ].add(
            int(
                label
            )
        )

    conflict_coords = [
        coord
        for coord, values
        in class_map.items()
        if len(
            values
        )
        > 1
    ]

    if conflict_coords:
        return {
            "conflict_coords":
                conflict_coords,
        }

    ordered_coords = sorted(
        class_map.keys()
    )

    supervision_coords = np.asarray(
        ordered_coords,
        dtype=np.int32,
    )

    supervision_labels = np.asarray(
        [
            next(
                iter(
                    class_map[
                        c
                    ]
                )
            )
            for c in ordered_coords
        ],
        dtype=np.int8,
    )

    fg_memberships = set()

    for coord, label, group_id in zip(
        coords_zyx,
        labels,
        group_ids,
    ):
        if int(
            label
        ) != 1:
            continue

        fg_memberships.add(
            (
                int(
                    coord[0]
                ),
                int(
                    coord[1]
                ),
                int(
                    coord[2]
                ),
                int(
                    group_id
                ),
            )
        )

    fg_memberships = sorted(
        fg_memberships
    )

    if fg_memberships:
        fg_memberships = np.asarray(
            fg_memberships,
            dtype=np.int32,
        )

        fg_membership_coords = (
            fg_memberships[
                :,
                :3
            ]
        )

        fg_membership_groups = (
            fg_memberships[
                :,
                3
            ]
        )

    else:
        fg_membership_coords = np.empty(
            (0, 3),
            dtype=np.int32,
        )

        fg_membership_groups = np.empty(
            (0,),
            dtype=np.int32,
        )

    return {
        "conflict_coords":
            [],

        "supervision_voxel_zyx":
            supervision_coords,

        "supervision_label":
            supervision_labels,

        "fg_membership_voxel_zyx":
            fg_membership_coords,

        "fg_membership_group_id":
            fg_membership_groups,
    }


# ==========================================================================================
# 4. VERIFY BLOCK-05 STATE + AUTHENTICATION
# ==========================================================================================

heading(
    "CORA-LUNG :: CODE BLOCK 06 :: TRAINING-GRID CACHE"
)

if not (
    REPO
    / ".git"
).exists():
    raise RuntimeError(
        "CORA-LUNG repository missing."
    )

state = json.loads(
    (
        REPO
        / "PROJECT_STATE.json"
    ).read_text(
        encoding="utf-8"
    )
)

if (
    state.get(
        "last_completed_block"
    )
    != "05"
    or state.get(
        "weak_label_generator"
    )
    != "PASS"
    or state.get(
        "dense_label_firewall"
    )
    != "PASS"
):
    raise RuntimeError(
        "Block 05 is not frozen as PASS."
    )

if not WEAK_MANIFEST.exists():
    raise RuntimeError(
        "Block-05 weak export is missing from this Kaggle session."
    )

weak_manifest = pd.read_csv(
    WEAK_MANIFEST
)

if len(
    weak_manifest
) != 260:
    raise RuntimeError(
        f"Expected 260 weak annotations; found {len(weak_manifest)}."
    )

volume_registry = pd.read_csv(
    REPO
    / "data/splits/primary_volume_split_registry.csv"
)

if len(
    volume_registry
) != 20:
    raise RuntimeError(
        "Frozen primary split registry is invalid."
    )

dataset_root = locate_dataset()

secrets = UserSecretsClient()

token = secrets.get_secret(
    "pushCora"
)

if not token:
    raise RuntimeError(
        "Kaggle secret 'pushCora' unavailable."
    )

token = token.strip()

askpass = Path(
    "/tmp/cora_git_askpass_block06.sh"
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

starting_commit = sh(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    cwd=REPO,
).stdout.strip()

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
        "Repository is not clean before Block 06."
    )

print(
    f"✓ Starting commit      : {starting_commit[:12]}"
)

print(
    f"✓ Primary CT volumes   : {len(volume_registry)}"
)

print(
    f"✓ Weak annotations     : {len(weak_manifest)}"
)

print(
    f"✓ Dataset root         : {dataset_root}"
)


# ==========================================================================================
# 5. FREEZE PREPROCESSING CONFIG
# ==========================================================================================

heading(
    "STEP 1/10 — FREEZE PREPROCESSING CONTRACT"
)

config_text = f"""
preprocessing:
  version: "{PREPROCESS_VERSION}"

geometry:
  internal_array_order: [z, y, x]
  target_spacing_zyx_mm: [3.0, 1.5, 1.5]
  nifti_resample_spacing_xyz_mm: [1.5, 1.5, 3.0]
  canonical_orientation_before_resample: true
  ct_interpolation: linear
  sparse_coordinate_transfer: physical_world_coordinate_nearest_voxel

intensity:
  hu_clip: [-1000, 400]
  output_range: [-1.0, 1.0]
  cache_dtype: float16

crop:
  dense_masks_used: false
  scribbles_used: false
  ct_only: true
  body_threshold_hu: {BODY_THRESHOLD_HU}
  lung_air_threshold_hu: {LUNG_AIR_THRESHOLD_HU}
  minimum_lung_air_component_ml: {MIN_LUNG_AIR_COMPONENT_ML}
  minimum_total_lung_air_ml: {MIN_TOTAL_LUNG_AIR_ML}
  physical_margin_mm: {CROP_MARGIN_MM}
  fallback: body_envelope

sparse_collision_policy:
  same_class_spatial_collision: merge_for_direct_supervision
  foreground_multi_group_collision: preserve_all_group_memberships
  foreground_background_collision: hard_failure
  out_of_volume_coordinate: hard_failure
  out_of_crop_coordinate: hard_failure
  group_loss: hard_failure

causal_budget_audit:
  evaluate_after_training_grid_transfer: true
  do_not_silently_rebalance: true
"""

write_text(
    REPO
    / "configs/preprocess_primary.yaml",
    config_text,
)

print(
    "✓ configs/preprocess_primary.yaml"
)


# ==========================================================================================
# 6. INITIALIZE CLEAN TRAINING CACHE
# ==========================================================================================

heading(
    "STEP 2/10 — INITIALIZE RESTARTABLE LOCAL CACHE"
)

if CACHE_ROOT.exists():
    import shutil

    shutil.rmtree(
        CACHE_ROOT
    )

IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

ANNOTATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print(
    f"✓ Cache root: {CACHE_ROOT}"
)


# ==========================================================================================
# 7. PREPROCESS 20 CT VOLUMES
# ==========================================================================================

heading(
    "STEP 3/10 — RESAMPLE, NORMALIZE AND IMAGE-CROP CT VOLUMES"
)

case_rows = []
case_geometry = {}

for _, row in tqdm(
    volume_registry.iterrows(),
    total=len(
        volume_registry
    ),
    desc="Preprocessing CT volumes",
    unit="volume",
):
    case_id = str(
        row[
            "case_id"
        ]
    )

    ct_relative = str(
        row[
            "ct_scan"
        ]
    )

    ct_path = (
        dataset_root
        / ct_relative
    )

    if not ct_path.exists():
        raise RuntimeError(
            f"CT missing: {ct_path}"
        )

    native_img = nib.load(
        str(
            ct_path
        ),
        mmap=True,
    )

    native_hu = np.asarray(
        native_img.dataobj,
        dtype=np.float32,
    )

    native_shape_xyz = tuple(
        int(x)
        for x in native_hu.shape[
            :3
        ]
    )

    native_hu = np.clip(
        native_hu,
        HU_MIN,
        HU_MAX,
    )

    header = native_img.header.copy()

    header.set_data_dtype(
        np.float32
    )

    clipped_img = nib.Nifti1Image(
        native_hu,
        native_img.affine,
        header=header,
    )

    canonical_img = nib.as_closest_canonical(
        clipped_img
    )

    resampled_img = resample_to_output(
        canonical_img,
        voxel_sizes=tuple(
            TARGET_SPACING_XYZ.tolist()
        ),
        order=1,
        mode="constant",
        cval=HU_MIN,
    )

    hu_xyz = np.asarray(
        resampled_img.dataobj,
        dtype=np.float32,
    )

    actual_spacing_xyz = np.asarray(
        resampled_img.header.get_zooms()[
            :3
        ],
        dtype=np.float64,
    )

    if not np.allclose(
        actual_spacing_xyz,
        TARGET_SPACING_XYZ,
        atol=1e-4,
        rtol=1e-4,
    ):
        raise RuntimeError(
            f"Target spacing mismatch for {case_id}: "
            f"{actual_spacing_xyz}"
        )

    hu_zyx = np.transpose(
        hu_xyz,
        (
            2,
            1,
            0,
        ),
    )

    full_shape_zyx = np.asarray(
        hu_zyx.shape,
        dtype=np.int32,
    )

    crop_info = derive_image_only_crop(
        hu_zyx,
        TARGET_SPACING_ZYX,
    )

    crop_lo = crop_info[
        "lo_zyx"
    ]

    crop_hi = crop_info[
        "hi_zyx"
    ]

    crop_slices = tuple(
        slice(
            int(
                crop_lo[a]
            ),
            int(
                crop_hi[a]
            ),
        )
        for a in range(3)
    )

    cropped_hu = hu_zyx[
        crop_slices
    ]

    if min(
        cropped_hu.shape
    ) <= 0:
        raise RuntimeError(
            f"Invalid crop for {case_id}."
        )

    normalized = (
        2.0
        * (
            cropped_hu
            - HU_MIN
        )
        / (
            HU_MAX
            - HU_MIN
        )
        - 1.0
    )

    normalized = np.clip(
        normalized,
        -1.0,
        1.0,
    ).astype(
        np.float16
    )

    image_file = (
        IMAGE_DIR
        / f"{case_id}.npz"
    )

    affine_xyz = np.asarray(
        resampled_img.affine,
        dtype=np.float64,
    )

    np.savez_compressed(
        image_file,
        ct_zyx=normalized,
        spacing_zyx_mm=TARGET_SPACING_ZYX.astype(
            np.float32
        ),
        resampled_affine_xyz=affine_xyz,
        crop_origin_zyx=crop_lo.astype(
            np.int32
        ),
        full_shape_zyx=full_shape_zyx,
        crop_shape_zyx=np.asarray(
            normalized.shape,
            dtype=np.int32,
        ),
        hu_window=np.asarray(
            [
                HU_MIN,
                HU_MAX,
            ],
            dtype=np.float32,
        ),
    )

    image_semantic_sha = semantic_hash(
        normalized,
        TARGET_SPACING_ZYX.astype(
            np.float32
        ),
        affine_xyz,
        crop_lo.astype(
            np.int32
        ),
        full_shape_zyx,
    )

    full_voxels = int(
        np.prod(
            full_shape_zyx
        )
    )

    crop_voxels = int(
        np.prod(
            normalized.shape
        )
    )

    crop_fraction = (
        crop_voxels
        / full_voxels
    )

    case_geometry[
        case_id
    ] = {
        "affine_xyz":
            affine_xyz,

        "inverse_affine_xyz":
            np.linalg.inv(
                affine_xyz
            ),

        "full_shape_zyx":
            full_shape_zyx,

        "crop_lo_zyx":
            crop_lo,

        "crop_hi_zyx":
            crop_hi,

        "crop_shape_zyx":
            np.asarray(
                normalized.shape,
                dtype=np.int32,
            ),

        "image_file":
            image_file,
    }

    case_rows.append(
        {
            "case_id":
                case_id,

            "source_subject_key":
                str(
                    row[
                        "source_subject_key"
                    ]
                ),

            "split_role":
                str(
                    row[
                        "role"
                    ]
                ),

            "outer_fold":
                row[
                    "outer_fold"
                ],

            "native_shape_xyz":
                "x".join(
                    map(
                        str,
                        native_shape_xyz,
                    )
                ),

            "training_full_shape_zyx":
                "x".join(
                    map(
                        str,
                        full_shape_zyx.tolist(),
                    )
                ),

            "training_crop_shape_zyx":
                "x".join(
                    map(
                        str,
                        normalized.shape,
                    )
                ),

            "crop_method":
                crop_info[
                    "method"
                ],

            "crop_fraction_of_resampled_volume":
                float(
                    crop_fraction
                ),

            "image_derived_lung_air_ml":
                float(
                    crop_info[
                        "lung_air_ml"
                    ]
                ),

            "image_cache_bytes":
                int(
                    image_file.stat().st_size
                ),

            "image_semantic_sha256":
                image_semantic_sha,

            "preprocess_version":
                PREPROCESS_VERSION,
        }
    )

    del native_hu
    del hu_xyz
    del hu_zyx
    del cropped_hu
    del normalized
    del clipped_img
    del canonical_img
    del resampled_img

    gc.collect()


case_df = pd.DataFrame(
    case_rows
)

print(
    "\nCrop method distribution:"
)

print(
    case_df[
        "crop_method"
    ].value_counts().to_string()
)

print(
    "\nMedian retained resampled volume fraction: "
    f"{case_df['crop_fraction_of_resampled_volume'].median():.3f}"
)


# ==========================================================================================
# 8. TRANSFER ALL 260 SPARSE ANNOTATIONS
# ==========================================================================================

heading(
    "STEP 4/10 — TRANSFER SPARSE SUPERVISION THROUGH PHYSICAL COORDINATES"
)

transfer_rows = []

critical_conflicts = []
out_of_full_failures = []
out_of_crop_failures = []
group_loss_failures = []

nearest_voxel_half_diagonal_mm = float(
    0.5
    * np.linalg.norm(
        TARGET_SPACING_XYZ
    )
)

for _, row in tqdm(
    weak_manifest.iterrows(),
    total=len(
        weak_manifest
    ),
    desc="Transferring sparse annotations",
    unit="condition",
):
    case_id = str(
        row[
            "case_id"
        ]
    )

    condition = str(
        row[
            "condition"
        ]
    )

    geom = case_geometry[
        case_id
    ]

    source_path = (
        WEAK_ROOT
        / str(
            row[
                "annotation_file"
            ]
        )
    )

    with np.load(
        source_path,
        allow_pickle=False,
    ) as data:
        world_xyz = np.asarray(
            data[
                "world_xyz_mm"
            ],
            dtype=np.float64,
        )

        native_labels = np.asarray(
            data[
                "label"
            ],
            dtype=np.int8,
        )

        native_groups = np.asarray(
            data[
                "group_id"
            ],
            dtype=np.int32,
        )

    # world -> full resampled XYZ continuous coordinates.
    full_xyz_float = nib.affines.apply_affine(
        geom[
            "inverse_affine_xyz"
        ],
        world_xyz,
    )

    full_xyz_int = np.rint(
        full_xyz_float
    ).astype(
        np.int32
    )

    full_shape_xyz = geom[
        "full_shape_zyx"
    ][
        ::-1
    ]

    inside_full = np.all(
        (
            full_xyz_int
            >= 0
        )
        & (
            full_xyz_int
            < full_shape_xyz[
                None,
                :
            ]
        ),
        axis=1,
    )

    out_full = int(
        (
            ~inside_full
        ).sum()
    )

    if out_full:
        out_of_full_failures.append(
            (
                case_id,
                condition,
                out_full,
            )
        )

    # World round-trip error at nearest training voxel.
    mapped_world = nib.affines.apply_affine(
        geom[
            "affine_xyz"
        ],
        full_xyz_int,
    )

    roundtrip_error_mm = np.linalg.norm(
        mapped_world
        - world_xyz,
        axis=1,
    )

    max_roundtrip_error_mm = float(
        roundtrip_error_mm.max()
        if len(
            roundtrip_error_mm
        )
        else 0.0
    )

    # XYZ -> ZYX.
    full_zyx = full_xyz_int[
        :,
        ::-1
    ]

    crop_zyx = (
        full_zyx
        - geom[
            "crop_lo_zyx"
        ][
            None,
            :
        ]
    )

    crop_shape = geom[
        "crop_shape_zyx"
    ]

    inside_crop = np.all(
        (
            crop_zyx
            >= 0
        )
        & (
            crop_zyx
            < crop_shape[
                None,
                :
            ]
        ),
        axis=1,
    )

    out_crop = int(
        (
            ~inside_crop
        ).sum()
    )

    if out_crop:
        out_of_crop_failures.append(
            (
                case_id,
                condition,
                out_crop,
            )
        )

    valid = (
        inside_full
        & inside_crop
    )

    transferred_coords = crop_zyx[
        valid
    ]

    transferred_labels = native_labels[
        valid
    ]

    transferred_groups = native_groups[
        valid
    ]

    resolved = resolve_sparse_training_rows(
        transferred_coords,
        transferred_labels,
        transferred_groups,
    )

    conflict_count = len(
        resolved[
            "conflict_coords"
        ]
    )

    if conflict_count:
        critical_conflicts.append(
            (
                case_id,
                condition,
                conflict_count,
            )
        )

        continue

    supervision_coords = resolved[
        "supervision_voxel_zyx"
    ]

    supervision_labels = resolved[
        "supervision_label"
    ]

    membership_coords = resolved[
        "fg_membership_voxel_zyx"
    ]

    membership_groups = resolved[
        "fg_membership_group_id"
    ]

    native_fg_groups = set(
        int(x)
        for x in np.unique(
            native_groups[
                native_labels
                == 1
            ]
        )
    )

    transferred_fg_groups = set(
        int(x)
        for x in np.unique(
            membership_groups
        )
    )

    lost_groups = sorted(
        native_fg_groups
        - transferred_fg_groups
    )

    if lost_groups:
        group_loss_failures.append(
            (
                case_id,
                condition,
                lost_groups,
            )
        )

    input_fg = int(
        (
            native_labels
            == 1
        ).sum()
    )

    input_bg = int(
        (
            native_labels
            == 0
        ).sum()
    )

    output_fg_unique = int(
        (
            supervision_labels
            == 1
        ).sum()
    )

    output_bg_unique = int(
        (
            supervision_labels
            == 0
        ).sum()
    )

    same_class_merges = int(
        len(
            transferred_labels
        )
        - len(
            supervision_labels
        )
    )

    annotation_file = (
        ANNOTATION_DIR
        / f"{case_id}__{condition}.npz"
    )

    np.savez_compressed(
        annotation_file,
        supervision_voxel_zyx=supervision_coords.astype(
            np.int32
        ),
        supervision_label=supervision_labels.astype(
            np.int8
        ),
        fg_membership_voxel_zyx=membership_coords.astype(
            np.int32
        ),
        fg_membership_group_id=membership_groups.astype(
            np.int32
        ),
    )

    ann_semantic_sha = semantic_hash(
        supervision_coords.astype(
            np.int32
        ),
        supervision_labels.astype(
            np.int8
        ),
        membership_coords.astype(
            np.int32
        ),
        membership_groups.astype(
            np.int32
        ),
    )

    bg_training_coords = supervision_coords[
        supervision_labels
        == 0
    ]

    bg_training_sha = semantic_hash(
        bg_training_coords.astype(
            np.int32
        )
    )

    transfer_rows.append(
        {
            "case_id":
                case_id,

            "source_subject_key":
                str(
                    row[
                        "source_subject_key"
                    ]
                ),

            "split_role":
                str(
                    row[
                        "split_role"
                    ]
                ),

            "outer_fold":
                row[
                    "outer_fold"
                ],

            "condition":
                condition,

            "source_generator_version":
                str(
                    row[
                        "generator_version"
                    ]
                ),

            "source_semantic_sha256":
                str(
                    row[
                        "semantic_sha256"
                    ]
                ),

            "image_file":
                f"images/{case_id}.npz",

            "annotation_file":
                (
                    f"annotations/"
                    f"{case_id}__{condition}.npz"
                ),

            "native_sparse_rows":
                int(
                    len(
                        native_labels
                    )
                ),

            "native_fg_voxels":
                input_fg,

            "native_bg_voxels":
                input_bg,

            "mapped_sparse_rows":
                int(
                    len(
                        transferred_labels
                    )
                ),

            "training_supervision_voxels":
                int(
                    len(
                        supervision_labels
                    )
                ),

            "training_fg_voxels":
                output_fg_unique,

            "training_bg_voxels":
                output_bg_unique,

            "fg_membership_rows":
                int(
                    len(
                        membership_groups
                    )
                ),

            "native_fg_groups":
                int(
                    len(
                        native_fg_groups
                    )
                ),

            "training_fg_groups":
                int(
                    len(
                        transferred_fg_groups
                    )
                ),

            "same_class_collision_merges":
                same_class_merges,

            "class_conflicts":
                conflict_count,

            "out_of_full_volume":
                out_full,

            "out_of_image_crop":
                out_crop,

            "fg_voxel_retention":
                (
                    output_fg_unique
                    / input_fg
                    if input_fg
                    else 1.0
                ),

            "bg_voxel_retention":
                (
                    output_bg_unique
                    / input_bg
                    if input_bg
                    else 1.0
                ),

            "group_survival_fraction":
                (
                    len(
                        transferred_fg_groups
                    )
                    / len(
                        native_fg_groups
                    )
                    if native_fg_groups
                    else 1.0
                ),

            "max_roundtrip_error_mm":
                max_roundtrip_error_mm,

            "background_training_sha256":
                bg_training_sha,

            "annotation_semantic_sha256":
                ann_semantic_sha,

            "preprocess_version":
                PREPROCESS_VERSION,
        }
    )


transfer_df = pd.DataFrame(
    transfer_rows
)

print(
    f"Transferred artifacts : {len(transfer_df)}/260"
)

print(
    f"FG/BG conflicts       : {len(critical_conflicts)}"
)

print(
    f"Out-of-volume cases   : {len(out_of_full_failures)}"
)

print(
    f"Out-of-crop cases     : {len(out_of_crop_failures)}"
)

print(
    f"Group-loss cases      : {len(group_loss_failures)}"
)


# ==========================================================================================
# 9. HARD GEOMETRY / FIREWALL-SAFETY AUDIT
# ==========================================================================================

heading(
    "STEP 5/10 — HARD TRANSFER CORRECTNESS AUDIT"
)

all_artifacts_transferred = (
    len(
        transfer_df
    )
    == 260
)

zero_conflicts = (
    len(
        critical_conflicts
    )
    == 0
)

zero_out_full = (
    len(
        out_of_full_failures
    )
    == 0
)

zero_out_crop = (
    len(
        out_of_crop_failures
    )
    == 0
)

all_groups_survive = (
    len(
        group_loss_failures
    )
    == 0
    and len(
        transfer_df
    )
    == 260
    and np.allclose(
        transfer_df[
            "group_survival_fraction"
        ],
        1.0,
    )
)

roundtrip_bound_pass = bool(
    len(
        transfer_df
    )
    == 260
    and (
        transfer_df[
            "max_roundtrip_error_mm"
        ]
        <= (
            nearest_voxel_half_diagonal_mm
            + 1e-3
        )
    ).all()
)

print(
    f"All 260 artifacts transferred : "
    f"{'PASS' if all_artifacts_transferred else 'FAIL'}"
)

print(
    f"FG/BG conflicts               : "
    f"{'PASS' if zero_conflicts else 'FAIL'}"
)

print(
    f"Coordinates inside volume     : "
    f"{'PASS' if zero_out_full else 'FAIL'}"
)

print(
    f"Coordinates inside crop       : "
    f"{'PASS' if zero_out_crop else 'FAIL'}"
)

print(
    f"Foreground group survival     : "
    f"{'PASS' if all_groups_survive else 'FAIL'}"
)

print(
    f"Nearest-voxel physical bound  : "
    f"{'PASS' if roundtrip_bound_pass else 'FAIL'}"
)

print(
    f"Theoretical half-diagonal     : "
    f"{nearest_voxel_half_diagonal_mm:.4f} mm"
)

if len(
    transfer_df
):
    print(
        f"Observed maximum roundtrip    : "
        f"{transfer_df['max_roundtrip_error_mm'].max():.4f} mm"
    )


# ==========================================================================================
# 10. VERIFY SAME BACKGROUND AFTER TRANSFER
# ==========================================================================================

heading(
    "STEP 6/10 — VERIFY PAIRED BACKGROUND AFTER TRAINING-GRID TRANSFER"
)

background_hash_counts = (
    transfer_df.groupby(
        "case_id"
    )[
        "background_training_sha256"
    ]
    .nunique()
)

background_transfer_exact = bool(
    len(
        background_hash_counts
    )
    == 20
    and (
        background_hash_counts
        == 1
    ).all()
)

print(
    "Identical transferred BG coordinates across conditions:"
)

print(
    "  "
    + (
        "PASS"
        if background_transfer_exact
        else "FAIL"
    )
)


# ==========================================================================================
# 11. AUDIT ACTUAL TRAINING-GRID CAUSAL BUDGETS
# ==========================================================================================

heading(
    "STEP 7/10 — AUDIT POST-TRANSFER CAUSAL BUDGET EQUIVALENCE"
)

budget_rows = []

natural_pixel_exact = True
fixed_pair_exact = True

natural_pixel_mismatches = []
fixed_pair_mismatches = []

for case_id in sorted(
    transfer_df[
        "case_id"
    ].unique()
):
    case = transfer_df[
        transfer_df[
            "case_id"
        ]
        == case_id
    ].set_index(
        "condition"
    )

    for cov in [
        75,
        50,
        25,
    ]:
        natural_name = (
            f"component_natural_{cov}"
        )

        pixel_name = (
            f"pixel_dropout_matched_{cov}"
        )

        fixed_name = (
            f"component_fixed_{cov}"
        )

        complete_fixed_name = (
            f"complete_fixed_{cov}"
        )

        n_fg = int(
            case.loc[
                natural_name,
                "training_fg_voxels",
            ]
        )

        p_fg = int(
            case.loc[
                pixel_name,
                "training_fg_voxels",
            ]
        )

        f_fg = int(
            case.loc[
                fixed_name,
                "training_fg_voxels",
            ]
        )

        c_fg = int(
            case.loc[
                complete_fixed_name,
                "training_fg_voxels",
            ]
        )

        np_match = (
            n_fg
            == p_fg
        )

        fixed_match = (
            f_fg
            == c_fg
        )

        budget_rows.append(
            {
                "case_id":
                    case_id,

                "coverage":
                    cov,

                "component_natural_training_fg":
                    n_fg,

                "pixel_dropout_training_fg":
                    p_fg,

                "natural_pixel_exact":
                    np_match,

                "component_fixed_training_fg":
                    f_fg,

                "complete_fixed_training_fg":
                    c_fg,

                "fixed_pair_exact":
                    fixed_match,
            }
        )

        if not np_match:
            natural_pixel_exact = False

            natural_pixel_mismatches.append(
                (
                    case_id,
                    cov,
                    n_fg,
                    p_fg,
                )
            )

        if not fixed_match:
            fixed_pair_exact = False

            fixed_pair_mismatches.append(
                (
                    case_id,
                    cov,
                    f_fg,
                    c_fg,
                )
            )


budget_df = pd.DataFrame(
    budget_rows
)

primary50_np = budget_df[
    budget_df[
        "coverage"
    ]
    == 50
][
    "natural_pixel_exact"
].all()

primary50_fixed = budget_df[
    budget_df[
        "coverage"
    ]
    == 50
][
    "fixed_pair_exact"
].all()

print(
    f"Natural vs pixel exact after transfer, all coverages : "
    f"{'PASS' if natural_pixel_exact else 'REVIEW REQUIRED'}"
)

print(
    f"Fixed pair exact after transfer, all coverages        : "
    f"{'PASS' if fixed_pair_exact else 'REVIEW REQUIRED'}"
)

print(
    f"PRIMARY 50% natural/pixel exact                       : "
    f"{'PASS' if primary50_np else 'REVIEW REQUIRED'}"
)

print(
    f"PRIMARY 50% fixed pair exact                          : "
    f"{'PASS' if primary50_fixed else 'REVIEW REQUIRED'}"
)

print(
    f"Natural/pixel mismatches                              : "
    f"{len(natural_pixel_mismatches)}/60"
)

print(
    f"Fixed-pair mismatches                                 : "
    f"{len(fixed_pair_mismatches)}/60"
)

if natural_pixel_mismatches:
    print(
        "\nFirst natural/pixel mismatches:"
    )

    for item in natural_pixel_mismatches[
        :12
    ]:
        print(
            " ",
            item,
        )

if fixed_pair_mismatches:
    print(
        "\nFirst fixed-pair mismatches:"
    )

    for item in fixed_pair_mismatches[
        :12
    ]:
        print(
            " ",
            item,
        )


# ==========================================================================================
# 12. BUILD TRAINING-CACHE MANIFEST + FIREWALL
# ==========================================================================================

heading(
    "STEP 8/10 — BUILD TRAINING-CACHE FIREWALL"
)

image_lookup = case_df.set_index(
    "case_id"
)

cache_manifest_rows = []

for _, row in transfer_df.iterrows():
    case_id = str(
        row[
            "case_id"
        ]
    )

    cache_manifest_rows.append(
        {
            "case_id":
                case_id,

            "source_subject_key":
                row[
                    "source_subject_key"
                ],

            "split_role":
                row[
                    "split_role"
                ],

            "outer_fold":
                row[
                    "outer_fold"
                ],

            "condition":
                row[
                    "condition"
                ],

            "image_file":
                row[
                    "image_file"
                ],

            "annotation_file":
                row[
                    "annotation_file"
                ],

            "image_semantic_sha256":
                image_lookup.loc[
                    case_id,
                    "image_semantic_sha256",
                ],

            "annotation_semantic_sha256":
                row[
                    "annotation_semantic_sha256"
                ],

            "source_annotation_semantic_sha256":
                row[
                    "source_semantic_sha256"
                ],

            "preprocess_version":
                PREPROCESS_VERSION,
        }
    )


cache_manifest_df = pd.DataFrame(
    cache_manifest_rows
)

cache_manifest_path = (
    CACHE_ROOT
    / "manifest.csv"
)

cache_manifest_df.to_csv(
    cache_manifest_path,
    index=False,
)


training_cache_firewall_source = r'''
"""Firewall for CORA-Lung model-fitting cache."""

from pathlib import Path
import numpy as np
import pandas as pd


ALLOWED_IMAGE_KEYS = {
    "ct_zyx",
    "spacing_zyx_mm",
    "resampled_affine_xyz",
    "crop_origin_zyx",
    "full_shape_zyx",
    "crop_shape_zyx",
    "hu_window",
}

ALLOWED_ANNOTATION_KEYS = {
    "supervision_voxel_zyx",
    "supervision_label",
    "fg_membership_voxel_zyx",
    "fg_membership_group_id",
}

ALLOWED_MANIFEST_COLUMNS = {
    "case_id",
    "source_subject_key",
    "split_role",
    "outer_fold",
    "condition",
    "image_file",
    "annotation_file",
    "image_semantic_sha256",
    "annotation_semantic_sha256",
    "source_annotation_semantic_sha256",
    "preprocess_version",
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
}


def _forbidden(text):
    text = str(text).lower()

    for token in FORBIDDEN_TOKENS:
        if token in text:
            raise RuntimeError(
                f"Forbidden trainer token: {token}"
            )


def validate_training_cache(root):
    root = Path(root)

    manifest = pd.read_csv(
        root / "manifest.csv"
    )

    if set(manifest.columns) != ALLOWED_MANIFEST_COLUMNS:
        raise RuntimeError(
            "Unexpected model-fitting manifest schema."
        )

    for col in manifest.columns:
        _forbidden(col)

    for _, row in manifest.iterrows():
        image_path = (
            root
            / row["image_file"]
        )

        ann_path = (
            root
            / row["annotation_file"]
        )

        _forbidden(
            row["image_file"]
        )

        _forbidden(
            row["annotation_file"]
        )

        with np.load(
            image_path,
            allow_pickle=False,
        ) as data:
            if set(
                data.files
            ) != ALLOWED_IMAGE_KEYS:
                raise RuntimeError(
                    f"Unsafe image cache keys: {image_path}"
                )

            ct = np.asarray(
                data["ct_zyx"]
            )

            if ct.ndim != 3:
                raise RuntimeError(
                    "Training CT must be 3-D."
                )

        with np.load(
            ann_path,
            allow_pickle=False,
        ) as data:
            if set(
                data.files
            ) != ALLOWED_ANNOTATION_KEYS:
                raise RuntimeError(
                    f"Unsafe annotation cache keys: {ann_path}"
                )

            coords = np.asarray(
                data["supervision_voxel_zyx"]
            )

            labels = np.asarray(
                data["supervision_label"]
            )

            fg_coords = np.asarray(
                data["fg_membership_voxel_zyx"]
            )

            fg_groups = np.asarray(
                data["fg_membership_group_id"]
            )

            if coords.shape != (
                len(labels),
                3,
            ):
                raise RuntimeError(
                    "Invalid supervision coordinate shape."
                )

            if fg_coords.shape != (
                len(fg_groups),
                3,
            ):
                raise RuntimeError(
                    "Invalid FG membership coordinate shape."
                )

            if not set(
                np.unique(
                    labels
                )
            ).issubset(
                {0, 1}
            ):
                raise RuntimeError(
                    "Training labels must be sparse binary labels."
                )

            if np.any(
                fg_groups
                <= 0
            ):
                raise RuntimeError(
                    "Replay FG group IDs must be positive."
                )

    return {
        "manifest_rows":
            int(
                len(
                    manifest
                )
            ),

        "cases":
            int(
                manifest[
                    "case_id"
                ].nunique()
            ),

        "conditions":
            int(
                manifest[
                    "condition"
                ].nunique()
            ),
    }
'''

write_text(
    REPO
    / "src/cora_lung/data/training_cache.py",
    training_cache_firewall_source,
)


preprocess_source = r'''
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
'''

write_text(
    REPO
    / "src/cora_lung/data/preprocess.py",
    preprocess_source,
)


tests = r'''
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
'''

write_text(
    REPO
    / "tests/test_physical_geometry.py",
    tests,
)


training_firewall_test = r'''
from pathlib import Path
import numpy as np
import pandas as pd

from cora_lung.data.training_cache import validate_training_cache


def test_minimal_training_cache(tmp_path):
    root = Path(tmp_path)

    (root / "images").mkdir()
    (root / "annotations").mkdir()

    np.savez_compressed(
        root / "images/case.npz",
        ct_zyx=np.zeros(
            (4, 5, 6),
            dtype=np.float16,
        ),
        spacing_zyx_mm=np.asarray(
            [3., 1.5, 1.5],
            dtype=np.float32,
        ),
        resampled_affine_xyz=np.eye(
            4,
            dtype=float,
        ),
        crop_origin_zyx=np.zeros(
            3,
            dtype=np.int32,
        ),
        full_shape_zyx=np.asarray(
            [4, 5, 6],
            dtype=np.int32,
        ),
        crop_shape_zyx=np.asarray(
            [4, 5, 6],
            dtype=np.int32,
        ),
        hu_window=np.asarray(
            [-1000., 400.],
            dtype=np.float32,
        ),
    )

    np.savez_compressed(
        root / "annotations/case.npz",
        supervision_voxel_zyx=np.asarray(
            [[1, 2, 3]],
            dtype=np.int32,
        ),
        supervision_label=np.asarray(
            [1],
            dtype=np.int8,
        ),
        fg_membership_voxel_zyx=np.asarray(
            [[1, 2, 3]],
            dtype=np.int32,
        ),
        fg_membership_group_id=np.asarray(
            [1],
            dtype=np.int32,
        ),
    )

    pd.DataFrame(
        [
            {
                "case_id": "case",
                "source_subject_key": "case",
                "split_role": "development",
                "outer_fold": float("nan"),
                "condition": "complete",
                "image_file": "images/case.npz",
                "annotation_file": "annotations/case.npz",
                "image_semantic_sha256": "a" * 64,
                "annotation_semantic_sha256": "b" * 64,
                "source_annotation_semantic_sha256": "c" * 64,
                "preprocess_version": "1.0",
            }
        ]
    ).to_csv(
        root / "manifest.csv",
        index=False,
    )

    report = validate_training_cache(
        root
    )

    assert report["manifest_rows"] == 1
'''

write_text(
    REPO
    / "tests/test_training_cache_firewall.py",
    training_firewall_test,
)


if str(
    REPO / "src"
) not in sys.path:
    sys.path.insert(
        0,
        str(
            REPO / "src"
        ),
    )

from cora_lung.data.training_cache import validate_training_cache

cache_firewall_report = validate_training_cache(
    CACHE_ROOT
)

print(
    f"✓ Training-cache firewall: PASS "
    f"({cache_firewall_report['manifest_rows']} rows)"
)

pytest_result = sh(
    [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_firewall.py",
        "tests/test_scribble_budget.py",
        "tests/test_physical_geometry.py",
        "tests/test_training_cache_firewall.py",
        "-q",
    ],
    cwd=REPO,
    check=False,
)

print(
    pytest_result.stdout.strip()
)

if pytest_result.stderr.strip():
    print(
        pytest_result.stderr.strip()
    )

tests_pass = (
    pytest_result.returncode
    == 0
)

print(
    f"Scientific/unit tests: "
    f"{'PASS' if tests_pass else 'FAIL'}"
)


# ==========================================================================================
# 13. CACHE SIZE + SEMANTIC HASH FREEZE
# ==========================================================================================

heading(
    "STEP 9/10 — FREEZE CACHE PROVENANCE AND AGGREGATE FIGURES"
)

cache_files = sorted(
    [
        p
        for p in CACHE_ROOT.rglob(
            "*"
        )
        if p.is_file()
    ],
    key=lambda p:
        str(
            p.relative_to(
                CACHE_ROOT
            )
        ),
)

cache_hash_rows = []

for path in tqdm(
    cache_files,
    desc="Hashing training cache",
    unit="file",
):
    cache_hash_rows.append(
        {
            "relative_path":
                str(
                    path.relative_to(
                        CACHE_ROOT
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
    )

cache_hash_df = pd.DataFrame(
    cache_hash_rows
)

cache_bytes = int(
    cache_hash_df[
        "bytes"
    ].sum()
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

manifest_dir.mkdir(
    parents=True,
    exist_ok=True,
)

audit_dir.mkdir(
    parents=True,
    exist_ok=True,
)

figure_dir.mkdir(
    parents=True,
    exist_ok=True,
)

case_df.to_csv(
    manifest_dir
    / "training_grid_case_summary.csv",
    index=False,
)

transfer_df.to_csv(
    manifest_dir
    / "training_grid_sparse_transfer_summary.csv",
    index=False,
)

budget_df.to_csv(
    manifest_dir
    / "training_grid_budget_equivalence.csv",
    index=False,
)

cache_hash_df.to_csv(
    manifest_dir
    / "training_cache_file_hashes.csv",
    index=False,
)


# ------------------------------------------------------------------------------------------
# Figure 06 — crop/storage reduction
# ------------------------------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "savefig.dpi": 600,
    }
)

plot_case = case_df.sort_values(
    "crop_fraction_of_resampled_volume"
).reset_index(
    drop=True
)

fig, ax = plt.subplots(
    figsize=(
        12.5,
        6.2,
    )
)

x = np.arange(
    len(
        plot_case
    )
)

ax.bar(
    x,
    100.0
    * plot_case[
        "crop_fraction_of_resampled_volume"
    ],
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    plot_case[
        "case_id"
    ],
    rotation=60,
    ha="right",
    fontweight="bold",
)

ax.set_ylabel(
    "Retained Resampled Volume (%)",
    fontweight="bold",
)

ax.set_xlabel(
    "Primary CT Volume",
    fontweight="bold",
)

ax.set_title(
    "Image-Only Thoracic Cropping Efficiency on the Frozen Anisotropic Training Grid\n"
    "(No Dense Lesion or Released Lung Mask Used for Crop Construction)",
    fontweight="bold",
)

ax.grid(
    axis="y",
    alpha=0.25,
)

fig.tight_layout()

fig.savefig(
    figure_dir
    / "fig06_image_only_training_crop_efficiency.png",
    dpi=600,
    bbox_inches="tight",
)

fig.savefig(
    figure_dir
    / "fig06_image_only_training_crop_efficiency.pdf",
    bbox_inches="tight",
)

plt.close(
    fig
)


# ------------------------------------------------------------------------------------------
# Figure 07 — sparse voxel retention
# ------------------------------------------------------------------------------------------

condition_retention = (
    transfer_df.groupby(
        "condition",
        as_index=False,
    )
    .agg(
        median_fg_retention=(
            "fg_voxel_retention",
            "median",
        ),

        median_bg_retention=(
            "bg_voxel_retention",
            "median",
        ),
    )
    .sort_values(
        "condition"
    )
)

fig, ax = plt.subplots(
    figsize=(
        13.5,
        6.5,
    )
)

x = np.arange(
    len(
        condition_retention
    )
)

width = 0.38

ax.bar(
    x
    - width / 2,
    100
    * condition_retention[
        "median_fg_retention"
    ],
    width=width,
    label="Foreground",
)

ax.bar(
    x
    + width / 2,
    100
    * condition_retention[
        "median_bg_retention"
    ],
    width=width,
    label="Explicit Background",
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    condition_retention[
        "condition"
    ],
    rotation=55,
    ha="right",
    fontweight="bold",
)

ax.set_ylabel(
    "Median Unique-Voxel Retention After Transfer (%)",
    fontweight="bold",
)

ax.set_xlabel(
    "Frozen Weak-Supervision Condition",
    fontweight="bold",
)

ax.set_title(
    "Sparse-Supervision Retention After Physical-Coordinate Transfer to the Training Grid\n"
    "(3.0 × 1.5 × 1.5 mm; Same-Class Spatial Collisions Merged)",
    fontweight="bold",
)

ax.grid(
    axis="y",
    alpha=0.25,
)

legend = ax.legend()

for text in legend.get_texts():
    text.set_fontweight(
        "bold"
    )

fig.tight_layout()

fig.savefig(
    figure_dir
    / "fig07_sparse_label_transfer_retention.png",
    dpi=600,
    bbox_inches="tight",
)

fig.savefig(
    figure_dir
    / "fig07_sparse_label_transfer_retention.pdf",
    bbox_inches="tight",
)

plt.close(
    fig
)

print(
    "✓ Publication aggregate figures generated."
)


# ==========================================================================================
# 14. FINAL SCIENTIFIC DECISION
# ==========================================================================================

hard_geometry_pass = bool(
    all_artifacts_transferred
    and zero_conflicts
    and zero_out_full
    and zero_out_crop
    and all_groups_survive
    and roundtrip_bound_pass
    and background_transfer_exact
    and tests_pass
)

post_transfer_budget_pass = bool(
    natural_pixel_exact
    and fixed_pair_exact
)

primary_50_budget_pass = bool(
    primary50_np
    and primary50_fixed
)

if not hard_geometry_pass:
    BLOCK06_STATUS = "FAIL"

elif not primary_50_budget_pass:
    BLOCK06_STATUS = (
        "REVIEW_REQUIRED_PRIMARY_CAUSAL_BUDGET"
    )

elif not post_transfer_budget_pass:
    BLOCK06_STATUS = (
        "PASS_PRIMARY_50_WITH_SECONDARY_BUDGET_REVIEW"
    )

else:
    BLOCK06_STATUS = "PASS"


block06_audit = {
    "project":
        PROJECT,

    "block":
        BLOCK_ID,

    "block_name":
        BLOCK_NAME,

    "generated_at_utc":
        NOW_ISO,

    "status":
        BLOCK06_STATUS,

    "preprocess_version":
        PREPROCESS_VERSION,

    "target_spacing_zyx_mm":
        TARGET_SPACING_ZYX.tolist(),

    "ct_cases":
        int(
            len(
                case_df
            )
        ),

    "sparse_artifacts":
        int(
            len(
                transfer_df
            )
        ),

    "cache": {
        "files":
            int(
                len(
                    cache_hash_df
                )
            ),

        "bytes":
            cache_bytes,

        "human_size":
            human_bytes(
                cache_bytes
            ),
    },

    "crop": {
        "dense_masks_used":
            False,

        "scribbles_used":
            False,

        "median_retained_volume_fraction":
            float(
                case_df[
                    "crop_fraction_of_resampled_volume"
                ].median()
            ),

        "method_counts":
            {
                str(k):
                    int(v)

                for k, v in (
                    case_df[
                        "crop_method"
                    ].value_counts()
                    .to_dict()
                    .items()
                )
            },
    },

    "geometry": {
        "all_260_transferred":
            bool(
                all_artifacts_transferred
            ),

        "fg_bg_conflicts":
            int(
                len(
                    critical_conflicts
                )
            ),

        "out_of_full_volume_conditions":
            int(
                len(
                    out_of_full_failures
                )
            ),

        "out_of_crop_conditions":
            int(
                len(
                    out_of_crop_failures
                )
            ),

        "group_loss_conditions":
            int(
                len(
                    group_loss_failures
                )
            ),

        "group_survival":
            bool(
                all_groups_survive
            ),

        "nearest_voxel_half_diagonal_mm":
            nearest_voxel_half_diagonal_mm,

        "max_observed_roundtrip_error_mm":
            float(
                transfer_df[
                    "max_roundtrip_error_mm"
                ].max()
                if len(
                    transfer_df
                )
                else float(
                    "nan"
                )
            ),

        "roundtrip_bound_pass":
            bool(
                roundtrip_bound_pass
            ),
    },

    "collision_statistics": {
        "total_same_class_merges":
            int(
                transfer_df[
                    "same_class_collision_merges"
                ].sum()
                if len(
                    transfer_df
                )
                else 0
            ),

        "median_fg_voxel_retention":
            float(
                transfer_df[
                    "fg_voxel_retention"
                ].median()
                if len(
                    transfer_df
                )
                else float(
                    "nan"
                )
            ),

        "median_bg_voxel_retention":
            float(
                transfer_df[
                    "bg_voxel_retention"
                ].median()
                if len(
                    transfer_df
                )
                else float(
                    "nan"
                )
            ),
    },

    "paired_background_after_transfer":
        bool(
            background_transfer_exact
        ),

    "training_grid_budget_equivalence": {
        "natural_pixel_all_60":
            bool(
                natural_pixel_exact
            ),

        "fixed_pair_all_60":
            bool(
                fixed_pair_exact
            ),

        "natural_pixel_mismatch_count":
            int(
                len(
                    natural_pixel_mismatches
                )
            ),

        "fixed_pair_mismatch_count":
            int(
                len(
                    fixed_pair_mismatches
                )
            ),

        "primary_50_natural_pixel":
            bool(
                primary50_np
            ),

        "primary_50_fixed_pair":
            bool(
                primary50_fixed
            ),
    },

    "training_cache_firewall":
        "PASS"
        if tests_pass
        else "FAIL",

    "dense_masks_used":
        False,

    "model_training_performed":
        False,
}

write_json(
    audit_dir
    / "block06_training_grid_cache.json",
    block06_audit,
)


# ==========================================================================================
# 15. CAPTURE EXECUTED BLOCK SOURCE
# ==========================================================================================

source_capture = "NOT_AVAILABLE"

try:
    ip = get_ipython()

    raw_cell = (
        ip.history_manager
        .input_hist_raw[-1]
    )

    if (
        "CORA-LUNG — CODE BLOCK 06"
        in raw_cell
    ):
        code_dir = (
            REPO
            / "scripts/code_blocks"
        )

        code_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            code_dir
            / "block06_build_training_cache.py"
        ).write_text(
            raw_cell,
            encoding="utf-8",
        )

        source_capture = "PASS"

except Exception:
    pass


# ==========================================================================================
# 16. PROJECT STATE
# ==========================================================================================

state = json.loads(
    (
        REPO
        / "PROJECT_STATE.json"
    ).read_text(
        encoding="utf-8"
    )
)

block_complete = (
    BLOCK06_STATUS
    in {
        "PASS",
        "PASS_PRIMARY_50_WITH_SECONDARY_BUDGET_REVIEW",
    }
)

state.update(
    {
        "last_attempted_block":
            BLOCK_ID,

        "last_completed_block":
            (
                BLOCK_ID
                if block_complete
                else "05"
            ),

        "last_completed_block_name":
            (
                BLOCK_NAME
                if block_complete
                else state.get(
                    "last_completed_block_name"
                )
            ),

        "current_stage":
            (
                "training_grid_cache_verified"
                if BLOCK06_STATUS
                == "PASS"
                else
                (
                    "training_grid_cache_primary_ready_secondary_review"
                    if BLOCK06_STATUS
                    == "PASS_PRIMARY_50_WITH_SECONDARY_BUDGET_REVIEW"
                    else
                    "training_grid_cache_requires_review"
                )
            ),

        "current_gate":
            "POST_GATE_A_PRE_GATE_B",

        "gate_a":
            "PASS",

        "gate_b":
            "NOT_RUN",

        "training_grid_cache":
            BLOCK06_STATUS,

        "preprocess_version":
            PREPROCESS_VERSION,

        "training_grid_spacing_zyx_mm":
            [
                3.0,
                1.5,
                1.5,
            ],

        "training_cache_firewall":
            (
                "PASS"
                if tests_pass
                else "FAIL"
            ),

        "primary_50_post_transfer_budget":
            (
                "PASS"
                if primary_50_budget_pass
                else "REVIEW_REQUIRED"
            ),

        "training_authorized":
            False,

        "next_action":
            (
                "Audit Block 06 output before Gate-B pilot training."
            ),

        "updated_at_utc":
            NOW_ISO,
    }
)

write_json(
    REPO
    / "PROJECT_STATE.json",
    state,
)


# ==========================================================================================
# 17. REPOSITORY MANIFEST
# ==========================================================================================

repo_manifest = []

repo_files = sorted(
    [
        p
        for p in REPO.rglob(
            "*"
        )
        if p.is_file()
        and ".git"
        not in p.parts
    ],
    key=lambda p:
        str(
            p.relative_to(
                REPO
            )
        ),
)

for path in tqdm(
    repo_files,
    desc="Refreshing repository manifest",
):
    repo_manifest.append(
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
    )

write_json(
    REPO
    / "REPOSITORY_MANIFEST.json",
    {
        "generated_at_utc":
            NOW_ISO,

        "completed_block":
            (
                BLOCK_ID
                if block_complete
                else "05"
            ),

        "last_attempted_block":
            BLOCK_ID,

        "files":
            repo_manifest,
    },
)


# ==========================================================================================
# 18. COMMIT + PUSH
# ==========================================================================================

heading(
    "STEP 10/10 — COMMIT AND SYNCHRONIZE PREPROCESSING AUDIT"
)

sh(
    [
        "git",
        "add",
        ".",
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

if status:
    changes = status.splitlines()

    print(
        f"Tracked changes: {len(changes)}"
    )

    for line in changes[
        :45
    ]:
        print(
            " ",
            line,
        )

    if len(
        changes
    ) > 45:
        print(
            f"  ... +{len(changes)-45} more"
        )

    if block_complete:
        commit_message = (
            "data: freeze firewall-safe CORA-Lung training-grid cache protocol"
        )

    else:
        commit_message = (
            "audit: record Block-06 training-grid transfer review state"
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

    print(
        f"✓ Commit created: {commit_message}"
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
        push.stderr or ""
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

print(
    "✓ GitHub synchronization: PASS"
)


# ==========================================================================================
# 19. CLEAN AUTH
# ==========================================================================================

try:
    askpass.unlink(
        missing_ok=True
    )
except Exception:
    pass

git_env.pop(
    "GITHUB_TOKEN",
    None,
)

token = None


# ==========================================================================================
# 20. FINAL REPORT
# ==========================================================================================

print("\n")
print("=" * 116)
print("CORA-LUNG CODE BLOCK 06 — TRAINING-GRID CACHE REPORT")
print("=" * 116)

print(f"""
PREPROCESSING
-------------
Version                                  : {PREPROCESS_VERSION}
Internal array order                     : z, y, x
Frozen training spacing                  : 3.0 × 1.5 × 1.5 mm
HU window                                : [{HU_MIN:.0f}, {HU_MAX:.0f}]
Intensity range                          : [-1, 1]
CT interpolation                         : Linear
Sparse transfer                          : Physical world -> nearest training voxel

IMAGE-ONLY CROPPING
-------------------
CT volumes processed                     : {len(case_df)}
Dense lesion masks used                  : NO
Released lung masks used                 : NO
Scribbles used to choose crop            : NO
Median retained volume fraction          : {case_df['crop_fraction_of_resampled_volume'].median():.3f}

Crop methods:
{case_df['crop_method'].value_counts().to_string()}

TRAINING CACHE
--------------
Image files                              : {len(list(IMAGE_DIR.glob('*.npz')))}
Sparse annotation files                  : {len(list(ANNOTATION_DIR.glob('*.npz')))}
Manifest rows                            : {len(cache_manifest_df)}
Cache files total                        : {len(cache_hash_df)}
Local cache size                         : {human_bytes(cache_bytes)}

GEOMETRY / TRANSFER QA
----------------------
All 260 conditions transferred           : {'PASS' if all_artifacts_transferred else 'FAIL'}
FG/BG collision count                    : {len(critical_conflicts)}
Out-of-resampled-volume conditions       : {len(out_of_full_failures)}
Out-of-image-crop conditions             : {len(out_of_crop_failures)}
Foreground-group-loss conditions         : {len(group_loss_failures)}
100% foreground-group survival           : {'PASS' if all_groups_survive else 'FAIL'}

Nearest-voxel theoretical error bound    : {nearest_voxel_half_diagonal_mm:.4f} mm
Maximum observed physical error          : {transfer_df['max_roundtrip_error_mm'].max() if len(transfer_df) else float('nan'):.4f} mm
Physical mapping bound                   : {'PASS' if roundtrip_bound_pass else 'FAIL'}

SPARSE VOXEL COLLISIONS
-----------------------
Total same-class merges                  : {int(transfer_df['same_class_collision_merges'].sum()) if len(transfer_df) else 0}
Median FG voxel retention                : {100.0 * transfer_df['fg_voxel_retention'].median() if len(transfer_df) else float('nan'):.2f}%
Median BG voxel retention                : {100.0 * transfer_df['bg_voxel_retention'].median() if len(transfer_df) else float('nan'):.2f}%
Foreground group identity preserved      : {'PASS' if all_groups_survive else 'FAIL'}

PAIRED BACKGROUND AFTER TRANSFER
--------------------------------
Exact same BG coordinate realization     : {'PASS' if background_transfer_exact else 'FAIL'}

ACTUAL TRAINING-GRID CAUSAL BUDGET
----------------------------------
Natural/pixel exact — all 60 pairs       : {'PASS' if natural_pixel_exact else 'REVIEW REQUIRED'}
Fixed-budget exact — all 60 pairs        : {'PASS' if fixed_pair_exact else 'REVIEW REQUIRED'}
Natural/pixel mismatches                 : {len(natural_pixel_mismatches)}/60
Fixed-budget mismatches                  : {len(fixed_pair_mismatches)}/60

PRIMARY 50% GATE-B INPUT
------------------------
Natural/pixel exact after transfer       : {'PASS' if primary50_np else 'REVIEW REQUIRED'}
Fixed pair exact after transfer          : {'PASS' if primary50_fixed else 'REVIEW REQUIRED'}

FIREWALL / TESTS
----------------
Training-cache firewall                  : PASS
Scientific/unit tests                    : {'PASS' if tests_pass else 'FAIL'}
Dense masks in trainer cache             : NO
Dense component information              : NO

REPRODUCIBILITY
---------------
Cache hashes frozen                      : YES
Case preprocessing metadata committed    : YES
Transfer audit committed                 : YES
Budget-equivalence audit committed       : YES
Exact Block-06 source captured           : {source_capture}
Large CT cache committed to GitHub        : NO
Cache reconstructible                    : YES

SCIENTIFIC STATUS
-----------------
Gate A                                   : PASS
Gate B                                   : NOT RUN
Block 06                                 : {BLOCK06_STATUS}
Training                                 : NOT STARTED
Training authorized                      : NO

GITHUB
------
Starting commit                          : {starting_commit[:12]}
Current commit                           : {current_commit[:12]}
Synchronization                          : PASS

NEXT
----
Send me the COMPLETE final report.

Pay particular attention to:

  • crop method counts
  • out-of-crop conditions
  • same-class collision count
  • foreground/background retention
  • foreground-group survival
  • PRIMARY 50% natural/pixel equality
  • PRIMARY 50% fixed-budget equality

We will not start Gate-B training until I audit those results.
""")

print("=" * 116)

if not hard_geometry_pass:
    raise RuntimeError(
        "BLOCK 06 HARD GEOMETRY/FIREWALL FAILURE. "
        "Do not proceed to training."
    )

if not primary_50_budget_pass:
    raise RuntimeError(
        "BLOCK 06 PRIMARY 50% CAUSAL-BUDGET REVIEW REQUIRED. "
        "Do not proceed to training."
    )