# ==========================================================================================
# CORA-LUNG — CODE BLOCK 05D
# Surgical Repair of Block-05 Exact-Budget Allocator
#
# FAILURE ROOT CAUSE
#   coronacases_003 / 25% component-fixed condition
#
#   Requested shared budget          : 376
#   Retained connected-path capacity : 376
#   Therefore mathematically feasible.
#
#   v1.0 allocator stopped because:
#       safety = 2505
#       arbitrary safety_limit = 2504
#       remaining = 5
#
# CORRECTION
#   Replace the arbitrary iteration cutoff with a capacity-aware cyclic allocator.
#
# IMPORTANT
#   The new allocator preserves the exact same allocation sequence as the old
#   round-robin allocator for every case that previously completed successfully.
#   The only behavioral change is that a feasible allocation is no longer aborted
#   by an arbitrary iteration counter.
#
# THIS BLOCK
#   1. verifies the exact known failure state;
#   2. patches the canonical generator to v1.1;
#   3. archives the originally executed v1.0 generator;
#   4. regenerates ONLY the two missing artifacts;
#   5. leaves all 258 existing sparse artifacts untouched;
#   6. verifies all 260 sparse artifacts;
#   7. verifies exact background coordinates;
#   8. verifies natural-vs-pixel matching at 75/50/25%;
#   9. verifies component-fixed vs complete-fixed matching at 75/50/25%;
#  10. verifies nested 25% ⊂ 50% ⊂ 75% supervision;
#  11. reruns firewall + regression tests;
#  12. freezes Block 05 as PASS and pushes GitHub.
#
# NO MODEL TRAINING.
# ==========================================================================================

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from collections import deque
import subprocess
import hashlib
import json
import os
import re
import sys
import math
import shutil
import textwrap

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from kaggle_secrets import UserSecretsClient

try:
    import nibabel as nib
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "nibabel"]
    )
    import nibabel as nib

try:
    from skimage.morphology import skeletonize
except ImportError:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "scikit-image"]
    )
    from skimage.morphology import skeletonize


# ==========================================================================================
# 0. CONSTANTS
# ==========================================================================================

PROJECT = "CORA-Lung"

BLOCK_ID = "05"
REPAIR_ID = "05D"

GENERATOR_VERSION_OLD = "1.0"
GENERATOR_VERSION_NEW = "1.1"

GENERATOR_SEED = 20260912

CASE_ID = "coronacases_003"
COVERAGE = 0.25
COVERAGE_TOKEN = 25

MIN_COMPONENT_ML = 0.1
INTERIOR_EROSION_MM = 1.0

BASE_FG_STROKE_VOXELS = 20
MAX_ALLOWED_PATH_VOXELS = 256

GITHUB_OWNER = "itsCodeBakery"
GITHUB_REPO = "CORA-LUNG"
REMOTE_URL = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}.git"

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")

REPO = WORK / GITHUB_REPO
WEAK_ROOT = WORK / "cora_weak_train_v1"

ANNOTATION_DIR = WEAK_ROOT / "annotations"
GEOMETRY_DIR = WEAK_ROOT / "geometry"
WEAK_MANIFEST = WEAK_ROOT / "manifest.csv"

NOW = datetime.now(timezone.utc)
NOW_ISO = NOW.strftime("%Y-%m-%dT%H:%M:%SZ")

OFFSETS_26 = np.asarray(
    [
        (i, j, k)
        for i in (-1, 0, 1)
        for j in (-1, 0, 1)
        for k in (-1, 0, 1)
        if not (i == 0 and j == 0 and k == 0)
    ],
    dtype=np.int16,
)


# ==========================================================================================
# 1. HELPERS
# ==========================================================================================

def heading(text):
    print("\n" + "=" * 114)
    print(text)
    print("=" * 114)


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
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps(obj, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_text(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        textwrap.dedent(text).strip() + "\n",
        encoding="utf-8",
    )


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)

    return h.hexdigest()


def stable_seed(*parts):
    payload = "|".join(str(x) for x in parts)

    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        "little",
        signed=False,
    ) % (2**32 - 1)


def semantic_annotation_hash(
    voxel_ijk,
    world_xyz_mm,
    labels,
    group_ids,
):
    h = hashlib.sha256()

    h.update(
        np.ascontiguousarray(
            voxel_ijk.astype(np.int32)
        ).tobytes()
    )

    h.update(
        np.ascontiguousarray(
            world_xyz_mm.astype(np.float32)
        ).tobytes()
    )

    h.update(
        np.ascontiguousarray(
            labels.astype(np.int8)
        ).tobytes()
    )

    h.update(
        np.ascontiguousarray(
            group_ids.astype(np.int32)
        ).tobytes()
    )

    return h.hexdigest()


def coordinate_set_hash(coords):
    coords = np.asarray(
        coords,
        dtype=np.int32,
    )

    if len(coords) == 0:
        return hashlib.sha256(
            b""
        ).hexdigest()

    order = np.lexsort(
        (
            coords[:, 2],
            coords[:, 1],
            coords[:, 0],
        )
    )

    return hashlib.sha256(
        np.ascontiguousarray(
            coords[order]
        ).tobytes()
    ).hexdigest()


def locate_dataset():
    candidates = [
        INPUT / "covid19-ct-scans",
        INPUT / "datasets" / "andrewmvd" / "covid19-ct-scans",
    ]

    for p in candidates:
        if p.exists():
            return p.resolve()

    for p in INPUT.rglob("metadata.csv"):
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
        "Primary COVID-19 CT dataset could not be located."
    )


# ==========================================================================================
# 2. EXACT ORIGINAL PATH GENERATION
# ==========================================================================================

def physical_ball(
    spacing_mm,
    radius_mm,
):
    spacing = np.asarray(
        spacing_mm,
        dtype=np.float64,
    )

    extents = np.ceil(
        radius_mm / spacing
    ).astype(int)

    grids = np.meshgrid(
        *[
            np.arange(-e, e + 1)
            for e in extents
        ],
        indexing="ij",
    )

    distance_sq = np.zeros_like(
        grids[0],
        dtype=np.float64,
    )

    for axis, grid in enumerate(grids):
        distance_sq += (
            grid.astype(float)
            * spacing[axis]
        ) ** 2

    structure = (
        distance_sq
        <= radius_mm**2 + 1e-9
    )

    if not structure.any():
        structure[
            tuple(extents)
        ] = True

    return structure


def largest_connected_part(mask):
    structure = ndi.generate_binary_structure(
        3,
        3,
    )

    labels, n = ndi.label(
        mask,
        structure=structure,
    )

    if n == 0:
        return np.zeros_like(
            mask,
            dtype=bool,
        )

    counts = np.bincount(
        labels.ravel()
    )

    counts[0] = 0

    return labels == int(
        np.argmax(counts)
    )


def bfs_farthest(
    coords,
    coord_to_index,
    start_index,
):
    n = len(coords)

    parent = np.full(
        n,
        -1,
        dtype=np.int32,
    )

    distance = np.full(
        n,
        -1,
        dtype=np.int32,
    )

    queue = deque(
        [int(start_index)]
    )

    distance[
        start_index
    ] = 0

    farthest = int(
        start_index
    )

    while queue:
        idx = queue.popleft()

        if (
            distance[idx]
            > distance[farthest]
        ):
            farthest = idx

        base = coords[idx]

        for offset in OFFSETS_26:
            nb = tuple(
                (
                    base
                    + offset
                ).tolist()
            )

            j = coord_to_index.get(
                nb,
                None,
            )

            if j is None:
                continue

            if distance[j] >= 0:
                continue

            distance[j] = (
                distance[idx]
                + 1
            )

            parent[j] = idx

            queue.append(j)

    return (
        farthest,
        parent,
        distance,
    )


def approximate_skeleton_diameter_path(
    skeleton_mask,
    rng,
):
    skeleton_mask = largest_connected_part(
        skeleton_mask.astype(bool)
    )

    coords = np.argwhere(
        skeleton_mask
    ).astype(np.int32)

    if len(coords) == 0:
        return np.empty(
            (0, 3),
            dtype=np.int32,
        )

    if len(coords) == 1:
        return coords.copy()

    coord_to_index = {
        tuple(c.tolist()): i
        for i, c in enumerate(coords)
    }

    start = int(
        rng.integers(
            0,
            len(coords),
        )
    )

    endpoint_a, _, _ = bfs_farthest(
        coords,
        coord_to_index,
        start,
    )

    endpoint_b, parent, _ = bfs_farthest(
        coords,
        coord_to_index,
        endpoint_a,
    )

    path_indices = []

    current = endpoint_b

    while current >= 0:
        path_indices.append(
            int(current)
        )

        if current == endpoint_a:
            break

        current = int(
            parent[current]
        )

    return coords[
        np.asarray(
            path_indices[::-1],
            dtype=np.int32,
        )
    ]


def random_walk_fallback(
    valid_mask,
    rng,
    max_length,
):
    coords = np.argwhere(
        valid_mask
    )

    if len(coords) == 0:
        return np.empty(
            (0, 3),
            dtype=np.int32,
        )

    start = coords[
        int(
            rng.integers(
                0,
                len(coords),
            )
        )
    ].astype(np.int32)

    path = [
        tuple(
            start.tolist()
        )
    ]

    used = {
        tuple(
            start.tolist()
        )
    }

    current = start.copy()

    for _ in range(
        max_length - 1
    ):
        neighbors = []

        for offset in OFFSETS_26:
            candidate = (
                current + offset
            )

            if np.any(
                candidate < 0
            ):
                continue

            if np.any(
                candidate
                >= np.asarray(
                    valid_mask.shape
                )
            ):
                continue

            t = tuple(
                candidate.tolist()
            )

            if t in used:
                continue

            if valid_mask[t]:
                neighbors.append(
                    candidate
                )

        if not neighbors:
            break

        current = neighbors[
            int(
                rng.integers(
                    0,
                    len(neighbors),
                )
            )
        ].astype(np.int32)

        t = tuple(
            current.tolist()
        )

        used.add(t)
        path.append(t)

    return np.asarray(
        path,
        dtype=np.int32,
    )


def contiguous_segment(
    path,
    requested_length,
    rng,
):
    path = np.asarray(
        path,
        dtype=np.int32,
    )

    requested_length = int(
        requested_length
    )

    if requested_length <= 0:
        return np.empty(
            (0, 3),
            dtype=np.int32,
        )

    if len(path) <= requested_length:
        return path.copy()

    max_start = (
        len(path)
        - requested_length
    )

    start = int(
        rng.integers(
            0,
            max_start + 1,
        )
    )

    return path[
        start:
        start + requested_length
    ].copy()


# ==========================================================================================
# 3. CORRECTED ALLOCATOR — v1.1
# ==========================================================================================

def allocate_budget_exact(
    total_budget,
    capacities,
    rng,
):
    """
    Exact capacity-aware cyclic allocation.

    This preserves the v1.0 shuffled cyclic allocation sequence but removes
    the arbitrary safety counter.

    Termination is mathematically guaranteed because:
      - total_budget <= sum(capacities) is checked first;
      - each successful cycle reduces `remaining`;
      - if a full cycle makes no progress, the state is inconsistent.
    """

    group_ids = sorted(
        capacities.keys()
    )

    if not group_ids:
        return {}, False

    total_budget = int(
        total_budget
    )

    minimum_required = len(
        group_ids
    )

    total_capacity = int(
        sum(
            int(
                capacities[g]
            )
            for g in group_ids
        )
    )

    if (
        total_budget
        < minimum_required
    ):
        return {}, False

    if (
        total_budget
        > total_capacity
    ):
        return {}, False

    allocation = {
        g: 1
        for g in group_ids
    }

    remaining = (
        total_budget
        - minimum_required
    )

    order = np.asarray(
        group_ids,
        dtype=np.int32,
    )

    rng.shuffle(
        order
    )

    while remaining > 0:
        progressed = False

        for g_raw in order:
            g = int(
                g_raw
            )

            if remaining == 0:
                break

            if (
                allocation[g]
                < capacities[g]
            ):
                allocation[g] += 1
                remaining -= 1
                progressed = True

        if not progressed:
            return {}, False

    return allocation, True


def build_contiguous_budget_annotation(
    allowed_paths,
    observed_groups,
    target_budget,
    rng,
):
    observed_groups = sorted(
        int(x)
        for x in observed_groups
    )

    capacities = {
        g: len(
            allowed_paths[g]
        )
        for g in observed_groups
    }

    allocation, feasible = allocate_budget_exact(
        int(target_budget),
        capacities,
        rng,
    )

    if not feasible:
        return (
            {},
            False,
            allocation,
        )

    result = {}

    for g in observed_groups:
        segment_rng = np.random.default_rng(
            stable_seed(
                GENERATOR_SEED,
                "segment",
                int(
                    rng.integers(
                        0,
                        2**31 - 1,
                    )
                ),
                g,
            )
        )

        result[g] = contiguous_segment(
            allowed_paths[g],
            allocation[g],
            segment_rng,
        )

    achieved = int(
        sum(
            len(v)
            for v in result.values()
        )
    )

    return (
        result,
        achieved
        == int(target_budget),
        allocation,
    )


# ==========================================================================================
# 4. ANNOTATION ASSEMBLY
# ==========================================================================================

def assemble_annotation(
    fg_by_group,
    bg_coords,
    affine,
):
    fg_coords = []
    fg_group_ids = []

    for group_id in sorted(
        fg_by_group.keys()
    ):
        coords = np.asarray(
            fg_by_group[
                group_id
            ],
            dtype=np.int32,
        )

        if len(coords) == 0:
            continue

        fg_coords.append(
            coords
        )

        fg_group_ids.append(
            np.full(
                len(coords),
                int(group_id),
                dtype=np.int32,
            )
        )

    if fg_coords:
        fg_coords = np.concatenate(
            fg_coords,
            axis=0,
        )

        fg_group_ids = np.concatenate(
            fg_group_ids,
            axis=0,
        )

    else:
        fg_coords = np.empty(
            (0, 3),
            dtype=np.int32,
        )

        fg_group_ids = np.empty(
            (0,),
            dtype=np.int32,
        )

    bg_coords = np.asarray(
        bg_coords,
        dtype=np.int32,
    )

    if len(bg_coords):
        bg_coords = np.unique(
            bg_coords,
            axis=0,
        )

    fg_set = {
        tuple(
            x.tolist()
        )
        for x in fg_coords
    }

    bg_set = {
        tuple(
            x.tolist()
        )
        for x in bg_coords
    }

    collision = (
        fg_set
        .intersection(
            bg_set
        )
    )

    if collision:
        raise RuntimeError(
            f"FG/BG collision: {len(collision)}"
        )

    voxel_ijk = np.concatenate(
        [
            fg_coords,
            bg_coords,
        ],
        axis=0,
    )

    labels = np.concatenate(
        [
            np.ones(
                len(fg_coords),
                dtype=np.int8,
            ),
            np.zeros(
                len(bg_coords),
                dtype=np.int8,
            ),
        ]
    )

    group_ids = np.concatenate(
        [
            fg_group_ids,
            np.full(
                len(bg_coords),
                -1,
                dtype=np.int32,
            ),
        ]
    )

    world_xyz_mm = nib.affines.apply_affine(
        affine,
        voxel_ijk.astype(
            np.float64
        ),
    ).astype(
        np.float32
    )

    return {
        "voxel_ijk":
            voxel_ijk.astype(
                np.int32
            ),

        "world_xyz_mm":
            world_xyz_mm,

        "label":
            labels.astype(
                np.int8
            ),

        "group_id":
            group_ids.astype(
                np.int32
            ),
    }


def save_npz(
    path,
    annotation,
):
    np.savez_compressed(
        path,
        voxel_ijk=annotation[
            "voxel_ijk"
        ],
        world_xyz_mm=annotation[
            "world_xyz_mm"
        ],
        label=annotation[
            "label"
        ],
        group_id=annotation[
            "group_id"
        ],
    )


# ==========================================================================================
# 5. VERIFY FAILURE STATE / GITHUB AUTH
# ==========================================================================================

heading(
    "CORA-LUNG BLOCK 05D — EXACT ALLOCATOR REPAIR"
)

if not (
    REPO / ".git"
).exists():
    raise RuntimeError(
        "CORA-LUNG repository missing."
    )

if not WEAK_MANIFEST.exists():
    raise RuntimeError(
        "Existing 258-artifact weak manifest missing."
    )

manifest = pd.read_csv(
    WEAK_MANIFEST
)

existing_pairs = set(
    zip(
        manifest[
            "case_id"
        ].astype(str),
        manifest[
            "condition"
        ].astype(str),
    )
)

expected_missing = {
    (
        CASE_ID,
        "component_fixed_25",
    ),
    (
        CASE_ID,
        "complete_fixed_25",
    ),
}

all_expected_conditions = [
    "complete",
    "component_natural_75",
    "pixel_dropout_matched_75",
    "component_fixed_75",
    "complete_fixed_75",
    "component_natural_50",
    "pixel_dropout_matched_50",
    "component_fixed_50",
    "complete_fixed_50",
    "component_natural_25",
    "pixel_dropout_matched_25",
    "component_fixed_25",
    "complete_fixed_25",
]

all_cases = sorted(
    manifest[
        "case_id"
    ].astype(str).unique()
)

actual_missing = set()

for case_id in all_cases:
    for condition in all_expected_conditions:
        if (
            case_id,
            condition,
        ) not in existing_pairs:
            actual_missing.add(
                (
                    case_id,
                    condition,
                )
            )

print(
    f"Existing weak artifacts : {len(manifest)}"
)

print(
    f"Detected missing        : {sorted(actual_missing)}"
)

if (
    actual_missing
    != expected_missing
):
    raise RuntimeError(
        "Failure state differs from the diagnosed Block-05C state."
    )

print(
    "✓ Exact known failure state verified."
)

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
    "/tmp/cora_git_askpass_block05d.sh"
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

print(
    f"✓ Starting commit      : {starting_commit[:12]}"
)


# ==========================================================================================
# 6. ARCHIVE v1.0 AND PATCH CANONICAL GENERATOR TO v1.1
# ==========================================================================================

heading(
    "STEP 1/9 — PATCH CANONICAL GENERATOR WITHOUT ERASING v1.0 PROVENANCE"
)

canonical_script = (
    REPO
    / "scripts/code_blocks/block05_generate_weak_labels.py"
)

archive_dir = (
    REPO
    / "scripts/code_blocks/archive"
)

archive_dir.mkdir(
    parents=True,
    exist_ok=True,
)

archive_script = (
    archive_dir
    / "block05_generate_weak_labels_v1_0_executed.py"
)

if not canonical_script.exists():
    raise RuntimeError(
        "Captured canonical Block-05 source is missing."
    )

original_text = canonical_script.read_text(
    encoding="utf-8"
)

if not archive_script.exists():
    shutil.copy2(
        canonical_script,
        archive_script,
    )

    print(
        "✓ Archived originally executed generator v1.0."
    )

else:
    print(
        "✓ v1.0 archive already exists."
    )

allocator_start = original_text.find(
    "def allocate_budget("
)

allocator_end_marker = (
    "\n\n# ==========================================================================================\n"
    "# 5. BACKGROUND PATH GENERATOR"
)

allocator_end = original_text.find(
    allocator_end_marker,
    allocator_start,
)

if (
    allocator_start < 0
    or allocator_end < 0
):
    raise RuntimeError(
        "Could not locate allocator boundaries in canonical generator."
    )

replacement_allocator = r'''def allocate_budget(
    total_budget,
    capacities,
    rng,
):
    """
    Exact capacity-aware cyclic budget allocator.

    Version 1.1 correction:
    preserves the original shuffled cyclic allocation order while removing
    the arbitrary safety-counter termination that could reject feasible
    allocations when many groups had already reached capacity.
    """
    group_ids = sorted(
        capacities.keys()
    )

    if not group_ids:
        return {}, False

    total_budget = int(
        total_budget
    )

    minimum_required = len(
        group_ids
    )

    total_capacity = int(
        sum(
            int(capacities[g])
            for g in group_ids
        )
    )

    if total_budget < minimum_required:
        return {}, False

    if total_budget > total_capacity:
        return {}, False

    allocation = {
        g: 1
        for g in group_ids
    }

    remaining = (
        total_budget
        - minimum_required
    )

    order = np.asarray(
        group_ids,
        dtype=np.int32,
    )

    rng.shuffle(order)

    while remaining > 0:
        progressed = False

        for g_raw in order:
            g = int(g_raw)

            if remaining == 0:
                break

            if allocation[g] < capacities[g]:
                allocation[g] += 1
                remaining -= 1
                progressed = True

        if not progressed:
            return {}, False

    return allocation, True
'''

patched_text = (
    original_text[
        :allocator_start
    ]
    + replacement_allocator
    + original_text[
        allocator_end:
    ]
)

patched_text = patched_text.replace(
    'GENERATOR_VERSION = "1.0"',
    'GENERATOR_VERSION = "1.1"',
    1,
)

canonical_script.write_text(
    patched_text,
    encoding="utf-8",
)

print(
    "✓ Canonical generator patched to v1.1."
)


# ==========================================================================================
# 7. CREATE REUSABLE ALLOCATOR MODULE + REGRESSION TEST
# ==========================================================================================

allocator_module = r'''
"""Weak-label budget utilities for CORA-Lung."""

from __future__ import annotations

import numpy as np


def allocate_budget_exact(total_budget, capacities, rng):
    """
    Allocate an exact integer annotation budget over groups.

    Every group receives at least one labelled voxel.

    The group visitation order is deterministically shuffled once, then
    revisited cyclically until the requested budget is exhausted.

    Unlike the original v1.0 implementation, this function has no arbitrary
    iteration cutoff. A full cycle without progress is treated as infeasible.
    """

    group_ids = sorted(capacities.keys())

    if not group_ids:
        return {}, False

    total_budget = int(total_budget)

    minimum_required = len(group_ids)

    total_capacity = int(
        sum(
            int(capacities[g])
            for g in group_ids
        )
    )

    if total_budget < minimum_required:
        return {}, False

    if total_budget > total_capacity:
        return {}, False

    allocation = {
        g: 1
        for g in group_ids
    }

    remaining = (
        total_budget
        - minimum_required
    )

    order = np.asarray(
        group_ids,
        dtype=np.int32,
    )

    rng.shuffle(order)

    while remaining > 0:
        progressed = False

        for g_raw in order:
            g = int(g_raw)

            if remaining == 0:
                break

            if allocation[g] < capacities[g]:
                allocation[g] += 1
                remaining -= 1
                progressed = True

        if not progressed:
            return {}, False

    return allocation, True
'''

write_text(
    REPO
    / "src/cora_lung/data/scribbles.py",
    allocator_module,
)

regression_test = r'''
import numpy as np

from cora_lung.data.scribbles import allocate_budget_exact


def test_coronacases003_25pct_capacity_regression():
    capacities = {
        5: 19,
        8: 3,
        9: 13,
        10: 9,
        13: 14,
        14: 6,
        24: 256,
        27: 38,
        30: 14,
        39: 4,
    }

    target = 376

    allocation, feasible = allocate_budget_exact(
        target,
        capacities,
        np.random.default_rng(20260912),
    )

    assert feasible
    assert sum(allocation.values()) == target

    # Since target == total capacity, every group must be saturated.
    assert allocation == capacities


def test_budget_above_capacity_rejected():
    capacities = {
        1: 3,
        2: 4,
    }

    allocation, feasible = allocate_budget_exact(
        8,
        capacities,
        np.random.default_rng(1),
    )

    assert not feasible
    assert allocation == {}


def test_budget_below_one_per_group_rejected():
    capacities = {
        1: 10,
        2: 10,
        3: 10,
    }

    allocation, feasible = allocate_budget_exact(
        2,
        capacities,
        np.random.default_rng(1),
    )

    assert not feasible
    assert allocation == {}
'''

write_text(
    REPO
    / "tests/test_scribble_budget.py",
    regression_test,
)

print(
    "✓ Reusable allocator implementation written."
)

print(
    "✓ Allocator regression tests written."
)


# ==========================================================================================
# 8. RECONSTRUCT THE SINGLE CASE EXACTLY
# ==========================================================================================

heading(
    "STEP 2/9 — RECONSTRUCT coronacases_003 PATHS"
)

split_registry = pd.read_csv(
    REPO
    / "data/splits/primary_volume_split_registry.csv"
)

case_df = split_registry[
    split_registry[
        "case_id"
    ].astype(str)
    == CASE_ID
]

if len(case_df) != 1:
    raise RuntimeError(
        f"Cannot uniquely resolve {CASE_ID}."
    )

case_record = case_df.iloc[0]

dataset_root = locate_dataset()

infection_path = (
    dataset_root
    / str(
        case_record[
            "infection_mask"
        ]
    )
)

img = nib.load(
    str(
        infection_path
    ),
    mmap=True,
)

lesion = np.asarray(
    img.dataobj
) > 0

affine = np.asarray(
    img.affine,
    dtype=np.float64,
)

spacing = np.asarray(
    img.header.get_zooms()[:3],
    dtype=np.float64,
)

voxel_volume_ml = (
    float(
        np.prod(
            spacing
        )
    )
    / 1000.0
)

component_labels, raw_count = ndi.label(
    lesion,
    structure=ndi.generate_binary_structure(
        3,
        3,
    ),
)

component_counts = np.bincount(
    component_labels.ravel()
)[1:]

component_volumes = (
    component_counts.astype(
        np.float64
    )
    * voxel_volume_ml
)

eligible_dense_ids = (
    np.where(
        component_volumes
        >= MIN_COMPONENT_ML
    )[0]
    + 1
)

K = len(
    eligible_dense_ids
)

if K != 39:
    raise RuntimeError(
        f"Expected 39 eligible components, reconstructed {K}."
    )

object_slices = ndi.find_objects(
    component_labels,
    max_label=raw_count,
)

erosion_structure = physical_ball(
    spacing,
    INTERIOR_EROSION_MM,
)

allowed_paths = {}
complete_fg_by_group = {}

fallback_groups = []

for group_id, dense_component_id in enumerate(
    eligible_dense_ids,
    start=1,
):
    dense_component_id = int(
        dense_component_id
    )

    slc = object_slices[
        dense_component_id - 1
    ]

    component_crop = (
        component_labels[
            slc
        ]
        == dense_component_id
    )

    eroded = ndi.binary_erosion(
        component_crop,
        structure=erosion_structure,
        border_value=0,
    )

    interior = (
        component_crop
        if not eroded.any()
        else eroded
    )

    component_rng = np.random.default_rng(
        stable_seed(
            GENERATOR_SEED,
            CASE_ID,
            "component_path",
            group_id,
        )
    )

    try:
        skeleton = skeletonize(
            interior.astype(bool)
        ).astype(bool)

    except Exception:
        skeleton = np.zeros_like(
            interior,
            dtype=bool,
        )

    local_path = approximate_skeleton_diameter_path(
        skeleton,
        component_rng,
    )

    used_fallback = False

    if len(local_path) == 0:
        used_fallback = True

        local_path = random_walk_fallback(
            interior,
            component_rng,
            MAX_ALLOWED_PATH_VOXELS,
        )

    if len(local_path) == 0:
        raise RuntimeError(
            f"Could not reconstruct group {group_id}."
        )

    if used_fallback:
        fallback_groups.append(
            group_id
        )

    starts = np.asarray(
        [
            slc[a].start
            for a in range(3)
        ],
        dtype=np.int32,
    )

    global_path = (
        local_path
        + starts[
            None,
            :
        ]
    )

    cap_rng = np.random.default_rng(
        stable_seed(
            GENERATOR_SEED,
            CASE_ID,
            "allowed_cap",
            group_id,
        )
    )

    global_path = contiguous_segment(
        global_path,
        min(
            len(global_path),
            MAX_ALLOWED_PATH_VOXELS,
        ),
        cap_rng,
    )

    allowed_paths[
        int(group_id)
    ] = global_path

    complete_rng = np.random.default_rng(
        stable_seed(
            GENERATOR_SEED,
            CASE_ID,
            "complete_stroke",
            group_id,
        )
    )

    complete_fg_by_group[
        int(group_id)
    ] = contiguous_segment(
        global_path,
        min(
            BASE_FG_STROKE_VOXELS,
            len(global_path),
        ),
        complete_rng,
    )

complete_budget = int(
    sum(
        len(v)
        for v in complete_fg_by_group.values()
    )
)

if complete_budget != 517:
    raise RuntimeError(
        f"Expected complete budget 517; reconstructed {complete_budget}."
    )

print(
    f"✓ Eligible groups       : {K}"
)

print(
    f"✓ Complete FG budget    : {complete_budget}"
)

print(
    f"✓ Skeleton fallback     : {fallback_groups}"
)


# ==========================================================================================
# 9. RECOVER RETAINED GROUPS + EXACT EXISTING BACKGROUND
# ==========================================================================================

heading(
    "STEP 3/9 — RECOVER FROZEN 25% MISSINGNESS AND BACKGROUND"
)

natural25_file = (
    ANNOTATION_DIR
    / f"{CASE_ID}__component_natural_25.npz"
)

complete_file = (
    ANNOTATION_DIR
    / f"{CASE_ID}__complete.npz"
)

with np.load(
    natural25_file,
    allow_pickle=False,
) as data:
    natural_label = np.asarray(
        data[
            "label"
        ]
    )

    natural_group = np.asarray(
        data[
            "group_id"
        ]
    )

retained_groups = sorted(
    int(x)
    for x in np.unique(
        natural_group[
            natural_label == 1
        ]
    )
)

expected_retained = max(
    1,
    math.ceil(
        COVERAGE
        * K
    ),
)

if len(retained_groups) != expected_retained:
    raise RuntimeError(
        "Frozen natural-25 retained group count changed."
    )

with np.load(
    complete_file,
    allow_pickle=False,
) as data:
    complete_voxel_ijk = np.asarray(
        data[
            "voxel_ijk"
        ],
        dtype=np.int32,
    )

    complete_label = np.asarray(
        data[
            "label"
        ],
        dtype=np.int8,
    )

background_coords = complete_voxel_ijk[
    complete_label == 0
]

background_count = len(
    background_coords
)

if background_count != 517:
    raise RuntimeError(
        f"Expected 517 background voxels; found {background_count}."
    )

retained_capacity = int(
    sum(
        len(
            allowed_paths[g]
        )
        for g in retained_groups
    )
)

shared_budget = min(
    complete_budget,
    retained_capacity,
)

print(
    f"Frozen retained groups : {retained_groups}"
)

print(
    f"Retained path capacity : {retained_capacity}"
)

print(
    f"Shared fixed budget    : {shared_budget}"
)

print(
    f"Background voxels      : {background_count}"
)

if shared_budget != 376:
    raise RuntimeError(
        f"Expected diagnosed shared budget 376; found {shared_budget}."
    )


# ==========================================================================================
# 10. GENERATE ONLY component_fixed_25
# ==========================================================================================

heading(
    "STEP 4/9 — GENERATE MISSING component_fixed_25"
)

component_rng = np.random.default_rng(
    stable_seed(
        GENERATOR_SEED,
        CASE_ID,
        "component_fixed",
        COVERAGE,
    )
)

(
    component_fixed_fg,
    component_ok,
    component_allocation,
) = build_contiguous_budget_annotation(
    allowed_paths,
    retained_groups,
    shared_budget,
    component_rng,
)

if not component_ok:
    raise RuntimeError(
        "Corrected allocator still failed component_fixed_25."
    )

component_fixed_count = int(
    sum(
        len(v)
        for v in component_fixed_fg.values()
    )
)

if component_fixed_count != 376:
    raise RuntimeError(
        f"component_fixed_25 budget {component_fixed_count} != 376."
    )

# Since target == retained total capacity, this must exactly saturate capacities.
for g in retained_groups:
    if (
        len(
            component_fixed_fg[g]
        )
        != len(
            allowed_paths[g]
        )
    ):
        raise RuntimeError(
            f"Group {g} was not saturated despite target == total capacity."
        )

component_annotation = assemble_annotation(
    component_fixed_fg,
    background_coords,
    affine,
)

component_file = (
    ANNOTATION_DIR
    / f"{CASE_ID}__component_fixed_25.npz"
)

save_npz(
    component_file,
    component_annotation,
)

component_hash = semantic_annotation_hash(
    component_annotation[
        "voxel_ijk"
    ],
    component_annotation[
        "world_xyz_mm"
    ],
    component_annotation[
        "label"
    ],
    component_annotation[
        "group_id"
    ],
)

print(
    f"✓ component_fixed_25 generated: {component_fixed_count} FG voxels"
)


# ==========================================================================================
# 11. GENERATE ONLY complete_fixed_25
# ==========================================================================================

heading(
    "STEP 5/9 — GENERATE MISSING complete_fixed_25"
)

complete_fixed_rng = np.random.default_rng(
    stable_seed(
        GENERATOR_SEED,
        CASE_ID,
        "complete_fixed_control",
        COVERAGE,
    )
)

(
    complete_fixed_fg,
    complete_control_ok,
    complete_allocation,
) = build_contiguous_budget_annotation(
    allowed_paths,
    sorted(
        allowed_paths.keys()
    ),
    shared_budget,
    complete_fixed_rng,
)

if not complete_control_ok:
    raise RuntimeError(
        "Corrected complete fixed-budget control unexpectedly failed."
    )

complete_fixed_count = int(
    sum(
        len(v)
        for v in complete_fixed_fg.values()
    )
)

if complete_fixed_count != 376:
    raise RuntimeError(
        f"complete_fixed_25 budget {complete_fixed_count} != 376."
    )

if len(
    complete_fixed_fg
) != 39:
    raise RuntimeError(
        "Complete fixed control does not preserve all 39 groups."
    )

complete_fixed_annotation = assemble_annotation(
    complete_fixed_fg,
    background_coords,
    affine,
)

complete_fixed_file = (
    ANNOTATION_DIR
    / f"{CASE_ID}__complete_fixed_25.npz"
)

save_npz(
    complete_fixed_file,
    complete_fixed_annotation,
)

complete_fixed_hash = semantic_annotation_hash(
    complete_fixed_annotation[
        "voxel_ijk"
    ],
    complete_fixed_annotation[
        "world_xyz_mm"
    ],
    complete_fixed_annotation[
        "label"
    ],
    complete_fixed_annotation[
        "group_id"
    ],
)

print(
    f"✓ complete_fixed_25 generated: {complete_fixed_count} FG voxels"
)


# ==========================================================================================
# 12. APPEND TWO MANIFEST ROWS
# ==========================================================================================

heading(
    "STEP 6/9 — UPDATE LOCAL TRAINER-SAFE MANIFEST"
)

if len(
    manifest
) != 258:
    raise RuntimeError(
        "Expected 258 pre-repair manifest rows."
    )

geometry_file = (
    f"geometry/{CASE_ID}.json"
)

common = {
    "case_id":
        CASE_ID,

    "source_subject_key":
        str(
            case_record[
                "source_subject_key"
            ]
        ),

    "split_role":
        str(
            case_record[
                "role"
            ]
        ),

    "outer_fold":
        case_record[
            "outer_fold"
        ],

    "ct_scan":
        str(
            case_record[
                "ct_scan"
            ]
        ),

    "geometry_file":
        geometry_file,

    "background_labelled_voxels":
        background_count,

    "generator_version":
        GENERATOR_VERSION_NEW,

    "generator_seed":
        GENERATOR_SEED,
}

new_rows = pd.DataFrame(
    [
        {
            **common,

            "condition":
                "component_fixed_25",

            "requested_coverage":
                0.25,

            "annotation_file":
                (
                    f"annotations/"
                    f"{CASE_ID}__component_fixed_25.npz"
                ),

            "foreground_labelled_voxels":
                component_fixed_count,

            "semantic_sha256":
                component_hash,
        },

        {
            **common,

            "condition":
                "complete_fixed_25",

            "requested_coverage":
                1.0,

            "annotation_file":
                (
                    f"annotations/"
                    f"{CASE_ID}__complete_fixed_25.npz"
                ),

            "foreground_labelled_voxels":
                complete_fixed_count,

            "semantic_sha256":
                complete_fixed_hash,
        },
    ]
)

# Preserve exact manifest schema/order.
new_rows = new_rows[
    manifest.columns
]

manifest = pd.concat(
    [
        manifest,
        new_rows,
    ],
    ignore_index=True,
)

if len(
    manifest
) != 260:
    raise RuntimeError(
        "Corrected manifest does not contain 260 rows."
    )

duplicates = manifest.duplicated(
    subset=[
        "case_id",
        "condition",
    ],
).sum()

if duplicates:
    raise RuntimeError(
        f"Duplicate case-condition rows after repair: {duplicates}"
    )

manifest = manifest.sort_values(
    [
        "case_id",
        "condition",
    ]
).reset_index(
    drop=True
)

manifest.to_csv(
    WEAK_MANIFEST,
    index=False,
)

print(
    "✓ Local weak_train manifest now contains 260 unique artifacts."
)


# ==========================================================================================
# 13. RUN FIREWALL + REGRESSION TESTS
# ==========================================================================================

heading(
    "STEP 7/9 — RUN FIREWALL AND ALLOCATOR REGRESSION TESTS"
)

SRC = REPO / "src"

if str(
    SRC
) not in sys.path:
    sys.path.insert(
        0,
        str(
            SRC
        ),
    )

from cora_lung.data.firewall import validate_weak_export

firewall_report = validate_weak_export(
    WEAK_ROOT
)

if (
    firewall_report[
        "annotation_files"
    ]
    != 260
):
    raise RuntimeError(
        "Firewall did not validate exactly 260 annotations."
    )

print(
    "✓ Runtime firewall: PASS"
)

pytest_result = sh(
    [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_firewall.py",
        "tests/test_scribble_budget.py",
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

if pytest_result.returncode != 0:
    raise RuntimeError(
        "Post-repair firewall/allocator tests FAILED."
    )

print(
    "✓ Firewall + allocator regression tests: PASS"
)


# ==========================================================================================
# 14. REAUDIT ALL 260 ARTIFACTS FROM DISK
# ==========================================================================================

heading(
    "STEP 8/9 — COMPLETE 260-ARTIFACT SCIENTIFIC AUDIT"
)

manifest = pd.read_csv(
    WEAK_MANIFEST
)

audit_rows = []

hash_failures = []

for _, row in tqdm(
    manifest.iterrows(),
    total=len(manifest),
    desc="Reauditing sparse artifacts",
    unit="artifact",
):
    path = (
        WEAK_ROOT
        / str(
            row[
                "annotation_file"
            ]
        )
    )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        voxel_ijk = np.asarray(
            data[
                "voxel_ijk"
            ],
            dtype=np.int32,
        )

        world_xyz = np.asarray(
            data[
                "world_xyz_mm"
            ],
            dtype=np.float32,
        )

        label = np.asarray(
            data[
                "label"
            ],
            dtype=np.int8,
        )

        group_id = np.asarray(
            data[
                "group_id"
            ],
            dtype=np.int32,
        )

    recomputed_hash = semantic_annotation_hash(
        voxel_ijk,
        world_xyz,
        label,
        group_id,
    )

    if (
        recomputed_hash
        != str(
            row[
                "semantic_sha256"
            ]
        )
    ):
        hash_failures.append(
            str(
                row[
                    "annotation_file"
                ]
            )
        )

    fg = (
        label == 1
    )

    bg = (
        label == 0
    )

    groups = sorted(
        int(x)
        for x in np.unique(
            group_id[
                fg
            ]
        )
    )

    audit_rows.append(
        {
            "case_id":
                str(
                    row[
                        "case_id"
                    ]
                ),

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
                str(
                    row[
                        "condition"
                    ]
                ),

            "foreground_voxels":
                int(
                    fg.sum()
                ),

            "background_voxels":
                int(
                    bg.sum()
                ),

            "observed_group_count":
                len(
                    groups
                ),

            "observed_groups":
                "|".join(
                    str(x)
                    for x in groups
                ),

            "background_hash":
                coordinate_set_hash(
                    voxel_ijk[
                        bg
                    ]
                ),

            "semantic_sha256":
                recomputed_hash,
        }
    )

if hash_failures:
    raise RuntimeError(
        "Semantic hash failures:\n"
        + "\n".join(
            hash_failures
        )
    )

audit_df = pd.DataFrame(
    audit_rows
)

if len(
    audit_df
) != 260:
    raise RuntimeError(
        f"Expected 260 audited artifacts; found {len(audit_df)}."
    )


# ------------------------------------------------------------------------------------------
# Full 13-condition matrix.
# ------------------------------------------------------------------------------------------

expected_conditions = set(
    all_expected_conditions
)

condition_failures = []

for case_id in sorted(
    audit_df[
        "case_id"
    ].unique()
):
    observed = set(
        audit_df.loc[
            audit_df[
                "case_id"
            ]
            == case_id,
            "condition",
        ]
    )

    if observed != expected_conditions:
        condition_failures.append(
            (
                case_id,
                sorted(
                    expected_conditions
                    - observed
                ),
            )
        )

if condition_failures:
    raise RuntimeError(
        f"Condition matrix incomplete: {condition_failures}"
    )

print(
    "✓ Complete 20 × 13 condition matrix: PASS"
)


# ------------------------------------------------------------------------------------------
# Exact background coordinate equality.
# ------------------------------------------------------------------------------------------

bg_group = (
    audit_df.groupby(
        "case_id"
    )[
        "background_hash"
    ]
    .nunique()
)

bg_exact = bool(
    (
        bg_group
        == 1
    ).all()
)

if not bg_exact:
    raise RuntimeError(
        "Background coordinate realization differs across paired conditions."
    )

print(
    "✓ Identical BG coordinates across all conditions: PASS"
)


# ------------------------------------------------------------------------------------------
# Nested component coverage + budget matching.
# ------------------------------------------------------------------------------------------

natural_pixel_pairs = 0
fixed_pairs = 0

nested_pass = True
natural_pixel_pass = True
fixed_pass = True
group_contract_pass = True

for case_id in sorted(
    audit_df[
        "case_id"
    ].unique()
):
    case = audit_df[
        audit_df[
            "case_id"
        ]
        == case_id
    ].set_index(
        "condition"
    )

    complete_groups = {
        int(x)
        for x in str(
            case.loc[
                "complete",
                "observed_groups",
            ]
        ).split("|")
        if x
    }

    K_case = len(
        complete_groups
    )

    natural_sets = {}

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

        natural_groups = {
            int(x)
            for x in str(
                case.loc[
                    natural_name,
                    "observed_groups",
                ]
            ).split("|")
            if x
        }

        fixed_groups = {
            int(x)
            for x in str(
                case.loc[
                    fixed_name,
                    "observed_groups",
                ]
            ).split("|")
            if x
        }

        pixel_groups = {
            int(x)
            for x in str(
                case.loc[
                    pixel_name,
                    "observed_groups",
                ]
            ).split("|")
            if x
        }

        complete_fixed_groups = {
            int(x)
            for x in str(
                case.loc[
                    complete_fixed_name,
                    "observed_groups",
                ]
            ).split("|")
            if x
        }

        expected_retained = max(
            1,
            math.ceil(
                (
                    cov / 100.0
                )
                * K_case
            ),
        )

        if (
            len(
                natural_groups
            )
            != expected_retained
        ):
            group_contract_pass = False

        if (
            natural_groups
            != fixed_groups
        ):
            group_contract_pass = False

        if (
            pixel_groups
            != complete_groups
        ):
            group_contract_pass = False

        if (
            complete_fixed_groups
            != complete_groups
        ):
            group_contract_pass = False

        natural_sets[
            cov
        ] = natural_groups

        # Natural component omission vs matched pixel thinning.
        natural_fg = int(
            case.loc[
                natural_name,
                "foreground_voxels",
            ]
        )

        pixel_fg = int(
            case.loc[
                pixel_name,
                "foreground_voxels",
            ]
        )

        natural_pixel_pairs += 1

        if natural_fg != pixel_fg:
            natural_pixel_pass = False

        # Fixed component omission vs complete fixed-budget control.
        fixed_fg = int(
            case.loc[
                fixed_name,
                "foreground_voxels",
            ]
        )

        complete_fixed_fg = int(
            case.loc[
                complete_fixed_name,
                "foreground_voxels",
            ]
        )

        fixed_pairs += 1

        if (
            fixed_fg
            != complete_fixed_fg
        ):
            fixed_pass = False

    if not (
        natural_sets[
            25
        ].issubset(
            natural_sets[
                50
            ]
        )
        and natural_sets[
            50
        ].issubset(
            natural_sets[
                75
            ]
        )
        and natural_sets[
            75
        ].issubset(
            complete_groups
        )
    ):
        nested_pass = False


if natural_pixel_pairs != 60:
    raise RuntimeError(
        f"Expected 60 natural/pixel pairs; found {natural_pixel_pairs}."
    )

if fixed_pairs != 60:
    raise RuntimeError(
        f"Expected 60 fixed-budget pairs; found {fixed_pairs}."
    )

if not natural_pixel_pass:
    raise RuntimeError(
        "At least one natural-vs-pixel budget mismatch exists."
    )

if not fixed_pass:
    raise RuntimeError(
        "At least one fixed-budget pair mismatch exists."
    )

if not nested_pass:
    raise RuntimeError(
        "Nested missingness contract failed."
    )

if not group_contract_pass:
    raise RuntimeError(
        "Foreground-group coverage contract failed."
    )

print(
    "✓ Natural-vs-pixel exact FG budgets: PASS (60/60)"
)

print(
    "✓ Fixed component-vs-complete exact FG budgets: PASS (60/60)"
)

print(
    "✓ Foreground group-preservation contracts: PASS"
)

print(
    "✓ Nested 25% ⊂ 50% ⊂ 75% missingness: PASS"
)


# ==========================================================================================
# 15. CALCULATE FINAL SUMMARY STATISTICS
# ==========================================================================================

complete_budget_df = (
    audit_df[
        audit_df[
            "condition"
        ]
        == "complete"
    ][
        [
            "case_id",
            "foreground_voxels",
        ]
    ]
    .rename(
        columns={
            "foreground_voxels":
                "complete_fg"
        }
    )
)

fixed_rows = audit_df[
    audit_df[
        "condition"
    ].str.startswith(
        "component_fixed_"
    )
].merge(
    complete_budget_df,
    on="case_id",
    how="left",
)

fixed_rows[
    "budget_reduced"
] = (
    fixed_rows[
        "foreground_voxels"
    ]
    < fixed_rows[
        "complete_fg"
    ]
)

fixed_budget_reductions = int(
    fixed_rows[
        "budget_reduced"
    ].sum()
)

previous_audit_path = (
    REPO
    / "experiments/audits/block05_weak_label_generator.json"
)

previous_audit = json.loads(
    previous_audit_path.read_text(
        encoding="utf-8"
    )
)

skeleton_fallbacks = (
    previous_audit
    .get(
        "generation_time_counters",
        {}
    )
    .get(
        "total_skeleton_fallbacks",
        None,
    )
)

background_fallbacks = (
    previous_audit
    .get(
        "generation_time_counters",
        {}
    )
    .get(
        "total_background_fallbacks",
        None,
    )
)

print()
print(
    f"Fixed-budget reductions after repair: {fixed_budget_reductions}/60"
)


# ==========================================================================================
# 16. FREEZE REPRODUCIBILITY TABLES
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


# No patient sparse coordinates are committed.
generation_summary = audit_df[
    [
        "case_id",
        "source_subject_key",
        "split_role",
        "outer_fold",
        "condition",
        "foreground_voxels",
        "background_voxels",
        "observed_group_count",
        "semantic_sha256",
    ]
].copy()

generation_summary.to_csv(
    manifest_dir
    / "weak_label_generation_summary.csv",
    index=False,
)


semantic_export = manifest[
    [
        "case_id",
        "source_subject_key",
        "split_role",
        "outer_fold",
        "condition",
        "requested_coverage",
        "foreground_labelled_voxels",
        "background_labelled_voxels",
        "generator_version",
        "generator_seed",
        "semantic_sha256",
    ]
].copy()

semantic_export.to_csv(
    manifest_dir
    / "weak_label_artifact_semantic_hashes.csv",
    index=False,
)


condition_summary = (
    audit_df.groupby(
        "condition",
        as_index=False,
    )
    .agg(
        cases=(
            "case_id",
            "nunique",
        ),

        median_foreground_voxels=(
            "foreground_voxels",
            "median",
        ),

        q25_foreground_voxels=(
            "foreground_voxels",
            lambda x:
                float(
                    np.quantile(
                        x,
                        0.25,
                    )
                ),
        ),

        q75_foreground_voxels=(
            "foreground_voxels",
            lambda x:
                float(
                    np.quantile(
                        x,
                        0.75,
                    )
                ),
        ),

        median_background_voxels=(
            "background_voxels",
            "median",
        ),

        median_observed_groups=(
            "observed_group_count",
            "median",
        ),
    )
)

condition_summary.to_csv(
    manifest_dir
    / "weak_label_condition_summary.csv",
    index=False,
)


# Missing-condition audit should now be EMPTY.
empty_missing = pd.DataFrame(
    columns=[
        "case_id",
        "condition",
        "coverage_token",
        "mathematically_infeasible_pixel_control",
        "justification",
    ]
)

empty_missing.to_csv(
    manifest_dir
    / "weak_label_missing_condition_audit.csv",
    index=False,
)


# ==========================================================================================
# 17. REGENERATE PUBLICATION FIGURE
# ==========================================================================================

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 9.5,
        "savefig.dpi": 600,
    }
)

records = []

complete_df = audit_df[
    audit_df[
        "condition"
    ]
    == "complete"
]

records.append(
    {
        "regime":
            "Component-Complete",

        "coverage":
            100,

        "median_fg":
            float(
                complete_df[
                    "foreground_voxels"
                ].median()
            ),
    }
)

for cov in [
    75,
    50,
    25,
]:
    mappings = [
        (
            f"component_natural_{cov}",
            "Whole-Component Omission — Natural Budget",
        ),
        (
            f"pixel_dropout_matched_{cov}",
            "Random-Pixel Dropout — Matched Natural Budget",
        ),
        (
            f"component_fixed_{cov}",
            "Whole-Component Omission — Fixed Pixel Budget",
        ),
        (
            f"complete_fixed_{cov}",
            "Component-Complete — Fixed Pixel Control",
        ),
    ]

    for condition, label in mappings:
        subset = audit_df[
            audit_df[
                "condition"
            ]
            == condition
        ]

        records.append(
            {
                "regime":
                    label,

                "coverage":
                    cov,

                "median_fg":
                    float(
                        subset[
                            "foreground_voxels"
                        ].median()
                    ),
            }
        )

plot_df = pd.DataFrame(
    records
)

fig, ax = plt.subplots(
    figsize=(
        11.5,
        6.8,
    )
)

for regime, group in plot_df.groupby(
    "regime",
    sort=False,
):
    group = group.sort_values(
        "coverage"
    )

    ax.plot(
        group[
            "coverage"
        ],
        group[
            "median_fg"
        ],
        marker="o",
        linewidth=2,
        markersize=6,
        label=regime,
    )

ax.set_xlabel(
    "Retained Lesion-Component Coverage (%)",
    fontweight="bold",
)

ax.set_ylabel(
    "Median Foreground-Labelled Voxels per CT Volume",
    fontweight="bold",
)

ax.set_title(
    "Annotation-Budget Equivalence Across the Frozen CORA-Lung Weak-Supervision Regimes\n"
    "Separating Whole-Component Missingness from Matched Pixel Sparsity",
    fontweight="bold",
)

ax.set_xticks(
    [
        25,
        50,
        75,
        100,
    ]
)

ax.grid(
    alpha=0.25
)

legend = ax.legend()

for text in legend.get_texts():
    text.set_fontweight(
        "bold"
    )

fig.tight_layout()

figure_stem = (
    figure_dir
    / "fig05_weak_supervision_annotation_budget"
)

fig.savefig(
    figure_stem.with_suffix(
        ".png"
    ),
    dpi=600,
    bbox_inches="tight",
)

fig.savefig(
    figure_stem.with_suffix(
        ".pdf"
    ),
    bbox_inches="tight",
)

plt.close(
    fig
)

print(
    "✓ Publication figure regenerated from complete 260-artifact set."
)


# ==========================================================================================
# 18. UPDATE GENERATOR CONFIG TO v1.1
# ==========================================================================================

config_path = (
    REPO
    / "configs/weak_labels_primary.yaml"
)

config_text = config_path.read_text(
    encoding="utf-8"
)

config_text = config_text.replace(
    'version: "1.0"',
    'version: "1.1"',
    1,
)

if "allocator:" not in config_text:
    config_text += """

allocator:
  version: "1.1"
  strategy: exact_capacity_aware_cyclic
  arbitrary_iteration_limit: false
  behavior_change: feasible_allocations_no_longer_aborted_by_safety_counter
  regression_case: coronacases_003_component_fixed_25
"""

config_path.write_text(
    config_text,
    encoding="utf-8",
)


# ==========================================================================================
# 19. UPDATE PROTOCOL DOCUMENTATION
# ==========================================================================================

protocol_path = (
    REPO
    / "docs/weak_label_protocol.md"
)

protocol = protocol_path.read_text(
    encoding="utf-8"
)

repair_note = f"""

## Generator v1.1 allocator correction

Before any model training, the Block-05 audit detected an implementation-only failure in
`coronacases_003` at the secondary 25% fixed-pixel condition.

The retained ten groups had exactly **376 voxels** of allowed connected-path capacity, and
the requested shared budget was also **376 voxels**. The comparison was therefore
mathematically feasible.

Generator v1.0 used a round-robin allocator with an arbitrary iteration safety threshold.
For this case it stopped at iteration 2505 when the safety threshold was 2504, with five
valid allocations still remaining.

Generator v1.1 removes the arbitrary iteration cutoff. It preserves the same deterministic
shuffled cyclic group visitation order and terminates only when:

1. the exact requested budget is allocated; or
2. a complete pass makes no progress, indicating true infeasibility.

The original executed v1.0 generator is preserved under
`scripts/code_blocks/archive/`.

No model had been trained when this correction was made.

After correction:

- all **260/260** expected weak-supervision artifacts are present;
- all **60/60** natural-component vs random-pixel budget pairs match exactly;
- all **60/60** component-fixed vs complete-fixed budget pairs match exactly;
- all paired background coordinate realizations are identical;
- 25%, 50%, and 75% component missingness remains nested;
- the dense-label firewall passes.
"""

if (
    "## Generator v1.1 allocator correction"
    not in protocol
):
    protocol += repair_note

protocol = protocol.replace(
    "**Generator version:** 1.0",
    "**Generator version:** 1.1",
)

protocol = protocol.replace(
    "**FAIL**",
    "**PASS**",
)

protocol_path.write_text(
    protocol,
    encoding="utf-8",
)


# ==========================================================================================
# 20. FINAL BLOCK-05 AUDIT
# ==========================================================================================

block05_audit = {
    "project":
        PROJECT,

    "block":
        BLOCK_ID,

    "corrective_block":
        REPAIR_ID,

    "generated_at_utc":
        NOW_ISO,

    "status":
        "PASS",

    "generator": {
        "version":
            GENERATOR_VERSION_NEW,

        "seed":
            GENERATOR_SEED,

        "allocator":
            "exact_capacity_aware_cyclic",

        "allocator_regression_case":
            (
                "coronacases_003/"
                "component_fixed_25"
            ),
    },

    "repair": {
        "model_training_occurred_before_repair":
            False,

        "pre_repair_artifacts":
            258,

        "artifacts_regenerated":
            0,

        "existing_artifacts_modified":
            0,

        "new_artifacts_created":
            2,

        "post_repair_artifacts":
            260,

        "root_cause":
            (
                "v1.0 arbitrary allocator safety limit "
                "aborted a mathematically feasible allocation"
            ),

        "diagnosed_target_budget":
            376,

        "diagnosed_retained_capacity":
            376,

        "v1_0_safety_limit":
            2504,

        "v1_0_abort_iteration":
            2505,

        "v1_0_remaining_voxels_at_abort":
            5,
    },

    "scientific_contract": {
        "complete_condition_matrix":
            "PASS_260_OF_260",

        "natural_pixel_exact_pairs":
            "PASS_60_OF_60",

        "fixed_budget_exact_pairs":
            "PASS_60_OF_60",

        "background_coordinate_identity":
            "PASS",

        "nested_component_missingness":
            "PASS",

        "foreground_group_contract":
            "PASS",

        "semantic_hash_validation":
            "PASS",

        "dense_label_firewall":
            "PASS",
    },

    "diagnostics": {
        "skeleton_fallbacks":
            skeleton_fallbacks,

        "background_sampling_fallbacks":
            background_fallbacks,

        "fixed_budget_reductions":
            fixed_budget_reductions,
    },

    "training_performed":
        False,

    "dense_masks_exported":
        False,

    "unknown_voxels_materialized_as_background":
        False,
}

write_json(
    audit_dir
    / "block05_weak_label_generator.json",
    block05_audit,
)


# ==========================================================================================
# 21. CAPTURE 05D SOURCE
# ==========================================================================================

source_capture = "NOT_AVAILABLE"

try:
    ip = get_ipython()

    raw_cell = (
        ip.history_manager
        .input_hist_raw[-1]
    )

    if (
        "CORA-LUNG — CODE BLOCK 05D"
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
            / "block05d_allocator_repair.py"
        ).write_text(
            raw_cell,
            encoding="utf-8",
        )

        source_capture = "PASS"

except Exception:
    pass


# ==========================================================================================
# 22. PROJECT STATE
# ==========================================================================================

state_path = (
    REPO
    / "PROJECT_STATE.json"
)

state = json.loads(
    state_path.read_text(
        encoding="utf-8"
    )
)

state.update(
    {
        "last_completed_block":
            "05",

        "last_attempted_block":
            "05",

        "last_completed_block_name":
            "deterministic_component_aware_weak_label_generator",

        "current_stage":
            "native_weak_labels_generated_and_firewall_verified",

        "current_gate":
            "POST_GATE_A_PRE_GATE_B",

        "gate_a":
            "PASS",

        "gate_b":
            "NOT_RUN",

        "weak_label_generator":
            "PASS",

        "weak_label_generator_version":
            GENERATOR_VERSION_NEW,

        "weak_label_generator_seed":
            GENERATOR_SEED,

        "weak_label_allocator":
            "exact_capacity_aware_cyclic_v1.1",

        "allocator_regression_repair":
            "PASS",

        "weak_annotation_artifacts":
            260,

        "primary_50_percent_matched_control":
            "PASS",

        "all_coverage_matched_controls":
            "PASS",

        "dense_label_firewall":
            "PASS",

        "weak_export_reconstructible":
            True,

        "training_authorized":
            False,

        "next_action":
            (
                "Code Block 06: construct the firewall-safe resampled CT "
                "and weak-label training cache with physical-coordinate "
                "transfer, collision auditing, and storage/RAM profiling."
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
# 23. REPOSITORY MANIFEST
# ==========================================================================================

repo_manifest = []

repo_files = sorted(
    [
        p for p in REPO.rglob("*")
        if p.is_file()
        and ".git" not in p.parts
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
            "05",

        "corrective_block":
            "05D",

        "files":
            repo_manifest,
    },
)


# ==========================================================================================
# 24. FINAL GIT COMMIT + PUSH
# ==========================================================================================

heading(
    "STEP 9/9 — COMMIT CORRECTED GENERATOR v1.1"
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

    commit_message = (
        "fix: repair exact weak-label budget allocator and complete Block 05"
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

else:
    print(
        "✓ No new repository changes."
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
# 25. CLEAN AUTH
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
# 26. FINAL REPORT
# ==========================================================================================

def median_budget(condition):
    x = audit_df.loc[
        audit_df[
            "condition"
        ]
        == condition,
        "foreground_voxels",
    ]

    return float(
        x.median()
    )


print("\n")
print("=" * 114)
print("CORA-LUNG CODE BLOCK 05D — FINAL GENERATOR v1.1 FREEZE REPORT")
print("=" * 114)

print(f"""
ALLOCATOR ROOT CAUSE
--------------------
Failed case                           : {CASE_ID}
Failed regime                         : component_fixed_25
Retained components                   : {len(retained_groups)}
Retained connected-path capacity      : {retained_capacity}
Required shared budget                : {shared_budget}
Mathematically feasible               : YES

v1.0 arbitrary safety limit           : 2504
v1.0 abort iteration                  : 2505
Voxels remaining at erroneous abort   : 5

CORRECTION
----------
Old generator                         : v{GENERATOR_VERSION_OLD}
Corrected generator                   : v{GENERATOR_VERSION_NEW}
Allocator                             : exact capacity-aware cyclic allocation
Existing annotations regenerated      : 0
Existing annotations modified         : 0
Missing annotations generated         : 2

FINAL WEAK SUPERVISION
----------------------
CT volumes                            : 20
Conditions per volume                 : 13
Expected annotation artifacts         : 260
Observed annotation artifacts         : {len(audit_df)}
Complete condition matrix             : PASS

PAIRWISE CAUSAL CONTROLS
------------------------
Natural-vs-pixel pairs                : {natural_pixel_pairs}/60
Exact natural/pixel FG budgets        : PASS
Fixed component/control pairs         : {fixed_pairs}/60
Exact fixed-pixel FG budgets          : PASS
Identical BG coordinate realization  : PASS
Nested 25% ⊂ 50% ⊂ 75% coverage      : PASS
Foreground group contracts            : PASS

50% PRIMARY GATE-B SUPERVISION
------------------------------
Complete median FG voxels             : {median_budget('complete'):.1f}
Natural component-omission median     : {median_budget('component_natural_50'):.1f}
Matched pixel-dropout median          : {median_budget('pixel_dropout_matched_50'):.1f}
Component-fixed median                : {median_budget('component_fixed_50'):.1f}
Complete-fixed median                 : {median_budget('complete_fixed_50'):.1f}

GENERATOR DIAGNOSTICS
---------------------
Skeleton fallbacks                    : {skeleton_fallbacks}
Background sampling fallbacks         : {background_fallbacks}
Fixed-budget reductions               : {fixed_budget_reductions}/60

FIREWALL / QA
-------------
Runtime dense-label firewall          : PASS
Firewall tests                        : PASS
Allocator regression tests            : PASS
Semantic annotation hashes            : PASS
Dense masks exported                  : NO
Unknown voxels converted to BG        : NO

REPRODUCIBILITY
---------------
Original executed v1.0 archived       : YES
Canonical generator patched to v1.1   : YES
05D correction source captured        : {source_capture}
Patient sparse arrays in GitHub       : NO
Sparse annotations reconstructible    : YES

SCIENTIFIC STATUS
-----------------
Gate A                                : PASS
Gate B                                : NOT RUN
Weak-label generator                  : PASS
Dense-label firewall                  : PASS
Training                              : NOT STARTED

GITHUB
------
Starting commit                       : {starting_commit[:12]}
Current commit                        : {current_commit[:12]}
Synchronization                       : PASS

NEXT
----
Do NOT train yet.

Send me this complete report.

If PASS, Code Block 06 will build the firewall-safe training-grid cache:
  • CT HU clipping and normalization
  • frozen 3.0 × 1.5 × 1.5 mm training spacing
  • image-derived crop only
  • physical-coordinate transfer of all sparse labels
  • same-class collision merging
  • conflicting-class collision rejection
  • labelled-voxel retention audit
  • group-ID survival audit
  • restartable preprocessing cache
  • storage/RAM/GPU-readiness profiling

Only after that cache passes will we start the Gate-B pilot.
""")

print("=" * 114)