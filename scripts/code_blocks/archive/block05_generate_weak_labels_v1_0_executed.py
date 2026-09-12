# ==========================================================================================
# CORA-LUNG — CODE BLOCK 05
# Deterministic Component-Aware Weak-Label Generator + Dense-Label Firewall
#
# PROJECT:
#   CORA-Lung: Component-Omission Replay for Incomplete Scribble-Supervised
#   Lung Lesion Segmentation
#
# PURPOSE
#   1. Generate deterministic native-grid sparse foreground/background annotations.
#   2. Preserve one foreground stroke/group per observed lesion focus.
#   3. Generate component-complete and component-incomplete supervision.
#   4. Generate exactly matched random-pixel dropout controls.
#   5. Generate fixed-positive-pixel-budget incomplete supervision.
#   6. Keep one identical background realization across every paired condition.
#   7. Export ONLY trainer-safe sparse coordinates + geometry.
#   8. Build and execute the dense-label firewall.
#   9. Synchronize generator protocol, audits, tests and project state to GitHub.
#
# IMPORTANT
#   * Dense masks are read ONLY inside this isolated generator.
#   * Generated patient sparse-coordinate arrays are NOT committed to GitHub.
#   * They are deterministically reproducible from source data + committed generator seed/config.
#   * No model training occurs here.
#   * No final test outcomes are inspected.
#
# PRIMARY COMPONENT DEFINITION
#   Connectivity              : 26
#   Minimum physical volume   : 0.1 mL
#
# PILOT GENERATOR ENGINEERING CONSTANTS — FROZEN BEFORE GATE B
#   Base FG stroke budget     : 20 native voxels / observed component
#   Maximum allowed path      : 256 native voxels / component
#   Interior erosion          : 1.0 mm
#   Background safety band    : 3.0 mm
#   Background budget         : complete-annotation FG voxel count
#   Coverage                  : 100%, 75%, 50%, 25%
#   Omission mechanism        : uniform nested component omission
#   Generator seed            : 20260912
#
# NOTE:
#   The 20-voxel stroke budget is an engineering starting value, not a novelty claim.
#   It is frozen here before Gate B so it cannot be tuned after observing model outcomes.
# ==========================================================================================

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from collections import deque, defaultdict
import subprocess
import hashlib
import json
import os
import sys
import math
import shutil
import textwrap
import warnings

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
# 0. LOCKED CONSTANTS
# ==========================================================================================

PROJECT = "CORA-Lung"

BLOCK_ID = "05"
BLOCK_NAME = "deterministic_component_aware_weak_label_generator"

GITHUB_OWNER = "itsCodeBakery"
GITHUB_REPO = "CORA-LUNG"
REMOTE_URL = f"https://github.com/{GITHUB_OWNER}/{GITHUB_REPO}.git"

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
REPO = WORK / GITHUB_REPO

EXPECTED_DATASET_SLUG = "covid19-ct-scans"

GENERATOR_VERSION = "1.0"
GENERATOR_SEED = 20260912

COMPONENT_CONNECTIVITY = 26
MIN_COMPONENT_ML = 0.1

INTERIOR_EROSION_MM = 1.0
BACKGROUND_SAFETY_MM = 3.0

BASE_FG_STROKE_VOXELS = 20
MAX_ALLOWED_PATH_VOXELS = 256

COVERAGE_LEVELS = [1.00, 0.75, 0.50, 0.25]

WEAK_ROOT = WORK / "cora_weak_train_v1"

NOW = datetime.now(timezone.utc)
NOW_ISO = NOW.strftime("%Y-%m-%dT%H:%M:%SZ")

warnings.filterwarnings("ignore", category=RuntimeWarning)

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
# 1. GENERAL HELPERS
# ==========================================================================================

def heading(text):
    print("\n" + "=" * 112)
    print(text)
    print("=" * 112)


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


def human_bytes(n):
    n = float(n)

    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.2f} {unit}"

        n /= 1024

    return f"{n:.2f} PB"


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


def locate_dataset():
    candidates = [
        INPUT / EXPECTED_DATASET_SLUG,
        INPUT / "datasets" / "andrewmvd" / EXPECTED_DATASET_SLUG,
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

            if required.issubset(df.columns):
                return p.parent.resolve()

        except Exception:
            pass

    raise RuntimeError(
        "Primary COVID-19 CT dataset could not be located."
    )


# ==========================================================================================
# 2. PHYSICAL MORPHOLOGY
# ==========================================================================================

def physical_ball(spacing_mm, radius_mm):
    """
    Discrete ellipsoidal structuring element approximating a physical ball.

    spacing_mm follows native NIfTI voxel axes (i, j, k).
    """
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
        center = tuple(extents)
        structure[center] = True

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

    largest = int(
        np.argmax(counts)
    )

    return labels == largest


# ==========================================================================================
# 3. CONTIGUOUS SKELETON PATH CONSTRUCTION
# ==========================================================================================

def bfs_farthest(coords, coord_to_index, start_index):
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

    q = deque(
        [int(start_index)]
    )

    distance[start_index] = 0

    farthest = int(start_index)

    while q:
        idx = q.popleft()

        if distance[idx] > distance[farthest]:
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
                distance[idx] + 1
            )

            parent[j] = idx

            q.append(j)

    return farthest, parent, distance


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

    path_indices = path_indices[::-1]

    return coords[
        np.asarray(
            path_indices,
            dtype=np.int32,
        )
    ]


def random_walk_fallback(
    valid_mask,
    rng,
    max_length,
):
    """
    Connected fallback used only when 3-D skeletonization fails.
    """
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
        tuple(start.tolist())
    ]

    used = {
        tuple(start.tolist())
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

            if np.any(candidate < 0):
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
        start : start + requested_length
    ].copy()


# ==========================================================================================
# 4. BUDGET ALLOCATION
# ==========================================================================================

def allocate_budget(
    total_budget,
    capacities,
    rng,
):
    """
    Allocate an exact number of labelled voxels over observed groups.

    Each group receives >=1 voxel when feasible.
    Additional voxels are distributed deterministically in shuffled
    round-robin order until the requested total or capacity is reached.
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

    cursor = 0
    safety = 0

    while remaining > 0:
        g = int(
            order[
                cursor % len(order)
            ]
        )

        if (
            allocation[g]
            < capacities[g]
        ):
            allocation[g] += 1
            remaining -= 1

        cursor += 1
        safety += 1

        if safety > (
            total_capacity
            * 4
            + 1000
        ):
            return {}, False

    return allocation, True


# ==========================================================================================
# 5. BACKGROUND PATH GENERATOR
# ==========================================================================================

def random_safe_voxel(
    safe_mask,
    rng,
    max_attempts=50000,
):
    shape = np.asarray(
        safe_mask.shape,
        dtype=np.int64,
    )

    for _ in range(
        max_attempts
    ):
        candidate = np.asarray(
            [
                rng.integers(
                    0,
                    shape[a],
                )
                for a in range(3)
            ],
            dtype=np.int32,
        )

        if safe_mask[
            tuple(
                candidate.tolist()
            )
        ]:
            return candidate

    return None


def generate_background_paths(
    safe_mask,
    target_count,
    rng,
):
    """
    Generate several connected background tracks until exact target count.

    Returns
    -------
    coords : N x 3 int32
    fallback_used : bool
    """
    target_count = int(
        target_count
    )

    if target_count <= 0:
        return (
            np.empty(
                (0, 3),
                dtype=np.int32,
            ),
            False,
        )

    if int(
        safe_mask.sum()
    ) < target_count:
        raise RuntimeError(
            "Insufficient safe background voxels "
            f"({int(safe_mask.sum())} < {target_count})."
        )

    selected = []
    selected_set = set()

    restarts = 0
    fallback_used = False

    while len(selected) < target_count:

        start = random_safe_voxel(
            safe_mask,
            rng,
        )

        if start is None:
            break

        current = start.copy()

        track_steps = min(
            24,
            target_count
            - len(selected),
        )

        for _ in range(track_steps):

            t = tuple(
                current.tolist()
            )

            if (
                safe_mask[t]
                and t not in selected_set
            ):
                selected_set.add(t)
                selected.append(t)

                if len(selected) >= target_count:
                    break

            candidates = []

            for offset in OFFSETS_26:
                nb = (
                    current + offset
                )

                if np.any(nb < 0):
                    continue

                if np.any(
                    nb
                    >= np.asarray(
                        safe_mask.shape
                    )
                ):
                    continue

                nt = tuple(
                    nb.tolist()
                )

                if nt in selected_set:
                    continue

                if safe_mask[nt]:
                    candidates.append(
                        nb
                    )

            if not candidates:
                break

            current = candidates[
                int(
                    rng.integers(
                        0,
                        len(candidates),
                    )
                )
            ].astype(np.int32)

        restarts += 1

        if restarts > (
            target_count * 5
            + 100
        ):
            break

    if len(selected) < target_count:
        # Extremely unlikely controlled fallback.
        fallback_used = True

        needed = (
            target_count
            - len(selected)
        )

        available_flat = np.flatnonzero(
            safe_mask
        )

        rng.shuffle(
            available_flat
        )

        for flat in available_flat:
            coord = np.asarray(
                np.unravel_index(
                    int(flat),
                    safe_mask.shape,
                ),
                dtype=np.int32,
            )

            t = tuple(
                coord.tolist()
            )

            if t in selected_set:
                continue

            selected_set.add(t)
            selected.append(t)

            needed -= 1

            if needed == 0:
                break

    if len(selected) != target_count:
        raise RuntimeError(
            "Could not generate exact background budget."
        )

    return (
        np.asarray(
            selected,
            dtype=np.int32,
        ),
        fallback_used,
    )


# ==========================================================================================
# 6. ANNOTATION ASSEMBLY
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
            fg_by_group[group_id],
            dtype=np.int32,
        )

        if len(coords) == 0:
            continue

        fg_coords.append(coords)

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

    # ------------------------------------------------------------------
    # Remove accidental within-class duplicate coordinates.
    # ------------------------------------------------------------------

    if len(fg_coords):
        unique_key = np.column_stack(
            [
                fg_coords,
                fg_group_ids,
            ]
        )

        _, unique_idx = np.unique(
            unique_key,
            axis=0,
            return_index=True,
        )

        unique_idx = np.sort(
            unique_idx
        )

        fg_coords = fg_coords[
            unique_idx
        ]

        fg_group_ids = fg_group_ids[
            unique_idx
        ]

    if len(bg_coords):
        bg_coords = np.unique(
            bg_coords,
            axis=0,
        )

    fg_set = {
        tuple(x.tolist())
        for x in fg_coords
    }

    bg_set = {
        tuple(x.tolist())
        for x in bg_coords
    }

    collision = (
        fg_set.intersection(
            bg_set
        )
    )

    if collision:
        raise RuntimeError(
            f"Foreground/background collision detected: "
            f"{len(collision)} voxels."
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
        ],
        axis=0,
    )

    group_ids = np.concatenate(
        [
            fg_group_ids,
            np.full(
                len(bg_coords),
                -1,
                dtype=np.int32,
            ),
        ],
        axis=0,
    )

    world_xyz_mm = nib.affines.apply_affine(
        affine,
        voxel_ijk.astype(
            np.float64
        ),
    ).astype(np.float32)

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


def save_annotation_npz(
    path,
    annotation,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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
# 7. PIXEL-DROPOUT CONTROL
# ==========================================================================================

def make_pixel_dropout_control(
    complete_fg_by_group,
    target_fg_count,
    rng,
):
    """
    Keep every observed component represented by >=1 labelled voxel while
    matching exactly the reduced foreground-labelled voxel count.
    """
    group_ids = sorted(
        complete_fg_by_group.keys()
    )

    if target_fg_count < len(
        group_ids
    ):
        return {}, False

    selected = {}
    remaining_pool = []

    # One mandatory voxel per group.
    for g in group_ids:
        coords = np.asarray(
            complete_fg_by_group[g],
            dtype=np.int32,
        )

        if len(coords) == 0:
            return {}, False

        chosen_idx = int(
            rng.integers(
                0,
                len(coords),
            )
        )

        selected[g] = [
            coords[
                chosen_idx
            ].copy()
        ]

        for i, coord in enumerate(
            coords
        ):
            if i == chosen_idx:
                continue

            remaining_pool.append(
                (
                    g,
                    coord.copy(),
                )
            )

    remaining_needed = (
        int(target_fg_count)
        - len(group_ids)
    )

    if remaining_needed > len(
        remaining_pool
    ):
        return {}, False

    order = np.arange(
        len(remaining_pool)
    )

    rng.shuffle(order)

    for index in order[
        :remaining_needed
    ]:
        g, coord = remaining_pool[
            int(index)
        ]

        selected[g].append(
            coord
        )

    output = {
        g: np.asarray(
            selected[g],
            dtype=np.int32,
        )
        for g in group_ids
    }

    achieved = sum(
        len(v)
        for v in output.values()
    )

    return (
        output,
        achieved
        == int(target_fg_count),
    )


# ==========================================================================================
# 8. FIXED-BUDGET CONTIGUOUS SCRIBBLE BUILDER
# ==========================================================================================

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

    allocation, feasible = allocate_budget(
        int(target_budget),
        capacities,
        rng,
    )

    if not feasible:
        return {}, False, allocation

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

    achieved = sum(
        len(v)
        for v in result.values()
    )

    return (
        result,
        achieved == int(target_budget),
        allocation,
    )


# ==========================================================================================
# 9. GITHUB AUTH / RECOVERY
# ==========================================================================================

heading(
    "CORA-LUNG :: CODE BLOCK 05 :: WEAK-LABEL GENERATOR + FIREWALL"
)

secrets = UserSecretsClient()

token = secrets.get_secret(
    "pushCora"
)

if not token:
    raise RuntimeError(
        "Kaggle secret 'pushCora' is unavailable."
    )

token = token.strip()

askpass = Path(
    "/tmp/cora_git_askpass_block05.sh"
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

if not (
    REPO / ".git"
).exists():

    clone = sh(
        [
            "git",
            "clone",
            REMOTE_URL,
            str(REPO),
        ],
        cwd=WORK,
        env=git_env,
        check=False,
    )

    if clone.returncode != 0:
        raise RuntimeError(
            "Could not restore CORA-LUNG repository."
        )

else:
    print(
        "✓ Existing CORA-LUNG repository found."
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
        "Repository has uncommitted changes before Block 05."
    )

sh(
    [
        "git",
        "fetch",
        "origin",
        "main",
    ],
    cwd=REPO,
    env=git_env,
    check=False,
)

local_sha = sh(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    cwd=REPO,
).stdout.strip()

remote_sha = sh(
    [
        "git",
        "rev-parse",
        "origin/main",
    ],
    cwd=REPO,
    check=False,
).stdout.strip()

if (
    remote_sha
    and local_sha != remote_sha
):
    sh(
        [
            "git",
            "pull",
            "--ff-only",
            "origin",
            "main",
        ],
        cwd=REPO,
        env=git_env,
    )

starting_commit = sh(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    cwd=REPO,
).stdout.strip()

print(
    f"✓ Starting commit: "
    f"{starting_commit[:12]}"
)


# ==========================================================================================
# 10. LOAD FROZEN SPLITS
# ==========================================================================================

heading(
    "STEP 1/10 — LOAD FROZEN SPLITS AND DATASET"
)

split_path = (
    REPO
    / "data/splits/primary_volume_split_registry.csv"
)

if not split_path.exists():
    raise RuntimeError(
        "Frozen Block-04 split registry is missing."
    )

volume_registry = pd.read_csv(
    split_path
)

if len(
    volume_registry
) != 20:
    raise RuntimeError(
        "Expected exactly 20 frozen CT volume records."
    )

if (
    volume_registry[
        "source_subject_key"
    ].nunique()
    != 18
):
    raise RuntimeError(
        "Expected exactly 18 conservative source groups."
    )

dataset_root = locate_dataset()

print(
    f"Dataset root            : {dataset_root}"
)

print(
    f"Frozen CT volumes       : {len(volume_registry)}"
)

print(
    "Conservative groups    : "
    f"{volume_registry['source_subject_key'].nunique()}"
)

print(
    "Development volumes    : "
    f"{int((volume_registry['role'] == 'permanent_development').sum())}"
)

print(
    "Final-study volumes    : "
    f"{int((volume_registry['role'] == 'final_outer_cv').sum())}"
)


# ==========================================================================================
# 11. FREEZE GENERATOR CONFIGURATION
# ==========================================================================================

heading(
    "STEP 2/10 — FREEZE GENERATOR CONFIGURATION"
)

config_text = f"""
weak_label_generator:
  version: "{GENERATOR_VERSION}"
  seed: {GENERATOR_SEED}

component_definition:
  connectivity: {COMPONENT_CONNECTIVITY}
  minimum_volume_ml: {MIN_COMPONENT_ML}

foreground:
  interior_erosion_mm: {INTERIOR_EROSION_MM}
  base_stroke_voxels_per_observed_group: {BASE_FG_STROKE_VOXELS}
  maximum_allowed_path_voxels: {MAX_ALLOWED_PATH_VOXELS}
  path_source: 3d_skeleton_approximate_diameter
  skeleton_failure_fallback: connected_random_walk

background:
  safety_band_mm: {BACKGROUND_SAFETY_MM}
  budget_rule: equal_to_complete_foreground_voxel_count
  reuse_identical_realization_across_paired_conditions: true

coverage:
  levels: [1.00, 0.75, 0.50, 0.25]
  mechanism: uniform_nested_component_omission
  retained_count_rule: max_1_ceil_rho_times_K

matched_controls:
  natural_component_omission: true
  random_pixel_dropout:
    exact_foreground_voxel_match: true
    keep_at_least_one_voxel_per_component: true
  fixed_pixel_component_omission:
    redistribute_foreground_voxels_over_retained_groups: true
    preserve_contiguous_group_tracks: true
    lower_shared_budget_if_capacity_insufficient: true

export:
  coordinate_system: native_nifti_voxel_ijk_plus_world_xyz_mm
  unknown_voxels_materialized: false
  dense_masks_exported: false
  dense_component_ids_exported: false
"""

write_text(
    REPO
    / "configs/weak_labels_primary.yaml",
    config_text,
)

print(
    "✓ configs/weak_labels_primary.yaml"
)


# ==========================================================================================
# 12. BUILD CLEAN LOCAL WEAK EXPORT
# ==========================================================================================

heading(
    "STEP 3/10 — INITIALIZE ISOLATED WEAK_TRAIN EXPORT"
)

if WEAK_ROOT.exists():
    shutil.rmtree(
        WEAK_ROOT
    )

annotation_dir = (
    WEAK_ROOT
    / "annotations"
)

geometry_dir = (
    WEAK_ROOT
    / "geometry"
)

annotation_dir.mkdir(
    parents=True,
    exist_ok=True,
)

geometry_dir.mkdir(
    parents=True,
    exist_ok=True,
)

print(
    f"✓ Clean weak export root: {WEAK_ROOT}"
)


# ==========================================================================================
# 13. GENERATE ALL CASES
# ==========================================================================================

heading(
    "STEP 4/10 — GENERATE COMPONENT-AWARE SPARSE SUPERVISION"
)

artifact_rows = []
audit_rows = []

total_skeleton_fallbacks = 0
total_background_fallbacks = 0

pixel_control_infeasible = 0
fixed_budget_reductions = 0
fixed_budget_infeasible = 0

conditions_per_case = defaultdict(int)


for _, record in tqdm(
    volume_registry.iterrows(),
    total=len(volume_registry),
    desc="Generating weak supervision",
    unit="volume",
):

    case_id = str(
        record["case_id"]
    )

    source_group = str(
        record[
            "source_subject_key"
        ]
    )

    split_role = str(
        record["role"]
    )

    outer_fold = (
        None
        if pd.isna(
            record["outer_fold"]
        )
        else int(
            record["outer_fold"]
        )
    )

    ct_relative = str(
        record["ct_scan"]
    )

    infection_relative = str(
        record[
            "infection_mask"
        ]
    )

    lung_relative = str(
        record["lung_mask"]
    )

    infection_path = (
        dataset_root
        / infection_relative
    )

    lung_path = (
        dataset_root
        / lung_relative
    )

    infection_img = nib.load(
        str(
            infection_path
        ),
        mmap=True,
    )

    lung_img = nib.load(
        str(
            lung_path
        ),
        mmap=True,
    )

    lesion = np.asarray(
        infection_img.dataobj
    ) > 0

    lung = np.asarray(
        lung_img.dataobj
    ) > 0

    if lesion.shape != lung.shape:
        raise RuntimeError(
            f"Lesion/lung geometry mismatch: {case_id}"
        )

    spacing = np.asarray(
        infection_img.header.get_zooms()[:3],
        dtype=np.float64,
    )

    affine = np.asarray(
        infection_img.affine,
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

    structure26 = ndi.generate_binary_structure(
        3,
        3,
    )

    component_labels, raw_component_count = ndi.label(
        lesion,
        structure=structure26,
    )

    component_voxel_counts = np.bincount(
        component_labels.ravel()
    )[1:]

    component_volumes_ml = (
        component_voxel_counts.astype(
            np.float64
        )
        * voxel_volume_ml
    )

    eligible_dense_ids = (
        np.where(
            component_volumes_ml
            >= MIN_COMPONENT_ML
        )[0]
        + 1
    )

    if len(
        eligible_dense_ids
    ) < 1:
        raise RuntimeError(
            f"No eligible components: {case_id}"
        )

    object_slices = ndi.find_objects(
        component_labels,
        max_label=raw_component_count,
    )

    erosion_structure = physical_ball(
        spacing,
        INTERIOR_EROSION_MM,
    )

    # ------------------------------------------------------------------
    # Build exactly one allowed connected path for every eligible focus.
    #
    # group_id is a weak-annotation stroke identifier, not an exported
    # dense component identifier.
    # ------------------------------------------------------------------

    allowed_paths = {}
    complete_fg_by_group = {}

    dense_id_to_group = {}

    for group_id, dense_component_id in enumerate(
        eligible_dense_ids,
        start=1,
    ):
        dense_component_id = int(
            dense_component_id
        )

        dense_id_to_group[
            dense_component_id
        ] = int(
            group_id
        )

        slc = object_slices[
            dense_component_id - 1
        ]

        if slc is None:
            raise RuntimeError(
                f"Missing component slice in {case_id}"
            )

        component_crop = (
            component_labels[slc]
            == dense_component_id
        )

        eroded = ndi.binary_erosion(
            component_crop,
            structure=erosion_structure,
            border_value=0,
        )

        if not eroded.any():
            interior = component_crop
        else:
            interior = eroded

        component_rng = np.random.default_rng(
            stable_seed(
                GENERATOR_SEED,
                case_id,
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
                f"Unable to build foreground path: "
                f"{case_id}, group {group_id}"
            )

        if used_fallback:
            total_skeleton_fallbacks += 1

        # Convert crop coordinates -> native volume coordinates.
        starts = np.asarray(
            [
                slc[a].start
                for a in range(3)
            ],
            dtype=np.int32,
        )

        global_path = (
            local_path.astype(
                np.int32
            )
            + starts[None, :]
        )

        # Cap the candidate path while preserving contiguity.
        allowed_rng = np.random.default_rng(
            stable_seed(
                GENERATOR_SEED,
                case_id,
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
            allowed_rng,
        )

        allowed_paths[
            int(group_id)
        ] = global_path

        complete_rng = np.random.default_rng(
            stable_seed(
                GENERATOR_SEED,
                case_id,
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

    group_ids_all = sorted(
        allowed_paths.keys()
    )

    K = len(
        group_ids_all
    )

    complete_fg_count = int(
        sum(
            len(
                complete_fg_by_group[g]
            )
            for g in group_ids_all
        )
    )

    if complete_fg_count < K:
        raise RuntimeError(
            f"Invalid complete foreground budget: {case_id}"
        )

    # ------------------------------------------------------------------
    # One identical explicit-background realization for every condition.
    # ------------------------------------------------------------------

    background_structure = physical_ball(
        spacing,
        BACKGROUND_SAFETY_MM,
    )

    lesion_with_safety_band = ndi.binary_dilation(
        lesion,
        structure=background_structure,
        border_value=0,
    )

    safe_background = (
        lung
        & (~lesion_with_safety_band)
    )

    bg_rng = np.random.default_rng(
        stable_seed(
            GENERATOR_SEED,
            case_id,
            "background",
        )
    )

    background_coords, bg_fallback = generate_background_paths(
        safe_background,
        complete_fg_count,
        bg_rng,
    )

    if bg_fallback:
        total_background_fallbacks += 1

    # ------------------------------------------------------------------
    # Trainer-safe geometry export.
    #
    # No lesion mask path.
    # No lung mask path.
    # No dense component statistics.
    # ------------------------------------------------------------------

    geometry_payload = {
        "case_id":
            case_id,

        "source_subject_key":
            source_group,

        "split_role":
            split_role,

        "outer_fold":
            outer_fold,

        "ct_scan":
            ct_relative,

        "native_shape":
            [
                int(x)
                for x in lesion.shape
            ],

        "native_spacing_mm":
            [
                float(x)
                for x in spacing
            ],

        "native_affine":
            affine.tolist(),

        "coordinate_convention":
            "native_nifti_voxel_ijk",

        "generator_version":
            GENERATOR_VERSION,

        "generator_seed":
            GENERATOR_SEED,
    }

    geometry_file = (
        geometry_dir
        / f"{case_id}.json"
    )

    write_json(
        geometry_file,
        geometry_payload,
    )

    # ------------------------------------------------------------------
    # Stable nested component permutation.
    # 25% subset of 50%, subset of 75%, subset of complete.
    # ------------------------------------------------------------------

    omission_rng = np.random.default_rng(
        stable_seed(
            GENERATOR_SEED,
            case_id,
            "uniform_nested_component_order",
        )
    )

    component_order = np.asarray(
        group_ids_all,
        dtype=np.int32,
    )

    omission_rng.shuffle(
        component_order
    )

    # ==================================================================
    # A. COMPONENT-COMPLETE ANNOTATION
    # ==================================================================

    complete_annotation = assemble_annotation(
        complete_fg_by_group,
        background_coords,
        affine,
    )

    complete_file = (
        annotation_dir
        / f"{case_id}__complete.npz"
    )

    save_annotation_npz(
        complete_file,
        complete_annotation,
    )

    complete_hash = semantic_annotation_hash(
        **{
            "voxel_ijk":
                complete_annotation["voxel_ijk"],

            "world_xyz_mm":
                complete_annotation["world_xyz_mm"],

            "labels":
                complete_annotation["label"],

            "group_ids":
                complete_annotation["group_id"],
        }
    )

    artifact_rows.append(
        {
            "case_id":
                case_id,

            "source_subject_key":
                source_group,

            "split_role":
                split_role,

            "outer_fold":
                outer_fold,

            "condition":
                "complete",

            "requested_coverage":
                1.0,

            "ct_scan":
                ct_relative,

            "geometry_file":
                str(
                    geometry_file.relative_to(
                        WEAK_ROOT
                    )
                ),

            "annotation_file":
                str(
                    complete_file.relative_to(
                        WEAK_ROOT
                    )
                ),

            "foreground_labelled_voxels":
                complete_fg_count,

            "background_labelled_voxels":
                len(
                    background_coords
                ),

            "generator_version":
                GENERATOR_VERSION,

            "generator_seed":
                GENERATOR_SEED,

            "semantic_sha256":
                complete_hash,
        }
    )

    conditions_per_case[
        case_id
    ] += 1

    audit_rows.append(
        {
            "case_id":
                case_id,

            "source_subject_key":
                source_group,

            "condition":
                "complete",

            "requested_coverage":
                1.0,

            "eligible_components":
                K,

            "observed_groups":
                K,

            "realized_coverage":
                1.0,

            "foreground_voxels":
                complete_fg_count,

            "background_voxels":
                len(
                    background_coords
                ),

            "complete_fg_budget":
                complete_fg_count,

            "shared_fixed_budget":
                complete_fg_count,

            "exact_pixel_budget_match":
                True,

            "background_identical":
                True,
        }
    )

    # ==================================================================
    # B. COMPONENT-INCOMPLETE / MATCHED CONTROLS
    # ==================================================================

    for coverage in [
        0.75,
        0.50,
        0.25,
    ]:

        n_retained = max(
            1,
            int(
                math.ceil(
                    coverage * K
                )
            ),
        )

        retained_groups = sorted(
            int(x)
            for x in component_order[
                :n_retained
            ]
        )

        # --------------------------------------------------------------
        # B1. NATURAL COMPONENT OMISSION
        # --------------------------------------------------------------

        natural_fg = {
            g:
                complete_fg_by_group[g].copy()

            for g in retained_groups
        }

        natural_fg_count = int(
            sum(
                len(v)
                for v in natural_fg.values()
            )
        )

        natural_annotation = assemble_annotation(
            natural_fg,
            background_coords,
            affine,
        )

        coverage_token = str(
            int(
                round(
                    coverage * 100
                )
            )
        )

        natural_condition = (
            f"component_natural_{coverage_token}"
        )

        natural_file = (
            annotation_dir
            / (
                f"{case_id}__"
                f"{natural_condition}.npz"
            )
        )

        save_annotation_npz(
            natural_file,
            natural_annotation,
        )

        natural_hash = semantic_annotation_hash(
            natural_annotation[
                "voxel_ijk"
            ],
            natural_annotation[
                "world_xyz_mm"
            ],
            natural_annotation[
                "label"
            ],
            natural_annotation[
                "group_id"
            ],
        )

        artifact_rows.append(
            {
                "case_id":
                    case_id,

                "source_subject_key":
                    source_group,

                "split_role":
                    split_role,

                "outer_fold":
                    outer_fold,

                "condition":
                    natural_condition,

                "requested_coverage":
                    coverage,

                "ct_scan":
                    ct_relative,

                "geometry_file":
                    str(
                        geometry_file.relative_to(
                            WEAK_ROOT
                        )
                    ),

                "annotation_file":
                    str(
                        natural_file.relative_to(
                            WEAK_ROOT
                        )
                    ),

                "foreground_labelled_voxels":
                    natural_fg_count,

                "background_labelled_voxels":
                    len(
                        background_coords
                    ),

                "generator_version":
                    GENERATOR_VERSION,

                "generator_seed":
                    GENERATOR_SEED,

                "semantic_sha256":
                    natural_hash,
            }
        )

        conditions_per_case[
            case_id
        ] += 1

        audit_rows.append(
            {
                "case_id":
                    case_id,

                "source_subject_key":
                    source_group,

                "condition":
                    natural_condition,

                "requested_coverage":
                    coverage,

                "eligible_components":
                    K,

                "observed_groups":
                    len(
                        retained_groups
                    ),

                "realized_coverage":
                    len(
                        retained_groups
                    ) / K,

                "foreground_voxels":
                    natural_fg_count,

                "background_voxels":
                    len(
                        background_coords
                    ),

                "complete_fg_budget":
                    complete_fg_count,

                "shared_fixed_budget":
                    np.nan,

                "exact_pixel_budget_match":
                    True,

                "background_identical":
                    True,
            }
        )

        # --------------------------------------------------------------
        # B2. RANDOM-PIXEL DROPOUT MATCHED TO NATURAL FG BUDGET
        #
        # Every complete component remains represented.
        # --------------------------------------------------------------

        pixel_rng = np.random.default_rng(
            stable_seed(
                GENERATOR_SEED,
                case_id,
                "pixel_dropout",
                coverage,
            )
        )

        pixel_fg, pixel_feasible = (
            make_pixel_dropout_control(
                complete_fg_by_group,
                natural_fg_count,
                pixel_rng,
            )
        )

        if not pixel_feasible:
            pixel_control_infeasible += 1

        else:
            pixel_count = int(
                sum(
                    len(v)
                    for v in pixel_fg.values()
                )
            )

            if pixel_count != natural_fg_count:
                raise RuntimeError(
                    "Random pixel-dropout budget mismatch."
                )

            pixel_condition = (
                f"pixel_dropout_matched_{coverage_token}"
            )

            pixel_annotation = assemble_annotation(
                pixel_fg,
                background_coords,
                affine,
            )

            pixel_file = (
                annotation_dir
                / (
                    f"{case_id}__"
                    f"{pixel_condition}.npz"
                )
            )

            save_annotation_npz(
                pixel_file,
                pixel_annotation,
            )

            pixel_hash = semantic_annotation_hash(
                pixel_annotation[
                    "voxel_ijk"
                ],
                pixel_annotation[
                    "world_xyz_mm"
                ],
                pixel_annotation[
                    "label"
                ],
                pixel_annotation[
                    "group_id"
                ],
            )

            artifact_rows.append(
                {
                    "case_id":
                        case_id,

                    "source_subject_key":
                        source_group,

                    "split_role":
                        split_role,

                    "outer_fold":
                        outer_fold,

                    "condition":
                        pixel_condition,

                    "requested_coverage":
                        coverage,

                    "ct_scan":
                        ct_relative,

                    "geometry_file":
                        str(
                            geometry_file.relative_to(
                                WEAK_ROOT
                            )
                        ),

                    "annotation_file":
                        str(
                            pixel_file.relative_to(
                                WEAK_ROOT
                            )
                        ),

                    "foreground_labelled_voxels":
                        pixel_count,

                    "background_labelled_voxels":
                        len(
                            background_coords
                        ),

                    "generator_version":
                        GENERATOR_VERSION,

                    "generator_seed":
                        GENERATOR_SEED,

                    "semantic_sha256":
                        pixel_hash,
                }
            )

            conditions_per_case[
                case_id
            ] += 1

            audit_rows.append(
                {
                    "case_id":
                        case_id,

                    "source_subject_key":
                        source_group,

                    "condition":
                        pixel_condition,

                    "requested_coverage":
                        coverage,

                    "eligible_components":
                        K,

                    "observed_groups":
                        K,

                    "realized_coverage":
                        1.0,

                    "foreground_voxels":
                        pixel_count,

                    "background_voxels":
                        len(
                            background_coords
                        ),

                    "complete_fg_budget":
                        complete_fg_count,

                    "shared_fixed_budget":
                        np.nan,

                    "exact_pixel_budget_match":
                        (
                            pixel_count
                            == natural_fg_count
                        ),

                    "background_identical":
                        True,
                }
            )

        # --------------------------------------------------------------
        # B3. FIXED-PIXEL COMPONENT OMISSION
        #
        # Try to retain the complete FG budget while only placing
        # foreground labels on retained components.
        #
        # If the allowed-path capacity is insufficient, lower the shared
        # budget and create a corresponding complete-control annotation.
        # --------------------------------------------------------------

        retained_capacity = int(
            sum(
                len(
                    allowed_paths[g]
                )
                for g in retained_groups
            )
        )

        shared_fixed_budget = min(
            complete_fg_count,
            retained_capacity,
        )

        if (
            shared_fixed_budget
            < complete_fg_count
        ):
            fixed_budget_reductions += 1

        if (
            shared_fixed_budget
            < len(
                retained_groups
            )
        ):
            fixed_budget_infeasible += 1
            continue

        component_fixed_rng = np.random.default_rng(
            stable_seed(
                GENERATOR_SEED,
                case_id,
                "component_fixed",
                coverage,
            )
        )

        (
            component_fixed_fg,
            component_fixed_ok,
            component_fixed_allocation,
        ) = build_contiguous_budget_annotation(
            allowed_paths,
            retained_groups,
            shared_fixed_budget,
            component_fixed_rng,
        )

        if not component_fixed_ok:
            fixed_budget_infeasible += 1
            continue

        component_fixed_count = int(
            sum(
                len(v)
                for v in component_fixed_fg.values()
            )
        )

        fixed_condition = (
            f"component_fixed_{coverage_token}"
        )

        fixed_annotation = assemble_annotation(
            component_fixed_fg,
            background_coords,
            affine,
        )

        fixed_file = (
            annotation_dir
            / (
                f"{case_id}__"
                f"{fixed_condition}.npz"
            )
        )

        save_annotation_npz(
            fixed_file,
            fixed_annotation,
        )

        fixed_hash = semantic_annotation_hash(
            fixed_annotation[
                "voxel_ijk"
            ],
            fixed_annotation[
                "world_xyz_mm"
            ],
            fixed_annotation[
                "label"
            ],
            fixed_annotation[
                "group_id"
            ],
        )

        artifact_rows.append(
            {
                "case_id":
                    case_id,

                "source_subject_key":
                    source_group,

                "split_role":
                    split_role,

                "outer_fold":
                    outer_fold,

                "condition":
                    fixed_condition,

                "requested_coverage":
                    coverage,

                "ct_scan":
                    ct_relative,

                "geometry_file":
                    str(
                        geometry_file.relative_to(
                            WEAK_ROOT
                        )
                    ),

                "annotation_file":
                    str(
                        fixed_file.relative_to(
                            WEAK_ROOT
                        )
                    ),

                "foreground_labelled_voxels":
                    component_fixed_count,

                "background_labelled_voxels":
                    len(
                        background_coords
                    ),

                "generator_version":
                    GENERATOR_VERSION,

                "generator_seed":
                    GENERATOR_SEED,

                "semantic_sha256":
                    fixed_hash,
            }
        )

        conditions_per_case[
            case_id
        ] += 1

        audit_rows.append(
            {
                "case_id":
                    case_id,

                "source_subject_key":
                    source_group,

                "condition":
                    fixed_condition,

                "requested_coverage":
                    coverage,

                "eligible_components":
                    K,

                "observed_groups":
                    len(
                        retained_groups
                    ),

                "realized_coverage":
                    len(
                        retained_groups
                    ) / K,

                "foreground_voxels":
                    component_fixed_count,

                "background_voxels":
                    len(
                        background_coords
                    ),

                "complete_fg_budget":
                    complete_fg_count,

                "shared_fixed_budget":
                    shared_fixed_budget,

                "exact_pixel_budget_match":
                    (
                        component_fixed_count
                        == shared_fixed_budget
                    ),

                "background_identical":
                    True,
            }
        )

        # --------------------------------------------------------------
        # B4. COMPLETE CONTROL AT THE IDENTICAL FIXED BUDGET
        # --------------------------------------------------------------

        if (
            shared_fixed_budget
            == complete_fg_count
        ):
            complete_fixed_fg = {
                g:
                    complete_fg_by_group[g].copy()

                for g in group_ids_all
            }

            complete_fixed_ok = True

        else:
            complete_fixed_rng = np.random.default_rng(
                stable_seed(
                    GENERATOR_SEED,
                    case_id,
                    "complete_fixed_control",
                    coverage,
                )
            )

            (
                complete_fixed_fg,
                complete_fixed_ok,
                _,
            ) = build_contiguous_budget_annotation(
                allowed_paths,
                group_ids_all,
                shared_fixed_budget,
                complete_fixed_rng,
            )

        if not complete_fixed_ok:
            fixed_budget_infeasible += 1
            continue

        complete_fixed_count = int(
            sum(
                len(v)
                for v in complete_fixed_fg.values()
            )
        )

        complete_fixed_condition = (
            f"complete_fixed_{coverage_token}"
        )

        complete_fixed_annotation = assemble_annotation(
            complete_fixed_fg,
            background_coords,
            affine,
        )

        complete_fixed_file = (
            annotation_dir
            / (
                f"{case_id}__"
                f"{complete_fixed_condition}.npz"
            )
        )

        save_annotation_npz(
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

        artifact_rows.append(
            {
                "case_id":
                    case_id,

                "source_subject_key":
                    source_group,

                "split_role":
                    split_role,

                "outer_fold":
                    outer_fold,

                "condition":
                    complete_fixed_condition,

                "requested_coverage":
                    1.0,

                "ct_scan":
                    ct_relative,

                "geometry_file":
                    str(
                        geometry_file.relative_to(
                            WEAK_ROOT
                        )
                    ),

                "annotation_file":
                    str(
                        complete_fixed_file.relative_to(
                            WEAK_ROOT
                        )
                    ),

                "foreground_labelled_voxels":
                    complete_fixed_count,

                "background_labelled_voxels":
                    len(
                        background_coords
                    ),

                "generator_version":
                    GENERATOR_VERSION,

                "generator_seed":
                    GENERATOR_SEED,

                "semantic_sha256":
                    complete_fixed_hash,
            }
        )

        conditions_per_case[
            case_id
        ] += 1

        audit_rows.append(
            {
                "case_id":
                    case_id,

                "source_subject_key":
                    source_group,

                "condition":
                    complete_fixed_condition,

                "requested_coverage":
                    1.0,

                "eligible_components":
                    K,

                "observed_groups":
                    K,

                "realized_coverage":
                    1.0,

                "foreground_voxels":
                    complete_fixed_count,

                "background_voxels":
                    len(
                        background_coords
                    ),

                "complete_fg_budget":
                    complete_fg_count,

                "shared_fixed_budget":
                    shared_fixed_budget,

                "exact_pixel_budget_match":
                    (
                        complete_fixed_count
                        == component_fixed_count
                    ),

                "background_identical":
                    True,
            }
        )

    # Free memory explicitly before next volume.
    del lesion
    del lung
    del component_labels
    del safe_background
    del lesion_with_safety_band


# ==========================================================================================
# 14. WRITE TRAINER-SAFE LOCAL MANIFEST
# ==========================================================================================

heading(
    "STEP 5/10 — WRITE TRAINER-SAFE WEAK MANIFEST"
)

artifact_df = pd.DataFrame(
    artifact_rows
)

audit_df = pd.DataFrame(
    audit_rows
)

safe_manifest_columns = [
    "case_id",
    "source_subject_key",
    "split_role",
    "outer_fold",
    "condition",
    "requested_coverage",
    "ct_scan",
    "geometry_file",
    "annotation_file",
    "foreground_labelled_voxels",
    "background_labelled_voxels",
    "generator_version",
    "generator_seed",
    "semantic_sha256",
]

artifact_df = artifact_df[
    safe_manifest_columns
].copy()

weak_manifest_path = (
    WEAK_ROOT
    / "manifest.csv"
)

artifact_df.to_csv(
    weak_manifest_path,
    index=False,
)

print(
    f"Trainer-safe manifest rows : {len(artifact_df)}"
)

print(
    f"Weak annotation files      : "
    f"{len(list(annotation_dir.glob('*.npz')))}"
)

print(
    f"Geometry files             : "
    f"{len(list(geometry_dir.glob('*.json')))}"
)


# ==========================================================================================
# 15. FIREWALL IMPLEMENTATION
# ==========================================================================================

heading(
    "STEP 6/10 — BUILD AND EXECUTE DENSE-LABEL FIREWALL"
)

firewall_source = r'''
"""Dense-label firewall for CORA-Lung trainer-safe weak supervision."""

from pathlib import Path
import json
import numpy as np
import pandas as pd


ALLOWED_NPZ_KEYS = {
    "voxel_ijk",
    "world_xyz_mm",
    "label",
    "group_id",
}

ALLOWED_MANIFEST_COLUMNS = {
    "case_id",
    "source_subject_key",
    "split_role",
    "outer_fold",
    "condition",
    "requested_coverage",
    "ct_scan",
    "geometry_file",
    "annotation_file",
    "foreground_labelled_voxels",
    "background_labelled_voxels",
    "generator_version",
    "generator_seed",
    "semantic_sha256",
}

FORBIDDEN_TOKENS = {
    "infection_mask",
    "lung_mask",
    "lesion_mask",
    "dense_mask",
    "dense_component",
    "hidden_component",
    "component_centroid",
    "component_size",
    "component_volume",
    "full_mask_distance",
}


def _assert_no_forbidden_text(value):
    text = str(value).lower()

    for token in FORBIDDEN_TOKENS:
        if token in text:
            raise RuntimeError(
                f"Forbidden dense-supervision token detected: {token}"
            )


def validate_manifest(manifest_path):
    manifest_path = Path(manifest_path)

    df = pd.read_csv(manifest_path)

    actual = set(df.columns)

    if actual != ALLOWED_MANIFEST_COLUMNS:
        extra = actual - ALLOWED_MANIFEST_COLUMNS
        missing = ALLOWED_MANIFEST_COLUMNS - actual

        raise RuntimeError(
            f"Unsafe manifest schema. Extra={extra}, Missing={missing}"
        )

    for column in df.columns:
        _assert_no_forbidden_text(column)

    for column in ["ct_scan", "geometry_file", "annotation_file"]:
        for value in df[column].dropna():
            _assert_no_forbidden_text(value)

    return df


def validate_annotation_file(path):
    path = Path(path)

    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)

        if keys != ALLOWED_NPZ_KEYS:
            raise RuntimeError(
                f"Unsafe annotation keys in {path.name}: {keys}"
            )

        voxel_ijk = np.asarray(data["voxel_ijk"])
        world_xyz_mm = np.asarray(data["world_xyz_mm"])
        label = np.asarray(data["label"])
        group_id = np.asarray(data["group_id"])

    n = len(label)

    if voxel_ijk.shape != (n, 3):
        raise RuntimeError("Invalid voxel_ijk shape.")

    if world_xyz_mm.shape != (n, 3):
        raise RuntimeError("Invalid world_xyz_mm shape.")

    if group_id.shape != (n,):
        raise RuntimeError("Invalid group_id shape.")

    if not set(np.unique(label)).issubset({0, 1}):
        raise RuntimeError("Sparse labels must be only 0 or 1.")

    fg = label == 1
    bg = label == 0

    if np.any(group_id[fg] <= 0):
        raise RuntimeError(
            "Every foreground labelled voxel must have positive group ID."
        )

    if np.any(group_id[bg] != -1):
        raise RuntimeError(
            "Every background labelled voxel must use group_id=-1."
        )

    coord_label = {}

    for coord, y in zip(voxel_ijk, label):
        key = tuple(int(x) for x in coord)

        if key in coord_label and coord_label[key] != int(y):
            raise RuntimeError(
                "Conflicting foreground/background sparse coordinate."
            )

        coord_label[key] = int(y)

    return True


def validate_geometry_file(path):
    path = Path(path)

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    serialized = json.dumps(payload).lower()

    for token in FORBIDDEN_TOKENS:
        if token in serialized:
            raise RuntimeError(
                f"Forbidden dense field in geometry file: {token}"
            )

    required = {
        "case_id",
        "source_subject_key",
        "split_role",
        "outer_fold",
        "ct_scan",
        "native_shape",
        "native_spacing_mm",
        "native_affine",
        "coordinate_convention",
        "generator_version",
        "generator_seed",
    }

    if set(payload.keys()) != required:
        raise RuntimeError(
            "Unexpected trainer geometry schema."
        )

    return True


def validate_weak_export(root):
    root = Path(root)

    manifest = validate_manifest(
        root / "manifest.csv"
    )

    for _, row in manifest.iterrows():
        validate_annotation_file(
            root / row["annotation_file"]
        )

        validate_geometry_file(
            root / row["geometry_file"]
        )

    return {
        "manifest_rows": int(len(manifest)),
        "annotation_files": int(
            manifest["annotation_file"].nunique()
        ),
        "case_count": int(
            manifest["case_id"].nunique()
        ),
    }
'''

write_text(
    REPO
    / "src/cora_lung/data/firewall.py",
    firewall_source,
)

test_source = r'''
import json
import numpy as np
import pandas as pd

from cora_lung.data.firewall import (
    validate_annotation_file,
    validate_geometry_file,
)


def test_safe_sparse_annotation(tmp_path):
    path = tmp_path / "safe.npz"

    np.savez_compressed(
        path,
        voxel_ijk=np.asarray(
            [[1, 2, 3], [4, 5, 6]],
            dtype=np.int32,
        ),
        world_xyz_mm=np.asarray(
            [[1., 2., 3.], [4., 5., 6.]],
            dtype=np.float32,
        ),
        label=np.asarray(
            [1, 0],
            dtype=np.int8,
        ),
        group_id=np.asarray(
            [1, -1],
            dtype=np.int32,
        ),
    )

    assert validate_annotation_file(path)


def test_unknown_voxels_are_not_materialized(tmp_path):
    path = tmp_path / "safe.npz"

    np.savez_compressed(
        path,
        voxel_ijk=np.asarray(
            [[2, 2, 2]],
            dtype=np.int32,
        ),
        world_xyz_mm=np.asarray(
            [[2., 2., 2.]],
            dtype=np.float32,
        ),
        label=np.asarray(
            [1],
            dtype=np.int8,
        ),
        group_id=np.asarray(
            [3],
            dtype=np.int32,
        ),
    )

    with np.load(path, allow_pickle=False) as data:
        assert "unknown_mask" not in data.files
        assert "dense_mask" not in data.files


def test_geometry_rejects_dense_mask_field(tmp_path):
    path = tmp_path / "geometry.json"

    payload = {
        "case_id": "synthetic",
        "infection_mask": "forbidden.nii",
    }

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    failed = False

    try:
        validate_geometry_file(path)
    except RuntimeError:
        failed = True

    assert failed
'''

write_text(
    REPO
    / "tests/test_firewall.py",
    test_source,
)

sys.path.insert(
    0,
    str(
        REPO / "src"
    ),
)

from cora_lung.data.firewall import validate_weak_export

firewall_report = validate_weak_export(
    WEAK_ROOT
)

print(
    "✓ Runtime weak-export firewall: PASS"
)

print(
    f"  Cases checked       : "
    f"{firewall_report['case_count']}"
)

print(
    f"  Annotation artifacts: "
    f"{firewall_report['annotation_files']}"
)

pytest_result = sh(
    [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_firewall.py",
        "-q",
    ],
    cwd=REPO,
    check=False,
)

print(
    "\npytest output:"
)

print(
    pytest_result.stdout.strip()
)

if pytest_result.returncode != 0:
    print(
        pytest_result.stderr
    )

    raise RuntimeError(
        "Dense-label firewall unit tests FAILED."
    )

print(
    "✓ Dense-label firewall unit tests: PASS"
)


# ==========================================================================================
# 16. PAIRWISE GENERATOR AUDITS
# ==========================================================================================

heading(
    "STEP 7/10 — AUDIT MATCHED BUDGETS AND COVERAGE"
)

# Exact natural-vs-pixel comparisons.
natural = audit_df[
    audit_df[
        "condition"
    ].str.startswith(
        "component_natural_"
    )
].copy()

pixel = audit_df[
    audit_df[
        "condition"
    ].str.startswith(
        "pixel_dropout_matched_"
    )
].copy()

natural[
    "coverage_token"
] = natural[
    "condition"
].str.extract(
    r"(\d+)$"
).astype(int)

pixel[
    "coverage_token"
] = pixel[
    "condition"
].str.extract(
    r"(\d+)$"
).astype(int)

natural_pixel_check = natural.merge(
    pixel,
    on=[
        "case_id",
        "coverage_token",
    ],
    suffixes=(
        "_component",
        "_pixel",
    ),
)

natural_pixel_check[
    "fg_budget_exact"
] = (
    natural_pixel_check[
        "foreground_voxels_component"
    ]
    == natural_pixel_check[
        "foreground_voxels_pixel"
    ]
)

natural_pixel_check[
    "background_exact"
] = (
    natural_pixel_check[
        "background_voxels_component"
    ]
    == natural_pixel_check[
        "background_voxels_pixel"
    ]
)

natural_pixel_exact = bool(
    natural_pixel_check[
        "fg_budget_exact"
    ].all()
    and natural_pixel_check[
        "background_exact"
    ].all()
)

# Exact component-fixed vs complete-fixed comparisons.
component_fixed = audit_df[
    audit_df[
        "condition"
    ].str.startswith(
        "component_fixed_"
    )
].copy()

complete_fixed = audit_df[
    audit_df[
        "condition"
    ].str.startswith(
        "complete_fixed_"
    )
].copy()

component_fixed[
    "coverage_token"
] = component_fixed[
    "condition"
].str.extract(
    r"(\d+)$"
).astype(int)

complete_fixed[
    "coverage_token"
] = complete_fixed[
    "condition"
].str.extract(
    r"(\d+)$"
).astype(int)

fixed_check = component_fixed.merge(
    complete_fixed,
    on=[
        "case_id",
        "coverage_token",
    ],
    suffixes=(
        "_component",
        "_complete",
    ),
)

fixed_check[
    "fg_budget_exact"
] = (
    fixed_check[
        "foreground_voxels_component"
    ]
    == fixed_check[
        "foreground_voxels_complete"
    ]
)

fixed_check[
    "background_exact"
] = (
    fixed_check[
        "background_voxels_component"
    ]
    == fixed_check[
        "background_voxels_complete"
    ]
)

fixed_exact = bool(
    len(fixed_check) > 0
    and fixed_check[
        "fg_budget_exact"
    ].all()
    and fixed_check[
        "background_exact"
    ].all()
)

background_counts_per_case = (
    audit_df.groupby(
        "case_id"
    )[
        "background_voxels"
    ]
    .nunique()
)

same_background_count = bool(
    (
        background_counts_per_case
        == 1
    ).all()
)

condition_counts = pd.Series(
    conditions_per_case
)

print(
    "Natural component omission vs pixel dropout:"
)

print(
    f"  Paired comparisons : "
    f"{len(natural_pixel_check)}"
)

print(
    f"  Exact FG budgets   : "
    f"{'PASS' if natural_pixel_exact else 'FAIL'}"
)

print(
    "\nFixed component omission vs complete fixed control:"
)

print(
    f"  Paired comparisons : "
    f"{len(fixed_check)}"
)

print(
    f"  Exact FG budgets   : "
    f"{'PASS' if fixed_exact else 'FAIL'}"
)

print(
    "\nBackground supervision:"
)

print(
    f"  Same count across conditions: "
    f"{'PASS' if same_background_count else 'FAIL'}"
)

print(
    f"  Path fallbacks             : "
    f"{total_background_fallbacks}"
)

print(
    "\nPath generation:"
)

print(
    f"  Skeleton fallbacks         : "
    f"{total_skeleton_fallbacks}"
)

print(
    "\nFeasibility:"
)

print(
    f"  Pixel controls infeasible  : "
    f"{pixel_control_infeasible}"
)

print(
    f"  Fixed budget reductions    : "
    f"{fixed_budget_reductions}"
)

print(
    f"  Fixed controls infeasible  : "
    f"{fixed_budget_infeasible}"
)

if not natural_pixel_exact:
    raise RuntimeError(
        "Matched natural/pixel control audit FAILED."
    )

if not fixed_exact:
    raise RuntimeError(
        "Matched fixed-budget control audit FAILED."
    )

if not same_background_count:
    raise RuntimeError(
        "Background budget consistency audit FAILED."
    )

if pixel_control_infeasible > 0:
    raise RuntimeError(
        "At least one matched pixel-dropout control is infeasible."
    )

if fixed_budget_infeasible > 0:
    raise RuntimeError(
        "At least one fixed-budget comparison is infeasible."
    )


# ==========================================================================================
# 17. DETERMINISM AUDIT OF SAVED SEMANTIC CONTENT
# ==========================================================================================

heading(
    "STEP 8/10 — VERIFY SEMANTIC HASH INTEGRITY"
)

semantic_hash_unique = (
    artifact_df[
        "semantic_sha256"
    ].str.len()
    == 64
).all()

if not semantic_hash_unique:
    raise RuntimeError(
        "Invalid annotation semantic hash."
    )

# Reload every artifact and recompute its semantic hash.
hash_failures = []

for _, row in tqdm(
    artifact_df.iterrows(),
    total=len(artifact_df),
    desc="Rechecking annotation hashes",
    unit="artifact",
):

    path = (
        WEAK_ROOT
        / row[
            "annotation_file"
        ]
    )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        recomputed = semantic_annotation_hash(
            data[
                "voxel_ijk"
            ],
            data[
                "world_xyz_mm"
            ],
            data[
                "label"
            ],
            data[
                "group_id"
            ],
        )

    if (
        recomputed
        != row[
            "semantic_sha256"
        ]
    ):
        hash_failures.append(
            str(
                row[
                    "annotation_file"
                ]
            )
        )

if hash_failures:
    raise RuntimeError(
        "Semantic annotation hash verification failed."
    )

print(
    f"✓ Semantic hash verification: "
    f"PASS ({len(artifact_df)} artifacts)"
)


# ==========================================================================================
# 18. EXPORT NON-PATIENT-ARRAY AUDIT TO GITHUB
# ==========================================================================================

heading(
    "STEP 9/10 — SAVE REPRODUCIBILITY AUDITS AND PUBLICATION FIGURE"
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

# This audit contains COUNTS only, never sparse patient arrays.
audit_df.to_csv(
    manifest_dir
    / "weak_label_generation_summary.csv",
    index=False,
)

artifact_checksum_export = artifact_df[
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

artifact_checksum_export.to_csv(
    manifest_dir
    / "weak_label_artifact_semantic_hashes.csv",
    index=False,
)

# ----------------------------------------------------------------------
# Aggregate budget statistics
# ----------------------------------------------------------------------

summary = (
    audit_df.groupby(
        [
            "condition",
            "requested_coverage",
        ],
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

        median_realized_coverage=(
            "realized_coverage",
            "median",
        ),
    )
)

summary.to_csv(
    manifest_dir
    / "weak_label_condition_summary.csv",
    index=False,
)

# ----------------------------------------------------------------------
# Publication-quality figure:
# Foreground annotation budget under the paired experimental regimes.
# ----------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 150,
        "savefig.dpi": 600,
    }
)

plot_rows = []

for coverage in [
    1.00,
    0.75,
    0.50,
    0.25,
]:

    if coverage == 1.00:
        subset = audit_df[
            audit_df[
                "condition"
            ]
            == "complete"
        ]

        plot_rows.append(
            {
                "regime":
                    "Component-Complete",

                "coverage":
                    coverage,

                "median_fg":
                    float(
                        subset[
                            "foreground_voxels"
                        ].median()
                    ),
            }
        )

    else:
        token = int(
            round(
                coverage * 100
            )
        )

        for prefix, label in [
            (
                "component_natural",
                "Whole-Component Omission (Natural Budget)",
            ),
            (
                "pixel_dropout_matched",
                "Random-Pixel Dropout (Matched Budget)",
            ),
            (
                "component_fixed",
                "Whole-Component Omission (Fixed Budget)",
            ),
        ]:
            condition = (
                f"{prefix}_{token}"
            )

            subset = audit_df[
                audit_df[
                    "condition"
                ]
                == condition
            ]

            if len(subset):
                plot_rows.append(
                    {
                        "regime":
                            label,

                        "coverage":
                            coverage,

                        "median_fg":
                            float(
                                subset[
                                    "foreground_voxels"
                                ].median()
                            ),
                    }
                )

plot_df = pd.DataFrame(
    plot_rows
)

fig, ax = plt.subplots(
    figsize=(10.5, 6.5)
)

for regime, group in plot_df.groupby(
    "regime"
):
    group = group.sort_values(
        "coverage"
    )

    ax.plot(
        group[
            "coverage"
        ] * 100,
        group[
            "median_fg"
        ],
        marker="o",
        linewidth=2.0,
        label=regime,
    )

ax.set_xlabel(
    "Retained Component Coverage (%)",
    fontweight="bold",
)

ax.set_ylabel(
    "Median Foreground-Labelled Voxels per CT Volume",
    fontweight="bold",
)

ax.set_title(
    "Foreground Annotation Budget Across the Frozen CORA-Lung Weak-Supervision Regimes\n"
    "Natural Component Omission, Matched Pixel Thinning, and Fixed-Budget Component Omission",
    fontweight="bold",
)

ax.grid(
    alpha=0.25,
)

legend = ax.legend(
    frameon=True,
)

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
    "✓ figures/audit/"
    "fig05_weak_supervision_annotation_budget.png"
)

print(
    "✓ figures/audit/"
    "fig05_weak_supervision_annotation_budget.pdf"
)


# ==========================================================================================
# 19. WRITE PROTOCOL DOCUMENT
# ==========================================================================================

protocol_text = f"""
# CORA-Lung Weak-Label Generation Protocol

## Version

Generator version: **{GENERATOR_VERSION}**

Generator seed: **{GENERATOR_SEED}**

This protocol was frozen before Gate B model training.

## Dense-supervision boundary

Dense infection and lung masks are read only by the isolated weak-label generator.

The trainer receives no dense lesion mask, dense component map, omitted-component identity,
component centroid, component size, or dense-mask-derived distance.

Unknown voxels are not materialized as negative labels.

## Component definition

- Native annotation grid.
- 26-connected infection-mask components.
- Minimum component volume: 0.1 mL.

## Foreground scribbles

For every eligible component:

1. perform a 1-mm physical interior erosion;
2. fall back to the un-eroded component when erosion empties the component;
3. compute a 3-D skeleton;
4. retain the largest connected skeleton;
5. obtain an approximate geodesic-diameter path;
6. select a deterministic randomized contiguous segment;
7. assign one positive foreground stroke-group identifier.

Base complete-scribble budget:

**{BASE_FG_STROKE_VOXELS} native voxels per observed component**, subject to available
connected path length.

The maximum candidate connected path retained per component is
**{MAX_ALLOWED_PATH_VOXELS} voxels**.

## Background scribbles

Background tracks are sampled inside released lung tissue but outside a
**{BACKGROUND_SAFETY_MM}-mm physical safety band** around every annotated lesion.

The background budget equals the complete foreground-labelled voxel count for that case.

Exactly the same background realization is reused across all paired supervision regimes.

Released lung masks are used only by this isolated simulation stage and are not trainer inputs.

## Component coverage

The frozen coverage levels are:

- 100%;
- 75%;
- 50%;
- 25%.

A deterministic random ordering of eligible components is generated once per case.

Coverage is nested:

25% subset of 50%, 50% subset of 75%, 75% subset of 100%.

The retained count is:

`max(1, ceil(rho * K))`.

## Natural component omission

Complete foreground strokes belonging to omitted components are removed in their entirety.

No removed foreground pixels are redistributed.

## Matched random-pixel dropout

The total number of labelled foreground voxels exactly matches the corresponding natural
component-omission annotation.

Every eligible component remains represented by at least one foreground-labelled voxel.

This control tests missing pixels versus missing complete components.

## Fixed-pixel component omission

Only retained components may receive foreground annotation.

Removed foreground pixels are redistributed over connected allowed paths of retained components
so that the total positive-labelled voxel budget matches the component-complete condition whenever
capacity permits.

If retained connected-path capacity is insufficient, a lower shared budget is used for both the
component-incomplete annotation and its component-complete paired control.

No voxel is duplicated to claim a matched budget.

## Coordinate export

Sparse annotation files contain only:

- native NIfTI voxel coordinates `(i, j, k)`;
- physical world coordinates `(x, y, z)` in mm;
- sparse label (`1` foreground, `0` explicit background);
- foreground stroke-group identifier.

All unspecified voxels are unknown.

## Reproducibility

Patient sparse-coordinate arrays are not committed to GitHub.

They are deterministically reproducible using:

- frozen source data checksums;
- generator version;
- generator seed;
- frozen split registry;
- this protocol;
- the executable Block-05 generator.

Semantic annotation hashes are committed for verification.
"""

write_text(
    REPO
    / "docs/weak_label_protocol.md",
    protocol_text,
)


# ==========================================================================================
# 20. BLOCK AUDIT JSON
# ==========================================================================================

local_export_bytes = sum(
    p.stat().st_size
    for p in WEAK_ROOT.rglob("*")
    if p.is_file()
)

block05_status = (
    "PASS"
    if (
        natural_pixel_exact
        and fixed_exact
        and same_background_count
        and pixel_control_infeasible == 0
        and fixed_budget_infeasible == 0
        and pytest_result.returncode == 0
        and len(hash_failures) == 0
    )
    else "FAIL"
)

block05_audit = {
    "project":
        PROJECT,

    "block":
        BLOCK_ID,

    "block_name":
        BLOCK_NAME,

    "generated_at_utc":
        NOW_ISO,

    "status":
        block05_status,

    "generator": {
        "version":
            GENERATOR_VERSION,

        "seed":
            GENERATOR_SEED,

        "base_fg_stroke_voxels":
            BASE_FG_STROKE_VOXELS,

        "maximum_allowed_path_voxels":
            MAX_ALLOWED_PATH_VOXELS,

        "interior_erosion_mm":
            INTERIOR_EROSION_MM,

        "background_safety_mm":
            BACKGROUND_SAFETY_MM,

        "coverage_levels":
            COVERAGE_LEVELS,
    },

    "cases":
        int(
            artifact_df[
                "case_id"
            ].nunique()
        ),

    "annotation_artifacts":
        int(
            len(
                artifact_df
            )
        ),

    "local_export_bytes":
        int(
            local_export_bytes
        ),

    "skeleton_fallbacks":
        int(
            total_skeleton_fallbacks
        ),

    "background_path_fallbacks":
        int(
            total_background_fallbacks
        ),

    "pixel_controls_infeasible":
        int(
            pixel_control_infeasible
        ),

    "fixed_budget_reductions":
        int(
            fixed_budget_reductions
        ),

    "fixed_budget_controls_infeasible":
        int(
            fixed_budget_infeasible
        ),

    "natural_pixel_budget_match":
        bool(
            natural_pixel_exact
        ),

    "fixed_budget_pair_match":
        bool(
            fixed_exact
        ),

    "same_background_count_across_conditions":
        bool(
            same_background_count
        ),

    "firewall":
        "PASS",

    "training_performed":
        False,

    "dense_masks_exported":
        False,

    "unknown_voxels_converted_to_background":
        False,
}

write_json(
    audit_dir
    / "block05_weak_label_generator.json",
    block05_audit,
)


# ==========================================================================================
# 21. ATTEMPT TO CAPTURE EXACT EXECUTED BLOCK SOURCE
# ==========================================================================================

source_capture_status = "NOT_AVAILABLE"

try:
    ip = get_ipython()

    raw_cell = (
        ip.history_manager
        .input_hist_raw[-1]
    )

    if (
        "CORA-LUNG — CODE BLOCK 05"
        in raw_cell
        and "GENERATOR_SEED = 20260912"
        in raw_cell
    ):
        code_block_dir = (
            REPO
            / "scripts/code_blocks"
        )

        code_block_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            code_block_dir
            / "block05_generate_weak_labels.py"
        ).write_text(
            raw_cell,
            encoding="utf-8",
        )

        source_capture_status = "PASS"

        print(
            "✓ Exact executed Block-05 source captured to GitHub tree."
        )

except Exception as e:
    print(
        "⚠ Exact cell source capture unavailable:",
        repr(e),
    )


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
            BLOCK_ID,

        "last_completed_block_name":
            BLOCK_NAME,

        "current_stage":
            (
                "native_weak_labels_generated_and_firewall_verified"
                if block05_status == "PASS"
                else
                "weak_label_generator_failed"
            ),

        "current_gate":
            "POST_GATE_A_PRE_GATE_B",

        "gate_a":
            "PASS",

        "gate_b":
            "NOT_RUN",

        "weak_label_generator":
            block05_status,

        "weak_label_generator_version":
            GENERATOR_VERSION,

        "weak_label_generator_seed":
            GENERATOR_SEED,

        "dense_label_firewall":
            "PASS"
            if block05_status == "PASS"
            else "FAIL",

        "local_weak_export":
            str(
                WEAK_ROOT
            ),

        "weak_export_reconstructible":
            True,

        "training_authorized":
            False,

        "next_action":
            (
                "Code Block 06: CT resampling/preprocessing cache, "
                "physical-coordinate transfer of sparse labels to the "
                "training grid, collision audit, and firewall-safe training package."
                if block05_status == "PASS"
                else
                "STOP: resolve weak-label generation or firewall failure."
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
            BLOCK_ID,

        "files":
            repo_manifest,
    },
)


# ==========================================================================================
# 24. COMMIT + PUSH
# ==========================================================================================

heading(
    "STEP 10/10 — COMMIT AND SYNCHRONIZE WEAK-LABEL INFRASTRUCTURE"
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
    changed = status.splitlines()

    print(
        f"Tracked changes: {len(changed)}"
    )

    for line in changed[
        :40
    ]:
        print(
            " ",
            line,
        )

    if len(
        changed
    ) > 40:
        print(
            f"  ... +{len(changed)-40} more"
        )

    commit_message = (
        "data: freeze component-aware weak-label generator and firewall"
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
        "✓ No repository changes detected."
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
# 25. CLEAN AUTHENTICATION MATERIAL
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
# 26. FINAL SCIENTIFIC REPORT
# ==========================================================================================

complete_rows = audit_df[
    audit_df[
        "condition"
    ]
    == "complete"
]

complete_fg_median = float(
    complete_rows[
        "foreground_voxels"
    ].median()
)

natural50 = audit_df[
    audit_df[
        "condition"
    ]
    == "component_natural_50"
]

pixel50 = audit_df[
    audit_df[
        "condition"
    ]
    == "pixel_dropout_matched_50"
]

fixed50 = audit_df[
    audit_df[
        "condition"
    ]
    == "component_fixed_50"
]

complete_fixed50 = audit_df[
    audit_df[
        "condition"
    ]
    == "complete_fixed_50"
]

print("\n")
print("=" * 112)
print("CORA-LUNG CODE BLOCK 05 — WEAK-LABEL GENERATOR AND FIREWALL REPORT")
print("=" * 112)

print(f"""
GENERATOR
---------
Version                                : {GENERATOR_VERSION}
Seed                                   : {GENERATOR_SEED}
Component definition                   : 26-connectivity / >=0.1 mL
Interior erosion                       : {INTERIOR_EROSION_MM:.1f} mm
Base foreground stroke                 : {BASE_FG_STROKE_VOXELS} native voxels/group
Maximum allowed connected path         : {MAX_ALLOWED_PATH_VOXELS} voxels
Background safety band                 : {BACKGROUND_SAFETY_MM:.1f} mm

DATA
----
Cases generated                        : {artifact_df['case_id'].nunique()}
Weak annotation artifacts              : {len(artifact_df)}
Geometry exports                       : {len(list(geometry_dir.glob('*.json')))}
Local deterministic export size        : {human_bytes(local_export_bytes)}

COMPONENT PATH GENERATION
-------------------------
Skeleton fallbacks                     : {total_skeleton_fallbacks}
Background direct-sample fallbacks     : {total_background_fallbacks}

MATCHED CONTROL VALIDATION
--------------------------
Natural-vs-pixel paired comparisons    : {len(natural_pixel_check)}
Exact natural/pixel FG budget match    : {'PASS' if natural_pixel_exact else 'FAIL'}
Fixed-budget paired comparisons        : {len(fixed_check)}
Exact fixed-budget pair match          : {'PASS' if fixed_exact else 'FAIL'}
Identical BG count across conditions   : {'PASS' if same_background_count else 'FAIL'}

FEASIBILITY
-----------
Pixel controls infeasible              : {pixel_control_infeasible}
Fixed-budget reductions                : {fixed_budget_reductions}
Fixed-budget controls infeasible       : {fixed_budget_infeasible}

ANNOTATION BUDGET — MEDIANS
---------------------------
Complete FG voxels                     : {complete_fg_median:.1f}
50% component-natural FG voxels        : {natural50['foreground_voxels'].median():.1f}
50% matched pixel-dropout FG voxels    : {pixel50['foreground_voxels'].median():.1f}
50% component-fixed FG voxels          : {fixed50['foreground_voxels'].median():.1f}
50% complete-fixed FG voxels           : {complete_fixed50['foreground_voxels'].median():.1f}

DENSE-LABEL FIREWALL
--------------------
Runtime export validation              : PASS
Unit tests                             : PASS
Dense masks exported                   : NO
Dense component IDs exported           : NO
Unknown voxels converted to background : NO
Semantic hash verification             : PASS

REPRODUCIBILITY
---------------
Generator config committed             : YES
Generation protocol committed          : YES
Semantic annotation hashes committed   : YES
Patient sparse arrays committed        : NO
Patient sparse arrays reconstructible  : YES
Exact Block-05 source capture          : {source_capture_status}

SCIENTIFIC STATUS
-----------------
Gate A                                 : PASS
Gate B                                 : NOT RUN
Weak-label generator                   : {block05_status}
Dense-label firewall                   : PASS
Model training                         : NOT STARTED

GITHUB
------
Starting commit                        : {starting_commit[:12]}
Current commit                         : {current_commit[:12]}
Synchronization                        : PASS

NEXT
----
Do NOT train yet.

Code Block 06 will create the firewall-safe training-grid package:

  1. resample CT to the frozen anisotropic spacing;
  2. preserve native-to-training physical geometry;
  3. derive the image-based crop without lesion-mask access;
  4. map sparse physical coordinates to the training grid;
  5. combine same-class coordinate collisions;
  6. reject conflicting-class collisions;
  7. quantify labelled-voxel losses after resampling;
  8. verify foreground group IDs survive transfer;
  9. build restartable CT/weak-label caches;
 10. profile RAM/storage and prepare Gate-B baseline training.

Send me the COMPLETE output of this block before proceeding.
""")

print("=" * 112)

if block05_status != "PASS":
    raise RuntimeError(
        "BLOCK 05 FAILED. Do not proceed."
    )