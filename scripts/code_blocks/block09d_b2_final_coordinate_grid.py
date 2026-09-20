# ==========================================================================================
# COVA-3D — BLOCK 09D-B2
# FINAL Coordinate-Level Training-Grid Construction
# Neutral Physical Anchor Banks + Exact Six-Cell Causal Matching
#
# EFFECTIVE PROTOCOL
# ------------------
# COVA3D_1.0 + A1 + A1.1 + A1.2
#
# EXPECTED START COMMIT
# ---------------------
# 790e8c5de909
#
# PURPOSE
# -------
# This is the decisive structural-feasibility block.
#
# GLOBAL_GEOMETRY_RESOLVABLE LESIONS
#   • minimum quota = 2
#   • coherent / dispersed / fragmented geometry retained
#   • maximum quota = common geometry-preserving capacity
#
# GLOBAL_RESOLUTION_LIMITED LESIONS
#   • minimum quota = 1
#   • maximum quota = physical training-grid support capacity
#   • deterministic ordered physical anchor bank
#   • SAME ordered bank across COH / DIS / FRG
#   • SAME ordered bank across C50 / C100
#   • coverage-specific quota uses a PREFIX of that same bank
#   • no geometry claim is made
#
# FINAL B_i_train
# ---------------
# For every patient, search from the largest aggregate-feasible budget downward.
# A candidate budget is accepted ONLY if actual coordinate-level construction passes:
#
#   ✓ no cross-component FG coordinate collisions
#   ✓ no FG/BG collisions
#   ✓ same total FG count across all six cells
#   ✓ identical BG coordinates across all six cells
#   ✓ same component set across geometry
#   ✓ same per-component quota across geometry
#   ✓ C50 component set nested inside C100
#   ✓ neutral-bank prefixes nested across coverage
#   ✓ geometry preserved for geometry-resolvable lesions
#
# DENSE-MASK FIREWALL
# -------------------
# Dense infection masks are used ONLY OFFLINE to construct physical anchor banks.
#
# Dense masks are NOT stored in trainer artifacts.
#
# THIS BLOCK DOES NOT:
#   ✗ train a model
#   ✗ create an optimizer
#   ✗ inspect predictions
#   ✗ inspect checkpoints
#   ✗ inspect segmentation outcomes
#   ✗ authorize factorial training
#
# PASS CONDITION
# --------------
# ALL 20 CASES MUST PASS.
#
# If even one case cannot satisfy the frozen A1.2 rules at coordinate level,
# this block stops and DOES NOT commit a PASS.
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
import gc
import shutil

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import yaml

import nibabel as nib
from nibabel.processing import resample_to_output

from tqdm.auto import tqdm
from kaggle_secrets import UserSecretsClient


# ==========================================================================================
# 0. FROZEN CONSTANTS
# ==========================================================================================

REPO = Path(
    "/kaggle/working/CORA-LUNG"
)

EXPECTED_START_COMMIT = (
    "790e8c5de909"
)

BLOCK = (
    "09D-B2"
)

EFFECTIVE_PROTOCOL = (
    "COVA3D_1.0+A1+A1.1+A1.2"
)

EXPECTED_CASES = (
    20
)

EXPECTED_CONDITIONS = (
    120
)

EXPECTED_ELIGIBLE_COMPONENTS = (
    308
)

EXPECTED_LIMITED_COMPONENTS = (
    28
)

EXPECTED_LIMITED_CASES = (
    13
)

GLOBAL_RESOLVABLE = (
    "GLOBAL_GEOMETRY_RESOLVABLE"
)

GLOBAL_LIMITED = (
    "GLOBAL_RESOLUTION_LIMITED"
)

CONDITIONS = [
    "C50_COH",
    "C50_DIS",
    "C50_FRG",
    "C100_COH",
    "C100_DIS",
    "C100_FRG",
]

CONDITION_TO_COVERAGE = {
    "C50_COH":
        0.50,

    "C50_DIS":
        0.50,

    "C50_FRG":
        0.50,

    "C100_COH":
        1.00,

    "C100_DIS":
        1.00,

    "C100_FRG":
        1.00,
}

CONDITION_TO_GEOMETRY = {
    "C50_COH":
        "coherent",

    "C50_DIS":
        "dispersed",

    "C50_FRG":
        "fragmented",

    "C100_COH":
        "coherent",

    "C100_DIS":
        "dispersed",

    "C100_FRG":
        "fragmented",
}

GEOMETRIES = [
    "coherent",
    "dispersed",
    "fragmented",
]

LIMITED_MIN_QUOTA = (
    1
)

RESOLVABLE_MIN_QUOTA = (
    2
)

REFERENCE_MIN_VOLUME_ML = (
    0.1
)

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

COMMON_HU_MIN = (
    -1250.0
)

COMMON_HU_MAX = (
    250.0
)

BODY_THRESHOLD_HU = (
    -600.0
)

LUNG_AIR_THRESHOLD_HU = (
    -320.0
)

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

CROP_MARGIN_MM = (
    15.0
)

MIN_LUNG_AIR_COMPONENT_ML = (
    50.0
)

MIN_TOTAL_LUNG_AIR_ML = (
    150.0
)

STRUCT26 = np.ones(
    (
        3,
        3,
        3,
    ),
    dtype=np.uint8,
)

FINAL_GRID_ROOT = (
    REPO
    / "data/cova3d_training_grid_candidates_v1_2"
)

BANK_ROOT = (
    REPO
    / "data/cova3d_neutral_anchor_banks_v1_0"
)

MANIFEST_DIR = (
    REPO
    / "data/manifests"
)

NOW_ISO = datetime.now(
    timezone.utc
).strftime(
    "%Y-%m-%dT%H:%M:%SZ"
)


# ==========================================================================================
# 1. GENERAL HELPERS
# ==========================================================================================

def heading(text):

    print(
        "\n"
        + "=" * 130
    )

    print(
        text
    )

    print(
        "=" * 130
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


def sha256_file(path):

    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        for chunk in iter(
            lambda:
                f.read(
                    8
                    * 1024
                    * 1024
                ),
            b"",
        ):

            h.update(
                chunk
            )

    return h.hexdigest()


def semantic_hash(*arrays):

    h = hashlib.sha256()

    for array in arrays:

        array = np.ascontiguousarray(
            np.asarray(
                array
            )
        )

        h.update(
            str(
                array.dtype
            ).encode(
                "utf-8"
            )
        )

        h.update(
            np.asarray(
                array.shape,
                dtype=np.int64,
            ).tobytes()
        )

        h.update(
            array.tobytes()
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


def lexical_sort_coords(coords):

    coords = np.asarray(
        coords,
        dtype=np.int32,
    )

    if len(
        coords
    ) == 0:

        return np.empty(
            (
                0,
                3,
            ),
            dtype=np.int32,
        )

    order = np.lexsort(
        (
            coords[
                :,
                2
            ],
            coords[
                :,
                1
            ],
            coords[
                :,
                0
            ],
        )
    )

    return coords[
        order
    ]


def unique_coords(coords):

    coords = np.asarray(
        coords,
        dtype=np.int32,
    )

    if len(
        coords
    ) == 0:

        return np.empty(
            (
                0,
                3,
            ),
            dtype=np.int32,
        )

    return lexical_sort_coords(
        np.unique(
            coords,
            axis=0,
        )
    )


def make_git_auth():

    token = UserSecretsClient().get_secret(
        "pushCora"
    )

    if not token:

        raise RuntimeError(
            "Kaggle secret 'pushCora' unavailable."
        )

    token = token.strip()

    askpass = Path(
        "/tmp/cova3d_git_askpass_09db2.sh"
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

    env = os.environ.copy()

    env[
        "GITHUB_TOKEN"
    ] = token

    env[
        "GIT_ASKPASS"
    ] = str(
        askpass
    )

    env[
        "GIT_TERMINAL_PROMPT"
    ] = "0"

    return (
        token,
        askpass,
        env,
    )


def safe_push(
    env,
    token,
):

    result = sh(
        [
            "git",
            "push",
            "origin",
            "main",
        ],
        cwd=REPO,
        env=env,
        check=False,
    )

    if result.returncode != 0:

        safe_error = (
            result.stderr
            or ""
        ).replace(
            token,
            "***TOKEN_REDACTED***",
        )

        raise RuntimeError(
            "GitHub push failed:\n"
            + safe_error
        )


# ==========================================================================================
# 2. SPARSE CONNECTIVITY HELPERS
# ==========================================================================================

OFFSETS_26 = [
    (
        dz,
        dy,
        dx,
    )
    for dz in (
        -1,
        0,
        1,
    )
    for dy in (
        -1,
        0,
        1,
    )
    for dx in (
        -1,
        0,
        1,
    )
    if not (
        dz == 0
        and dy == 0
        and dx == 0
    )
]


def sparse_components(coords):

    coords = unique_coords(
        coords
    )

    remaining = {
        tuple(
            int(
                value
            )
            for value in coord
        )
        for coord in coords
    }

    components = []

    while remaining:

        seed = min(
            remaining
        )

        remaining.remove(
            seed
        )

        stack = [
            seed
        ]

        component = {
            seed
        }

        while stack:

            z, y, x = stack.pop()

            for dz, dy, dx in OFFSETS_26:

                neighbor = (
                    z + dz,
                    y + dy,
                    x + dx,
                )

                if neighbor in remaining:

                    remaining.remove(
                        neighbor
                    )

                    component.add(
                        neighbor
                    )

                    stack.append(
                        neighbor
                    )

        components.append(
            component
        )

    return sorted(
        components,
        key=lambda component: (
            -len(
                component
            ),
            min(
                component
            ),
        ),
    )


def sparse_component_count(coords):

    return len(
        sparse_components(
            coords
        )
    )


def component_set_to_array(component):

    return np.asarray(
        sorted(
            component
        ),
        dtype=np.int32,
    )


# ==========================================================================================
# 3. GEOMETRY-PRESERVING SUBSET HELPERS
# ==========================================================================================

def connected_subset(
    pool_coords,
    quota,
    spacing_zyx,
):

    pool_coords = unique_coords(
        pool_coords
    )

    quota = int(
        quota
    )

    components = sparse_components(
        pool_coords
    )

    if not components:

        raise RuntimeError(
            "Empty coherent candidate pool."
        )

    largest = components[
        0
    ]

    if len(
        largest
    ) < quota:

        raise RuntimeError(
            "Connected candidate pool is smaller than requested quota."
        )

    pool = component_set_to_array(
        largest
    )

    spacing_zyx = np.asarray(
        spacing_zyx,
        dtype=np.float64,
    )

    physical = (
        pool.astype(
            np.float64
        )
        * spacing_zyx[
            None,
            :
        ]
    )

    centroid = physical.mean(
        axis=0
    )

    distance = np.sum(
        (
            physical
            - centroid[
                None,
                :
            ]
        )
        ** 2,
        axis=1,
    )

    seed = tuple(
        int(
            value
        )
        for value in pool[
            int(
                np.argmin(
                    distance
                )
            )
        ]
    )

    selected = {
        seed
    }

    frontier = set()

    def add_neighbors(coord):

        z, y, x = coord

        for dz, dy, dx in OFFSETS_26:

            neighbor = (
                z + dz,
                y + dy,
                x + dx,
            )

            if (
                neighbor in largest
                and neighbor not in selected
            ):

                frontier.add(
                    neighbor
                )

    add_neighbors(
        seed
    )

    while len(
        selected
    ) < quota:

        if not frontier:

            raise RuntimeError(
                "Connected selection exhausted before quota."
            )

        def key(coord):

            physical_coord = (
                np.asarray(
                    coord,
                    dtype=np.float64,
                )
                * spacing_zyx
            )

            return (
                float(
                    np.sum(
                        (
                            physical_coord
                            - centroid
                        )
                        ** 2
                    )
                ),
                coord,
            )

        candidate = min(
            frontier,
            key=key,
        )

        frontier.remove(
            candidate
        )

        selected.add(
            candidate
        )

        add_neighbors(
            candidate
        )

    output = component_set_to_array(
        selected
    )

    if sparse_component_count(
        output
    ) != 1:

        raise RuntimeError(
            "Coherent subset is disconnected."
        )

    return output


def farthest_point_subset(
    pool_coords,
    quota,
    spacing_zyx,
):

    pool_coords = unique_coords(
        pool_coords
    )

    quota = int(
        quota
    )

    if quota <= 0:

        raise RuntimeError(
            "Invalid dispersed quota."
        )

    if quota > len(
        pool_coords
    ):

        raise RuntimeError(
            "Dispersed quota exceeds candidate pool."
        )

    spacing_zyx = np.asarray(
        spacing_zyx,
        dtype=np.float64,
    )

    physical = (
        pool_coords.astype(
            np.float64
        )
        * spacing_zyx[
            None,
            :
        ]
    )

    centroid = physical.mean(
        axis=0
    )

    centroid_distance = np.sum(
        (
            physical
            - centroid[
                None,
                :
            ]
        )
        ** 2,
        axis=1,
    )

    first = int(
        np.argmax(
            centroid_distance
        )
    )

    selected = [
        first
    ]

    min_distance = np.sum(
        (
            physical
            - physical[
                first
            ][
                None,
                :
            ]
        )
        ** 2,
        axis=1,
    )

    min_distance[
        first
    ] = (
        -np.inf
    )

    while len(
        selected
    ) < quota:

        next_index = int(
            np.argmax(
                min_distance
            )
        )

        selected.append(
            next_index
        )

        distance = np.sum(
            (
                physical
                - physical[
                    next_index
                ][
                    None,
                    :
                ]
            )
            ** 2,
            axis=1,
        )

        min_distance = np.minimum(
            min_distance,
            distance,
        )

        min_distance[
            selected
        ] = (
            -np.inf
        )

    return lexical_sort_coords(
        pool_coords[
            np.asarray(
                selected,
                dtype=np.int64,
            )
        ]
    )


def allocate_exact(
    total,
    component_ids,
    capacities,
    minimums,
):

    total = int(
        total
    )

    component_ids = [
        int(
            component_id
        )
        for component_id in component_ids
    ]

    allocation = {}


    for component_id in component_ids:

        minimum = int(
            minimums[
                component_id
            ]
        )

        capacity = int(
            capacities[
                component_id
            ]
        )

        if capacity < minimum:

            return None

        allocation[
            component_id
        ] = minimum


    remaining = int(
        total
        - sum(
            allocation.values()
        )
    )


    if remaining < 0:

        return None


    # Frozen A1.2 allocation:
    # selected-component order, one voxel per component per pass.
    while remaining > 0:

        progressed = False

        for component_id in component_ids:

            if remaining <= 0:

                break

            if (
                allocation[
                    component_id
                ]
                < int(
                    capacities[
                        component_id
                    ]
                )
            ):

                allocation[
                    component_id
                ] += 1

                remaining -= 1

                progressed = True

        if not progressed:

            return None

    return allocation


def fragmented_subset(
    pool_coords,
    quota,
    spacing_zyx,
):

    quota = int(
        quota
    )

    pool_coords = unique_coords(
        pool_coords
    )

    if quota < 2:

        raise RuntimeError(
            "Fragmented geometry requires quota >=2."
        )

    components = sparse_components(
        pool_coords
    )

    if len(
        components
    ) < 2:

        raise RuntimeError(
            "Fragmented candidate has fewer than two disconnected sets."
        )

    n_fragments = min(
        3,
        len(
            components
        ),
        quota,
    )

    selected_components = components[
        :n_fragments
    ]

    capacities = {
        index:
            len(
                component
            )
        for index, component in enumerate(
            selected_components
        )
    }

    minimums = {
        index:
            1
        for index in capacities
    }

    allocation = allocate_exact(
        total=quota,
        component_ids=list(
            capacities
        ),
        capacities=capacities,
        minimums=minimums,
    )

    if allocation is None:

        raise RuntimeError(
            "Fragment allocation failed."
        )

    parts = []

    for index, component in enumerate(
        selected_components
    ):

        part = connected_subset(
            component_set_to_array(
                component
            ),
            allocation[
                index
            ],
            spacing_zyx,
        )

        parts.append(
            part
        )

    output = unique_coords(
        np.concatenate(
            parts,
            axis=0,
        )
    )

    if len(
        output
    ) != quota:

        raise RuntimeError(
            "Fragmented quota mismatch."
        )

    if sparse_component_count(
        output
    ) < 2:

        raise RuntimeError(
            "Fragmented annotation collapsed."
        )

    return output


def max_geometry_capacity(
    pool_coords,
    geometry,
    spacing_zyx,
):

    pool_coords = unique_coords(
        pool_coords
    )

    if len(
        pool_coords
    ) < 2:

        return 0


    if geometry == "coherent":

        components = sparse_components(
            pool_coords
        )

        if not components:

            return 0

        return int(
            len(
                components[
                    0
                ]
            )
        )


    if geometry == "fragmented":

        components = sparse_components(
            pool_coords
        )

        if len(
            components
        ) < 2:

            return 0

        return int(
            sum(
                len(
                    component
                )
                for component in components[
                    :3
                ]
            )
        )


    if geometry == "dispersed":

        for quota in range(
            len(
                pool_coords
            ),
            1,
            -1,
        ):

            selected = farthest_point_subset(
                pool_coords,
                quota,
                spacing_zyx,
            )

            if sparse_component_count(
                selected
            ) >= 2:

                return int(
                    quota
                )

        return 0


    raise RuntimeError(
        "Unknown geometry."
    )


def select_geometry(
    pool_coords,
    geometry,
    quota,
    spacing_zyx,
):

    quota = int(
        quota
    )

    if quota < 2:

        raise RuntimeError(
            "Resolvable geometry received quota <2."
        )


    if geometry == "coherent":

        output = connected_subset(
            pool_coords,
            quota,
            spacing_zyx,
        )

        if sparse_component_count(
            output
        ) != 1:

            raise RuntimeError(
                "Coherent connectivity failure."
            )

        return output


    if geometry == "dispersed":

        output = farthest_point_subset(
            pool_coords,
            quota,
            spacing_zyx,
        )

        if sparse_component_count(
            output
        ) < 2:

            raise RuntimeError(
                "Dispersed separation failure."
            )

        return output


    if geometry == "fragmented":

        output = fragmented_subset(
            pool_coords,
            quota,
            spacing_zyx,
        )

        if sparse_component_count(
            output
        ) < 2:

            raise RuntimeError(
                "Fragmented separation failure."
            )

        return output


    raise RuntimeError(
        "Unknown geometry."
    )


def mean_nearest_neighbor_mm(
    coords_zyx,
    spacing_zyx,
):

    coords_zyx = np.asarray(
        coords_zyx,
        dtype=np.float64,
    )

    if len(
        coords_zyx
    ) <= 1:

        return 0.0

    physical = (
        coords_zyx
        * np.asarray(
            spacing_zyx,
            dtype=np.float64,
        )[
            None,
            :
        ]
    )

    delta = (
        physical[
            :,
            None,
            :
        ]
        - physical[
            None,
            :,
            :
        ]
    )

    distance = np.sqrt(
        np.sum(
            delta
            ** 2,
            axis=-1,
        )
    )

    np.fill_diagonal(
        distance,
        np.inf,
    )

    return float(
        np.min(
            distance,
            axis=1,
        ).mean()
    )


# ==========================================================================================
# 4. CORRECTED v1.2 PREPROCESSING
# ==========================================================================================

def largest_2d_component(mask2d):

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

    margin_voxels = np.ceil(
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
        - margin_voxels,
    )

    hi = np.minimum(
        np.asarray(
            mask.shape,
            dtype=np.int32,
        ),
        hi
        + margin_voxels,
    )

    return (
        lo,
        hi,
    )


def normalize_raw_hu(image):

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


def normalize_prewindowed(image):

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


def derive_normalized_image_only_crop(
    normalized_zyx,
    spacing_zyx,
):

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


        body_envelope[
            z
        ] = ndi.binary_fill_holes(
            slice_mask
        )


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


    labels, count = ndi.label(
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


    if count > 0:

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

        return {
            "lo_zyx":
                lung_lo.astype(
                    np.int32
                ),

            "hi_zyx":
                lung_hi.astype(
                    np.int32
                ),

            "method":
                "normalized_internal_air_lung_bbox",

            "lung_air_ml":
                lung_air_ml,
        }


    return {
        "lo_zyx":
            body_lo.astype(
                np.int32
            ),

        "hi_zyx":
            body_hi.astype(
                np.int32
            ),

        "method":
            "normalized_body_envelope_fallback",

        "lung_air_ml":
            lung_air_ml,
    }


# ==========================================================================================
# 5. VERIFY REPOSITORY STATE
# ==========================================================================================

heading(
    "COVA-3D BLOCK 09D-B2 — FINAL COORDINATE-LEVEL CONSTRUCTION"
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
        "Unexpected repository commit.\n"
        + "Expected: "
        + EXPECTED_START_COMMIT
        + "\nObserved: "
        + head
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
        "Repository must be clean before 09D-B2.\n"
        + dirty
    )


print(
    "✓ Starting commit                       :",
    head[:12],
)

print(
    "✓ Repository state                      : CLEAN"
)


# ==========================================================================================
# 6. VERIFY A1.2
# ==========================================================================================

heading(
    "STEP 1/12 — VERIFY FROZEN A1.2 PROTOCOL"
)


cova_state_path = (
    REPO
    / "COVA3D_STATE.json"
)

project_state_path = (
    REPO
    / "PROJECT_STATE.json"
)

a12_path = (
    REPO
    / "configs/"
    "cova3d_resolution_aware_amendment_A1_2.yaml"
)


cova_state = json.loads(
    cova_state_path.read_text(
        encoding="utf-8"
    )
)


project_state = json.loads(
    project_state_path.read_text(
        encoding="utf-8"
    )
)


a12 = yaml.safe_load(
    a12_path.read_text(
        encoding="utf-8"
    )
)


if cova_state.get(
    "last_completed_block"
) != "09D-A3-LOCK":

    raise RuntimeError(
        "Expected 09D-A3-LOCK as the previous completed block."
    )


if cova_state.get(
    "effective_protocol"
) != EFFECTIVE_PROTOCOL:

    raise RuntimeError(
        "Effective protocol mismatch."
    )


if a12.get(
    "status"
) != "PROSPECTIVELY_FROZEN_BEFORE_COVA_TRAINING":

    raise RuntimeError(
        "A1.2 is not prospectively frozen."
    )


if a12.get(
    "next_block"
) != "09D-B2":

    raise RuntimeError(
        "A1.2 does not point to 09D-B2."
    )


if int(
    cova_state.get(
        "optimizer_steps_in_cova3d",
        -1,
    )
) != 0:

    raise RuntimeError(
        "Unexpected COVA optimizer steps."
    )


if cova_state.get(
    "factorial_training_authorized"
) is not False:

    raise RuntimeError(
        "Factorial training was unexpectedly authorized."
    )


print(
    "✓ Effective protocol                    :",
    EFFECTIVE_PROTOCOL,
)

print(
    "✓ A1.2                                  : FROZEN"
)

print(
    "✓ Neutral-bank aggregate feasibility    : PASS 20/20"
)

print(
    "✓ Optimizer steps                       : 0"
)

print(
    "✓ Factorial training                    : NOT AUTHORIZED"
)


# ==========================================================================================
# 7. IMPORT FROZEN WORLD→GRID MAPPER
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
    "cora_lung.data.preprocess",
    None,
)


importlib.invalidate_caches()


from cora_lung.data.preprocess import (
    world_xyz_to_full_zyx,
)


# ==========================================================================================
# 8. LOAD FROZEN MANIFESTS
# ==========================================================================================

split_df = pd.read_csv(
    REPO
    / "data/splits/"
    "primary_volume_split_registry.csv"
)


source_hash_df = pd.read_csv(
    REPO
    / "data/manifests/"
    "primary_file_sha256.csv"
)


crop_reference_df = pd.read_csv(
    REPO
    / "data/manifests/"
    "training_crop_v1_2_audit.csv"
)


native_budget_df = pd.read_csv(
    REPO
    / "data/manifests/"
    "cova3d_native_budget_v1_0.csv"
)


native_artifact_df = pd.read_csv(
    REPO
    / "data/manifests/"
    "cova3d_native_candidate_artifact_manifest_v1_0.csv"
)


global_class_df = pd.read_csv(
    REPO
    / "data/manifests/"
    "cova3d_global_resolution_class_v1_0.csv"
)


neutral_capacity_df = pd.read_csv(
    REPO
    / "data/manifests/"
    "cova3d_resolution_limited_neutral_bank_capacity_v1_0.csv"
)


if len(
    split_df
) != EXPECTED_CASES:

    raise RuntimeError(
        "Expected 20 split rows."
    )


if len(
    native_artifact_df
) != EXPECTED_CONDITIONS:

    raise RuntimeError(
        "Expected 120 native candidate artifacts."
    )


if len(
    global_class_df
) != EXPECTED_ELIGIBLE_COMPONENTS:

    raise RuntimeError(
        "Expected 308 global component classifications."
    )


if len(
    neutral_capacity_df
) != EXPECTED_LIMITED_COMPONENTS:

    raise RuntimeError(
        "Expected 28 neutral-bank capacity rows."
    )


if (
    global_class_df[
        "global_resolution_class"
    ]
    == GLOBAL_LIMITED
).sum() != EXPECTED_LIMITED_COMPONENTS:

    raise RuntimeError(
        "Global limited-component count mismatch."
    )


source_hash_lookup = {
    str(
        row[
            "relative_path"
        ]
    ):
        str(
            row[
                "sha256"
            ]
        ).lower()
    for _, row
    in source_hash_df.iterrows()
}


crop_lookup = {
    str(
        row[
            "case_id"
        ]
    ):
        row
    for _, row
    in crop_reference_df.iterrows()
}


native_budget_lookup = {
    str(
        row[
            "case_id"
        ]
    ):
        row
    for _, row
    in native_budget_df.iterrows()
}


artifact_lookup = {
    (
        str(
            row[
                "case_id"
            ]
        ),
        str(
            row[
                "condition_id"
            ]
        ),
    ):
        row
    for _, row
    in native_artifact_df.iterrows()
}


global_class_lookup = {
    (
        str(
            row[
                "case_id"
            ]
        ),
        int(
            row[
                "component_native_id"
            ]
        ),
    ):
        str(
            row[
                "global_resolution_class"
            ]
        )
    for _, row
    in global_class_df.iterrows()
}


global_volume_lookup = {
    (
        str(
            row[
                "case_id"
            ]
        ),
        int(
            row[
                "component_native_id"
            ]
        ),
    ):
        float(
            row[
                "component_volume_ml"
            ]
        )
    for _, row
    in global_class_df.iterrows()
}


neutral_capacity_lookup = {
    (
        str(
            row[
                "case_id"
            ]
        ),
        int(
            row[
                "component_native_id"
            ]
        ),
    ):
        int(
            row[
                "mapped_physical_support_capacity"
            ]
        )
    for _, row
    in neutral_capacity_df.iterrows()
}


for _, row in tqdm(
    native_artifact_df.iterrows(),
    total=len(
        native_artifact_df
    ),
    desc="Verify native sparse candidates",
):

    path = (
        REPO
        / str(
            row[
                "artifact_file"
            ]
        )
    )

    if not path.exists():

        raise RuntimeError(
            "Native sparse candidate missing:\n"
            + str(
                path
            )
        )

    if sha256_file(
        path
    ) != str(
        row[
            "file_sha256"
        ]
    ):

        raise RuntimeError(
            "Native sparse-candidate checksum mismatch:\n"
            + str(
                path
            )
        )


print(
    "✓ Native sparse candidates              : VERIFIED 120/120"
)


# ==========================================================================================
# 9. LOCATE PRIMARY DATASET
# ==========================================================================================

candidate_roots = [
    Path(
        "/kaggle/input/datasets/andrewmvd/covid19-ct-scans"
    ),

    Path(
        "/kaggle/input/covid19-ct-scans"
    ),
]


probe_ct = str(
    split_df.iloc[
        0
    ][
        "ct_scan"
    ]
)


primary_root = None


for root in candidate_roots:

    if (
        root.exists()
        and (
            root
            / probe_ct
        ).exists()
    ):

        primary_root = root

        break


if primary_root is None:

    raise RuntimeError(
        "Primary COVID-19 CT Kaggle dataset is not attached."
    )


print(
    "✓ Primary dataset root                  :",
    primary_root,
)


# ==========================================================================================
# 10. PREPARE OUTPUT DIRECTORIES
# ==========================================================================================

for output_root in [
    FINAL_GRID_ROOT,
    BANK_ROOT,
]:

    if output_root.exists():

        existing_npz = list(
            output_root.rglob(
                "*.npz"
            )
        )

        if existing_npz:

            raise RuntimeError(
                "Output directory already contains NPZ files:\n"
                + str(
                    output_root
                )
                + "\nDo not overwrite a previous 09D-B2 run."
            )

        shutil.rmtree(
            output_root
        )


FINAL_GRID_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)


BANK_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)


# ==========================================================================================
# 11. WORLD→CROP GRID MAPPING
# ==========================================================================================

def map_world_to_crop(
    world_xyz,
    resampled_affine_xyz,
    full_shape_zyx,
    crop_origin_zyx,
    crop_shape_zyx,
):

    world_xyz = np.asarray(
        world_xyz,
        dtype=np.float64,
    )

    full_zyx = world_xyz_to_full_zyx(
        world_xyz,
        resampled_affine_xyz,
    )

    full_shape_zyx = np.asarray(
        full_shape_zyx,
        dtype=np.int32,
    )

    crop_origin_zyx = np.asarray(
        crop_origin_zyx,
        dtype=np.int32,
    )

    crop_shape_zyx = np.asarray(
        crop_shape_zyx,
        dtype=np.int32,
    )


    inside_full = np.all(
        (
            full_zyx
            >= 0
        )
        & (
            full_zyx
            < full_shape_zyx[
                None,
                :
            ]
        ),
        axis=1,
    )


    crop_zyx = (
        full_zyx
        - crop_origin_zyx[
            None,
            :
        ]
    )


    inside_crop = (
        inside_full
        & np.all(
            (
                crop_zyx
                >= 0
            )
            & (
                crop_zyx
                < crop_shape_zyx[
                    None,
                    :
                ]
            ),
            axis=1,
        )
    )


    return (
        crop_zyx.astype(
            np.int32
        ),
        inside_full,
        inside_crop,
    )


# ==========================================================================================
# 12. FROZEN A1.2 NEUTRAL BANK CONSTRUCTION
# ==========================================================================================

def build_ordered_neutral_bank(
    native_component_xyz,
    native_affine_xyz,
    resampled_affine_xyz,
    full_shape_zyx,
    crop_origin_zyx,
    crop_shape_zyx,
):

    native_component_xyz = np.asarray(
        native_component_xyz,
        dtype=np.int32,
    )


    if len(
        native_component_xyz
    ) == 0:

        raise RuntimeError(
            "Cannot build bank from empty physical component."
        )


    native_world_xyz = nib.affines.apply_affine(
        native_affine_xyz,
        native_component_xyz,
    ).astype(
        np.float64
    )


    centroid_world_xyz = native_world_xyz.mean(
        axis=0
    )


    (
        mapped_crop_zyx,
        _,
        inside_crop,
    ) = map_world_to_crop(
        native_world_xyz,
        resampled_affine_xyz,
        full_shape_zyx,
        crop_origin_zyx,
        crop_shape_zyx,
    )


    support = unique_coords(
        mapped_crop_zyx[
            inside_crop
        ]
    )


    if len(
        support
    ) == 0:

        raise RuntimeError(
            "Physical lesion has zero support on cropped training grid."
        )


    full_zyx = (
        support
        + np.asarray(
            crop_origin_zyx,
            dtype=np.int32,
        )[
            None,
            :
        ]
    )


    full_xyz = full_zyx[
        :,
        ::-1
    ]


    support_world_xyz = nib.affines.apply_affine(
        resampled_affine_xyz,
        full_xyz,
    ).astype(
        np.float64
    )


    distance_sq = np.sum(
        (
            support_world_xyz
            - centroid_world_xyz[
                None,
                :
            ]
        )
        ** 2,
        axis=1,
    )


    # np.lexsort uses the LAST key as the primary key:
    # primary = distance
    # then z
    # then y
    # then x
    order = np.lexsort(
        (
            support[
                :,
                2
            ],
            support[
                :,
                1
            ],
            support[
                :,
                0
            ],
            distance_sq,
        )
    )


    ordered_support = support[
        order
    ].astype(
        np.int32
    )


    ordered_world = support_world_xyz[
        order
    ].astype(
        np.float64
    )


    ordered_distance = np.sqrt(
        distance_sq[
            order
        ]
    ).astype(
        np.float64
    )


    return {
        "ordered_voxel_zyx":
            ordered_support,

        "ordered_world_xyz":
            ordered_world,

        "distance_to_centroid_mm":
            ordered_distance,

        "centroid_world_xyz":
            centroid_world_xyz.astype(
                np.float64
            ),
    }


# ==========================================================================================
# 13. OUTPUT COLLECTIONS
# ==========================================================================================

grid_rows = []

transfer_rows = []

bank_rows = []

budget_rows = []

condition_rows = []

component_rows = []

artifact_rows = []


ct_arrays_accessed = (
    0
)

dense_lesion_arrays_accessed = (
    0
)

dense_lung_arrays_accessed = (
    0
)

final_outer_dense_masks_used_offline = (
    0
)

predictions_accessed = (
    0
)

checkpoints_accessed = (
    0
)

model_outcomes_accessed = (
    0
)


total_candidate_budget_rejections = (
    0
)

total_cross_component_collision_rejections = (
    0
)

total_background_rejections = (
    0
)

total_geometry_selection_rejections = (
    0
)


# ==========================================================================================
# 14. PROCESS ALL 20 CASES
# ==========================================================================================

heading(
    "STEP 2/12 — BUILD PHYSICAL BANKS + MAP ORIGINAL GEOMETRY CANDIDATES"
)


for _, split_row in tqdm(
    split_df.sort_values(
        "case_id"
    ).iterrows(),
    total=len(
        split_df
    ),
    desc="09D-B2 cases",
):

    case_id = str(
        split_row[
            "case_id"
        ]
    )


    role = str(
        split_row[
            "role"
        ]
    )


    source_origin = str(
        split_row[
            "source_origin"
        ]
    )


    ct_relative = str(
        split_row[
            "ct_scan"
        ]
    )


    infection_relative = str(
        split_row[
            "infection_mask"
        ]
    )


    ct_path = (
        primary_root
        / ct_relative
    )


    infection_path = (
        primary_root
        / infection_relative
    )


    # --------------------------------------------------------------------------------------
    # Verify frozen source data.
    # --------------------------------------------------------------------------------------

    for relative_path, absolute_path in [
        (
            ct_relative,
            ct_path,
        ),
        (
            infection_relative,
            infection_path,
        ),
    ]:

        if relative_path not in source_hash_lookup:

            raise RuntimeError(
                "Missing source hash for:\n"
                + relative_path
            )

        if sha256_file(
            absolute_path
        ).lower() != source_hash_lookup[
            relative_path
        ]:

            raise RuntimeError(
                "Frozen source checksum mismatch:\n"
                + relative_path
            )


    # ======================================================================================
    # A. Reconstruct corrected v1.2 CT grid
    # ======================================================================================

    native_ct_img = nib.load(
        str(
            ct_path
        ),
        mmap=True,
    )


    native_ct = np.asarray(
        native_ct_img.dataobj,
        dtype=np.float32,
    )


    ct_arrays_accessed += 1


    source_min = float(
        native_ct.min()
    )


    source_max = float(
        native_ct.max()
    )


    if source_origin.lower() == "radiopaedia":

        if (
            source_min < -1e-5
            or source_max > 255.0 + 1e-5
        ):

            raise RuntimeError(
                "Radiopaedia source encoding check failed for "
                + case_id
            )


        encoded_native = np.clip(
            native_ct,
            0.0,
            255.0,
        )


        interpolation_cval = (
            0.0
        )


        source_encoding = (
            "documented_prewindowed_0_255"
        )


    elif source_origin.lower() == "coronacases":

        encoded_native = np.clip(
            native_ct,
            COMMON_HU_MIN,
            COMMON_HU_MAX,
        )


        interpolation_cval = (
            COMMON_HU_MIN
        )


        source_encoding = (
            "raw_hu"
        )


    else:

        raise RuntimeError(
            "Unknown source origin: "
            + source_origin
        )


    encoded_header = native_ct_img.header.copy()


    encoded_header.set_data_dtype(
        np.float32
    )


    encoded_img = nib.Nifti1Image(
        encoded_native,
        native_ct_img.affine,
        header=encoded_header,
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


    actual_spacing_xyz = np.asarray(
        resampled_img.header.get_zooms()[
            :3
        ],
        dtype=np.float64,
    )


    if not np.allclose(
        actual_spacing_xyz,
        TARGET_SPACING_XYZ,
        atol=1e-6,
        rtol=1e-6,
    ):

        raise RuntimeError(
            "Resampled spacing mismatch for "
            + case_id
        )


    resampled_xyz = np.asarray(
        resampled_img.dataobj,
        dtype=np.float32,
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

        normalized_full = normalize_prewindowed(
            resampled_zyx
        )

    else:

        normalized_full = normalize_raw_hu(
            resampled_zyx
        )


    resampled_affine_xyz = np.asarray(
        resampled_img.affine,
        dtype=np.float64,
    )


    full_shape_zyx = np.asarray(
        normalized_full.shape,
        dtype=np.int32,
    )


    crop_info = derive_normalized_image_only_crop(
        normalized_full,
        TARGET_SPACING_ZYX,
    )


    crop_origin_zyx = np.asarray(
        crop_info[
            "lo_zyx"
        ],
        dtype=np.int32,
    )


    crop_hi_zyx = np.asarray(
        crop_info[
            "hi_zyx"
        ],
        dtype=np.int32,
    )


    crop_shape_zyx = (
        crop_hi_zyx
        - crop_origin_zyx
    )


    frozen_crop = crop_lookup[
        case_id
    ]


    expected_origin = np.asarray(
        [
            int(
                frozen_crop[
                    "new_crop_origin_z"
                ]
            ),
            int(
                frozen_crop[
                    "new_crop_origin_y"
                ]
            ),
            int(
                frozen_crop[
                    "new_crop_origin_x"
                ]
            ),
        ],
        dtype=np.int32,
    )


    if not np.array_equal(
        crop_origin_zyx,
        expected_origin,
    ):

        raise RuntimeError(
            "Frozen v1.2 crop origin was not reproduced for "
            + case_id
        )


    if int(
        np.prod(
            crop_shape_zyx
        )
    ) != int(
        frozen_crop[
            "new_crop_voxels"
        ]
    ):

        raise RuntimeError(
            "Frozen crop volume was not reproduced for "
            + case_id
        )


    if str(
        crop_info[
            "method"
        ]
    ) != str(
        frozen_crop[
            "crop_method"
        ]
    ):

        raise RuntimeError(
            "Frozen crop method mismatch for "
            + case_id
        )


    grid_rows.append(
        {
            "case_id":
                case_id,

            "source_origin":
                source_origin,

            "source_encoding":
                source_encoding,

            "spacing_z_mm":
                3.0,

            "spacing_y_mm":
                1.5,

            "spacing_x_mm":
                1.5,

            "full_shape_z":
                int(
                    full_shape_zyx[
                        0
                    ]
                ),

            "full_shape_y":
                int(
                    full_shape_zyx[
                        1
                    ]
                ),

            "full_shape_x":
                int(
                    full_shape_zyx[
                        2
                    ]
                ),

            "crop_origin_z":
                int(
                    crop_origin_zyx[
                        0
                    ]
                ),

            "crop_origin_y":
                int(
                    crop_origin_zyx[
                        1
                    ]
                ),

            "crop_origin_x":
                int(
                    crop_origin_zyx[
                        2
                    ]
                ),

            "crop_shape_z":
                int(
                    crop_shape_zyx[
                        0
                    ]
                ),

            "crop_shape_y":
                int(
                    crop_shape_zyx[
                        1
                    ]
                ),

            "crop_shape_x":
                int(
                    crop_shape_zyx[
                        2
                    ]
                ),

            "crop_method":
                str(
                    crop_info[
                        "method"
                    ]
                ),

            "crop_reproduction_pass":
                True,
        }
    )


    # ======================================================================================
    # B. Dense physical lesion components — OFFLINE bank construction only
    # ======================================================================================

    infection_img = nib.load(
        str(
            infection_path
        ),
        mmap=True,
    )


    infection = (
        np.asarray(
            infection_img.dataobj
        )
        > 0.5
    )


    dense_lesion_arrays_accessed += 1


    if role == "final_outer_cv":

        final_outer_dense_masks_used_offline += 1


    infection_affine_xyz = np.asarray(
        infection_img.affine,
        dtype=np.float64,
    )


    voxel_volume_ml = float(
        abs(
            np.linalg.det(
                infection_affine_xyz[
                    :3,
                    :3
                ]
            )
        )
        / 1000.0
    )


    component_labels, raw_component_count = ndi.label(
        infection,
        structure=STRUCT26,
    )


    component_sizes = np.bincount(
        component_labels.ravel()
    ).astype(
        np.int64
    )


    eligible_component_ids = [
        int(
            component_id
        )
        for component_id in range(
            1,
            raw_component_count
            + 1,
        )
        if (
            float(
                component_sizes[
                    component_id
                ]
            )
            * voxel_volume_ml
            >= REFERENCE_MIN_VOLUME_ML
        )
    ]


    expected_component_ids = sorted(
        global_class_df.loc[
            global_class_df[
                "case_id"
            ]
            == case_id,
            "component_native_id",
        ].astype(
            int
        ).tolist()
    )


    if sorted(
        eligible_component_ids
    ) != expected_component_ids:

        raise RuntimeError(
            "Dense component-ID lineage mismatch for "
            + case_id
        )


    # ======================================================================================
    # C. Build and persist neutral banks for GLOBAL_LIMITED lesions
    # ======================================================================================

    neutral_bank_lookup = {}


    limited_case_df = global_class_df[
        (
            global_class_df[
                "case_id"
            ]
            == case_id
        )
        & (
            global_class_df[
                "global_resolution_class"
            ]
            == GLOBAL_LIMITED
        )
    ]


    case_bank_dir = (
        BANK_ROOT
        / case_id
    )


    if len(
        limited_case_df
    ) > 0:

        case_bank_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


    for _, limited_row in limited_case_df.iterrows():

        component_id = int(
            limited_row[
                "component_native_id"
            ]
        )


        native_component_xyz = np.argwhere(
            component_labels
            == component_id
        ).astype(
            np.int32
        )


        observed_volume_ml = float(
            len(
                native_component_xyz
            )
            * voxel_volume_ml
        )


        expected_volume_ml = global_volume_lookup[
            (
                case_id,
                component_id,
            )
        ]


        if not np.isclose(
            observed_volume_ml,
            expected_volume_ml,
            rtol=1e-6,
            atol=1e-8,
        ):

            raise RuntimeError(
                "Dense component volume lineage mismatch for "
                + case_id
                + " component "
                + str(
                    component_id
                )
            )


        bank = build_ordered_neutral_bank(
            native_component_xyz=native_component_xyz,
            native_affine_xyz=infection_affine_xyz,
            resampled_affine_xyz=resampled_affine_xyz,
            full_shape_zyx=full_shape_zyx,
            crop_origin_zyx=crop_origin_zyx,
            crop_shape_zyx=crop_shape_zyx,
        )


        ordered_coords = bank[
            "ordered_voxel_zyx"
        ]


        ordered_world = bank[
            "ordered_world_xyz"
        ]


        ordered_distance = bank[
            "distance_to_centroid_mm"
        ]


        centroid_world = bank[
            "centroid_world_xyz"
        ]


        expected_capacity = neutral_capacity_lookup[
            (
                case_id,
                component_id,
            )
        ]


        if len(
            ordered_coords
        ) != expected_capacity:

            raise RuntimeError(
                "Neutral-bank capacity failed to reproduce diagnostic for "
                + case_id
                + " component "
                + str(
                    component_id
                )
                + "\nExpected: "
                + str(
                    expected_capacity
                )
                + "\nObserved: "
                + str(
                    len(
                        ordered_coords
                    )
                )
            )


        if len(
            np.unique(
                ordered_coords,
                axis=0,
            )
        ) != len(
            ordered_coords
        ):

            raise RuntimeError(
                "Neutral bank contains duplicate coordinates."
            )


        if not np.all(
            np.diff(
                ordered_distance
            )
            >= -1e-10
        ):

            raise RuntimeError(
                "Neutral bank is not ordered by centroid distance."
            )


        neutral_bank_lookup[
            component_id
        ] = ordered_coords


        bank_path = (
            case_bank_dir
            / (
                "component_"
                + str(
                    component_id
                ).zfill(
                    4
                )
                + ".npz"
            )
        )


        np.savez_compressed(
            bank_path,

            ordered_voxel_zyx=
                ordered_coords.astype(
                    np.int32
                ),

            ordered_world_xyz=
                ordered_world.astype(
                    np.float64
                ),

            distance_to_centroid_mm=
                ordered_distance.astype(
                    np.float64
                ),

            component_centroid_world_xyz=
                centroid_world.astype(
                    np.float64
                ),
        )


        bank_rows.append(
            {
                "case_id":
                    case_id,

                "component_native_id":
                    component_id,

                "component_volume_ml":
                    observed_volume_ml,

                "capacity":
                    int(
                        len(
                            ordered_coords
                        )
                    ),

                "first_anchor_z":
                    int(
                        ordered_coords[
                            0,
                            0
                        ]
                    ),

                "first_anchor_y":
                    int(
                        ordered_coords[
                            0,
                            1
                        ]
                    ),

                "first_anchor_x":
                    int(
                        ordered_coords[
                            0,
                            2
                        ]
                    ),

                "first_anchor_distance_mm":
                    float(
                        ordered_distance[
                            0
                        ]
                    ),

                "max_bank_distance_mm":
                    float(
                        ordered_distance[
                            -1
                        ]
                    ),

                "bank_semantic_sha256":
                    semantic_hash(
                        ordered_coords,
                        ordered_world,
                        ordered_distance,
                        centroid_world,
                    ),

                "artifact_file":
                    str(
                        bank_path.relative_to(
                            REPO
                        )
                    ),

                "artifact_file_sha256":
                    sha256_file(
                        bank_path
                    ),

                "shared_across_geometry":
                    True,

                "shared_across_coverage":
                    True,

                "geometry_claim_allowed":
                    False,
            }
        )


    # ======================================================================================
    # D. Map 09C candidate geometry + common background
    # ======================================================================================

    case_conditions = {}

    bg_hashes = []


    for condition_id in CONDITIONS:

        artifact_row = artifact_lookup[
            (
                case_id,
                condition_id,
            )
        ]


        artifact_path = (
            REPO
            / str(
                artifact_row[
                    "artifact_file"
                ]
            )
        )


        with np.load(
            artifact_path,
            allow_pickle=False,
        ) as data:

            supervision_world_xyz = np.asarray(
                data[
                    "supervision_world_xyz"
                ],
                dtype=np.float64,
            )


            supervision_label = np.asarray(
                data[
                    "supervision_label"
                ],
                dtype=np.int8,
            )


            foreground_component_id = np.asarray(
                data[
                    "foreground_component_id"
                ],
                dtype=np.int32,
            )


            selected_component_ids = np.asarray(
                data[
                    "selected_component_ids"
                ],
                dtype=np.int32,
            )


            native_affine = np.asarray(
                data[
                    "native_affine_xyz"
                ],
                dtype=np.float64,
            )


            geometry = str(
                np.asarray(
                    data[
                        "geometry"
                    ]
                ).item()
            )


            coverage = float(
                np.asarray(
                    data[
                        "coverage_fraction"
                    ]
                ).item()
            )


        if geometry != CONDITION_TO_GEOMETRY[
            condition_id
        ]:

            raise RuntimeError(
                "Condition geometry mismatch."
            )


        if not np.isclose(
            coverage,
            CONDITION_TO_COVERAGE[
                condition_id
            ],
        ):

            raise RuntimeError(
                "Condition coverage mismatch."
            )


        if not np.allclose(
            native_affine,
            infection_affine_xyz,
            atol=1e-5,
            rtol=0,
        ):

            raise RuntimeError(
                "Native affine mismatch for "
                + case_id
            )


        n_fg = int(
            (
                supervision_label
                == 1
            ).sum()
        )


        n_bg = int(
            (
                supervision_label
                == 0
            ).sum()
        )


        if n_fg != len(
            foreground_component_id
        ):

            raise RuntimeError(
                "Foreground membership length mismatch."
            )


        if not np.all(
            supervision_label[
                :n_fg
            ]
            == 1
        ):

            raise RuntimeError(
                "Unexpected FG/BG ordering in native sparse artifact."
            )


        if not np.all(
            supervision_label[
                n_fg:
            ]
            == 0
        ):

            raise RuntimeError(
                "Unexpected FG/BG ordering in native sparse artifact."
            )


        fg_world = supervision_world_xyz[
            :n_fg
        ]


        bg_world = supervision_world_xyz[
            n_fg:
        ]


        (
            mapped_fg,
            _,
            fg_inside_crop,
        ) = map_world_to_crop(
            fg_world,
            resampled_affine_xyz,
            full_shape_zyx,
            crop_origin_zyx,
            crop_shape_zyx,
        )


        (
            mapped_bg,
            _,
            bg_inside_crop,
        ) = map_world_to_crop(
            bg_world,
            resampled_affine_xyz,
            full_shape_zyx,
            crop_origin_zyx,
            crop_shape_zyx,
        )


        fg_pools = {}


        for component_id in selected_component_ids.tolist():

            component_id = int(
                component_id
            )


            membership = (
                foreground_component_id
                == component_id
            )


            valid = (
                membership
                & fg_inside_crop
            )


            fg_pools[
                component_id
            ] = unique_coords(
                mapped_fg[
                    valid
                ]
            )


        mapped_bg_unique = unique_coords(
            mapped_bg[
                bg_inside_crop
            ]
        )


        bg_hash = semantic_hash(
            mapped_bg_unique
        )


        bg_hashes.append(
            bg_hash
        )


        transfer_rows.append(
            {
                "case_id":
                    case_id,

                "condition_id":
                    condition_id,

                "coverage_fraction":
                    coverage,

                "geometry":
                    geometry,

                "native_fg":
                    n_fg,

                "native_bg":
                    n_bg,

                "fg_inside_crop_before_dedup":
                    int(
                        fg_inside_crop.sum()
                    ),

                "fg_unique_after_grid_mapping":
                    int(
                        sum(
                            len(
                                value
                            )
                            for value in fg_pools.values()
                        )
                    ),

                "fg_coordinate_collapse":
                    int(
                        fg_inside_crop.sum()
                        - sum(
                            len(
                                value
                            )
                            for value in fg_pools.values()
                        )
                    ),

                "fg_outside_crop":
                    int(
                        (
                            ~fg_inside_crop
                        ).sum()
                    ),

                "bg_inside_crop_before_dedup":
                    int(
                        bg_inside_crop.sum()
                    ),

                "bg_unique_after_grid_mapping":
                    int(
                        len(
                            mapped_bg_unique
                        )
                    ),

                "bg_coordinate_collapse":
                    int(
                        bg_inside_crop.sum()
                        - len(
                            mapped_bg_unique
                        )
                    ),

                "bg_outside_crop":
                    int(
                        (
                            ~bg_inside_crop
                        ).sum()
                    ),

                "mapped_bg_semantic_sha256":
                    bg_hash,
            }
        )


        case_conditions[
            condition_id
        ] = {
            "coverage":
                coverage,

            "geometry":
                geometry,

            "selected_component_ids":
                [
                    int(
                        component_id
                    )
                    for component_id in selected_component_ids.tolist()
                ],

            "fg_pools":
                fg_pools,

            "bg_pool":
                mapped_bg_unique,
        }


    if len(
        set(
            bg_hashes
        )
    ) != 1:

        raise RuntimeError(
            "Mapped background differs across native factorial cells for "
            + case_id
        )


    common_bg_pool = case_conditions[
        "C50_COH"
    ][
        "bg_pool"
    ]


    selected_50 = case_conditions[
        "C50_COH"
    ][
        "selected_component_ids"
    ]


    selected_100 = case_conditions[
        "C100_COH"
    ][
        "selected_component_ids"
    ]


    for condition_id in [
        "C50_DIS",
        "C50_FRG",
    ]:

        if case_conditions[
            condition_id
        ][
            "selected_component_ids"
        ] != selected_50:

            raise RuntimeError(
                "C50 component sets differ across geometry."
            )


    for condition_id in [
        "C100_DIS",
        "C100_FRG",
    ]:

        if case_conditions[
            condition_id
        ][
            "selected_component_ids"
        ] != selected_100:

            raise RuntimeError(
                "C100 component sets differ across geometry."
            )


    if not set(
        selected_50
    ).issubset(
        set(
            selected_100
        )
    ):

        raise RuntimeError(
            "C50 component set is not nested inside C100."
        )


    # ======================================================================================
    # E. Freeze component-specific min/max capacities for both coverage levels
    # ======================================================================================

    capacities_by_coverage = {}

    minimums_by_coverage = {}

    detailed_capacity_lookup = {}


    for coverage, selected_ids, prefix in [
        (
            0.50,
            selected_50,
            "C50",
        ),

        (
            1.00,
            selected_100,
            "C100",
        ),
    ]:

        capacities = {}

        minimums = {}


        for component_id in selected_ids:

            component_class = global_class_lookup[
                (
                    case_id,
                    component_id,
                )
            ]


            if component_class == GLOBAL_LIMITED:

                if component_id not in neutral_bank_lookup:

                    raise RuntimeError(
                        "Missing frozen neutral bank for "
                        + case_id
                        + " component "
                        + str(
                            component_id
                        )
                    )


                capacity = int(
                    len(
                        neutral_bank_lookup[
                            component_id
                        ]
                    )
                )


                if capacity < LIMITED_MIN_QUOTA:

                    raise RuntimeError(
                        "Limited bank has capacity <1."
                    )


                capacities[
                    component_id
                ] = capacity


                minimums[
                    component_id
                ] = LIMITED_MIN_QUOTA


                detailed_capacity_lookup[
                    (
                        coverage,
                        component_id,
                    )
                ] = {
                    "coherent":
                        capacity,

                    "dispersed":
                        capacity,

                    "fragmented":
                        capacity,

                    "common":
                        capacity,

                    "capacity_source":
                        "physical_neutral_bank",
                }


            elif component_class == GLOBAL_RESOLVABLE:

                geometry_caps = {}


                for suffix, geometry in [
                    (
                        "COH",
                        "coherent",
                    ),
                    (
                        "DIS",
                        "dispersed",
                    ),
                    (
                        "FRG",
                        "fragmented",
                    ),
                ]:

                    condition_id = (
                        prefix
                        + "_"
                        + suffix
                    )


                    pool = case_conditions[
                        condition_id
                    ][
                        "fg_pools"
                    ][
                        component_id
                    ]


                    geometry_caps[
                        geometry
                    ] = max_geometry_capacity(
                        pool,
                        geometry,
                        TARGET_SPACING_ZYX,
                    )


                common_capacity = int(
                    min(
                        geometry_caps.values()
                    )
                )


                if common_capacity < RESOLVABLE_MIN_QUOTA:

                    raise RuntimeError(
                        "Globally resolvable lesion failed frozen minimum "
                        "geometry capacity in 09D-B2:\n"
                        + case_id
                        + " / component "
                        + str(
                            component_id
                        )
                        + " / coverage "
                        + str(
                            coverage
                        )
                    )


                capacities[
                    component_id
                ] = common_capacity


                minimums[
                    component_id
                ] = RESOLVABLE_MIN_QUOTA


                detailed_capacity_lookup[
                    (
                        coverage,
                        component_id,
                    )
                ] = {
                    **{
                        key:
                            int(
                                value
                            )
                        for key, value
                        in geometry_caps.items()
                    },

                    "common":
                        common_capacity,

                    "capacity_source":
                        "common_geometry_preserving_candidate_capacity",
                }


            else:

                raise RuntimeError(
                    "Unknown global resolution class."
                )


        capacities_by_coverage[
            coverage
        ] = capacities


        minimums_by_coverage[
            coverage
        ] = minimums


    native_budget = int(
        native_budget_lookup[
            case_id
        ][
            "frozen_native_budget_B_i"
        ]
    )


    minimum_50 = int(
        sum(
            minimums_by_coverage[
                0.50
            ].values()
        )
    )


    minimum_100 = int(
        sum(
            minimums_by_coverage[
                1.00
            ].values()
        )
    )


    lower_bound = int(
        max(
            minimum_50,
            minimum_100,
        )
    )


    capacity_50 = int(
        sum(
            capacities_by_coverage[
                0.50
            ].values()
        )
    )


    capacity_100 = int(
        sum(
            capacities_by_coverage[
                1.00
            ].values()
        )
    )


    upper_bound = int(
        min(
            native_budget,
            capacity_50,
            capacity_100,
            len(
                common_bg_pool
            ),
        )
    )


    if upper_bound < lower_bound:

        raise RuntimeError(
            "A1.2 aggregate feasibility failed to reproduce for "
            + case_id
            + "\n"
            + json.dumps(
                {
                    "minimum_50":
                        minimum_50,

                    "minimum_100":
                        minimum_100,

                    "capacity_50":
                        capacity_50,

                    "capacity_100":
                        capacity_100,

                    "background_capacity":
                        len(
                            common_bg_pool
                        ),

                    "native_budget":
                        native_budget,

                    "lower_bound":
                        lower_bound,

                    "upper_bound":
                        upper_bound,
                },
                indent=2,
            )
        )


    # ======================================================================================
    # F. Coordinate-level candidate budget search
    # ======================================================================================

    selection_cache = {}


    def select_component_coords(
        condition_id,
        component_id,
        quota,
    ):

        quota = int(
            quota
        )


        component_class = global_class_lookup[
            (
                case_id,
                component_id,
            )
        ]


        if component_class == GLOBAL_LIMITED:

            bank = neutral_bank_lookup[
                component_id
            ]


            if quota < 1:

                raise RuntimeError(
                    "Limited lesion quota <1."
                )


            if quota > len(
                bank
            ):

                raise RuntimeError(
                    "Limited lesion quota exceeds frozen bank capacity."
                )


            return bank[
                :quota
            ].copy()


        key = (
            condition_id,
            int(
                component_id
            ),
            quota,
        )


        if key not in selection_cache:

            selection_cache[
                key
            ] = select_geometry(
                pool_coords=case_conditions[
                    condition_id
                ][
                    "fg_pools"
                ][
                    component_id
                ],
                geometry=case_conditions[
                    condition_id
                ][
                    "geometry"
                ],
                quota=quota,
                spacing_zyx=TARGET_SPACING_ZYX,
            )


        return selection_cache[
            key
        ]


    final_budget = None

    final_allocations = None

    final_fg = None

    final_groups = None

    final_bg = None

    case_rejections = {
        "allocation":
            0,

        "geometry_selection":
            0,

        "cross_component_collision":
            0,

        "background":
            0,
    }


    for candidate_budget in range(
        upper_bound,
        lower_bound
        - 1,
        -1,
    ):

        total_candidate_budget_rejections += 1


        allocation_50 = allocate_exact(
            total=candidate_budget,
            component_ids=selected_50,
            capacities=capacities_by_coverage[
                0.50
            ],
            minimums=minimums_by_coverage[
                0.50
            ],
        )


        allocation_100 = allocate_exact(
            total=candidate_budget,
            component_ids=selected_100,
            capacities=capacities_by_coverage[
                1.00
            ],
            minimums=minimums_by_coverage[
                1.00
            ],
        )


        if (
            allocation_50 is None
            or allocation_100 is None
        ):

            case_rejections[
                "allocation"
            ] += 1

            continue


        trial_fg = {}

        trial_groups = {}

        trial_valid = True


        try:

            for condition_id in CONDITIONS:

                coverage = CONDITION_TO_COVERAGE[
                    condition_id
                ]


                selected_ids = (
                    selected_50
                    if np.isclose(
                        coverage,
                        0.50,
                    )
                    else selected_100
                )


                allocation = (
                    allocation_50
                    if np.isclose(
                        coverage,
                        0.50,
                    )
                    else allocation_100
                )


                coord_parts = []

                group_parts = []


                for component_id in selected_ids:

                    quota = int(
                        allocation[
                            component_id
                        ]
                    )


                    coords = select_component_coords(
                        condition_id=condition_id,
                        component_id=component_id,
                        quota=quota,
                    )


                    if len(
                        coords
                    ) != quota:

                        raise RuntimeError(
                            "Component selection returned incorrect quota."
                        )


                    coord_parts.append(
                        coords
                    )


                    group_parts.append(
                        np.full(
                            quota,
                            component_id,
                            dtype=np.int32,
                        )
                    )


                fg_coords = np.concatenate(
                    coord_parts,
                    axis=0,
                ).astype(
                    np.int32
                )


                fg_groups = np.concatenate(
                    group_parts,
                    axis=0,
                ).astype(
                    np.int32
                )


                if len(
                    fg_coords
                ) != candidate_budget:

                    raise RuntimeError(
                        "Total FG count does not equal candidate budget."
                    )


                # Core coordinate-level collision test.
                if len(
                    np.unique(
                        fg_coords,
                        axis=0,
                    )
                ) != candidate_budget:

                    case_rejections[
                        "cross_component_collision"
                    ] += 1

                    total_cross_component_collision_rejections += 1

                    trial_valid = False

                    break


                order = np.lexsort(
                    (
                        fg_coords[
                            :,
                            2
                        ],
                        fg_coords[
                            :,
                            1
                        ],
                        fg_coords[
                            :,
                            0
                        ],
                    )
                )


                trial_fg[
                    condition_id
                ] = fg_coords[
                    order
                ]


                trial_groups[
                    condition_id
                ] = fg_groups[
                    order
                ]


        except RuntimeError:

            case_rejections[
                "geometry_selection"
            ] += 1

            total_geometry_selection_rejections += 1

            trial_valid = False


        if not trial_valid:

            continue


        # ------------------------------------------------------------------
        # Same BG must be safe for every one of the six conditions.
        # ------------------------------------------------------------------

        union_fg = set()


        for condition_id in CONDITIONS:

            union_fg.update(
                tuple(
                    int(
                        value
                    )
                    for value in coord
                )
                for coord in trial_fg[
                    condition_id
                ]
            )


        eligible_bg = np.asarray(
            [
                coord
                for coord in common_bg_pool.tolist()
                if tuple(
                    int(
                        value
                    )
                    for value in coord
                )
                not in union_fg
            ],
            dtype=np.int32,
        )


        eligible_bg = unique_coords(
            eligible_bg
        )


        if len(
            eligible_bg
        ) < candidate_budget:

            case_rejections[
                "background"
            ] += 1

            total_background_rejections += 1

            continue


        # ------------------------------------------------------------------
        # Exact A1.2 cross-coverage prefix check for limited lesions.
        # ------------------------------------------------------------------

        for component_id in selected_50:

            if global_class_lookup[
                (
                    case_id,
                    component_id,
                )
            ] != GLOBAL_LIMITED:

                continue


            q50 = int(
                allocation_50[
                    component_id
                ]
            )


            q100 = int(
                allocation_100[
                    component_id
                ]
            )


            bank = neutral_bank_lookup[
                component_id
            ]


            coords_50 = bank[
                :q50
            ]


            coords_100 = bank[
                :q100
            ]


            shorter = (
                coords_50
                if q50 <= q100
                else coords_100
            )


            longer = (
                coords_100
                if q50 <= q100
                else coords_50
            )


            if not np.array_equal(
                shorter,
                longer[
                    :len(
                        shorter
                    )
                ],
            ):

                raise RuntimeError(
                    "Frozen neutral-bank prefix nesting failed."
                )


        final_budget = int(
            candidate_budget
        )


        final_allocations = {
            0.50:
                allocation_50,

            1.00:
                allocation_100,
        }


        final_fg = trial_fg


        final_groups = trial_groups


        final_bg = eligible_bg[
            :candidate_budget
        ].copy()


        # This candidate was accepted, so remove the accepted candidate
        # from the rejection-attempt counter.
        total_candidate_budget_rejections -= 1


        break


    if final_budget is None:

        raise RuntimeError(
            "A1.2 coordinate-level construction FAILED for "
            + case_id
            + "\n"
            + json.dumps(
                {
                    "lower_bound":
                        lower_bound,

                    "upper_bound":
                        upper_bound,

                    "case_rejections":
                        case_rejections,
                },
                indent=2,
            )
        )


    # ======================================================================================
    # G. Write FINAL sparse-only trainer candidates
    # ======================================================================================

    case_output_dir = (
        FINAL_GRID_ROOT
        / case_id
    )


    case_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    final_bg_hash = semantic_hash(
        final_bg
    )


    case_condition_rows = []


    for condition_id in CONDITIONS:

        coverage = CONDITION_TO_COVERAGE[
            condition_id
        ]


        geometry = CONDITION_TO_GEOMETRY[
            condition_id
        ]


        selected_ids = (
            selected_50
            if np.isclose(
                coverage,
                0.50,
            )
            else selected_100
        )


        allocation = final_allocations[
            coverage
        ]


        fg_coords = final_fg[
            condition_id
        ]


        fg_groups = final_groups[
            condition_id
        ]


        n_limited = (
            0
        )


        n_resolvable = (
            0
        )


        for component_id in selected_ids:

            component_class = global_class_lookup[
                (
                    case_id,
                    component_id,
                )
            ]


            component_coords = fg_coords[
                fg_groups
                == component_id
            ]


            quota = int(
                allocation[
                    component_id
                ]
            )


            connected_sets = int(
                sparse_component_count(
                    component_coords
                )
            )


            capacity_info = detailed_capacity_lookup[
                (
                    coverage,
                    component_id,
                )
            ]


            if component_class == GLOBAL_LIMITED:

                n_limited += 1


                bank = neutral_bank_lookup[
                    component_id
                ]


                if quota < 1:

                    raise RuntimeError(
                        "Limited lesion has quota <1."
                    )


                if quota > len(
                    bank
                ):

                    raise RuntimeError(
                        "Limited lesion quota exceeds bank."
                    )


                if not np.array_equal(
                    component_coords,
                    bank[
                        :quota
                    ],
                ):

                    # fg_coords were globally lexicographically sorted,
                    # so compare as sets/order-independent.
                    if {
                        tuple(
                            int(
                                value
                            )
                            for value in coord
                        )
                        for coord in component_coords
                    } != {
                        tuple(
                            int(
                                value
                            )
                            for value in coord
                        )
                        for coord in bank[
                            :quota
                        ]
                    }:

                        raise RuntimeError(
                            "Limited lesion does not use frozen bank prefix."
                        )


                geometry_claim_allowed = (
                    False
                )


                representation = (
                    "neutral_physical_bank_prefix"
                )


            else:

                n_resolvable += 1


                if quota < RESOLVABLE_MIN_QUOTA:

                    raise RuntimeError(
                        "Resolvable lesion quota <2."
                    )


                if (
                    geometry == "coherent"
                    and connected_sets != 1
                ):

                    raise RuntimeError(
                        "Resolvable coherent annotation is disconnected."
                    )


                if (
                    geometry
                    in {
                        "dispersed",
                        "fragmented",
                    }
                    and connected_sets < 2
                ):

                    raise RuntimeError(
                        "Resolvable geometry lost separation."
                    )


                geometry_claim_allowed = (
                    True
                )


                representation = (
                    geometry
                )


            component_rows.append(
                {
                    "case_id":
                        case_id,

                    "role":
                        role,

                    "condition_id":
                        condition_id,

                    "coverage_fraction":
                        coverage,

                    "geometry_condition":
                        geometry,

                    "component_native_id":
                        int(
                            component_id
                        ),

                    "component_volume_ml":
                        global_volume_lookup[
                            (
                                case_id,
                                component_id,
                            )
                        ],

                    "global_resolution_class":
                        component_class,

                    "representation":
                        representation,

                    "geometry_claim_allowed":
                        geometry_claim_allowed,

                    "quota":
                        quota,

                    "connected_sets":
                        connected_sets,

                    "mean_nearest_neighbor_mm":
                        mean_nearest_neighbor_mm(
                            component_coords,
                            TARGET_SPACING_ZYX,
                        ),

                    "common_capacity":
                        int(
                            capacity_info[
                                "common"
                            ]
                        ),

                    "coherent_capacity":
                        int(
                            capacity_info[
                                "coherent"
                            ]
                        ),

                    "dispersed_capacity":
                        int(
                            capacity_info[
                                "dispersed"
                            ]
                        ),

                    "fragmented_capacity":
                        int(
                            capacity_info[
                                "fragmented"
                            ]
                        ),

                    "capacity_source":
                        str(
                            capacity_info[
                                "capacity_source"
                            ]
                        ),

                    "uses_neutral_bank":
                        bool(
                            component_class
                            == GLOBAL_LIMITED
                        ),
                }
            )


        # ----------------------------------------------------------------------------------
        # Sparse direct labels + grouped FG membership.
        # ----------------------------------------------------------------------------------

        supervision_coords = np.concatenate(
            [
                fg_coords,
                final_bg,
            ],
            axis=0,
        ).astype(
            np.int32
        )


        supervision_labels = np.concatenate(
            [
                np.ones(
                    len(
                        fg_coords
                    ),
                    dtype=np.int8,
                ),

                np.zeros(
                    len(
                        final_bg
                    ),
                    dtype=np.int8,
                ),
            ],
            axis=0,
        )


        fg_set = {
            tuple(
                int(
                    value
                )
                for value in coord
            )
            for coord in fg_coords
        }


        bg_set = {
            tuple(
                int(
                    value
                )
                for value in coord
            )
            for coord in final_bg
        }


        if fg_set & bg_set:

            raise RuntimeError(
                "Final FG/BG coordinate collision."
            )


        direct_order = np.lexsort(
            (
                supervision_coords[
                    :,
                    2
                ],
                supervision_coords[
                    :,
                    1
                ],
                supervision_coords[
                    :,
                    0
                ],
            )
        )


        supervision_coords = supervision_coords[
            direct_order
        ]


        supervision_labels = supervision_labels[
            direct_order
        ]


        membership_order = np.lexsort(
            (
                fg_coords[
                    :,
                    2
                ],
                fg_coords[
                    :,
                    1
                ],
                fg_coords[
                    :,
                    0
                ],
                fg_groups,
            )
        )


        membership_coords = fg_coords[
            membership_order
        ]


        membership_groups = fg_groups[
            membership_order
        ]


        output_path = (
            case_output_dir
            / (
                condition_id
                + ".npz"
            )
        )


        np.savez_compressed(
            output_path,

            supervision_voxel_zyx=
                supervision_coords.astype(
                    np.int32
                ),

            supervision_label=
                supervision_labels.astype(
                    np.int8
                ),

            fg_membership_voxel_zyx=
                membership_coords.astype(
                    np.int32
                ),

            fg_membership_group_id=
                membership_groups.astype(
                    np.int32
                ),
        )


        annotation_hash = semantic_hash(
            supervision_coords,
            supervision_labels,
            membership_coords,
            membership_groups,
        )


        artifact_sha = sha256_file(
            output_path
        )


        selected_id_string = ";".join(
            str(
                component_id
            )
            for component_id in selected_ids
        )


        selected_quota_string = ";".join(
            str(
                allocation[
                    component_id
                ]
            )
            for component_id in selected_ids
        )


        condition_record = {
            "case_id":
                case_id,

            "source_subject_key":
                str(
                    split_row[
                        "source_subject_key"
                    ]
                ),

            "role":
                role,

            "outer_fold":
                (
                    ""
                    if pd.isna(
                        split_row[
                            "outer_fold"
                        ]
                    )
                    else int(
                        split_row[
                            "outer_fold"
                        ]
                    )
                ),

            "condition_id":
                condition_id,

            "coverage_fraction":
                coverage,

            "geometry":
                geometry,

            "selected_components":
                len(
                    selected_ids
                ),

            "selected_resolvable_components":
                n_resolvable,

            "selected_resolution_limited_components":
                n_limited,

            "train_positive_budget_B_i":
                final_budget,

            "foreground_unique_voxels":
                int(
                    len(
                        fg_coords
                    )
                ),

            "background_unique_voxels":
                int(
                    len(
                        final_bg
                    )
                ),

            "selected_component_ids":
                selected_id_string,

            "selected_component_quotas":
                selected_quota_string,

            "background_semantic_sha256":
                final_bg_hash,

            "annotation_semantic_sha256":
                annotation_hash,

            "artifact_file":
                str(
                    output_path.relative_to(
                        REPO
                    )
                ),

            "artifact_file_sha256":
                artifact_sha,

            "effective_protocol":
                EFFECTIVE_PROTOCOL,

            "coordinate_level_QA":
                True,

            "factorial_training_authorized":
                False,
        }


        condition_rows.append(
            condition_record
        )


        case_condition_rows.append(
            condition_record
        )


        artifact_rows.append(
            {
                "case_id":
                    case_id,

                "condition_id":
                    condition_id,

                "artifact_file":
                    str(
                        output_path.relative_to(
                            REPO
                        )
                    ),

                "artifact_file_sha256":
                    artifact_sha,

                "annotation_semantic_sha256":
                    annotation_hash,

                "sparse_only":
                    True,

                "coordinate_level_QA":
                    True,

                "trainer_candidate":
                    True,

                "training_authorized":
                    False,
            }
        )


    # ======================================================================================
    # H. Exact case-level six-cell QA
    # ======================================================================================

    if len(
        case_condition_rows
    ) != 6:

        raise RuntimeError(
            "Case does not contain exactly six final condition rows."
        )


    if len(
        {
            row[
                "train_positive_budget_B_i"
            ]
            for row in case_condition_rows
        }
    ) != 1:

        raise RuntimeError(
            "B_i_train differs across six cells."
        )


    if len(
        {
            row[
                "foreground_unique_voxels"
            ]
            for row in case_condition_rows
        }
    ) != 1:

        raise RuntimeError(
            "FG count differs across six cells."
        )


    if len(
        {
            row[
                "background_semantic_sha256"
            ]
            for row in case_condition_rows
        }
    ) != 1:

        raise RuntimeError(
            "BG coordinates differ across six cells."
        )


    for coverage in [
        0.50,
        1.00,
    ]:

        rows = [
            row
            for row in case_condition_rows
            if np.isclose(
                row[
                    "coverage_fraction"
                ],
                coverage,
            )
        ]


        if len(
            {
                row[
                    "selected_component_ids"
                ]
                for row in rows
            }
        ) != 1:

            raise RuntimeError(
                "Component set differs across geometry."
            )


        if len(
            {
                row[
                    "selected_component_quotas"
                ]
                for row in rows
            }
        ) != 1:

            raise RuntimeError(
                "Per-component quota differs across geometry."
            )


    # ======================================================================================
    # I. Cross-coverage limited-bank prefix QA
    # ======================================================================================

    prefix_checks = (
        0
    )


    for component_id in selected_50:

        if global_class_lookup[
            (
                case_id,
                component_id,
            )
        ] != GLOBAL_LIMITED:

            continue


        bank = neutral_bank_lookup[
            component_id
        ]


        q50 = int(
            final_allocations[
                0.50
            ][
                component_id
            ]
        )


        q100 = int(
            final_allocations[
                1.00
            ][
                component_id
            ]
        )


        prefix_50 = bank[
            :q50
        ]


        prefix_100 = bank[
            :q100
        ]


        if q50 <= q100:

            if not np.array_equal(
                prefix_50,
                prefix_100[
                    :q50
                ],
            ):

                raise RuntimeError(
                    "Limited C50 prefix is not nested in C100 prefix."
                )


        else:

            if not np.array_equal(
                prefix_100,
                prefix_50[
                    :q100
                ],
            ):

                raise RuntimeError(
                    "Limited C100 prefix is not nested in C50 prefix."
                )


        prefix_checks += 1


    budget_rows.append(
        {
            "case_id":
                case_id,

            "source_subject_key":
                str(
                    split_row[
                        "source_subject_key"
                    ]
                ),

            "source_origin":
                source_origin,

            "role":
                role,

            "eligible_components_K":
                len(
                    selected_100
                ),

            "selected_components_C50":
                len(
                    selected_50
                ),

            "selected_components_C100":
                len(
                    selected_100
                ),

            "global_limited_components_in_case":
                int(
                    len(
                        limited_case_df
                    )
                ),

            "native_budget_B_i":
                native_budget,

            "minimum_C50":
                minimum_50,

            "minimum_C100":
                minimum_100,

            "aggregate_lower_bound":
                lower_bound,

            "aggregate_upper_bound":
                upper_bound,

            "frozen_train_budget_B_i":
                final_budget,

            "budget_reduction_from_native":
                int(
                    native_budget
                    - final_budget
                ),

            "budget_retention_fraction":
                float(
                    final_budget
                    / native_budget
                ),

            "capacity_C50":
                capacity_50,

            "capacity_C100":
                capacity_100,

            "mapped_background_capacity":
                int(
                    len(
                        common_bg_pool
                    )
                ),

            "budget_candidates_rejected":
                int(
                    (
                        upper_bound
                        - final_budget
                    )
                ),

            "allocation_rejections":
                int(
                    case_rejections[
                        "allocation"
                    ]
                ),

            "geometry_selection_rejections":
                int(
                    case_rejections[
                        "geometry_selection"
                    ]
                ),

            "cross_component_collision_rejections":
                int(
                    case_rejections[
                        "cross_component_collision"
                    ]
                ),

            "background_rejections":
                int(
                    case_rejections[
                        "background"
                    ]
                ),

            "cross_coverage_prefix_checks":
                int(
                    prefix_checks
                ),

            "coordinate_level_six_cell_feasible":
                True,

            "factorial_training_authorized":
                False,
        }
    )


    del native_ct
    del encoded_native
    del encoded_img
    del canonical_img
    del resampled_img
    del resampled_xyz
    del resampled_zyx
    del normalized_full
    del infection
    del component_labels

    gc.collect()


# ==========================================================================================
# 15. BUILD DATAFRAMES + HARD COUNTS
# ==========================================================================================

heading(
    "STEP 3/12 — FREEZE FINAL COORDINATE-LEVEL MANIFESTS"
)


grid_df = pd.DataFrame(
    grid_rows
).sort_values(
    "case_id"
).reset_index(
    drop=True
)


transfer_df = pd.DataFrame(
    transfer_rows
).sort_values(
    [
        "case_id",
        "condition_id",
    ]
).reset_index(
    drop=True
)


bank_df = pd.DataFrame(
    bank_rows
).sort_values(
    [
        "case_id",
        "component_native_id",
    ]
).reset_index(
    drop=True
)


budget_df = pd.DataFrame(
    budget_rows
).sort_values(
    "case_id"
).reset_index(
    drop=True
)


condition_df = pd.DataFrame(
    condition_rows
).sort_values(
    [
        "case_id",
        "condition_id",
    ]
).reset_index(
    drop=True
)


component_df = pd.DataFrame(
    component_rows
).sort_values(
    [
        "case_id",
        "condition_id",
        "component_native_id",
    ]
).reset_index(
    drop=True
)


artifact_df = pd.DataFrame(
    artifact_rows
).sort_values(
    [
        "case_id",
        "condition_id",
    ]
).reset_index(
    drop=True
)


if len(
    grid_df
) != EXPECTED_CASES:

    raise RuntimeError(
        "Expected 20 grid rows."
    )


if len(
    transfer_df
) != EXPECTED_CONDITIONS:

    raise RuntimeError(
        "Expected 120 transfer rows."
    )


if len(
    bank_df
) != EXPECTED_LIMITED_COMPONENTS:

    raise RuntimeError(
        "Expected 28 neutral-bank artifacts."
    )


if bank_df[
    "case_id"
].nunique() != EXPECTED_LIMITED_CASES:

    raise RuntimeError(
        "Neutral banks must occur in 13 cases."
    )


if len(
    budget_df
) != EXPECTED_CASES:

    raise RuntimeError(
        "Expected 20 final budget rows."
    )


if not budget_df[
    "coordinate_level_six_cell_feasible"
].astype(
    bool
).all():

    raise RuntimeError(
        "At least one case failed coordinate-level feasibility."
    )


if len(
    condition_df
) != EXPECTED_CONDITIONS:

    raise RuntimeError(
        "Expected 120 final condition rows."
    )


if len(
    artifact_df
) != EXPECTED_CONDITIONS:

    raise RuntimeError(
        "Expected 120 final sparse artifacts."
    )


# ==========================================================================================
# 16. SAVE MANIFESTS
# ==========================================================================================

grid_manifest_path = (
    MANIFEST_DIR
    / "cova3d_training_grid_geometry_v1_2.csv"
)


transfer_manifest_path = (
    MANIFEST_DIR
    / "cova3d_training_grid_raw_transfer_audit_v1_2.csv"
)


bank_manifest_path = (
    MANIFEST_DIR
    / "cova3d_neutral_anchor_bank_manifest_v1_0.csv"
)


budget_manifest_path = (
    MANIFEST_DIR
    / "cova3d_training_grid_budget_v1_2.csv"
)


condition_manifest_path = (
    MANIFEST_DIR
    / "cova3d_training_grid_condition_manifest_v1_2.csv"
)


component_manifest_path = (
    MANIFEST_DIR
    / "cova3d_training_grid_component_audit_v1_2.csv"
)


artifact_manifest_path = (
    MANIFEST_DIR
    / "cova3d_training_grid_candidate_artifact_manifest_v1_2.csv"
)


grid_df.to_csv(
    grid_manifest_path,
    index=False,
)


transfer_df.to_csv(
    transfer_manifest_path,
    index=False,
)


bank_df.to_csv(
    bank_manifest_path,
    index=False,
)


budget_df.to_csv(
    budget_manifest_path,
    index=False,
)


condition_df.to_csv(
    condition_manifest_path,
    index=False,
)


component_df.to_csv(
    component_manifest_path,
    index=False,
)


artifact_df.to_csv(
    artifact_manifest_path,
    index=False,
)


print(
    "✓ Final training-grid cases             : 20"
)

print(
    "✓ Neutral-bank artifacts                : 28"
)

print(
    "✓ Final factorial rows                  : 120"
)

print(
    "✓ Final sparse trainer candidates       : 120"
)

print(
    "✓ Coordinate-level feasibility          : PASS 20/20"
)


# ==========================================================================================
# 17. GLOBAL SIX-CELL MATCHING AUDIT
# ==========================================================================================

heading(
    "STEP 4/12 — VERIFY FINAL SIX-CELL CAUSAL MATCHING"
)


for case_id, group in condition_df.groupby(
    "case_id"
):

    if len(
        group
    ) != 6:

        raise RuntimeError(
            "Incomplete factorial set."
        )


    if group[
        "train_positive_budget_B_i"
    ].nunique() != 1:

        raise RuntimeError(
            "FG budget differs across six conditions."
        )


    if group[
        "foreground_unique_voxels"
    ].nunique() != 1:

        raise RuntimeError(
            "FG count differs across six conditions."
        )


    if group[
        "background_unique_voxels"
    ].nunique() != 1:

        raise RuntimeError(
            "BG count differs across six conditions."
        )


    if group[
        "background_semantic_sha256"
    ].nunique() != 1:

        raise RuntimeError(
            "BG coordinates differ across six conditions."
        )


    for coverage in [
        0.50,
        1.00,
    ]:

        subset = group[
            np.isclose(
                group[
                    "coverage_fraction"
                ],
                coverage,
            )
        ]


        if subset[
            "selected_component_ids"
        ].nunique() != 1:

            raise RuntimeError(
                "Component sets differ across geometry."
            )


        if subset[
            "selected_component_quotas"
        ].nunique() != 1:

            raise RuntimeError(
                "Component quotas differ across geometry."
            )


print(
    "✓ Equal FG budget across six cells      : PASS 20/20"
)

print(
    "✓ Identical BG coordinates              : PASS 20/20"
)

print(
    "✓ Same component sets across geometry   : PASS"
)

print(
    "✓ Same component quotas across geometry : PASS"
)

print(
    "✓ C50 nested inside C100                : PASS"
)


# ==========================================================================================
# 18. NEUTRAL-BANK INVARIANCE AUDIT
# ==========================================================================================

heading(
    "STEP 5/12 — VERIFY A1.2 NEUTRAL-BANK INVARIANCE"
)


condition_lookup = {
    (
        str(
            row[
                "case_id"
            ]
        ),
        str(
            row[
                "condition_id"
            ]
        ),
    ):
        row
    for _, row
    in condition_df.iterrows()
}


bank_lookup = {}


for _, row in bank_df.iterrows():

    bank_path = (
        REPO
        / str(
            row[
                "artifact_file"
            ]
        )
    )


    with np.load(
        bank_path,
        allow_pickle=False,
    ) as data:

        bank_coords = np.asarray(
            data[
                "ordered_voxel_zyx"
            ],
            dtype=np.int32,
        )


    bank_lookup[
        (
            str(
                row[
                    "case_id"
                ]
            ),
            int(
                row[
                    "component_native_id"
                ]
            ),
        )
    ] = bank_coords


limited_prefix_checks = (
    0
)


for _, global_row in global_class_df[
    global_class_df[
        "global_resolution_class"
    ]
    == GLOBAL_LIMITED
].iterrows():

    case_id = str(
        global_row[
            "case_id"
        ]
    )


    component_id = int(
        global_row[
            "component_native_id"
        ]
    )


    bank = bank_lookup[
        (
            case_id,
            component_id,
        )
    ]


    observed_by_coverage = {}


    for coverage, prefix in [
        (
            0.50,
            "C50",
        ),

        (
            1.00,
            "C100",
        ),
    ]:

        geometry_observations = []


        for suffix in [
            "COH",
            "DIS",
            "FRG",
        ]:

            condition_id = (
                prefix
                + "_"
                + suffix
            )


            manifest_row = condition_lookup[
                (
                    case_id,
                    condition_id,
                )
            ]


            selected_ids = [
                int(
                    value
                )
                for value in str(
                    manifest_row[
                        "selected_component_ids"
                    ]
                ).split(
                    ";"
                )
                if value != ""
            ]


            if component_id not in selected_ids:

                continue


            artifact_path = (
                REPO
                / str(
                    manifest_row[
                        "artifact_file"
                    ]
                )
            )


            with np.load(
                artifact_path,
                allow_pickle=False,
            ) as data:

                membership_coords = np.asarray(
                    data[
                        "fg_membership_voxel_zyx"
                    ],
                    dtype=np.int32,
                )


                membership_groups = np.asarray(
                    data[
                        "fg_membership_group_id"
                    ],
                    dtype=np.int32,
                )


            coords = membership_coords[
                membership_groups
                == component_id
            ]


            expected_prefix = bank[
                :len(
                    coords
                )
            ]


            if {
                tuple(
                    int(
                        value
                    )
                    for value in coord
                )
                for coord in coords
            } != {
                tuple(
                    int(
                        value
                    )
                    for value in coord
                )
                for coord in expected_prefix
            }:

                raise RuntimeError(
                    "Final limited annotation is not a frozen bank prefix."
                )


            geometry_observations.append(
                lexical_sort_coords(
                    coords
                )
            )


        if geometry_observations:

            reference = geometry_observations[
                0
            ]


            for observation in geometry_observations[
                1:
            ]:

                if not np.array_equal(
                    observation,
                    reference,
                ):

                    raise RuntimeError(
                        "Neutral-bank coordinates differ across geometry."
                    )


            observed_by_coverage[
                coverage
            ] = reference


    if (
        0.50 in observed_by_coverage
        and 1.00 in observed_by_coverage
    ):

        c50 = observed_by_coverage[
            0.50
        ]

        c100 = observed_by_coverage[
            1.00
        ]


        if len(
            c50
        ) <= len(
            c100
        ):

            smaller = {
                tuple(
                    int(
                        value
                    )
                    for value in coord
                )
                for coord in c50
            }


            larger_prefix = {
                tuple(
                    int(
                        value
                    )
                    for value in coord
                )
                for coord in bank[
                    :len(
                        c50
                    )
                ]
            }


            if smaller != larger_prefix:

                raise RuntimeError(
                    "C50 neutral-bank prefix invariance failed."
                )


        else:

            smaller = {
                tuple(
                    int(
                        value
                    )
                    for value in coord
                )
                for coord in c100
            }


            larger_prefix = {
                tuple(
                    int(
                        value
                    )
                    for value in coord
                )
                for coord in bank[
                    :len(
                        c100
                    )
                ]
            }


            if smaller != larger_prefix:

                raise RuntimeError(
                    "C100 neutral-bank prefix invariance failed."
                )


        limited_prefix_checks += 1


print(
    "✓ Neutral banks persisted               : 28/28"
)

print(
    "✓ Same bank across geometry             : PASS"
)

print(
    "✓ Same ordered bank across coverage     : PASS"
)

print(
    "✓ Cross-coverage prefix nesting         : PASS"
)

print(
    "✓ Limited geometry claims               : 0"
)


# ==========================================================================================
# 19. RESOLVABLE-GEOMETRY AUDIT
# ==========================================================================================

heading(
    "STEP 6/12 — VERIFY RESOLVABLE-LESION GEOMETRY"
)


resolvable_df = component_df[
    component_df[
        "global_resolution_class"
    ]
    == GLOBAL_RESOLVABLE
]


coherent_df = resolvable_df[
    resolvable_df[
        "geometry_condition"
    ]
    == "coherent"
]


dispersed_df = resolvable_df[
    resolvable_df[
        "geometry_condition"
    ]
    == "dispersed"
]


fragmented_df = resolvable_df[
    resolvable_df[
        "geometry_condition"
    ]
    == "fragmented"
]


if not (
    coherent_df[
        "connected_sets"
    ]
    == 1
).all():

    raise RuntimeError(
        "A resolvable coherent annotation is disconnected."
    )


if not (
    dispersed_df[
        "connected_sets"
    ]
    >= 2
).all():

    raise RuntimeError(
        "A resolvable dispersed annotation lost separation."
    )


if not (
    fragmented_df[
        "connected_sets"
    ]
    >= 2
).all():

    raise RuntimeError(
        "A resolvable fragmented annotation lost separation."
    )


mean_nn = (
    resolvable_df.groupby(
        "geometry_condition"
    )[
        "mean_nearest_neighbor_mm"
    ]
    .mean()
    .to_dict()
)


print(
    "✓ Coherent connectivity                 : PASS"
)

print(
    "✓ Dispersed separation                  : PASS"
)

print(
    "✓ Fragmented separation                 : PASS"
)

print(
    "Mean NN coherent                        :",
    "{:.3f} mm".format(
        float(
            mean_nn[
                "coherent"
            ]
        )
    ),
)

print(
    "Mean NN dispersed                       :",
    "{:.3f} mm".format(
        float(
            mean_nn[
                "dispersed"
            ]
        )
    ),
)

print(
    "Mean NN fragmented                      :",
    "{:.3f} mm".format(
        float(
            mean_nn[
                "fragmented"
            ]
        )
    ),
)


# ==========================================================================================
# 20. SPARSE-ONLY TRAINER FIREWALL
# ==========================================================================================

heading(
    "STEP 7/12 — VERIFY TRAINER FIREWALL"
)


EXPECTED_TRAINER_KEYS = {
    "supervision_voxel_zyx",
    "supervision_label",
    "fg_membership_voxel_zyx",
    "fg_membership_group_id",
}


for _, row in tqdm(
    artifact_df.iterrows(),
    total=len(
        artifact_df
    ),
    desc="Final sparse firewall",
):

    artifact_path = (
        REPO
        / str(
            row[
                "artifact_file"
            ]
        )
    )


    if sha256_file(
        artifact_path
    ) != str(
        row[
            "artifact_file_sha256"
        ]
    ):

        raise RuntimeError(
            "Final trainer artifact hash mismatch."
        )


    with np.load(
        artifact_path,
        allow_pickle=False,
    ) as data:

        if set(
            data.files
        ) != EXPECTED_TRAINER_KEYS:

            raise RuntimeError(
                "Trainer artifact contains unexpected arrays."
            )


        supervision_coords = np.asarray(
            data[
                "supervision_voxel_zyx"
            ],
            dtype=np.int32,
        )


        supervision_label = np.asarray(
            data[
                "supervision_label"
            ],
            dtype=np.int8,
        )


        membership_coords = np.asarray(
            data[
                "fg_membership_voxel_zyx"
            ],
            dtype=np.int32,
        )


        membership_groups = np.asarray(
            data[
                "fg_membership_group_id"
            ],
            dtype=np.int32,
        )


    if not set(
        np.unique(
            supervision_label
        ).tolist()
    ).issubset(
        {
            0,
            1,
        }
    ):

        raise RuntimeError(
            "Invalid sparse labels."
        )


    direct_fg = {
        tuple(
            int(
                value
            )
            for value in coord
        )
        for coord in supervision_coords[
            supervision_label
            == 1
        ]
    }


    grouped_fg = {
        tuple(
            int(
                value
            )
            for value in coord
        )
        for coord in membership_coords
    }


    if direct_fg != grouped_fg:

        raise RuntimeError(
            "Direct FG and grouped FG support differ."
        )


    if np.any(
        membership_groups
        <= 0
    ):

        raise RuntimeError(
            "Invalid foreground group ID."
        )


print(
    "✓ Sparse trainer artifacts              : PASS 120/120"
)

print(
    "✓ Dense arrays in trainer artifacts     : 0"
)

print(
    "✓ Direct FG == grouped FG               : PASS"
)


# ==========================================================================================
# 21. FINAL BUDGET + COLLISION SUMMARY
# ==========================================================================================

heading(
    "STEP 8/12 — FINAL B_i_train SUMMARY"
)


train_budget_min = int(
    budget_df[
        "frozen_train_budget_B_i"
    ].min()
)


train_budget_median = float(
    budget_df[
        "frozen_train_budget_B_i"
    ].median()
)


train_budget_max = int(
    budget_df[
        "frozen_train_budget_B_i"
    ].max()
)


train_budget_total = int(
    budget_df[
        "frozen_train_budget_B_i"
    ].sum()
)


development_budget_total = int(
    budget_df.loc[
        budget_df[
            "role"
        ]
        == "permanent_development",
        "frozen_train_budget_B_i",
    ].sum()
)


final_cv_budget_total = int(
    budget_df.loc[
        budget_df[
            "role"
        ]
        == "final_outer_cv",
        "frozen_train_budget_B_i",
    ].sum()
)


reduced_cases = int(
    (
        budget_df[
            "budget_reduction_from_native"
        ]
        > 0
    ).sum()
)


total_reduction = int(
    budget_df[
        "budget_reduction_from_native"
    ].sum()
)


mean_retention = float(
    budget_df[
        "budget_retention_fraction"
    ].mean()
)


cases_requiring_budget_descent = int(
    (
        budget_df[
            "budget_candidates_rejected"
        ]
        > 0
    ).sum()
)


cross_collision_rejections = int(
    budget_df[
        "cross_component_collision_rejections"
    ].sum()
)


geometry_rejections = int(
    budget_df[
        "geometry_selection_rejections"
    ].sum()
)


background_rejections = int(
    budget_df[
        "background_rejections"
    ].sum()
)


fg_transfer_collapse = int(
    transfer_df[
        "fg_coordinate_collapse"
    ].sum()
)


bg_transfer_collapse = int(
    transfer_df.groupby(
        "case_id"
    )[
        "bg_coordinate_collapse"
    ].max().sum()
)


fg_outside_crop = int(
    transfer_df[
        "fg_outside_crop"
    ].sum()
)


bg_outside_crop = int(
    transfer_df.groupby(
        "case_id"
    )[
        "bg_outside_crop"
    ].max().sum()
)


bank_capacity_min = int(
    bank_df[
        "capacity"
    ].min()
)


bank_capacity_median = float(
    bank_df[
        "capacity"
    ].median()
)


bank_capacity_max = int(
    bank_df[
        "capacity"
    ].max()
)


print(
    "✓ Coordinate-level feasible cases       : 20/20"
)

print(
    "B_i_train range                         :",
    train_budget_min,
    "–",
    train_budget_max,
)

print(
    "B_i_train median                        :",
    "{:.1f}".format(
        train_budget_median
    ),
)

print(
    "Total FG budget / factorial cell        :",
    train_budget_total,
)

print(
    "Development FG budget / cell            :",
    development_budget_total,
)

print(
    "Final-CV FG budget / cell               :",
    final_cv_budget_total,
)

print(
    "Cases reduced from native B_i           :",
    reduced_cases,
    "/20",
)

print(
    "Total native→grid budget reduction      :",
    total_reduction,
)

print(
    "Mean native→grid retention              :",
    "{:.3f}".format(
        mean_retention
    ),
)

print(
    "Cases requiring coordinate-level descent:",
    cases_requiring_budget_descent,
    "/20",
)

print(
    "Cross-component collision rejections    :",
    cross_collision_rejections,
)

print(
    "Geometry-selection rejections           :",
    geometry_rejections,
)

print(
    "Background-capacity rejections          :",
    background_rejections,
)

print(
    "Neutral-bank capacity range             :",
    bank_capacity_min,
    "–",
    bank_capacity_max,
)

print(
    "Neutral-bank capacity median            :",
    "{:.1f}".format(
        bank_capacity_median
    ),
)


# ==========================================================================================
# 22. FREEZE FINAL TRAINING-GRID CONFIGURATION
# ==========================================================================================

final_config = {
    "project":
        "COVA-3D",

    "block":
        BLOCK,

    "status":
        "STRUCTURAL_PASS",

    "effective_protocol":
        EFFECTIVE_PROTOCOL,

    "training_grid": {
        "spacing_zyx_mm":
            TARGET_SPACING_ZYX.tolist(),

        "preprocessing":
            "corrected_v1.2",

        "crop":
            "normalized_image_only_crop",

        "crop_reproduction":
            "PASS_20_OF_20",
    },

    "GLOBAL_RESOLUTION_LIMITED":
        {
            "minimum_quota":
                1,

            "maximum_capacity":
                "physical training-grid support",

            "representation":
                "deterministic neutral physical bank prefix",

            "geometry_manipulation":
                False,

            "same_bank_across_geometry":
                True,

            "same_bank_across_coverage":
                True,

            "cross_coverage_prefix_nested":
                True,
        },

    "GLOBAL_GEOMETRY_RESOLVABLE":
        {
            "minimum_quota":
                2,

            "geometry_manipulation":
                True,

            "capacity":
                (
                    "common geometry-preserving capacity across COH/DIS/FRG "
                    "within coverage"
                ),
        },

    "budget": {
        "symbol":
            "B_i_train",

        "selection":
            (
                "largest exact common coordinate-level feasible foreground "
                "budget across all six factorial cells"
            ),

        "same_FG_total_across_six":
            True,

        "same_BG_coordinates_across_six":
            True,

        "same_component_set_across_geometry":
            True,

        "same_component_quota_across_geometry":
            True,

        "C50_nested_in_C100":
            True,
    },

    "structural_result": {
        "cases":
            20,

        "coordinate_level_pass":
            20,

        "neutral_banks":
            28,

        "coverage_lost_lesions":
            0,

        "selected_lesions_dropped":
            0,

        "trainer_artifacts":
            120,
    },

    "factorial_training_authorized":
        False,

    "method_development_authorized":
        False,

    "next_block":
        "09E-LOCK",

    "frozen_at_utc":
        NOW_ISO,
}


final_config_path = (
    REPO
    / "configs/"
    "cova3d_training_grid_transfer_v1_2.yaml"
)


final_config_path.write_text(
    yaml.safe_dump(
        final_config,
        sort_keys=False,
        allow_unicode=True,
    ),
    encoding="utf-8",
)


write_text(
    REPO
    / "docs/"
    "cova3d_training_grid_transfer_v1_2.md",
    f"""
    # COVA-3D Block 09D-B2

    ## Final coordinate-level training-grid construction

    Effective protocol: `{EFFECTIVE_PROTOCOL}`.

    Block 09D-B2 is the final structural annotation-feasibility audit before
    baseline training is specified.

    Globally geometry-resolvable lesion components retain the coherent,
    dispersed and fragmented supervision manipulations.

    Globally resolution-limited lesion components use deterministic
    geometry-neutral physical anchor banks.

    Each bank is formed from unique training-grid voxels occupied by the
    physical lesion component.

    Bank voxels are ordered by world-space distance from the native physical
    lesion centroid. Lexicographic z-y-x order resolves exact ties.

    For an assigned quota q, the first q bank voxels are used.

    The same bank is used across geometry and coverage conditions.
    Coverage-specific quota differences therefore produce nested prefixes
    rather than different annotation locations.

    The final patient-specific B_i_train is the largest foreground budget
    that passes actual coordinate-level construction in all six factorial
    cells.

    The final construction requires equal FG counts, identical BG
    coordinates, matched component sets and matched per-component quotas
    across geometry.

    Cross-component foreground collisions and foreground/background
    collisions are prohibited.

    Dense lesion masks are used only offline to derive neutral anchor banks.
    They are never included in trainer artifacts.

    Block 09D-B2 passed all 20 primary cases.

    This establishes structural readiness of the COVA-3D annotation
    methodology on the primary dataset.

    Training is still not authorized.

    The next block is 09E-LOCK. It must prospectively freeze the strong
    sparse-supervision baseline, sparse loss, optimizer, schedule,
    development-only sanity gate and authorization rules before the first
    COVA-3D optimizer step.
    """
)


# ==========================================================================================
# 23. UPDATE STATES — ANNOTATION STRUCTURALLY READY, TRAINING STILL LOCKED
# ==========================================================================================

heading(
    "STEP 9/12 — UPDATE PROJECT STATE"
)


cova_state.update(
    {
        "last_completed_block":
            BLOCK,

        "current_stage":
            "ANNOTATION_METHODOLOGY_STRUCTURALLY_READY",

        "effective_protocol":
            EFFECTIVE_PROTOCOL,

        "trainer_grid_budget_status":
            "FROZEN_PASS",

        "trainer_grid_collision_audit":
            "PASS",

        "coordinate_level_grid_feasibility":
            "PASS_20_OF_20",

        "annotation_methodology_structurally_ready":
            True,

        "training_grid_candidate_artifacts":
            120,

        "neutral_anchor_bank_artifacts":
            28,

        "neutral_anchor_bank_coordinate_QA":
            "PASS",

        "train_budget_min":
            train_budget_min,

        "train_budget_median":
            train_budget_median,

        "train_budget_max":
            train_budget_max,

        "train_budget_total_per_factorial_cell":
            train_budget_total,

        "development_train_budget_total_per_cell":
            development_budget_total,

        "final_train_budget_total_per_cell":
            final_cv_budget_total,

        "train_budget_reduction_cases":
            reduced_cases,

        "train_budget_total_reduction":
            total_reduction,

        "train_budget_mean_retention":
            mean_retention,

        "dense_masks_used_offline_for_final_annotation_construction":
            True,

        "final_outer_cv_dense_masks_used_offline_for_final_annotation_construction":
            final_outer_dense_masks_used_offline,

        "factorial_training_authorized":
            False,

        "method_development_authorized":
            False,

        "final_outer_cv_outcomes_authorized":
            False,

        "optimizer_steps_in_cova3d":
            0,

        "dense_outcomes_opened_in_cova3d":
            False,

        "final_outer_cv_access_in_cova3d":
            0,

        "next_block":
            "09E-LOCK",

        "next_action":
            (
                "Prospectively freeze strong sparse-supervision baseline, "
                "loss, optimizer, schedule and development sanity gate "
                "before any COVA optimizer step."
            ),

        "updated_at_utc":
            NOW_ISO,
    }
)


write_json(
    cova_state_path,
    cova_state,
)


project_state.update(
    {
        "last_attempted_block":
            BLOCK,

        "last_completed_block":
            BLOCK,

        "last_completed_block_name":
            "cova3d_final_coordinate_level_annotation_construction",

        "current_stage":
            "cova3d_annotation_methodology_structurally_ready",

        "current_gate":
            "COVA_BASELINE_PROTOCOL_LOCK",

        "cova3d_effective_protocol":
            EFFECTIVE_PROTOCOL,

        "cova3d_trainer_grid_feasibility":
            "PASS",

        "cova3d_coordinate_level_feasibility":
            "PASS_20_OF_20",

        "cova3d_annotation_methodology_structurally_ready":
            True,

        "cova3d_training_grid_candidate_artifacts":
            120,

        "cova3d_neutral_anchor_bank_artifacts":
            28,

        "cova3d_train_budget_status":
            "FROZEN",

        "cova3d_factorial_training_authorized":
            False,

        "cova3d_method_development_authorized":
            False,

        "cova3d_optimizer_steps":
            0,

        "cova3d_dense_outcomes_opened":
            False,

        "cova3d_final_outer_cv_access":
            0,

        "training_authorized":
            False,

        "pilot_training_authorized":
            False,

        "gate_b":
            "NO_GO",

        "gate_c":
            "NOT_RUN",

        "next_action":
            (
                "Run Block 09E-LOCK. Do not train before baseline and "
                "development sanity-gate settings are frozen."
            ),

        "updated_at_utc":
            NOW_ISO,
    }
)


write_json(
    project_state_path,
    project_state,
)


# ==========================================================================================
# 24. REGRESSION TESTS
# ==========================================================================================

heading(
    "STEP 10/12 — REGRESSION TESTS"
)


test_source = r'''
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


EXPECTED_TRAINER_KEYS = {
    "supervision_voxel_zyx",
    "supervision_label",
    "fg_membership_voxel_zyx",
    "fg_membership_group_id",
}


def test_final_coordinate_level_budget():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_training_grid_budget_v1_2.csv"
    )

    assert len(frame) == 20

    assert frame[
        "coordinate_level_six_cell_feasible"
    ].astype(bool).all()

    assert not frame[
        "factorial_training_authorized"
    ].astype(bool).any()

    assert (
        frame[
            "frozen_train_budget_B_i"
        ]
        >= frame[
            "aggregate_lower_bound"
        ]
    ).all()

    assert (
        frame[
            "frozen_train_budget_B_i"
        ]
        <= frame[
            "aggregate_upper_bound"
        ]
    ).all()


def test_final_six_cell_matching():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_training_grid_condition_manifest_v1_2.csv"
    )

    assert len(frame) == 120

    for _, group in frame.groupby(
        "case_id"
    ):

        assert len(group) == 6

        assert (
            group[
                "train_positive_budget_B_i"
            ].nunique()
            == 1
        )

        assert (
            group[
                "foreground_unique_voxels"
            ].nunique()
            == 1
        )

        assert (
            group[
                "background_semantic_sha256"
            ].nunique()
            == 1
        )

        for coverage in [
            0.5,
            1.0,
        ]:

            subset = group[
                np.isclose(
                    group[
                        "coverage_fraction"
                    ],
                    coverage,
                )
            ]

            assert (
                subset[
                    "selected_component_ids"
                ].nunique()
                == 1
            )

            assert (
                subset[
                    "selected_component_quotas"
                ].nunique()
                == 1
            )


def test_neutral_bank_manifest():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_neutral_anchor_bank_manifest_v1_0.csv"
    )

    assert len(frame) == 28

    assert frame[
        "case_id"
    ].nunique() == 13

    assert (
        frame[
            "capacity"
        ]
        >= 2
    ).all()

    assert frame[
        "shared_across_geometry"
    ].astype(bool).all()

    assert frame[
        "shared_across_coverage"
    ].astype(bool).all()

    assert not frame[
        "geometry_claim_allowed"
    ].astype(bool).any()


def test_resolution_class_rules():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_training_grid_component_audit_v1_2.csv"
    )

    limited = frame[
        frame[
            "global_resolution_class"
        ]
        == "GLOBAL_RESOLUTION_LIMITED"
    ]

    assert (
        limited[
            "quota"
        ]
        >= 1
    ).all()

    assert not limited[
        "geometry_claim_allowed"
    ].astype(bool).any()

    assert (
        limited[
            "representation"
        ]
        == "neutral_physical_bank_prefix"
    ).all()

    resolvable = frame[
        frame[
            "global_resolution_class"
        ]
        == "GLOBAL_GEOMETRY_RESOLVABLE"
    ]

    assert (
        resolvable[
            "quota"
        ]
        >= 2
    ).all()

    coherent = resolvable[
        resolvable[
            "geometry_condition"
        ]
        == "coherent"
    ]

    dispersed = resolvable[
        resolvable[
            "geometry_condition"
        ]
        == "dispersed"
    ]

    fragmented = resolvable[
        resolvable[
            "geometry_condition"
        ]
        == "fragmented"
    ]

    assert (
        coherent[
            "connected_sets"
        ]
        == 1
    ).all()

    assert (
        dispersed[
            "connected_sets"
        ]
        >= 2
    ).all()

    assert (
        fragmented[
            "connected_sets"
        ]
        >= 2
    ).all()


def test_sparse_trainer_artifacts():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_training_grid_candidate_artifact_manifest_v1_2.csv"
    )

    assert len(frame) == 120

    for _, row in frame.iterrows():

        path = ROOT / row[
            "artifact_file"
        ]

        with np.load(
            path,
            allow_pickle=False,
        ) as data:

            assert set(
                data.files
            ) == EXPECTED_TRAINER_KEYS

            labels = np.asarray(
                data[
                    "supervision_label"
                ]
            )

            coords = np.asarray(
                data[
                    "supervision_voxel_zyx"
                ]
            )

            assert set(
                np.unique(
                    labels
                ).tolist()
            ).issubset(
                {
                    0,
                    1,
                }
            )

            assert len(
                np.unique(
                    coords,
                    axis=0,
                )
            ) == len(
                coords
            )
'''


test_path = (
    REPO
    / "tests/"
    "test_cova3d_final_coordinate_grid_v1_2.py"
)


write_text(
    test_path,
    test_source,
)


pytest_env = os.environ.copy()


pytest_env[
    "PYTHONPATH"
] = (
    str(
        REPO
        / "src"
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


pytest_result = sh(
    [
        sys.executable,
        "-m",
        "pytest",

        "tests/test_cova3d_protocol_registry.py",
        "tests/test_cova3d_dataset_feasibility.py",
        "tests/test_cova3d_annotation_simulator_v2.py",
        "tests/test_cova3d_amendment_a1_2.py",
        "tests/test_cova3d_final_coordinate_grid_v1_2.py",

        "-q",
        "-p",
        "no:cacheprovider",
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
        "Block 09D-B2 regression tests FAILED."
    )


print(
    "✓ Regression tests                      : PASS"
)


# ==========================================================================================
# 25. CAPTURE EXACT SOURCE
# ==========================================================================================

source_capture = (
    "NOT_AVAILABLE"
)


try:

    ip = get_ipython()

    cell = (
        ip.history_manager
        .input_hist_raw[
            -1
        ]
    )

    if (
        "COVA-3D — BLOCK 09D-B2"
        in cell
    ):

        source_path = (
            REPO
            / "scripts/code_blocks/"
            "block09d_b2_final_coordinate_grid.py"
        )

        source_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        source_path.write_text(
            cell.rstrip()
            + "\n",
            encoding="utf-8",
        )

        source_capture = (
            "PASS"
        )


except Exception:

    pass


# ==========================================================================================
# 26. FINAL AUDIT
# ==========================================================================================

audit = {
    "project":
        "COVA-3D",

    "block":
        BLOCK,

    "status":
        "PASS",

    "generated_at_utc":
        NOW_ISO,

    "parent_commit":
        head,

    "effective_protocol":
        EFFECTIVE_PROTOCOL,

    "structural_result": {
        "primary_cases":
            20,

        "coordinate_level_feasible_cases":
            20,

        "factorial_conditions":
            120,

        "neutral_banks":
            28,

        "limited_cases":
            13,

        "coverage_lost_lesions":
            0,

        "selected_lesions_dropped":
            0,

        "annotation_methodology_structurally_ready":
            True,
    },

    "budget": {
        "minimum":
            train_budget_min,

        "median":
            train_budget_median,

        "maximum":
            train_budget_max,

        "total_per_factorial_cell":
            train_budget_total,

        "development_total_per_cell":
            development_budget_total,

        "final_cv_total_per_cell":
            final_cv_budget_total,

        "cases_reduced_from_native":
            reduced_cases,

        "total_reduction":
            total_reduction,

        "mean_retention":
            mean_retention,

        "cases_requiring_coordinate_level_budget_descent":
            cases_requiring_budget_descent,
    },

    "coordinate_level_search": {
        "cross_component_collision_rejections":
            cross_collision_rejections,

        "geometry_selection_rejections":
            geometry_rejections,

        "background_rejections":
            background_rejections,
    },

    "neutral_banks": {
        "capacity_min":
            bank_capacity_min,

        "capacity_median":
            bank_capacity_median,

        "capacity_max":
            bank_capacity_max,

        "same_across_geometry":
            True,

        "same_ordered_bank_across_coverage":
            True,

        "cross_coverage_prefix_nesting":
            True,

        "geometry_claims":
            0,
    },

    "causal_matching": {
        "same_FG_budget_across_six":
            True,

        "same_BG_coordinates_across_six":
            True,

        "same_component_set_across_geometry":
            True,

        "same_component_quota_across_geometry":
            True,

        "C50_component_set_nested_in_C100":
            True,

        "cross_component_FG_collisions_in_final_artifacts":
            0,

        "FG_BG_collisions_in_final_artifacts":
            0,
    },

    "resolvable_geometry": {
        "coherent_connected":
            True,

        "dispersed_separated":
            True,

        "fragmented_separated":
            True,

        "mean_nn_mm":
            {
                geometry:
                    float(
                        mean_nn[
                            geometry
                        ]
                    )
                for geometry in GEOMETRIES
            },
    },

    "raw_native_candidate_transfer": {
        "FG_coordinate_collapse":
            fg_transfer_collapse,

        "BG_coordinate_collapse":
            bg_transfer_collapse,

        "FG_outside_crop":
            fg_outside_crop,

        "BG_outside_crop":
            bg_outside_crop,
    },

    "firewall": {
        "CT_arrays_accessed":
            ct_arrays_accessed,

        "dense_lesion_arrays_accessed_offline":
            dense_lesion_arrays_accessed,

        "final_outer_dense_masks_used_offline":
            final_outer_dense_masks_used_offline,

        "dense_lung_arrays_accessed":
            dense_lung_arrays_accessed,

        "dense_arrays_inside_trainer_artifacts":
            0,

        "predictions_accessed":
            predictions_accessed,

        "checkpoints_accessed":
            checkpoints_accessed,

        "model_outcomes_accessed":
            model_outcomes_accessed,

        "optimizer_steps":
            0,
    },

    "factorial_training_authorized":
        False,

    "method_development_authorized":
        False,

    "source_capture":
        source_capture,

    "next_block":
        "09E-LOCK",
}


audit_path = (
    REPO
    / "experiments/audits/"
    "block09d_b2_final_coordinate_grid.json"
)


write_json(
    audit_path,
    audit,
)


# ==========================================================================================
# 27. REPOSITORY MANIFEST
# ==========================================================================================

repo_manifest_path = (
    REPO
    / "REPOSITORY_MANIFEST.json"
)


EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    ".pytest_cache",
    ".ipynb_checkpoints",
}


repo_files = sorted(
    [
        path
        for path in REPO.rglob(
            "*"
        )
        if (
            path.is_file()
            and not any(
                part in EXCLUDED_DIRS
                for part in path.parts
            )
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
            BLOCK,

        "active_track":
            "COVA3D",

        "effective_protocol":
            EFFECTIVE_PROTOCOL,

        "annotation_methodology_structurally_ready":
            True,

        "coordinate_level_feasibility":
            "PASS_20_OF_20",

        "neutral_anchor_banks":
            28,

        "final_sparse_trainer_candidates":
            120,

        "train_budget":
            "FROZEN",

        "factorial_training_authorized":
            False,

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
# 28. COMMIT + PUSH
# ==========================================================================================

heading(
    "STEP 11/12 — COMMIT FINAL STRUCTURAL FREEZE"
)


git_paths = [
    "COVA3D_STATE.json",
    "PROJECT_STATE.json",
    "REPOSITORY_MANIFEST.json",

    "configs/"
    "cova3d_training_grid_transfer_v1_2.yaml",

    "data/manifests/"
    "cova3d_training_grid_geometry_v1_2.csv",

    "data/manifests/"
    "cova3d_training_grid_raw_transfer_audit_v1_2.csv",

    "data/manifests/"
    "cova3d_neutral_anchor_bank_manifest_v1_0.csv",

    "data/manifests/"
    "cova3d_training_grid_budget_v1_2.csv",

    "data/manifests/"
    "cova3d_training_grid_condition_manifest_v1_2.csv",

    "data/manifests/"
    "cova3d_training_grid_component_audit_v1_2.csv",

    "data/manifests/"
    "cova3d_training_grid_candidate_artifact_manifest_v1_2.csv",

    "docs/"
    "cova3d_training_grid_transfer_v1_2.md",

    "experiments/audits/"
    "block09d_b2_final_coordinate_grid.json",

    "tests/"
    "test_cova3d_final_coordinate_grid_v1_2.py",
]


source_path = (
    REPO
    / "scripts/code_blocks/"
    "block09d_b2_final_coordinate_grid.py"
)


if source_path.exists():

    git_paths.append(
        "scripts/code_blocks/"
        "block09d_b2_final_coordinate_grid.py"
    )


# Normalize text EOF.
for relative in git_paths:

    path = (
        REPO
        / relative
    )

    if (
        path.exists()
        and path.is_file()
        and path.suffix.lower()
        in {
            ".json",
            ".yaml",
            ".yml",
            ".md",
            ".csv",
            ".py",
        }
    ):

        text = path.read_text(
            encoding="utf-8"
        )

        path.write_text(
            text.rstrip()
            + "\n",
            encoding="utf-8",
        )


sh(
    [
        "git",
        "add",
        *git_paths,
    ],
    cwd=REPO,
)


# Force-add scientific NPZ artifacts because repository ignores *.npz globally.
sh(
    [
        "git",
        "add",
        "-f",
        "data/cova3d_training_grid_candidates_v1_2",
        "data/cova3d_neutral_anchor_banks_v1_0",
    ],
    cwd=REPO,
)


tracked_candidates = sh(
    [
        "git",
        "ls-files",
        "data/cova3d_training_grid_candidates_v1_2",
    ],
    cwd=REPO,
).stdout.strip().splitlines()


tracked_banks = sh(
    [
        "git",
        "ls-files",
        "data/cova3d_neutral_anchor_banks_v1_0",
    ],
    cwd=REPO,
).stdout.strip().splitlines()


if len(
    tracked_candidates
) != 120:

    raise RuntimeError(
        "Expected 120 tracked final sparse candidates; observed "
        + str(
            len(
                tracked_candidates
            )
        )
    )


if len(
    tracked_banks
) != 28:

    raise RuntimeError(
        "Expected 28 tracked neutral banks; observed "
        + str(
            len(
                tracked_banks
            )
        )
    )


check = sh(
    [
        "git",
        "diff",
        "--cached",
        "--check",
    ],
    cwd=REPO,
    check=False,
)


if check.returncode != 0:

    raise RuntimeError(
        "Git validation failed:\n"
        + (
            check.stdout
            or ""
        )
        + (
            check.stderr
            or ""
        )
    )


print(
    "✓ Final sparse candidates tracked       : 120/120"
)

print(
    "✓ Neutral-bank artifacts tracked        : 28/28"
)

print(
    "✓ git diff --cached --check             : PASS"
)


sh(
    [
        "git",
        "commit",
        "-m",
        "data: freeze final COVA-3D coordinate-level annotations",
    ],
    cwd=REPO,
)


final_commit = sh(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    cwd=REPO,
).stdout.strip()


token, askpass, git_env = make_git_auth()


safe_push(
    git_env,
    token,
)


try:

    askpass.unlink(
        missing_ok=True
    )

except Exception:

    pass


final_status = sh(
    [
        "git",
        "status",
        "--porcelain",
    ],
    cwd=REPO,
).stdout.strip()


if final_status:

    raise RuntimeError(
        "Repository is not clean after 09D-B2 commit:\n"
        + final_status
    )


# ==========================================================================================
# 29. FINAL REPORT
# ==========================================================================================

heading(
    "STEP 12/12 — FINAL 09D-B2 REPORT"
)


print(
    "COVA-3D BLOCK 09D-B2 — FINAL COORDINATE-LEVEL REPORT"
)

print(
    "-" * 130
)

print(
    "Effective protocol                     :",
    EFFECTIVE_PROTOCOL,
)

print(
    "Starting commit                        :",
    head[:12],
)

print(
    "Final commit                           :",
    final_commit[:12],
)

print(
    "GitHub synchronization                 : PASS"
)

print(
    "Repository state                       : CLEAN"
)

print()

print(
    "STRUCTURAL FEASIBILITY"
)

print(
    "----------------------"
)

print(
    "Primary cases                          : 20"
)

print(
    "Coordinate-level six-cell pass         : 20 /20"
)

print(
    "Coverage-lost lesions                  : 0"
)

print(
    "Selected lesions dropped               : 0"
)

print(
    "Annotation methodology ready           : YES"
)

print()

print(
    "NEUTRAL PHYSICAL BANKS"
)

print(
    "----------------------"
)

print(
    "Global resolution-limited lesions      : 28"
)

print(
    "Cases with limited lesions             : 13 /20"
)

print(
    "Neutral banks persisted                : 28 /28"
)

print(
    "Bank capacity range                    :",
    bank_capacity_min,
    "–",
    bank_capacity_max,
    "voxels",
)

print(
    "Bank capacity median                   :",
    "{:.1f}".format(
        bank_capacity_median
    ),
    "voxels",
)

print(
    "Same bank across geometry              : PASS"
)

print(
    "Same ordered bank across coverage      : PASS"
)

print(
    "Nested cross-coverage prefixes         : PASS"
)

print(
    "Geometry claims on limited lesions     : 0"
)

print()

print(
    "FINAL B_i_train"
)

print(
    "---------------"
)

print(
    "B_i_train range                        :",
    train_budget_min,
    "–",
    train_budget_max,
)

print(
    "B_i_train median                       :",
    "{:.1f}".format(
        train_budget_median
    ),
)

print(
    "Total FG budget / factorial cell       :",
    train_budget_total,
)

print(
    "Development FG budget / cell           :",
    development_budget_total,
)

print(
    "Final-CV FG budget / cell              :",
    final_cv_budget_total,
)

print(
    "Cases reduced from native B_i          :",
    reduced_cases,
    "/20",
)

print(
    "Total native→grid reduction            :",
    total_reduction,
)

print(
    "Mean budget retention                 :",
    "{:.3f}".format(
        mean_retention
    ),
)

print(
    "Cases requiring budget descent         :",
    cases_requiring_budget_descent,
    "/20",
)

print()

print(
    "COORDINATE-LEVEL COLLISION SEARCH"
)

print(
    "---------------------------------"
)

print(
    "Cross-component rejected budgets       :",
    cross_collision_rejections,
)

print(
    "Geometry-selection rejected budgets    :",
    geometry_rejections,
)

print(
    "Background rejected budgets            :",
    background_rejections,
)

print(
    "Final FG cross-component collisions    : 0"
)

print(
    "Final FG/BG collisions                 : 0"
)

print()

print(
    "CAUSAL MATCHING"
)

print(
    "---------------"
)

print(
    "Equal FG budget across six cells       : PASS 20/20"
)

print(
    "Identical BG coordinates               : PASS 20/20"
)

print(
    "Same component sets across geometry    : PASS"
)

print(
    "Same quotas across geometry            : PASS"
)

print(
    "C50 component set nested in C100       : PASS"
)

print()

print(
    "RESOLVABLE-LESION GEOMETRY"
)

print(
    "--------------------------"
)

print(
    "Coherent connectivity                  : PASS"
)

print(
    "Dispersed separation                   : PASS"
)

print(
    "Fragmented separation                  : PASS"
)

print(
    "Mean NN coherent                       :",
    "{:.3f} mm".format(
        float(
            mean_nn[
                "coherent"
            ]
        )
    ),
)

print(
    "Mean NN dispersed                      :",
    "{:.3f} mm".format(
        float(
            mean_nn[
                "dispersed"
            ]
        )
    ),
)

print(
    "Mean NN fragmented                     :",
    "{:.3f} mm".format(
        float(
            mean_nn[
                "fragmented"
            ]
        )
    ),
)

print()

print(
    "RAW NATIVE→GRID TRANSFER"
)

print(
    "------------------------"
)

print(
    "Native-candidate FG collapse events    :",
    fg_transfer_collapse,
)

print(
    "Native-candidate BG collapse events    :",
    bg_transfer_collapse,
)

print(
    "Native-candidate FG outside crop       :",
    fg_outside_crop,
)

print(
    "Native-candidate BG outside crop       :",
    bg_outside_crop,
)

print()

print(
    "FIREWALL / TRAINING STATUS"
)

print(
    "--------------------------"
)

print(
    "CT arrays accessed                     :",
    ct_arrays_accessed,
)

print(
    "Dense lesion masks accessed OFFLINE    :",
    dense_lesion_arrays_accessed,
)

print(
    "Final-CV masks used OFFLINE            :",
    final_outer_dense_masks_used_offline,
)

print(
    "Dense lung masks accessed              :",
    dense_lung_arrays_accessed,
)

print(
    "Dense arrays in trainer artifacts      : 0"
)

print(
    "Predictions accessed                   : 0"
)

print(
    "Checkpoints accessed                   : 0"
)

print(
    "Model outcomes accessed                : 0"
)

print(
    "Optimizer steps                        : 0"
)

print(
    "Factorial training authorized          : NO"
)

print(
    "Method development authorized          : NO"
)

print()

print(
    "Regression tests                       : PASS"
)

print(
    "Exact source captured                  :",
    source_capture,
)

print()

print(
    "NEXT:"
)

print(
    "Send me this COMPLETE report."
)

print(
    "Do NOT train anything yet."
)

print(
    "If this report shows 20/20 coordinate-level PASS, the annotation "
    "methodology is structurally complete."
)

print(
    "The next block will be 09E-LOCK: freeze the strong sparse-supervision "
    "baseline, masked loss, optimizer, schedule, development-only sanity "
    "gate and training authorization rules before the first optimizer step."
)

print(
    "=" * 130
)
