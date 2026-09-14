# ==========================================================================================
# COVA-3D — CODE BLOCK 09C
# Annotation Simulator v2 + Exact Native-Space Six-Cell Feasibility
#
# PURPOSE
#
# Implement the three prospectively frozen annotation geometries:
#
#   1. COHERENT
#      Connected compact interior trace produced by deterministic 26-connected
#      growth from an interior seed.
#
#   2. DISPERSED
#      Deterministic farthest-point sampling within the same lesion component.
#
#   3. FRAGMENTED
#      Two/three spatially separated compact connected fragments.
#
# For every patient/case:
#
#   • 50% coverage uses ceil(0.5*K) lesion components.
#   • 100% coverage uses all K lesion components.
#   • 50% selected components are a deterministic nested subset of 100%.
#   • the SAME selected components are used across all three geometries
#     within a coverage level.
#   • the SAME per-component positive quota is used across all three
#     geometries within a coverage level.
#   • all SIX cells use the SAME total number B_i of unique FG voxels.
#   • all SIX cells use EXACTLY the SAME BG coordinates.
#
# IMPORTANT BUDGET SAFEGUARD
#
# Block 09C freezes:
#
#     B_i_native
#
# = the largest common feasible unique-positive budget in native physical
# space under the six annotation conditions.
#
# These artifacts are explicitly:
#
#     trainer_eligible = FALSE
#
# Block 09D MUST transfer all sparse coordinates through the frozen v1.2
# training-grid geometry and check for resampling collisions.
#
# If different geometries collapse differently after mapping to the training
# grid, Block 09D will prospectively define:
#
#     B_i_train <= B_i_native
#
# using ONLY transfer feasibility, never model performance.
#
# THIS BLOCK:
#
#   • MAY read dense lesion/lung masks offline for annotation simulation.
#   • DOES NOT read CT voxel arrays.
#   • DOES NOT read model predictions.
#   • DOES NOT read checkpoints.
#   • DOES NOT train.
#   • DOES NOT instantiate an optimizer.
#   • DOES NOT open final-CV MODEL OUTCOMES.
#
# Dense masks from final-CV cases are used ONLY as offline annotation sources,
# which is permitted by the frozen firewall.
# ==========================================================================================


from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import subprocess
import hashlib
import importlib
import json
import math
import os
import sys
import textwrap
import gc

import numpy as np
import pandas as pd
import yaml

import nibabel as nib

from scipy import ndimage as ndi

from tqdm.auto import tqdm

from kaggle_secrets import UserSecretsClient


# ==========================================================================================
# 0. CONSTANTS
# ==========================================================================================

REPO = Path(
    "/kaggle/working/CORA-LUNG"
)

REPOSITORY_URL = (
    "https://github.com/itsCodeBakery/CORA-LUNG.git"
)

EXPECTED_START_COMMIT = (
    "79be23b11400"
)

BLOCK = (
    "09C"
)

TRACK_ID = (
    "COVA3D"
)

SIMULATOR_VERSION = (
    "2.0-native-candidate"
)

PROTOCOL_VERSION = (
    "1.0"
)

PRIMARY_DATASET_ID = (
    "COVID19_CT_Seg_20"
)

COMPONENT_SELECTION_MASTER_SEED = (
    20260914
)

PRIMARY_CONNECTIVITY = (
    26
)

REFERENCE_MIN_VOLUME_ML = (
    0.10
)

LOW_COVERAGE = (
    0.50
)

MAX_NATIVE_POSITIVE_QUOTA_PER_COMPONENT = (
    20
)

MIN_NATIVE_POSITIVE_QUOTA_PER_COMPONENT = (
    3
)

INTERIOR_EROSION_MM = (
    1.0
)

BACKGROUND_EXCLUSION_MM = (
    3.0
)

BACKGROUND_TO_FOREGROUND_RATIO = (
    1.0
)

GEOMETRIES = [
    "coherent",
    "dispersed",
    "fragmented",
]

GEOMETRY_CODE = {
    "coherent":
        "COH",

    "dispersed":
        "DIS",

    "fragmented":
        "FRG",
}

COVERAGES = [
    0.50,
    1.00,
]

CONDITION_IDS = {
    (
        0.50,
        "coherent",
    ):
        "C50_COH",

    (
        0.50,
        "dispersed",
    ):
        "C50_DIS",

    (
        0.50,
        "fragmented",
    ):
        "C50_FRG",

    (
        1.00,
        "coherent",
    ):
        "C100_COH",

    (
        1.00,
        "dispersed",
    ):
        "C100_DIS",

    (
        1.00,
        "fragmented",
    ):
        "C100_FRG",
}

EXPECTED_CASES = (
    20
)

EXPECTED_CONDITIONS_PER_CASE = (
    6
)

EXPECTED_ANNOTATION_ARTIFACTS = (
    EXPECTED_CASES
    * EXPECTED_CONDITIONS_PER_CASE
)

NATIVE_CANDIDATE_ROOT = (
    REPO
    / "data/cova3d_native_candidates_v1_0"
)

STRUCT26 = ndi.generate_binary_structure(
    3,
    3,
)

NEIGHBOR_OFFSETS_26 = np.asarray(
    [
        (
            dx,
            dy,
            dz,
        )
        for dx in (
            -1,
            0,
            1,
        )
        for dy in (
            -1,
            0,
            1,
        )
        for dz in (
            -1,
            0,
            1,
        )
        if not (
            dx == 0
            and dy == 0
            and dz == 0
        )
    ],
    dtype=np.int32,
)

NOW_ISO = datetime.now(
    timezone.utc
).strftime(
    "%Y-%m-%dT%H:%M:%SZ"
)


# ==========================================================================================
# 1. HELPERS
# ==========================================================================================

def heading(
    text,
):

    print(
        "\n"
        + "=" * 126
    )

    print(
        text
    )

    print(
        "=" * 126
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
            str(
                cwd
            )
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
        and result.returncode
        != 0
    ):

        raise RuntimeError(
            "COMMAND FAILED\n"
            + " ".join(
                map(
                    str,
                    cmd,
                )
            )
            + "\nSTDOUT:\n"
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


def semantic_array_hash(
    *arrays,
):

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


def stable_seed(
    *parts,
):

    text = "|".join(
        str(
            value
        )
        for value in parts
    )

    digest = hashlib.sha256(
        text.encode(
            "utf-8"
        )
    ).digest()

    return int.from_bytes(
        digest[
            :8
        ],
        "little",
        signed=False,
    ) % (
        2**32
        - 1
    )


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
        "/tmp/cova3d_git_askpass_09c.sh"
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


def safe_git_push(
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


def physical_ball(
    radius_mm,
    spacing_xyz,
):

    spacing_xyz = np.asarray(
        spacing_xyz,
        dtype=np.float64,
    )

    radii = np.ceil(
        float(
            radius_mm
        )
        / spacing_xyz
    ).astype(
        int
    )

    ranges = [
        np.arange(
            -radius,
            radius + 1,
            dtype=np.float64,
        )
        for radius in radii
    ]

    x, y, z = np.meshgrid(
        ranges[
            0
        ],
        ranges[
            1
        ],
        ranges[
            2
        ],
        indexing="ij",
    )

    distance_squared = (
        (
            x
            * spacing_xyz[
                0
            ]
        )
        ** 2
        + (
            y
            * spacing_xyz[
                1
            ]
        )
        ** 2
        + (
            z
            * spacing_xyz[
                2
            ]
        )
        ** 2
    )

    return (
        distance_squared
        <= float(
            radius_mm
        )
        ** 2
        + 1e-12
    )


def lexical_sort_coords(
    coords,
):

    coords = np.asarray(
        coords,
        dtype=np.int32,
    )

    if len(
        coords
    ) == 0:

        return coords.reshape(
            0,
            3,
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


def point_component_count(
    coords_xyz,
):

    coords_xyz = np.asarray(
        coords_xyz,
        dtype=np.int32,
    )

    if len(
        coords_xyz
    ) == 0:

        return 0

    minimum = coords_xyz.min(
        axis=0
    )

    maximum = coords_xyz.max(
        axis=0
    )

    shape = (
        maximum
        - minimum
        + 1
    )

    local = np.zeros(
        tuple(
            int(
                x
            )
            for x in shape
        ),
        dtype=bool,
    )

    shifted = (
        coords_xyz
        - minimum[
            None,
            :
        ]
    )

    local[
        shifted[
            :,
            0
        ],
        shifted[
            :,
            1
        ],
        shifted[
            :,
            2
        ],
    ] = True

    _, count = ndi.label(
        local,
        structure=STRUCT26,
    )

    return int(
        count
    )


def mean_nearest_neighbor_mm(
    coords_xyz,
    spacing_xyz,
):

    coords_xyz = np.asarray(
        coords_xyz,
        dtype=np.float64,
    )

    if len(
        coords_xyz
    ) <= 1:

        return 0.0

    physical = (
        coords_xyz
        * np.asarray(
            spacing_xyz,
            dtype=np.float64,
        )[
            None,
            :
        ]
    )

    difference = (
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
            difference
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
# 2. REPOSITORY / PROTOCOL PRECONDITIONS
# ==========================================================================================

heading(
    "COVA-3D BLOCK 09C — ANNOTATION SIMULATOR v2"
)


token, askpass, git_env = make_git_auth()


if not (
    REPO
    / ".git"
).exists():

    if REPO.exists():

        raise RuntimeError(
            "Non-git directory exists at repository location."
        )

    result = sh(
        [
            "git",
            "clone",
            REPOSITORY_URL,
            str(
                REPO
            ),
        ],
        env=git_env,
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
            "Repository clone failed:\n"
            + safe_error
        )


sh(
    [
        "git",
        "config",
        "user.name",
        "COVA-3D Kaggle Runner",
    ],
    cwd=REPO,
)


sh(
    [
        "git",
        "config",
        "user.email",
        "cova3d@local.invalid",
    ],
    cwd=REPO,
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
        "Unexpected starting commit.\n"
        + "Expected: "
        + EXPECTED_START_COMMIT
        + "\nObserved: "
        + starting_commit
    )


if sh(
    [
        "git",
        "status",
        "--porcelain",
    ],
    cwd=REPO,
).stdout.strip():

    raise RuntimeError(
        "Repository must be clean before Block 09C."
    )


cova_state_path = (
    REPO
    / "COVA3D_STATE.json"
)


project_state_path = (
    REPO
    / "PROJECT_STATE.json"
)


protocol_path = (
    REPO
    / "configs/"
    "cova3d_protocol_v1_0.yaml"
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


protocol = yaml.safe_load(
    protocol_path.read_text(
        encoding="utf-8"
    )
)


if cova_state.get(
    "last_completed_block"
) != "09B":

    raise RuntimeError(
        "Expected Block 09B as the last completed COVA block."
    )


if cova_state.get(
    "primary_dataset_structural_feasibility"
) != "PASS":

    raise RuntimeError(
        "Primary structural feasibility did not pass."
    )


if cova_state.get(
    "factorial_training_authorized"
) is not False:

    raise RuntimeError(
        "Factorial training must remain unauthorized."
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


if project_state.get(
    "gate_b"
) != "NO_GO":

    raise RuntimeError(
        "Historical CORA Gate-B result changed."
    )


if project_state.get(
    "gate_c"
) != "NOT_RUN":

    raise RuntimeError(
        "Historical CORA Gate-C status changed."
    )


print(
    "✓ Starting commit                      :",
    starting_commit[:12],
)

print(
    "✓ COVA protocol                        : FROZEN"
)

print(
    "✓ 09B structural feasibility           : PASS"
)

print(
    "✓ Factorial training                   : NOT AUTHORIZED"
)

print(
    "✓ COVA optimizer steps                 : 0"
)


# ==========================================================================================
# 3. LOCATE PRIMARY DATASET + VERIFY FROZEN SOURCE HASHES
# ==========================================================================================

heading(
    "STEP 1/9 — LOCATE SOURCE MASKS AND VERIFY FROZEN HASH MANIFEST"
)


split_df = pd.read_csv(
    REPO
    / "data/splits/"
    "primary_volume_split_registry.csv"
)


component_summary_df = pd.read_csv(
    REPO
    / "data/manifests/"
    "primary_case_component_summary.csv"
)


structural_df = pd.read_csv(
    REPO
    / "data/manifests/"
    "cova3d_primary_factorial_structural_feasibility_v1_0.csv"
)


source_hash_df = pd.read_csv(
    REPO
    / "data/manifests/"
    "primary_file_sha256.csv"
)


if len(
    split_df
) != EXPECTED_CASES:

    raise RuntimeError(
        "Unexpected primary split size."
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


candidate_roots = [
    Path(
        "/kaggle/input/datasets/andrewmvd/covid19-ct-scans"
    ),
    Path(
        "/kaggle/input/covid19-ct-scans"
    ),
]


probe_relative = str(
    split_df.iloc[
        0
    ][
        "infection_mask"
    ]
)


primary_root = None


for root in candidate_roots:

    if (
        root.exists()
        and (
            root
            / probe_relative
        ).exists()
    ):

        primary_root = root

        break


if primary_root is None:

    raise RuntimeError(
        "Primary Kaggle dataset is not attached."
    )


for _, row in split_df.iterrows():

    for column in [
        "infection_mask",
        "lung_mask",
    ]:

        relative_path = str(
            row[
                column
            ]
        )

        if relative_path not in source_hash_lookup:

            raise RuntimeError(
                "Missing frozen source hash for "
                + relative_path
            )


print(
    "✓ Primary dataset root                 :",
    primary_root,
)

print(
    "✓ Split cases                          : 20"
)

print(
    "✓ Infection-mask hash entries          : 20"
)

print(
    "✓ Lung-mask hash entries               : 20"
)

print(
    "✓ CT arrays accessed                   : 0"
)


# ==========================================================================================
# 4. GEOMETRY ALGORITHMS
# ==========================================================================================

heading(
    "STEP 2/9 — FREEZE ANNOTATION GEOMETRY ALGORITHMS"
)


def build_component_pool(
    component_submask,
    spacing_xyz,
    global_offset_xyz,
):

    component_submask = np.asarray(
        component_submask,
        dtype=bool,
    )

    spacing_xyz = np.asarray(
        spacing_xyz,
        dtype=np.float64,
    )

    global_offset_xyz = np.asarray(
        global_offset_xyz,
        dtype=np.int32,
    )


    erosion_structure = physical_ball(
        INTERIOR_EROSION_MM,
        spacing_xyz,
    )


    eroded = ndi.binary_erosion(
        component_submask,
        structure=erosion_structure,
        border_value=0,
    )


    eroded_labels, eroded_count = ndi.label(
        eroded,
        structure=STRUCT26,
    )


    pool_mask = None

    pool_source = None


    if eroded_count > 0:

        eroded_sizes = np.bincount(
            eroded_labels.reshape(
                -1
            )
        )

        if len(
            eroded_sizes
        ) > 1:

            largest_id = int(
                1
                + np.argmax(
                    eroded_sizes[
                        1:
                    ]
                )
            )

            largest_mask = (
                eroded_labels
                == largest_id
            )

            if int(
                largest_mask.sum()
            ) >= MAX_NATIVE_POSITIVE_QUOTA_PER_COMPONENT:

                pool_mask = largest_mask

                pool_source = (
                    "largest_1mm_eroded_component"
                )


    if pool_mask is None:

        pool_mask = component_submask.copy()

        pool_source = (
            "full_component_fallback"
        )


    pool_local = np.argwhere(
        pool_mask
    ).astype(
        np.int32
    )


    pool_global = (
        pool_local
        + global_offset_xyz[
            None,
            :
        ]
    )


    distance_map = ndi.distance_transform_edt(
        component_submask,
        sampling=spacing_xyz,
    )


    pool_scores = distance_map[
        pool_mask
    ].astype(
        np.float64
    )


    if len(
        pool_global
    ) < MAX_NATIVE_POSITIVE_QUOTA_PER_COMPONENT:

        raise RuntimeError(
            "Eligible component has fewer than "
            + str(
                MAX_NATIVE_POSITIVE_QUOTA_PER_COMPONENT
            )
            + " candidate voxels after fallback."
        )


    return {
        "coords":
            pool_global,

        "scores":
            pool_scores,

        "source":
            pool_source,

        "capacity":
            int(
                min(
                    MAX_NATIVE_POSITIVE_QUOTA_PER_COMPONENT,
                    len(
                        pool_global
                    ),
                )
            ),
    }


def coherent_select(
    pool_coords,
    pool_scores,
    quota,
):

    pool_coords = np.asarray(
        pool_coords,
        dtype=np.int32,
    )

    pool_scores = np.asarray(
        pool_scores,
        dtype=np.float64,
    )

    quota = int(
        quota
    )


    if quota <= 0:

        raise ValueError(
            "quota must be positive"
        )


    coord_to_score = {
        tuple(
            int(
                x
            )
            for x in coord
        ):
            float(
                score
            )
        for coord, score in zip(
            pool_coords,
            pool_scores,
        )
    }


    pool_set = set(
        coord_to_score
    )


    seed_index = int(
        np.argmax(
            pool_scores
        )
    )


    seed = tuple(
        int(
            x
        )
        for x in pool_coords[
            seed_index
        ]
    )


    selected = {
        seed
    }


    frontier = set()


    def add_neighbors(
        coord,
    ):

        base = np.asarray(
            coord,
            dtype=np.int32,
        )

        neighbors = (
            base[
                None,
                :
            ]
            + NEIGHBOR_OFFSETS_26
        )

        for neighbor_array in neighbors:

            neighbor = tuple(
                int(
                    x
                )
                for x in neighbor_array
            )

            if (
                neighbor in pool_set
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
                "Connected coherent growth exhausted before quota."
            )


        candidate = sorted(
            frontier,
            key=lambda coord: (
                -coord_to_score[
                    coord
                ],
                coord,
            ),
        )[
            0
        ]


        frontier.remove(
            candidate
        )

        selected.add(
            candidate
        )

        add_neighbors(
            candidate
        )


    coords = np.asarray(
        sorted(
            selected
        ),
        dtype=np.int32,
    )


    if point_component_count(
        coords
    ) != 1:

        raise RuntimeError(
            "Coherent geometry is not 26-connected."
        )


    return coords


def farthest_point_select(
    pool_coords,
    pool_scores,
    quota,
    spacing_xyz,
):

    pool_coords = np.asarray(
        pool_coords,
        dtype=np.int32,
    )

    pool_scores = np.asarray(
        pool_scores,
        dtype=np.float64,
    )

    spacing_xyz = np.asarray(
        spacing_xyz,
        dtype=np.float64,
    )

    quota = int(
        quota
    )


    if quota > len(
        pool_coords
    ):

        raise RuntimeError(
            "Farthest-point quota exceeds candidate pool."
        )


    physical = (
        pool_coords.astype(
            np.float64
        )
        * spacing_xyz[
            None,
            :
        ]
    )


    first = int(
        np.argmax(
            pool_scores
        )
    )


    selected_indices = [
        first
    ]


    difference = (
        physical
        - physical[
            first
        ][
            None,
            :
        ]
    )


    min_distance_sq = np.sum(
        difference
        ** 2,
        axis=1,
    )


    min_distance_sq[
        first
    ] = (
        -np.inf
    )


    while len(
        selected_indices
    ) < quota:

        next_index = int(
            np.argmax(
                min_distance_sq
            )
        )


        if not np.isfinite(
            min_distance_sq[
                next_index
            ]
        ):

            raise RuntimeError(
                "Farthest-point sampling exhausted."
            )


        selected_indices.append(
            next_index
        )


        difference = (
            physical
            - physical[
                next_index
            ][
                None,
                :
            ]
        )


        distance_sq = np.sum(
            difference
            ** 2,
            axis=1,
        )


        min_distance_sq = np.minimum(
            min_distance_sq,
            distance_sq,
        )


        min_distance_sq[
            selected_indices
        ] = (
            -np.inf
        )


    return lexical_sort_coords(
        pool_coords[
            np.asarray(
                selected_indices,
                dtype=np.int64,
            )
        ]
    )


def _grow_fragment_cluster(
    pool_set,
    seed,
    quota,
    spacing_xyz,
    forbidden,
):

    quota = int(
        quota
    )

    spacing_xyz = np.asarray(
        spacing_xyz,
        dtype=np.float64,
    )


    if seed in forbidden:

        return None


    selected = {
        seed
    }


    frontier = set()


    seed_array = np.asarray(
        seed,
        dtype=np.float64,
    )


    def add_neighbors(
        coord,
    ):

        base = np.asarray(
            coord,
            dtype=np.int32,
        )

        neighbors = (
            base[
                None,
                :
            ]
            + NEIGHBOR_OFFSETS_26
        )


        for neighbor_array in neighbors:

            neighbor = tuple(
                int(
                    x
                )
                for x in neighbor_array
            )

            if (
                neighbor in pool_set
                and neighbor not in selected
                and neighbor not in forbidden
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

            return None


        def key(
            coord,
        ):

            coordinate = np.asarray(
                coord,
                dtype=np.float64,
            )

            physical_delta = (
                (
                    coordinate
                    - seed_array
                )
                * spacing_xyz
            )

            return (
                float(
                    np.sum(
                        physical_delta
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


    return selected


def fragmented_select(
    pool_coords,
    pool_scores,
    quota,
    spacing_xyz,
):

    pool_coords = np.asarray(
        pool_coords,
        dtype=np.int32,
    )

    pool_scores = np.asarray(
        pool_scores,
        dtype=np.float64,
    )

    quota = int(
        quota
    )


    if quota < 2:

        raise RuntimeError(
            "Fragmented geometry requires quota >=2."
        )


    pool_set = {
        tuple(
            int(
                x
            )
            for x in coord
        )
        for coord in pool_coords
    }


    for fragment_count in [
        min(
            3,
            quota,
        ),
        2,
    ]:

        if fragment_count > quota:

            continue


        seed_coords = farthest_point_select(
            pool_coords,
            pool_scores,
            fragment_count,
            spacing_xyz,
        )


        seeds = [
            tuple(
                int(
                    x
                )
                for x in coord
            )
            for coord in seed_coords
        ]


        base = (
            quota
            // fragment_count
        )

        remainder = (
            quota
            % fragment_count
        )


        fragment_quotas = [
            base
            + (
                1
                if index < remainder
                else 0
            )
            for index in range(
                fragment_count
            )
        ]


        used = set()

        clusters = []

        success = True


        for fragment_index, (
            seed,
            fragment_quota,
        ) in enumerate(
            zip(
                seeds,
                fragment_quotas,
            )
        ):

            future_seeds = set(
                seeds[
                    fragment_index
                    + 1:
                ]
            )


            forbidden = (
                used
                | future_seeds
            )


            cluster = _grow_fragment_cluster(
                pool_set=pool_set,
                seed=seed,
                quota=fragment_quota,
                spacing_xyz=spacing_xyz,
                forbidden=forbidden,
            )


            if cluster is None:

                success = False

                break


            clusters.append(
                cluster
            )

            used.update(
                cluster
            )


        if not success:

            continue


        union = set().union(
            *clusters
        )


        if len(
            union
        ) != quota:

            continue


        coords = np.asarray(
            sorted(
                union
            ),
            dtype=np.int32,
        )


        component_count = point_component_count(
            coords
        )


        if (
            component_count >= 2
            and component_count <= fragment_count
        ):

            return coords


    raise RuntimeError(
        "Could not construct spatially fragmented annotation."
    )


def geometry_select(
    geometry,
    component,
    quota,
    spacing_xyz,
):

    if geometry == "coherent":

        return coherent_select(
            component[
                "pool_coords"
            ],
            component[
                "pool_scores"
            ],
            quota,
        )


    if geometry == "dispersed":

        coords = farthest_point_select(
            component[
                "pool_coords"
            ],
            component[
                "pool_scores"
            ],
            quota,
            spacing_xyz,
        )


        if (
            quota >= 2
            and point_component_count(
                coords
            )
            < 2
        ):

            raise RuntimeError(
                "Dispersed geometry collapsed into one connected set."
            )


        return coords


    if geometry == "fragmented":

        return fragmented_select(
            component[
                "pool_coords"
            ],
            component[
                "pool_scores"
            ],
            quota,
            spacing_xyz,
        )


    raise ValueError(
        "Unknown geometry: "
        + str(
            geometry
        )
    )


def balanced_allocate(
    total_budget,
    component_order,
    capacity_lookup,
):

    total_budget = int(
        total_budget
    )

    component_order = [
        int(
            value
        )
        for value in component_order
    ]


    number_components = len(
        component_order
    )


    minimum_total = (
        MIN_NATIVE_POSITIVE_QUOTA_PER_COMPONENT
        * number_components
    )


    if total_budget < minimum_total:

        return None


    allocation = {
        component_id:
            MIN_NATIVE_POSITIVE_QUOTA_PER_COMPONENT
        for component_id in component_order
    }


    for component_id in component_order:

        if capacity_lookup[
            component_id
        ] < MIN_NATIVE_POSITIVE_QUOTA_PER_COMPONENT:

            return None


    remaining = (
        total_budget
        - minimum_total
    )


    while remaining > 0:

        progress = False


        for component_id in component_order:

            if remaining <= 0:

                break


            if (
                allocation[
                    component_id
                ]
                < capacity_lookup[
                    component_id
                ]
            ):

                allocation[
                    component_id
                ] += 1

                remaining -= 1

                progress = True


        if not progress:

            return None


    return allocation


print(
    "✓ Coherent algorithm                   : CONNECTED INTERIOR GROWTH"
)

print(
    "✓ Dispersed algorithm                  : PHYSICAL FARTHEST-POINT SAMPLING"
)

print(
    "✓ Fragmented algorithm                 : 2–3 SEPARATED COMPACT FRAGMENTS"
)

print(
    "✓ Per-component native quota ceiling   :",
    MAX_NATIVE_POSITIVE_QUOTA_PER_COMPONENT,
)

print(
    "✓ Per-component native quota minimum   :",
    MIN_NATIVE_POSITIVE_QUOTA_PER_COMPONENT,
)


# ==========================================================================================
# 5. PROCESS ALL 20 CASES
# ==========================================================================================

heading(
    "STEP 3/9 — GENERATE SIX NATIVE-SPACE CANDIDATE CELLS PER CASE"
)


if NATIVE_CANDIDATE_ROOT.exists():

    raise RuntimeError(
        "Native candidate directory already exists.\n"
        "Do not overwrite a prior Block-09C freeze."
    )


NATIVE_CANDIDATE_ROOT.mkdir(
    parents=True,
    exist_ok=False,
)


component_count_lookup = {
    str(
        row[
            "case_id"
        ]
    ):
        int(
            row[
                "eligible_components_26conn_ge0p1ml"
            ]
        )
    for _, row in component_summary_df.iterrows()
}


structural_lookup = {
    str(
        row[
            "case_id"
        ]
    ):
        row
    for _, row in structural_df.iterrows()
}


split_df[
    "case_id"
] = split_df[
    "case_id"
].astype(
    str
)


case_budget_rows = []

condition_rows = []

component_geometry_rows = []

artifact_rows = []


dense_lesion_arrays_accessed = 0

dense_lung_arrays_accessed = 0

final_dense_masks_used_for_annotation_generation = 0


def save_candidate_npz(
    path,
    *,
    case_id,
    condition_id,
    coverage,
    geometry,
    native_affine_xyz,
    fg_xyz,
    fg_component_ids,
    bg_xyz,
    selected_component_ids,
    selected_component_quotas,
):

    fg_xyz = lexical_sort_coords(
        fg_xyz
    )


    # Align component IDs after lexical sorting.
    original_pairs = {
        tuple(
            int(
                x
            )
            for x in coord
        ):
            int(
                component_id
            )
        for coord, component_id in zip(
            np.asarray(
                fg_xyz,
                dtype=np.int32,
            ),
            np.asarray(
                fg_component_ids,
                dtype=np.int32,
            ),
        )
    }


    # The dictionary construction above is safe only when fg_xyz passed in is
    # already aligned. We therefore rebuild using caller-supplied pairs below.
    # This helper receives arrays already sorted/aligned by the caller.
    del original_pairs


    supervision_xyz = np.concatenate(
        [
            fg_xyz,
            bg_xyz,
        ],
        axis=0,
    ).astype(
        np.int32
    )


    supervision_label = np.concatenate(
        [
            np.ones(
                len(
                    fg_xyz
                ),
                dtype=np.int8,
            ),
            np.zeros(
                len(
                    bg_xyz
                ),
                dtype=np.int8,
            ),
        ],
        axis=0,
    )


    supervision_world_xyz = nib.affines.apply_affine(
        np.asarray(
            native_affine_xyz,
            dtype=np.float64,
        ),
        supervision_xyz,
    ).astype(
        np.float64
    )


    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )


    np.savez_compressed(
        path,
        case_id=np.asarray(
            case_id
        ),
        condition_id=np.asarray(
            condition_id
        ),
        coverage_fraction=np.asarray(
            coverage,
            dtype=np.float32,
        ),
        geometry=np.asarray(
            geometry
        ),
        trainer_eligible=np.asarray(
            False,
            dtype=np.bool_,
        ),
        native_affine_xyz=np.asarray(
            native_affine_xyz,
            dtype=np.float64,
        ),
        supervision_voxel_xyz=supervision_xyz,
        supervision_world_xyz=supervision_world_xyz,
        supervision_label=supervision_label,
        foreground_voxel_xyz=np.asarray(
            fg_xyz,
            dtype=np.int32,
        ),
        foreground_component_id=np.asarray(
            fg_component_ids,
            dtype=np.int32,
        ),
        background_voxel_xyz=np.asarray(
            bg_xyz,
            dtype=np.int32,
        ),
        selected_component_ids=np.asarray(
            selected_component_ids,
            dtype=np.int32,
        ),
        selected_component_quotas=np.asarray(
            selected_component_quotas,
            dtype=np.int32,
        ),
    )


for _, split_row in tqdm(
    split_df.sort_values(
        "case_id"
    ).iterrows(),
    total=len(
        split_df
    ),
    desc="COVA-3D cases",
):

    case_id = str(
        split_row[
            "case_id"
        ]
    )


    infection_relative = str(
        split_row[
            "infection_mask"
        ]
    )


    lung_relative = str(
        split_row[
            "lung_mask"
        ]
    )


    infection_path = (
        primary_root
        / infection_relative
    )


    lung_path = (
        primary_root
        / lung_relative
    )


    for relative_path, path in [
        (
            infection_relative,
            infection_path,
        ),
        (
            lung_relative,
            lung_path,
        ),
    ]:

        observed_sha = sha256_file(
            path
        ).lower()


        expected_sha = source_hash_lookup[
            relative_path
        ]


        if observed_sha != expected_sha:

            raise RuntimeError(
                "Frozen source checksum mismatch:\n"
                + relative_path
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


    if infection_img.shape != lung_img.shape:

        raise RuntimeError(
            "Infection/lung shape mismatch for "
            + case_id
        )


    if not np.allclose(
        infection_img.affine,
        lung_img.affine,
        rtol=0,
        atol=1e-5,
    ):

        raise RuntimeError(
            "Infection/lung affine mismatch for "
            + case_id
        )


    lesion = (
        np.asarray(
            infection_img.dataobj
        )
        > 0.5
    )


    dense_lesion_arrays_accessed += 1


    lung = (
        np.asarray(
            lung_img.dataobj
        )
        > 0.5
    )


    dense_lung_arrays_accessed += 1


    if str(
        split_row[
            "role"
        ]
    ) == "final_outer_cv":

        final_dense_masks_used_for_annotation_generation += 1


    affine_xyz = np.asarray(
        infection_img.affine,
        dtype=np.float64,
    )


    spacing_xyz = np.linalg.norm(
        affine_xyz[
            :3,
            :3
        ],
        axis=0,
    )


    voxel_volume_ml = float(
        abs(
            np.linalg.det(
                affine_xyz[
                    :3,
                    :3
                ]
            )
        )
        / 1000.0
    )


    component_labels, raw_component_count = ndi.label(
        lesion,
        structure=STRUCT26,
    )


    component_sizes = np.bincount(
        component_labels.reshape(
            -1
        )
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
            + 1
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


    expected_eligible_count = component_count_lookup[
        case_id
    ]


    if len(
        eligible_component_ids
    ) != expected_eligible_count:

        raise RuntimeError(
            "Eligible-component count mismatch for "
            + case_id
            + "\nObserved="
            + str(
                len(
                    eligible_component_ids
                )
            )
            + " Expected="
            + str(
                expected_eligible_count
            )
        )


    component_slices = ndi.find_objects(
        component_labels
    )


    components = {}


    for component_id in eligible_component_ids:

        component_slice = component_slices[
            component_id
            - 1
        ]


        if component_slice is None:

            raise RuntimeError(
                "Missing component bounding box."
            )


        local_component = (
            component_labels[
                component_slice
            ]
            == component_id
        )


        offset_xyz = np.asarray(
            [
                component_slice[
                    axis
                ].start
                for axis in range(
                    3
                )
            ],
            dtype=np.int32,
        )


        pool = build_component_pool(
            local_component,
            spacing_xyz,
            offset_xyz,
        )


        components[
            component_id
        ] = {
            "native_id":
                component_id,

            "voxel_count":
                int(
                    component_sizes[
                        component_id
                    ]
                ),

            "volume_ml":
                float(
                    component_sizes[
                        component_id
                    ]
                    * voxel_volume_ml
                ),

            "pool_coords":
                pool[
                    "coords"
                ],

            "pool_scores":
                pool[
                    "scores"
                ],

            "pool_source":
                pool[
                    "source"
                ],

            "capacity":
                pool[
                    "capacity"
                ],
        }


    K = len(
        eligible_component_ids
    )


    expected_K = int(
        structural_lookup[
            case_id
        ][
            "eligible_components"
        ]
    )


    if K != expected_K:

        raise RuntimeError(
            "Structural manifest mismatch."
        )


    k50 = int(
        math.ceil(
            LOW_COVERAGE
            * K
        )
    )


    rng = np.random.default_rng(
        stable_seed(
            COMPONENT_SELECTION_MASTER_SEED,
            case_id,
            "component-selection",
        )
    )


    full_order = np.asarray(
        eligible_component_ids,
        dtype=np.int32,
    ).copy()


    rng.shuffle(
        full_order
    )


    order_100 = [
        int(
            value
        )
        for value in full_order.tolist()
    ]


    order_50 = order_100[
        :k50
    ]


    if not set(
        order_50
    ).issubset(
        set(
            order_100
        )
    ):

        raise RuntimeError(
            "Nested component-selection rule failed."
        )


    capacity_lookup = {
        component_id:
            int(
                components[
                    component_id
                ][
                    "capacity"
                ]
            )
        for component_id in order_100
    }


    maximum_native_budget = int(
        sum(
            capacity_lookup[
                component_id
            ]
            for component_id in order_50
        )
    )


    minimum_native_budget = int(
        MIN_NATIVE_POSITIVE_QUOTA_PER_COMPONENT
        * K
    )


    selection_cache = {}


    def cached_geometry(
        component_id,
        geometry,
        quota,
    ):

        key = (
            int(
                component_id
            ),
            str(
                geometry
            ),
            int(
                quota
            ),
        )


        if key not in selection_cache:

            selection_cache[
                key
            ] = geometry_select(
                geometry=geometry,
                component=components[
                    component_id
                ],
                quota=quota,
                spacing_xyz=spacing_xyz,
            )


        return selection_cache[
            key
        ]


    chosen_budget = None

    chosen_allocations = None

    chosen_coordinates = None


    for candidate_budget in range(
        maximum_native_budget,
        minimum_native_budget
        - 1,
        -1,
    ):

        allocation_50 = balanced_allocate(
            candidate_budget,
            order_50,
            capacity_lookup,
        )


        allocation_100 = balanced_allocate(
            candidate_budget,
            order_100,
            capacity_lookup,
        )


        if (
            allocation_50 is None
            or allocation_100 is None
        ):

            continue


        trial_coordinates = {}

        trial_ok = True


        try:

            for coverage, component_order, allocation in [
                (
                    0.50,
                    order_50,
                    allocation_50,
                ),
                (
                    1.00,
                    order_100,
                    allocation_100,
                ),
            ]:

                for geometry in GEOMETRIES:

                    geometry_component_coords = {}


                    for component_id in component_order:

                        quota = int(
                            allocation[
                                component_id
                            ]
                        )


                        coords = cached_geometry(
                            component_id,
                            geometry,
                            quota,
                        )


                        if len(
                            coords
                        ) != quota:

                            raise RuntimeError(
                                "Geometry quota mismatch."
                            )


                        connected_sets = point_component_count(
                            coords
                        )


                        if (
                            geometry == "coherent"
                            and connected_sets != 1
                        ):

                            raise RuntimeError(
                                "Coherent set not connected."
                            )


                        if (
                            geometry == "dispersed"
                            and quota >= 2
                            and connected_sets < 2
                        ):

                            raise RuntimeError(
                                "Dispersed set insufficiently separated."
                            )


                        if (
                            geometry == "fragmented"
                            and quota >= 2
                            and connected_sets < 2
                        ):

                            raise RuntimeError(
                                "Fragmented set insufficiently separated."
                            )


                        geometry_component_coords[
                            component_id
                        ] = coords


                    trial_coordinates[
                        (
                            coverage,
                            geometry,
                        )
                    ] = geometry_component_coords


        except RuntimeError:

            trial_ok = False


        if trial_ok:

            chosen_budget = int(
                candidate_budget
            )

            chosen_allocations = {
                0.50:
                    allocation_50,

                1.00:
                    allocation_100,
            }

            chosen_coordinates = (
                trial_coordinates
            )

            break


    if chosen_budget is None:

        raise RuntimeError(
            "No common six-cell native budget is feasible for "
            + case_id
        )


    # ----------------------------------------------------------------------
    # Identical BG realization for all six cells.
    # ----------------------------------------------------------------------

    background_exclusion_structure = physical_ball(
        BACKGROUND_EXCLUSION_MM,
        spacing_xyz,
    )


    lesion_band = ndi.binary_dilation(
        lesion,
        structure=background_exclusion_structure,
    )


    background_candidate = (
        lung
        & np.logical_not(
            lesion_band
        )
    )


    background_flat = np.flatnonzero(
        background_candidate.reshape(
            -1
        )
    )


    required_background = int(
        round(
            BACKGROUND_TO_FOREGROUND_RATIO
            * chosen_budget
        )
    )


    if len(
        background_flat
    ) < required_background:

        raise RuntimeError(
            "Insufficient BG capacity for "
            + case_id
        )


    bg_rng = np.random.default_rng(
        stable_seed(
            COMPONENT_SELECTION_MASTER_SEED,
            case_id,
            "background",
        )
    )


    selected_background_flat = bg_rng.choice(
        background_flat,
        size=required_background,
        replace=False,
    )


    background_xyz = np.column_stack(
        np.unravel_index(
            selected_background_flat,
            lesion.shape,
        )
    ).astype(
        np.int32
    )


    background_xyz = lexical_sort_coords(
        background_xyz
    )


    background_semantic_sha = semantic_array_hash(
        background_xyz
    )


    # ----------------------------------------------------------------------
    # Write six candidate artifacts.
    # ----------------------------------------------------------------------

    case_dir = (
        NATIVE_CANDIDATE_ROOT
        / case_id
    )


    case_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    condition_fg_counts = []

    condition_bg_hashes = []

    per_condition_component_sets = {}


    for coverage in COVERAGES:

        component_order = (
            order_50
            if np.isclose(
                coverage,
                0.50,
            )
            else order_100
        )


        allocation = chosen_allocations[
            coverage
        ]


        selected_ids_array = np.asarray(
            component_order,
            dtype=np.int32,
        )


        selected_quotas_array = np.asarray(
            [
                int(
                    allocation[
                        component_id
                    ]
                )
                for component_id in component_order
            ],
            dtype=np.int32,
        )


        for geometry in GEOMETRIES:

            condition_id = CONDITION_IDS[
                (
                    coverage,
                    geometry,
                )
            ]


            fg_coords_list = []

            fg_component_list = []


            for component_id in component_order:

                coords = chosen_coordinates[
                    (
                        coverage,
                        geometry,
                    )
                ][
                    component_id
                ]


                fg_coords_list.append(
                    coords
                )


                fg_component_list.append(
                    np.full(
                        len(
                            coords
                        ),
                        component_id,
                        dtype=np.int32,
                    )
                )


                component_geometry_rows.append(
                    {
                        "case_id":
                            case_id,

                        "role":
                            str(
                                split_row[
                                    "role"
                                ]
                            ),

                        "condition_id":
                            condition_id,

                        "coverage_fraction":
                            float(
                                coverage
                            ),

                        "geometry":
                            geometry,

                        "component_native_id":
                            int(
                                component_id
                            ),

                        "component_volume_ml":
                            float(
                                components[
                                    component_id
                                ][
                                    "volume_ml"
                                ]
                            ),

                        "pool_source":
                            components[
                                component_id
                            ][
                                "pool_source"
                            ],

                        "pool_size":
                            int(
                                len(
                                    components[
                                        component_id
                                    ][
                                        "pool_coords"
                                    ]
                                )
                            ),

                        "quota":
                            int(
                                allocation[
                                    component_id
                                ]
                            ),

                        "selected_point_components":
                            int(
                                point_component_count(
                                    coords
                                )
                            ),

                        "mean_nearest_neighbor_mm":
                            float(
                                mean_nearest_neighbor_mm(
                                    coords,
                                    spacing_xyz,
                                )
                            ),
                    }
                )


            foreground_xyz_unsorted = np.concatenate(
                fg_coords_list,
                axis=0,
            ).astype(
                np.int32
            )


            foreground_component_unsorted = np.concatenate(
                fg_component_list,
                axis=0,
            ).astype(
                np.int32
            )


            sort_order = np.lexsort(
                (
                    foreground_xyz_unsorted[
                        :,
                        2
                    ],
                    foreground_xyz_unsorted[
                        :,
                        1
                    ],
                    foreground_xyz_unsorted[
                        :,
                        0
                    ],
                )
            )


            foreground_xyz = (
                foreground_xyz_unsorted[
                    sort_order
                ]
            )


            foreground_component_ids = (
                foreground_component_unsorted[
                    sort_order
                ]
            )


            if len(
                foreground_xyz
            ) != chosen_budget:

                raise RuntimeError(
                    "Condition FG total differs from B_i."
                )


            if len(
                np.unique(
                    foreground_xyz,
                    axis=0,
                )
            ) != chosen_budget:

                raise RuntimeError(
                    "Duplicate FG coordinates detected."
                )


            if set(
                map(
                    tuple,
                    foreground_xyz.tolist(),
                )
            ) & set(
                map(
                    tuple,
                    background_xyz.tolist(),
                )
            ):

                raise RuntimeError(
                    "FG/BG overlap detected."
                )


            artifact_path = (
                case_dir
                / (
                    condition_id
                    + ".npz"
                )
            )


            # Save directly here so FG component IDs remain aligned.
            supervision_xyz = np.concatenate(
                [
                    foreground_xyz,
                    background_xyz,
                ],
                axis=0,
            ).astype(
                np.int32
            )


            supervision_label = np.concatenate(
                [
                    np.ones(
                        len(
                            foreground_xyz
                        ),
                        dtype=np.int8,
                    ),
                    np.zeros(
                        len(
                            background_xyz
                        ),
                        dtype=np.int8,
                    ),
                ],
                axis=0,
            )


            supervision_world_xyz = nib.affines.apply_affine(
                affine_xyz,
                supervision_xyz,
            ).astype(
                np.float64
            )


            np.savez_compressed(
                artifact_path,
                case_id=np.asarray(
                    case_id
                ),
                condition_id=np.asarray(
                    condition_id
                ),
                simulator_version=np.asarray(
                    SIMULATOR_VERSION
                ),
                coverage_fraction=np.asarray(
                    coverage,
                    dtype=np.float32,
                ),
                geometry=np.asarray(
                    geometry
                ),
                trainer_eligible=np.asarray(
                    False,
                    dtype=np.bool_,
                ),
                native_affine_xyz=affine_xyz,
                supervision_voxel_xyz=supervision_xyz,
                supervision_world_xyz=supervision_world_xyz,
                supervision_label=supervision_label,
                foreground_voxel_xyz=foreground_xyz,
                foreground_component_id=foreground_component_ids,
                background_voxel_xyz=background_xyz,
                selected_component_ids=selected_ids_array,
                selected_component_quotas=selected_quotas_array,
            )


            foreground_hash = semantic_array_hash(
                foreground_xyz,
                foreground_component_ids,
            )


            annotation_hash = semantic_array_hash(
                supervision_xyz,
                supervision_label,
                selected_ids_array,
                selected_quotas_array,
            )


            artifact_file_sha = sha256_file(
                artifact_path
            )


            condition_fg_counts.append(
                len(
                    foreground_xyz
                )
            )


            condition_bg_hashes.append(
                background_semantic_sha
            )


            per_condition_component_sets[
                condition_id
            ] = {
                "ids":
                    selected_ids_array.copy(),

                "quotas":
                    selected_quotas_array.copy(),
            }


            geometry_rows_for_condition = [
                row
                for row in component_geometry_rows
                if (
                    row[
                        "case_id"
                    ]
                    == case_id
                    and row[
                        "condition_id"
                    ]
                    == condition_id
                )
            ]


            mean_nn = float(
                np.mean(
                    [
                        row[
                            "mean_nearest_neighbor_mm"
                        ]
                        for row in geometry_rows_for_condition
                    ]
                )
            )


            mean_point_cc = float(
                np.mean(
                    [
                        row[
                            "selected_point_components"
                        ]
                        for row in geometry_rows_for_condition
                    ]
                )
            )


            condition_rows.append(
                {
                    "case_id":
                        case_id,

                    "source_subject_key":
                        str(
                            split_row[
                                "source_subject_key"
                            ]
                        ),

                    "role":
                        str(
                            split_row[
                                "role"
                            ]
                        ),

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
                        float(
                            coverage
                        ),

                    "geometry":
                        geometry,

                    "eligible_components":
                        K,

                    "selected_components":
                        len(
                            component_order
                        ),

                    "native_positive_budget_B_i":
                        chosen_budget,

                    "foreground_unique_voxels":
                        len(
                            foreground_xyz
                        ),

                    "background_unique_voxels":
                        len(
                            background_xyz
                        ),

                    "mean_component_annotation_nn_mm":
                        mean_nn,

                    "mean_annotation_connected_sets_per_component":
                        mean_point_cc,

                    "selected_component_ids":
                        ";".join(
                            str(
                                value
                            )
                            for value in selected_ids_array.tolist()
                        ),

                    "selected_component_quotas":
                        ";".join(
                            str(
                                value
                            )
                            for value in selected_quotas_array.tolist()
                        ),

                    "foreground_semantic_sha256":
                        foreground_hash,

                    "background_semantic_sha256":
                        background_semantic_sha,

                    "annotation_semantic_sha256":
                        annotation_hash,

                    "artifact_file":
                        str(
                            artifact_path.relative_to(
                                REPO
                            )
                        ),

                    "artifact_file_sha256":
                        artifact_file_sha,

                    "trainer_eligible":
                        False,
                }
            )


            artifact_rows.append(
                {
                    "case_id":
                        case_id,

                    "condition_id":
                        condition_id,

                    "artifact_file":
                        str(
                            artifact_path.relative_to(
                                REPO
                            )
                        ),

                    "file_sha256":
                        artifact_file_sha,

                    "semantic_sha256":
                        annotation_hash,

                    "trainer_eligible":
                        False,
                }
            )


    # ----------------------------------------------------------------------
    # Cross-cell matching assertions for this case.
    # ----------------------------------------------------------------------

    if set(
        condition_fg_counts
    ) != {
        chosen_budget
    }:

        raise RuntimeError(
            "Six-cell FG budget mismatch."
        )


    if len(
        set(
            condition_bg_hashes
        )
    ) != 1:

        raise RuntimeError(
            "BG realization differs across six cells."
        )


    for coverage_prefix in [
        "C50",
        "C100",
    ]:

        relevant = [
            condition_id
            for condition_id in CONDITION_IDS.values()
            if condition_id.startswith(
                coverage_prefix
            )
        ]


        reference_condition = relevant[
            0
        ]


        reference_ids = per_condition_component_sets[
            reference_condition
        ][
            "ids"
        ]


        reference_quotas = per_condition_component_sets[
            reference_condition
        ][
            "quotas"
        ]


        for condition_id in relevant[
            1:
        ]:

            if not np.array_equal(
                reference_ids,
                per_condition_component_sets[
                    condition_id
                ][
                    "ids"
                ],
            ):

                raise RuntimeError(
                    "Geometry conditions use different component sets."
                )


            if not np.array_equal(
                reference_quotas,
                per_condition_component_sets[
                    condition_id
                ][
                    "quotas"
                ],
            ):

                raise RuntimeError(
                    "Geometry conditions use different per-component quotas."
                )


    ids_50 = set(
        per_condition_component_sets[
            "C50_COH"
        ][
            "ids"
        ].tolist()
    )


    ids_100 = set(
        per_condition_component_sets[
            "C100_COH"
        ][
            "ids"
        ].tolist()
    )


    if not ids_50.issubset(
        ids_100
    ):

        raise RuntimeError(
            "50% components are not nested inside 100% components."
        )


    case_budget_rows.append(
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
                str(
                    split_row[
                        "source_origin"
                    ]
                ),

            "role":
                str(
                    split_row[
                        "role"
                    ]
                ),

            "eligible_components_K":
                K,

            "selected_components_50":
                len(
                    order_50
                ),

            "selected_components_100":
                len(
                    order_100
                ),

            "maximum_candidate_native_budget":
                maximum_native_budget,

            "frozen_native_budget_B_i":
                chosen_budget,

            "budget_reduction_from_max":
                int(
                    maximum_native_budget
                    - chosen_budget
                ),

            "background_budget":
                required_background,

            "background_candidates":
                int(
                    len(
                        background_flat
                    )
                ),

            "native_six_cell_feasible":
                True,

            "trainer_grid_feasibility":
                "PENDING_09D",

            "trainer_eligible":
                False,
        }
    )


    del lesion
    del lung
    del component_labels
    del background_candidate
    del background_flat
    del lesion_band
    del components

    gc.collect()


# ==========================================================================================
# 6. FREEZE MANIFESTS
# ==========================================================================================

heading(
    "STEP 4/9 — FREEZE NATIVE BUDGET + CONDITION MANIFESTS"
)


case_budget_df = pd.DataFrame(
    case_budget_rows
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


component_geometry_df = pd.DataFrame(
    component_geometry_rows
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
    case_budget_df
) != EXPECTED_CASES:

    raise RuntimeError(
        "Expected 20 case-budget rows."
    )


if len(
    condition_df
) != EXPECTED_ANNOTATION_ARTIFACTS:

    raise RuntimeError(
        "Expected 120 condition rows."
    )


if len(
    artifact_df
) != EXPECTED_ANNOTATION_ARTIFACTS:

    raise RuntimeError(
        "Expected 120 candidate artifacts."
    )


if not case_budget_df[
    "native_six_cell_feasible"
].astype(
    bool
).all():

    raise RuntimeError(
        "One or more cases failed native six-cell feasibility."
    )


manifest_dir = (
    REPO
    / "data/manifests"
)


case_budget_path = (
    manifest_dir
    / "cova3d_native_budget_v1_0.csv"
)


condition_manifest_path = (
    manifest_dir
    / "cova3d_native_condition_manifest_v1_0.csv"
)


geometry_manifest_path = (
    manifest_dir
    / "cova3d_native_geometry_component_audit_v1_0.csv"
)


artifact_manifest_path = (
    manifest_dir
    / "cova3d_native_candidate_artifact_manifest_v1_0.csv"
)


case_budget_df.to_csv(
    case_budget_path,
    index=False,
)


condition_df.to_csv(
    condition_manifest_path,
    index=False,
)


component_geometry_df.to_csv(
    geometry_manifest_path,
    index=False,
)


artifact_df.to_csv(
    artifact_manifest_path,
    index=False,
)


print(
    "✓ Case native budgets frozen           : 20/20"
)

print(
    "✓ Six-cell condition rows              : 120"
)

print(
    "✓ Sparse candidate artifacts           : 120"
)

print(
    "✓ All artifacts trainer eligible       : NO"
)


# ==========================================================================================
# 7. GLOBAL SIX-CELL QA
# ==========================================================================================

heading(
    "STEP 5/9 — GLOBAL SIX-CELL CAUSAL-MATCHING QA"
)


budget_count_failures = 0

background_hash_failures = 0

coverage_component_failures = 0

quota_pairing_failures = 0

trainer_eligible_failures = 0


for case_id, group in condition_df.groupby(
    "case_id"
):

    if group[
        "foreground_unique_voxels"
    ].nunique() != 1:

        budget_count_failures += 1


    if group[
        "background_semantic_sha256"
    ].nunique() != 1:

        background_hash_failures += 1


    if group[
        "trainer_eligible"
    ].astype(
        bool
    ).any():

        trainer_eligible_failures += 1


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

            coverage_component_failures += 1


        if subset[
            "selected_component_quotas"
        ].nunique() != 1:

            quota_pairing_failures += 1


if budget_count_failures:

    raise RuntimeError(
        "Global FG budget pairing failed."
    )


if background_hash_failures:

    raise RuntimeError(
        "Global BG pairing failed."
    )


if coverage_component_failures:

    raise RuntimeError(
        "Component-set pairing failed."
    )


if quota_pairing_failures:

    raise RuntimeError(
        "Per-component quota pairing failed."
    )


if trainer_eligible_failures:

    raise RuntimeError(
        "Candidate artifacts were incorrectly marked trainer eligible."
    )


# Geometry structural checks.
coherent_geometry = component_geometry_df[
    component_geometry_df[
        "geometry"
    ]
    == "coherent"
]


dispersed_geometry = component_geometry_df[
    component_geometry_df[
        "geometry"
    ]
    == "dispersed"
]


fragmented_geometry = component_geometry_df[
    component_geometry_df[
        "geometry"
    ]
    == "fragmented"
]


if not (
    coherent_geometry[
        "selected_point_components"
    ]
    == 1
).all():

    raise RuntimeError(
        "A coherent component annotation is disconnected."
    )


if not (
    dispersed_geometry[
        "selected_point_components"
    ]
    >= 2
).all():

    raise RuntimeError(
        "A dispersed component annotation is not spatially separated."
    )


if not (
    fragmented_geometry[
        "selected_point_components"
    ]
    >= 2
).all():

    raise RuntimeError(
        "A fragmented component annotation is not fragmented."
    )


mean_nn_by_geometry = (
    component_geometry_df.groupby(
        "geometry"
    )[
        "mean_nearest_neighbor_mm"
    ]
    .mean()
    .to_dict()
)


print(
    "✓ Exact same FG count / six cells     : PASS 20/20"
)

print(
    "✓ Exact same BG realization           : PASS 20/20"
)

print(
    "✓ Geometry component sets paired      : PASS"
)

print(
    "✓ Geometry per-component quotas paired: PASS"
)

print(
    "✓ 50% components nested in 100%       : PASS"
)

print(
    "✓ Coherent annotations connected      : PASS"
)

print(
    "✓ Dispersed annotations separated     : PASS"
)

print(
    "✓ Fragmented annotations separated    : PASS"
)

print()
print(
    "Mean nearest-neighbor distance by geometry:"
)

for geometry in GEOMETRIES:

    print(
        "  {:11s}: {:.3f} mm".format(
            geometry,
            float(
                mean_nn_by_geometry[
                    geometry
                ]
            ),
        )
    )


# ==========================================================================================
# 8. BUDGET SUMMARY
# ==========================================================================================

heading(
    "STEP 6/9 — SUMMARIZE NATIVE COMMON B_i"
)


native_budget_total = int(
    case_budget_df[
        "frozen_native_budget_B_i"
    ].sum()
)


native_budget_min = int(
    case_budget_df[
        "frozen_native_budget_B_i"
    ].min()
)


native_budget_median = float(
    case_budget_df[
        "frozen_native_budget_B_i"
    ].median()
)


native_budget_max = int(
    case_budget_df[
        "frozen_native_budget_B_i"
    ].max()
)


budget_reduction_cases = int(
    (
        case_budget_df[
            "budget_reduction_from_max"
        ]
        > 0
    ).sum()
)


development_budget_total = int(
    case_budget_df.loc[
        case_budget_df[
            "role"
        ]
        == "permanent_development",
        "frozen_native_budget_B_i",
    ].sum()
)


final_budget_total = int(
    case_budget_df.loc[
        case_budget_df[
            "role"
        ]
        == "final_outer_cv",
        "frozen_native_budget_B_i",
    ].sum()
)


print(
    "Native B_i range                      :",
    native_budget_min,
    "–",
    native_budget_max,
)

print(
    "Native B_i median                     :",
    "{:.1f}".format(
        native_budget_median
    ),
)

print(
    "Total native FG budget / one cell     :",
    native_budget_total,
)

print(
    "Development native FG budget / cell   :",
    development_budget_total,
)

print(
    "Final-CV native FG budget / cell      :",
    final_budget_total,
)

print(
    "Cases requiring reduction from max    :",
    budget_reduction_cases,
    "/20",
)

print(
    "Trainer-grid common budget            : PENDING 09D"
)


# ==========================================================================================
# 9. WRITE SIMULATOR CONFIG / SCIENTIFIC NOTE
# ==========================================================================================

heading(
    "STEP 7/9 — FREEZE SIMULATOR SPECIFICATION"
)


simulator_config = {
    "project":
        "COVA-3D",

    "simulator_version":
        SIMULATOR_VERSION,

    "status":
        "NATIVE_CANDIDATE_FROZEN",

    "protocol_version":
        PROTOCOL_VERSION,

    "source_dataset":
        PRIMARY_DATASET_ID,

    "component_definition": {
        "connectivity":
            PRIMARY_CONNECTIVITY,

        "minimum_volume_ml":
            REFERENCE_MIN_VOLUME_ML,
    },

    "coverage": {
        "levels": [
            0.50,
            1.00,
        ],

        "low_coverage_count":
            "ceil(0.5*K)",

        "selection":
            "uniform seeded permutation",

        "selection_seed":
            COMPONENT_SELECTION_MASTER_SEED,

        "nesting":
            "C50 subset of C100",
    },

    "native_budget": {
        "symbol":
            "B_i_native",

        "maximum_quota_per_selected_component":
            MAX_NATIVE_POSITIVE_QUOTA_PER_COMPONENT,

        "minimum_quota_per_selected_component":
            MIN_NATIVE_POSITIVE_QUOTA_PER_COMPONENT,

        "selection":
            (
                "largest feasible integer budget common to all six "
                "native-space cells"
            ),

        "same_total_fg_all_six":
            True,

        "same_component_quota_across_geometries_within_coverage":
            True,
    },

    "geometry": {
        "candidate_pool":
            (
                "largest connected component after 1mm physical erosion "
                "when >=20 voxels; otherwise full lesion-component fallback"
            ),

        "coherent":
            (
                "deterministic 26-connected compact interior growth "
                "from maximum-distance interior seed"
            ),

        "dispersed":
            (
                "deterministic physical farthest-point sampling from "
                "the same candidate pool"
            ),

        "fragmented":
            (
                "2–3 spatially separated compact connected fragments "
                "initialized by physical farthest-point seeds"
            ),
    },

    "background": {
        "candidate":
            "lung voxels outside 3mm physical lesion band",

        "count":
            "B_i_native",

        "selection":
            "deterministic uniform sample without replacement",

        "identical_across_six_cells":
            True,
    },

    "firewall": {
        "dense_masks_used_offline_for_annotation_generation":
            True,

        "candidate_artifacts_contain_dense_masks":
            False,

        "candidate_artifacts_trainer_eligible":
            False,

        "final_outer_cv_model_outcomes_accessed":
            False,
    },

    "mandatory_next_stage": {
        "block":
            "09D",

        "purpose":
            (
                "training-grid transfer, collision audit, and freeze "
                "B_i_train before any training"
            ),

        "native_budget_can_be_reduced":
            True,

        "reduction_basis":
            (
                "transfer uniqueness / causal matching only; "
                "never model performance"
            ),
    },

    "frozen_at_utc":
        NOW_ISO,
}


simulator_config_path = (
    REPO
    / "configs/"
    "cova3d_annotation_simulator_v2.yaml"
)


simulator_config_path.write_text(
    yaml.safe_dump(
        simulator_config,
        sort_keys=False,
        allow_unicode=True,
    ),
    encoding="utf-8",
)


scientific_note = f"""
# COVA-3D Block 09C — Annotation Simulator v2

## Status

Native-space six-cell annotation feasibility: **PASS**

Trainer-grid feasibility: **PENDING BLOCK 09D**

No model training has occurred.

## Factorial construction

For each case with `K` eligible lesion components:

- 50% coverage selects `ceil(0.5 K)` components;
- 100% coverage selects all `K`;
- the 50% set is a seeded nested subset of the 100% set.

Three within-instance geometries are constructed:

1. coherent;
2. dispersed;
3. fragmented.

Within a coverage level, all three geometries use the same selected
components and the same per-component positive quotas.

Across all six cells, the total foreground count equals a common
case-specific native budget `B_i_native`.

All six cells also reuse the identical background coordinates.

## Geometry definitions

**Coherent** annotations are deterministic 26-connected compact interior
traces.

**Dispersed** annotations use physical farthest-point sampling.

**Fragmented** annotations use two or three separated compact fragments.

All geometries are generated from the same component-specific candidate pool.

## Native annotation budget

A maximum of {MAX_NATIVE_POSITIVE_QUOTA_PER_COMPONENT} positive voxels per
selected component is permitted at the native-candidate stage.

At least {MIN_NATIVE_POSITIVE_QUOTA_PER_COMPONENT} voxels per selected
component are required so all three geometries remain meaningful.

The largest feasible exact common native budget is frozen independently for
each case.

## Critical training-grid safeguard

These artifacts are **not trainer eligible**.

Sparse points that are unique in native space may map to the same voxel after
resampling to the COVA training grid. Coherent annotations are particularly
susceptible because adjacent native voxels may collapse.

Therefore Block 09D must:

1. reconstruct the frozen corrected preprocessing/training grid;
2. map every candidate point by physical coordinates;
3. measure unique FG/BG counts after transfer;
4. verify that geometry conditions remain budget matched;
5. if necessary, lower the shared budget using a deterministic
   transfer-feasibility rule;
6. freeze `B_i_train`;
7. only then build trainer-eligible sparse caches.

No model outcome may influence that reduction.

## Firewall

Dense infection/lung masks were accessed only as offline annotation sources.

CT voxel arrays were not read.

Model predictions/checkpoints were not read.

Final outer-CV model outcomes remain unopened.

## Next

Block 09D — exact training-grid causal-matching and collision audit.
"""


scientific_note_path = (
    REPO
    / "docs/"
    "cova3d_annotation_simulator_v2.md"
)


write_text(
    scientific_note_path,
    scientific_note,
)


# ==========================================================================================
# 10. REGRESSION TESTS
# ==========================================================================================

heading(
    "STEP 8/9 — RUN ANNOTATION-SIMULATOR REGRESSION TESTS"
)


test_source = r'''
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_six_cells_per_case_and_equal_native_budget():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_native_condition_manifest_v1_0.csv"
    )

    assert frame[
        "case_id"
    ].nunique() == 20

    assert len(frame) == 120

    for _, group in frame.groupby(
        "case_id"
    ):

        assert len(group) == 6

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

        assert not group[
            "trainer_eligible"
        ].astype(bool).any()


def test_geometry_component_sets_and_quotas_are_paired():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_native_condition_manifest_v1_0.csv"
    )

    for _, case_group in frame.groupby(
        "case_id"
    ):

        for coverage in [
            0.5,
            1.0,
        ]:

            subset = case_group[
                np.isclose(
                    case_group[
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


def test_native_artifacts_have_no_dense_arrays():

    manifest = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_native_candidate_artifact_manifest_v1_0.csv"
    )

    forbidden_tokens = {
        "infection",
        "lesion_mask",
        "lung_mask",
        "dense_mask",
        "reference_mask",
    }

    expected_keys = {
        "case_id",
        "condition_id",
        "simulator_version",
        "coverage_fraction",
        "geometry",
        "trainer_eligible",
        "native_affine_xyz",
        "supervision_voxel_xyz",
        "supervision_world_xyz",
        "supervision_label",
        "foreground_voxel_xyz",
        "foreground_component_id",
        "background_voxel_xyz",
        "selected_component_ids",
        "selected_component_quotas",
    }

    for _, row in manifest.iterrows():

        path = ROOT / row[
            "artifact_file"
        ]

        assert path.exists()

        with np.load(
            path,
            allow_pickle=False,
        ) as data:

            assert set(
                data.files
            ) == expected_keys

            assert not bool(
                data[
                    "trainer_eligible"
                ]
            )

            joined = " ".join(
                data.files
            ).lower()

            for token in forbidden_tokens:

                assert token not in joined


def test_geometry_structural_distinction():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_native_geometry_component_audit_v1_0.csv"
    )

    coherent = frame[
        frame[
            "geometry"
        ]
        == "coherent"
    ]

    dispersed = frame[
        frame[
            "geometry"
        ]
        == "dispersed"
    ]

    fragmented = frame[
        frame[
            "geometry"
        ]
        == "fragmented"
    ]

    assert (
        coherent[
            "selected_point_components"
        ]
        == 1
    ).all()

    assert (
        dispersed[
            "selected_point_components"
        ]
        >= 2
    ).all()

    assert (
        fragmented[
            "selected_point_components"
        ]
        >= 2
    ).all()


def test_native_budget_table_is_fully_feasible():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_native_budget_v1_0.csv"
    )

    assert len(frame) == 20

    assert frame[
        "native_six_cell_feasible"
    ].astype(bool).all()

    assert not frame[
        "trainer_eligible"
    ].astype(bool).any()

    assert set(
        frame[
            "trainer_grid_feasibility"
        ]
    ) == {
        "PENDING_09D"
    }
'''


test_path = (
    REPO
    / "tests/"
    "test_cova3d_annotation_simulator_v2.py"
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
        "Block-09C regression tests FAILED."
    )


print(
    "✓ Annotation-simulator tests           : PASS"
)


# ==========================================================================================
# 11. UPDATE STATE / AUDIT
# ==========================================================================================

cova_state.update(
    {
        "last_completed_block":
            BLOCK,

        "current_stage":
            "NATIVE_ANNOTATION_GEOMETRY_FEASIBILITY_PASS",

        "next_block":
            "09D",

        "annotation_simulator_version":
            SIMULATOR_VERSION,

        "native_annotation_geometry_feasibility":
            "PASS",

        "native_six_cell_feasible_cases":
            EXPECTED_CASES,

        "native_candidate_artifacts":
            EXPECTED_ANNOTATION_ARTIFACTS,

        "native_budget_total_per_factorial_cell":
            native_budget_total,

        "native_budget_min":
            native_budget_min,

        "native_budget_median":
            native_budget_median,

        "native_budget_max":
            native_budget_max,

        "native_budget_reduction_cases":
            budget_reduction_cases,

        "trainer_grid_budget_status":
            "PENDING_09D",

        "trainer_grid_collision_audit":
            "NOT_RUN",

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

        "dense_masks_used_offline_for_annotation_generation":
            True,

        "final_outer_cv_dense_masks_used_for_annotation_generation":
            final_dense_masks_used_for_annotation_generation,

        "final_outer_cv_access_in_cova3d":
            0,

        "next_action":
            (
                "Run Block 09D training-grid transfer/collision audit. "
                "Do not train."
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
            "cova3d_annotation_simulator_v2",

        "active_research_track":
            TRACK_ID,

        "current_stage":
            "cova3d_native_annotation_geometry_feasibility_pass",

        "current_gate":
            "COVA_ANNOTATION_FEASIBILITY",

        "cova3d_annotation_simulator":
            "PASS_NATIVE_CANDIDATE_V2",

        "cova3d_native_six_cell_feasibility":
            "PASS",

        "cova3d_trainer_grid_feasibility":
            "PENDING_09D",

        "cova3d_native_candidate_artifacts":
            EXPECTED_ANNOTATION_ARTIFACTS,

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

        "gate_b":
            "NO_GO",

        "gate_c":
            "NOT_RUN",

        "training_authorized":
            False,

        "pilot_training_authorized":
            False,

        "next_action":
            (
                "Run Block 09D exact training-grid transfer and causal "
                "matching QA. Do not train."
            ),

        "updated_at_utc":
            NOW_ISO,
    }
)


write_json(
    project_state_path,
    project_state,
)


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
        "COVA-3D — CODE BLOCK 09C"
        in cell
    ):

        source_path = (
            REPO
            / "scripts/code_blocks/"
            "block09c_cova3d_annotation_simulator_v2.py"
        )

        source_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        source_path.write_text(
            cell,
            encoding="utf-8",
        )

        source_capture = (
            "PASS"
        )


except Exception:

    pass


audit_payload = {
    "project":
        "COVA-3D",

    "block":
        BLOCK,

    "status":
        "PASS",

    "generated_at_utc":
        NOW_ISO,

    "parent_commit":
        starting_commit,

    "simulator_version":
        SIMULATOR_VERSION,

    "purpose":
        (
            "implement three annotation geometries and freeze "
            "exact common native-space six-cell feasibility"
        ),

    "cases":
        EXPECTED_CASES,

    "conditions_per_case":
        EXPECTED_CONDITIONS_PER_CASE,

    "candidate_artifacts":
        EXPECTED_ANNOTATION_ARTIFACTS,

    "native_six_cell_feasible_cases":
        int(
            case_budget_df[
                "native_six_cell_feasible"
            ].sum()
        ),

    "native_budget": {
        "minimum":
            native_budget_min,

        "median":
            native_budget_median,

        "maximum":
            native_budget_max,

        "total_per_factorial_cell":
            native_budget_total,

        "development_total_per_cell":
            development_budget_total,

        "final_total_per_cell":
            final_budget_total,

        "cases_reduced_from_maximum":
            budget_reduction_cases,
    },

    "causal_matching": {
        "same_fg_count_across_six_cells":
            True,

        "same_bg_coordinates_across_six_cells":
            True,

        "same_component_set_across_geometry_within_coverage":
            True,

        "same_component_quota_across_geometry_within_coverage":
            True,

        "nested_50_in_100":
            True,
    },

    "geometry": {
        "coherent_connected":
            True,

        "dispersed_separated":
            True,

        "fragmented_separated":
            True,

        "mean_nearest_neighbor_mm":
            {
                key:
                    float(
                        value
                    )
                for key, value in mean_nn_by_geometry.items()
            },
    },

    "firewall": {
        "ct_arrays_accessed":
            0,

        "dense_lesion_arrays_accessed":
            dense_lesion_arrays_accessed,

        "dense_lung_arrays_accessed":
            dense_lung_arrays_accessed,

        "dense_masks_used_only_offline_for_annotation_generation":
            True,

        "prediction_arrays_accessed":
            0,

        "checkpoints_accessed":
            0,

        "development_model_outcomes_accessed":
            0,

        "final_model_outcomes_accessed":
            0,

        "final_outer_cv_dense_masks_used_for_annotation_generation":
            final_dense_masks_used_for_annotation_generation,
    },

    "trainer_eligibility": {
        "native_candidates_trainer_eligible":
            False,

        "reason":
            (
                "training-grid collision/equalization audit is mandatory "
                "before sparse artifacts may enter trainer cache"
            ),

        "next_block":
            "09D",
    },

    "new_training_performed":
        False,

    "optimizer_steps":
        0,

    "factorial_training_authorized":
        False,

    "method_development_authorized":
        False,

    "source_capture":
        source_capture,

    "next_action":
        (
            "Map all native sparse candidates through corrected v1.2 "
            "training geometry, detect geometry-specific voxel collapse, "
            "and freeze B_i_train before any training."
        ),
}


audit_path = (
    REPO
    / "experiments/audits/"
    "block09c_cova3d_annotation_simulator_v2.json"
)


write_json(
    audit_path,
    audit_payload,
)


# ==========================================================================================
# 12. REPOSITORY MANIFEST
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
            BLOCK,

        "active_track":
            TRACK_ID,

        "cora_gate_b":
            "NO_GO",

        "cova3d_native_geometry_feasibility":
            "PASS",

        "cova3d_trainer_grid_feasibility":
            "PENDING_09D",

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
# 13. COMMIT + PUSH
# ==========================================================================================

heading(
    "STEP 9/9 — COMMIT NATIVE ANNOTATION CANDIDATE FREEZE"
)


git_paths = [
    "COVA3D_STATE.json",
    "PROJECT_STATE.json",
    "REPOSITORY_MANIFEST.json",
    "configs/cova3d_annotation_simulator_v2.yaml",
    "data/cova3d_native_candidates_v1_0",
    "data/manifests/cova3d_native_budget_v1_0.csv",
    "data/manifests/cova3d_native_condition_manifest_v1_0.csv",
    "data/manifests/cova3d_native_geometry_component_audit_v1_0.csv",
    "data/manifests/cova3d_native_candidate_artifact_manifest_v1_0.csv",
    "docs/cova3d_annotation_simulator_v2.md",
    "experiments/audits/block09c_cova3d_annotation_simulator_v2.json",
    "tests/test_cova3d_annotation_simulator_v2.py",
]


source_path = (
    REPO
    / "scripts/code_blocks/"
    "block09c_cova3d_annotation_simulator_v2.py"
)


if source_path.exists():

    git_paths.append(
        "scripts/code_blocks/"
        "block09c_cova3d_annotation_simulator_v2.py"
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
        "No Block-09C artifacts available to commit."
    )


print(
    status
)


sh(
    [
        "git",
        "commit",
        "-m",
        "data: freeze COVA-3D native annotation geometry candidates",
    ],
    cwd=REPO,
)


safe_git_push(
    git_env,
    token,
)


final_commit = sh(
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
# 14. FINAL REPORT
# ==========================================================================================

print(
    "\n"
    + "=" * 126
)

print(
    "COVA-3D CODE BLOCK 09C — FINAL ANNOTATION-SIMULATOR REPORT"
)

print(
    "=" * 126
)


print(
    "Simulator version                     :",
    SIMULATOR_VERSION,
)

print(
    "Primary cases                         :",
    EXPECTED_CASES,
)

print(
    "Factorial cells / case                : 6"
)

print(
    "Sparse native candidate artifacts     :",
    EXPECTED_ANNOTATION_ARTIFACTS,
)

print()

print(
    "NATIVE SIX-CELL FEASIBILITY"
)

print(
    "---------------------------"
)

print(
    "Feasible cases                        :",
    int(
        case_budget_df[
            "native_six_cell_feasible"
        ].sum()
    ),
    "/20",
)

print(
    "Native B_i range                      :",
    native_budget_min,
    "–",
    native_budget_max,
)

print(
    "Native B_i median                     :",
    "{:.1f}".format(
        native_budget_median
    ),
)

print(
    "Native FG budget / one cell           :",
    native_budget_total,
)

print(
    "Development native FG budget / cell   :",
    development_budget_total,
)

print(
    "Final-CV native FG budget / cell      :",
    final_budget_total,
)

print(
    "Cases reduced from maximum budget     :",
    budget_reduction_cases,
    "/20",
)

print()

print(
    "CAUSAL MATCHING"
)

print(
    "---------------"
)

print(
    "Same FG count across six cells        : PASS 20/20"
)

print(
    "Same BG coordinates across six cells  : PASS 20/20"
)

print(
    "Same components across geometries     : PASS"
)

print(
    "Same quotas across geometries         : PASS"
)

print(
    "50% nested inside 100%                : PASS"
)

print()

print(
    "GEOMETRY QA"
)

print(
    "-----------"
)

print(
    "Coherent annotations connected        : PASS"
)

print(
    "Dispersed annotations separated       : PASS"
)

print(
    "Fragmented annotations separated      : PASS"
)

print(
    "Mean NN distance — coherent           :",
    "{:.3f} mm".format(
        float(
            mean_nn_by_geometry[
                "coherent"
            ]
        )
    ),
)

print(
    "Mean NN distance — dispersed          :",
    "{:.3f} mm".format(
        float(
            mean_nn_by_geometry[
                "dispersed"
            ]
        )
    ),
)

print(
    "Mean NN distance — fragmented         :",
    "{:.3f} mm".format(
        float(
            mean_nn_by_geometry[
                "fragmented"
            ]
        )
    ),
)

print()

print(
    "FIREWALL / TRAINING STATUS"
)

print(
    "--------------------------"
)

print(
    "CT arrays accessed                    : 0"
)

print(
    "Dense lesion arrays accessed          :",
    dense_lesion_arrays_accessed,
)

print(
    "Dense lung arrays accessed            :",
    dense_lung_arrays_accessed,
)

print(
    "Dense masks used only for simulation  : YES"
)

print(
    "Final-CV dense masks used for simulator:",
    final_dense_masks_used_for_annotation_generation,
)

print(
    "Final-CV MODEL outcomes accessed      : 0"
)

print(
    "Prediction arrays accessed            : 0"
)

print(
    "Checkpoint arrays accessed            : 0"
)

print(
    "New optimizer steps                   : 0"
)

print(
    "Native candidate trainer eligibility  : FALSE"
)

print(
    "Factorial training authorized         : NO"
)

print(
    "Method development authorized         : NO"
)

print()

print(
    "TRAINING-GRID SAFEGUARD"
)

print(
    "-----------------------"
)

print(
    "Native budget B_i frozen              : YES"
)

print(
    "Trainer-grid budget B_i_train frozen  : NO"
)

print(
    "Trainer-grid collision audit          : PENDING 09D"
)

print()

print(
    "Regression tests                      : PASS"
)

print(
    "Exact Block-09C source captured       :",
    source_capture,
)

print(
    "Starting commit                       :",
    starting_commit[:12],
)

print(
    "Final audit commit                    :",
    final_commit[:12],
)

print(
    "GitHub synchronization                : PASS"
)

print()

print(
    "NEXT:"
)

print(
    "Send me this COMPLETE report."
)

print(
    "Do NOT train anything."
)

print(
    "Next block: 09D — training-grid transfer, collision audit, "
    "exact post-transfer causal matching, and final B_i_train freeze."
)

print(
    "=" * 126
)