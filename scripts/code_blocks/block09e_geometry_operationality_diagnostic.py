# ==========================================================================================
# COVA-3D — BLOCK 09E-GEOM-OP-DIAG
# Pre-Outcome Geometry Operationality Diagnostic
#
# EXPECTED START COMMIT
# ---------------------
# 09822ec327f6
#
# PURPOSE
# -------
# Verify that the frozen factorial geometry interventions are ACTUALLY distinct
# at the final training-grid coordinate level.
#
# PRIMARY QUESTION
# ----------------
# For every GLOBAL_GEOMETRY_RESOLVABLE case × coverage × lesion component:
#
#     Are C*_COH, C*_DIS and C*_FRG distinct coordinate interventions?
#
# Particular concern:
#
#     quota = 2
#
# where dispersed and fragmented supervision may potentially collapse to the
# same coordinate set.
#
# DECISION RULE — PRE-SPECIFIED BEFORE READING RESULTS
# ----------------------------------------------------
# STRICT PASS requires:
#
# 1. 120/120 final sparse artifacts reproduce frozen hashes.
# 2. Same FG budget across all six cells within each case.
# 3. Same BG coordinates across all six cells within each case.
# 4. Same selected lesion set and same per-lesion quota across the three
#    geometries within a coverage level.
# 5. GLOBAL_RESOLUTION_LIMITED lesions:
#       • use identical coordinates across COH/DIS/FRG
#       • have geometry_claim_allowed == False
# 6. GLOBAL_GEOMETRY_RESOLVABLE lesions:
#       • COH has exactly 1 26-connected set
#       • DIS has >=2 26-connected sets
#       • FRG has >=2 26-connected sets
#       • COH != DIS
#       • COH != FRG
#       • DIS != FRG
# 7. No whole-case geometry pair is coordinate-identical within a coverage level.
#
# Any resolvable pairwise coordinate alias -> NO outcome opening.
#
# This block additionally computes:
#   • alias counts by quota, coverage and role
#   • pairwise coordinate-set Jaccard overlap
#   • actual 26-connected-set profiles
#   • mean nearest-neighbour physical spacing
#   • a NON-BINDING minimum-quota-3 feasibility screen
#
# FIREWALL
# --------
# NO CT arrays
# NO dense lesion masks
# NO lung masks
# NO predictions
# NO model/checkpoint loading
# NO optimizer step
# NO final outer-CV outcomes
#
# A FAIL is a scientific structural result, not a runtime failure.
# The diagnostic is committed either way.
# ==========================================================================================


from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import hashlib
import json
import math
import os
import subprocess
import sys
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from tqdm.auto import tqdm
from kaggle_secrets import UserSecretsClient


# ==========================================================================================
# 0. CONSTANTS
# ==========================================================================================

REPO = Path(
    "/kaggle/working/CORA-LUNG"
)

EXPECTED_START_COMMIT = (
    "09822ec327f6"
)

BLOCK = (
    "09E-GEOM-OP-DIAG"
)

EFFECTIVE_PROTOCOL = (
    "COVA3D_1.0+A1+A1.1+A1.2+N1+N2"
)

ARTIFACT_MANIFEST_PATH = (
    REPO
    / "data/manifests/"
    "cova3d_training_grid_candidate_artifact_manifest_v1_2.csv"
)

COMPONENT_AUDIT_PATH = (
    REPO
    / "data/manifests/"
    "cova3d_training_grid_component_audit_v1_2.csv"
)

CONDITION_MANIFEST_PATH = (
    REPO
    / "data/manifests/"
    "cova3d_training_grid_condition_manifest_v1_2.csv"
)

BUDGET_MANIFEST_PATH = (
    REPO
    / "data/manifests/"
    "cova3d_training_grid_budget_v1_2.csv"
)

GLOBAL_CLASS_PATH = (
    REPO
    / "data/manifests/"
    "cova3d_global_resolution_class_v1_0.csv"
)

SPLIT_PATH = (
    REPO
    / "data/splits/"
    "primary_volume_split_registry.csv"
)

COVA_STATE_PATH = (
    REPO
    / "COVA3D_STATE.json"
)

PROJECT_STATE_PATH = (
    REPO
    / "PROJECT_STATE.json"
)

AUDIT_DIR = (
    REPO
    / "experiments/audits"
)

MANIFEST_DIR = (
    REPO
    / "data/manifests"
)

FIGURE_DIR = (
    REPO
    / "figures/audit"
)

CONFIG_DIR = (
    REPO
    / "configs"
)

TARGET_SPACING_ZYX = np.asarray(
    [
        3.0,
        1.5,
        1.5,
    ],
    dtype=np.float64,
)

CONDITIONS = [
    "C50_COH",
    "C50_DIS",
    "C50_FRG",
    "C100_COH",
    "C100_DIS",
    "C100_FRG",
]

COVERAGES = [
    0.50,
    1.00,
]

GEOMETRIES = [
    "COH",
    "DIS",
    "FRG",
]

CONDITION_LOOKUP = {
    0.50:
        {
            "COH":
                "C50_COH",

            "DIS":
                "C50_DIS",

            "FRG":
                "C50_FRG",
        },

    1.00:
        {
            "COH":
                "C100_COH",

            "DIS":
                "C100_DIS",

            "FRG":
                "C100_FRG",
        },
}

GLOBAL_RESOLVABLE = (
    "GLOBAL_GEOMETRY_RESOLVABLE"
)

GLOBAL_LIMITED = (
    "GLOBAL_RESOLUTION_LIMITED"
)

EXPECTED_CASES = (
    20
)

EXPECTED_ARTIFACTS = (
    120
)

EXPECTED_DEV_CASES = (
    4
)

EXPECTED_FINAL_CASES = (
    16
)

EXPECTED_FINAL_MODEL_SHA = (
    "fce56f781cbd4ffd3e733cccb04c81f3b5bb2a664aceed4b8c0688e06ac4891c"
)

NOW_ISO = datetime.now(
    timezone.utc
).strftime(
    "%Y-%m-%dT%H:%M:%SZ"
)


# ==========================================================================================
# 1. PRE-SPECIFIED DECISION RULE
# ==========================================================================================

DECISION_RULE = {
    "rule_id":
        "COVA3D_GEOMETRY_OPERATIONALITY_V1",

    "timing":
        "PRE_DENSE_OUTCOME_OPENING",

    "population":
        (
            "All final training-grid sparse annotations from the 20-volume "
            "primary dataset."
        ),

    "unit":
        "case_x_coverage_x_selected_component",

    "strict_pass_requirements":
        {
            "artifact_integrity":
                "120_of_120",

            "same_case_budget_across_six_cells":
                True,

            "same_background_across_six_cells":
                True,

            "same_component_set_across_geometry_within_coverage":
                True,

            "same_component_quota_across_geometry_within_coverage":
                True,

            "resolution_limited_geometry_identity":
                True,

            "resolution_limited_geometry_claim_allowed":
                False,

            "resolvable_COH_connected_sets":
                1,

            "resolvable_DIS_min_connected_sets":
                2,

            "resolvable_FRG_min_connected_sets":
                2,

            "resolvable_COH_equals_DIS":
                0,

            "resolvable_COH_equals_FRG":
                0,

            "resolvable_DIS_equals_FRG":
                0,

            "whole_case_geometry_aliases":
                0,
        },

    "failure_action":
        (
            "Keep dense outcomes sealed and freeze a prospective annotation "
            "repair before any factorial comparison."
        ),

    "pass_action":
        (
            "Proceed to the already-planned dense development sanity evaluation."
        ),

    "minimum_quota_3_screen":
        (
            "Advisory only. It does not modify the current protocol."
        ),
}


# ==========================================================================================
# 2. HELPERS
# ==========================================================================================

def heading(text):

    print(
        "\n"
        + "=" * 132
    )

    print(
        text
    )

    print(
        "=" * 132
    )


def sh(
    cmd,
    *,
    env=None,
    check=True,
):

    result = subprocess.run(
        cmd,
        cwd=str(
            REPO
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
    ) as handle:

        for chunk in iter(
            lambda:
                handle.read(
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


def write_json(
    path,
    payload,
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
            payload,
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


def as_bool(value):

    if isinstance(
        value,
        (
            bool,
            np.bool_,
        ),
    ):

        return bool(
            value
        )

    return str(
        value
    ).strip().lower() in {
        "true",
        "1",
        "yes",
    }


def coord_set(coords):

    coords = np.asarray(
        coords,
        dtype=np.int32,
    )

    return frozenset(
        tuple(
            int(
                value
            )
            for value in row
        )
        for row in coords
    )


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


def connected_component_sizes(coords):

    remaining = set(
        coord_set(
            coords
        )
    )

    sizes = []


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

        size = (
            1
        )


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

                    stack.append(
                        neighbor
                    )

                    size += (
                        1
                    )


        sizes.append(
            size
        )


    return sorted(
        sizes,
        reverse=True,
    )


def jaccard(
    set_a,
    set_b,
):

    union = (
        set_a
        | set_b
    )

    if not union:

        return (
            1.0
        )

    return float(
        len(
            set_a
            & set_b
        )
        / len(
            union
        )
    )


def mean_nearest_neighbor_mm(coords):

    coords = np.asarray(
        coords,
        dtype=np.float64,
    )


    if len(
        coords
    ) < 2:

        return (
            np.nan
        )


    physical = (
        coords
        * TARGET_SPACING_ZYX[
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
            axis=2,
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


def parse_semicolon_ints(value):

    text = str(
        value
    ).strip()


    if (
        not text
        or text.lower()
        == "nan"
    ):

        return []


    return [
        int(
            token
        )
        for token in text.split(
            ";"
        )
        if token != ""
    ]


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
        "/tmp/cova3d_git_askpass_geom_op.sh"
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
# 3. VERIFY REPOSITORY / FIREWALL
# ==========================================================================================

heading(
    "COVA-3D 09E-GEOM-OP-DIAG — VERIFY PRE-OUTCOME STATE"
)


head = sh(
    [
        "git",
        "rev-parse",
        "HEAD",
    ]
).stdout.strip()


if not head.startswith(
    EXPECTED_START_COMMIT
):

    raise RuntimeError(
        "Unexpected Git HEAD.\n"
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
    ]
).stdout.strip()


if dirty:

    raise RuntimeError(
        "Repository must be clean before geometry operationality audit:\n"
        + dirty
    )


cova_state = json.loads(
    COVA_STATE_PATH.read_text(
        encoding="utf-8"
    )
)


project_state = json.loads(
    PROJECT_STATE_PATH.read_text(
        encoding="utf-8"
    )
)


if cova_state.get(
    "last_completed_block"
) != "09E-SANITY-FIT-R2":

    raise RuntimeError(
        "Expected completed R2 as previous scientific block."
    )


if cova_state.get(
    "next_block"
) != BLOCK:

    raise RuntimeError(
        "Project state does not authorize 09E-GEOM-OP-DIAG."
    )


if cova_state.get(
    "effective_protocol"
) != EFFECTIVE_PROTOCOL:

    raise RuntimeError(
        "Effective protocol mismatch."
    )


if int(
    cova_state.get(
        "optimizer_steps_retained_for_current_frozen_run",
        -1,
    )
) != 1000:

    raise RuntimeError(
        "Completed R2 retained-step count is not 1000."
    )


if str(
    cova_state.get(
        "sanity_fit_R2_final_model_sha256"
    )
) != EXPECTED_FINAL_MODEL_SHA:

    raise RuntimeError(
        "Frozen R2 model SHA changed."
    )


if cova_state.get(
    "dense_outcomes_opened_in_cova3d"
) is not False:

    raise RuntimeError(
        "Dense outcomes have already been opened."
    )


if int(
    cova_state.get(
        "final_outer_cv_access_in_cova3d",
        -1,
    )
) != 0:

    raise RuntimeError(
        "Final outer-CV access is not zero."
    )


if cova_state.get(
    "factorial_training_authorized"
) is not False:

    raise RuntimeError(
        "Factorial training must remain locked."
    )


print(
    "✓ Starting commit                      :",
    head[:12],
)

print(
    "✓ R2 retained optimizer steps          : 1000"
)

print(
    "✓ R2 frozen model SHA                  :",
    EXPECTED_FINAL_MODEL_SHA,
)

print(
    "✓ Dense outcomes                       : SEALED"
)

print(
    "✓ Final outer-CV access                : 0"
)

print(
    "✓ Factorial training                   : LOCKED"
)

print(
    "✓ CT arrays opened by this block       : 0"
)

print(
    "✓ Model/checkpoint opened              : 0"
)


# ==========================================================================================
# 4. WRITE DECISION RULE BEFORE ANALYSIS
# ==========================================================================================

heading(
    "STEP 1/10 — FREEZE DIAGNOSTIC DECISION RULE IN WORKING TREE"
)


CONFIG_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


decision_rule_path = (
    CONFIG_DIR
    / "cova3d_geometry_operationality_decision_rule_v1_0.yaml"
)


decision_rule_path.write_text(
    yaml.safe_dump(
        DECISION_RULE,
        sort_keys=False,
        allow_unicode=True,
    ),
    encoding="utf-8",
)


print(
    "✓ Decision rule ID                     :",
    DECISION_RULE[
        "rule_id"
    ],
)

print(
    "✓ Resolvable pairwise alias tolerance  : ZERO"
)

print(
    "✓ Whole-case geometry alias tolerance  : ZERO"
)

print(
    "✓ Resolution-limited identity expected : YES"
)


# ==========================================================================================
# 5. LOAD FROZEN SPARSE MANIFESTS
# ==========================================================================================

heading(
    "STEP 2/10 — LOAD FROZEN SPARSE MANIFESTS"
)


for required_path in [
    ARTIFACT_MANIFEST_PATH,
    COMPONENT_AUDIT_PATH,
    CONDITION_MANIFEST_PATH,
    BUDGET_MANIFEST_PATH,
    GLOBAL_CLASS_PATH,
    SPLIT_PATH,
]:

    if not required_path.exists():

        raise RuntimeError(
            "Required frozen manifest missing:\n"
            + str(
                required_path
            )
        )


artifact_manifest = pd.read_csv(
    ARTIFACT_MANIFEST_PATH
)


component_audit = pd.read_csv(
    COMPONENT_AUDIT_PATH
)


condition_manifest = pd.read_csv(
    CONDITION_MANIFEST_PATH
)


budget_manifest = pd.read_csv(
    BUDGET_MANIFEST_PATH
)


global_class_df = pd.read_csv(
    GLOBAL_CLASS_PATH
)


split_df = pd.read_csv(
    SPLIT_PATH
)


if len(
    artifact_manifest
) != EXPECTED_ARTIFACTS:

    raise RuntimeError(
        "Expected exactly 120 final sparse artifacts."
    )


cases = sorted(
    artifact_manifest[
        "case_id"
    ].astype(
        str
    ).unique().tolist()
)


if len(
    cases
) != EXPECTED_CASES:

    raise RuntimeError(
        "Expected exactly 20 primary cases."
    )


if sorted(
    artifact_manifest[
        "condition_id"
    ].astype(
        str
    ).unique().tolist()
) != sorted(
    CONDITIONS
):

    raise RuntimeError(
        "Unexpected factorial condition set."
    )


role_lookup = (
    split_df
    .drop_duplicates(
        "case_id"
    )
    .set_index(
        "case_id"
    )[
        "role"
    ]
    .astype(
        str
    )
    .to_dict()
)


dev_cases = sorted(
    [
        case_id
        for case_id in cases
        if role_lookup[
            case_id
        ]
        == "permanent_development"
    ]
)


final_cases = sorted(
    [
        case_id
        for case_id in cases
        if role_lookup[
            case_id
        ]
        == "final_outer_cv"
    ]
)


if len(
    dev_cases
) != EXPECTED_DEV_CASES:

    raise RuntimeError(
        "Development case count mismatch."
    )


if len(
    final_cases
) != EXPECTED_FINAL_CASES:

    raise RuntimeError(
        "Final outer-CV case count mismatch."
    )


print(
    "✓ Primary cases                        : 20"
)

print(
    "✓ Sparse artifacts                     : 120"
)

print(
    "✓ Permanent-development cases          : 4"
)

print(
    "✓ Final outer-CV cases                 : 16"
)


# ==========================================================================================
# 6. VERIFY ALL 120 ARTIFACT HASHES + LOAD ONLY SPARSE COORDINATES
# ==========================================================================================

heading(
    "STEP 3/10 — VERIFY 120 ARTIFACTS AND LOAD SPARSE COORDINATES"
)


artifact_data = {}


for _, row in tqdm(
    artifact_manifest.iterrows(),
    total=len(
        artifact_manifest
    ),
    desc="Sparse artifacts",
):

    case_id = str(
        row[
            "case_id"
        ]
    )

    condition_id = str(
        row[
            "condition_id"
        ]
    )

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
            "Sparse artifact missing:\n"
            + str(
                path
            )
        )


    observed_sha = sha256_file(
        path
    )


    expected_sha = str(
        row[
            "artifact_file_sha256"
        ]
    )


    if observed_sha != expected_sha:

        raise RuntimeError(
            "Sparse artifact SHA mismatch:\n"
            + str(
                path
            )
        )


    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        expected_keys = {
            "supervision_voxel_zyx",
            "supervision_label",
            "fg_membership_voxel_zyx",
            "fg_membership_group_id",
        }


        if not expected_keys.issubset(
            set(
                data.files
            )
        ):

            raise RuntimeError(
                "Final sparse artifact key mismatch:\n"
                + str(
                    path
                )
                + "\nObserved keys: "
                + str(
                    data.files
                )
            )


        direct_coords = np.asarray(
            data[
                "supervision_voxel_zyx"
            ],
            dtype=np.int32,
        )


        direct_labels = np.asarray(
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


    if len(
        direct_coords
    ) != len(
        direct_labels
    ):

        raise RuntimeError(
            "Direct sparse coordinate/label length mismatch."
        )


    if len(
        membership_coords
    ) != len(
        membership_groups
    ):

        raise RuntimeError(
            "FG membership coordinate/group length mismatch."
        )


    if not set(
        np.unique(
            direct_labels
        ).tolist()
    ).issubset(
        {
            0,
            1,
        }
    ):

        raise RuntimeError(
            "Unexpected direct supervision label."
        )


    if len(
        np.unique(
            membership_coords,
            axis=0,
        )
    ) != len(
        membership_coords
    ):

        raise RuntimeError(
            "Duplicate FG membership coordinates."
        )


    direct_fg = coord_set(
        direct_coords[
            direct_labels
            == 1
        ]
    )


    grouped_fg = coord_set(
        membership_coords
    )


    if direct_fg != grouped_fg:

        raise RuntimeError(
            "Direct FG and grouped FG semantic sets differ."
        )


    direct_bg = coord_set(
        direct_coords[
            direct_labels
            == 0
        ]
    )


    if direct_fg & direct_bg:

        raise RuntimeError(
            "FG/BG coordinate collision."
        )


    grouped_by_component = {}


    for component_id in sorted(
        np.unique(
            membership_groups
        ).tolist()
    ):

        component_id = int(
            component_id
        )


        grouped_by_component[
            component_id
        ] = np.asarray(
            membership_coords[
                membership_groups
                == component_id
            ],
            dtype=np.int32,
        )


    artifact_data[
        (
            case_id,
            condition_id,
        )
    ] = {
        "path":
            path,

        "direct_fg":
            direct_fg,

        "direct_bg":
            direct_bg,

        "grouped_by_component":
            grouped_by_component,

        "fg_count":
            len(
                direct_fg
            ),

        "bg_count":
            len(
                direct_bg
            ),
    }


print(
    "✓ SHA-verified artifacts               : 120 / 120"
)

print(
    "✓ Dense arrays loaded                  : 0"
)

print(
    "✓ CT arrays loaded                     : 0"
)

print(
    "✓ Sparse coordinate artifacts loaded   : 120"
)


# ==========================================================================================
# 7. BUILD LOOKUPS / VERIFY MANIFEST AGREEMENT
# ==========================================================================================

heading(
    "STEP 4/10 — VERIFY CASE / COVERAGE CAUSAL MATCHING"
)


component_lookup = {}


for _, row in component_audit.iterrows():

    key = (
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
        int(
            row[
                "component_native_id"
            ]
        ),
    )


    if key in component_lookup:

        raise RuntimeError(
            "Duplicate component-audit key."
        )


    component_lookup[
        key
    ] = row


condition_lookup = {}


for _, row in condition_manifest.iterrows():

    key = (
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
    )


    condition_lookup[
        key
    ] = row


causal_matching_rows = []


for case_id in cases:

    six = [
        condition_lookup[
            (
                case_id,
                condition_id,
            )
        ]
        for condition_id in CONDITIONS
    ]


    budgets = {
        int(
            row[
                "train_positive_budget_B_i"
            ]
        )
        for row in six
    }


    fg_counts = {
        int(
            row[
                "foreground_unique_voxels"
            ]
        )
        for row in six
    }


    bg_hashes = {
        str(
            row[
                "background_semantic_sha256"
            ]
        )
        for row in six
    }


    actual_bg_sets = {
        artifact_data[
            (
                case_id,
                condition_id,
            )
        ][
            "direct_bg"
        ]
        for condition_id in CONDITIONS
    }


    if len(
        budgets
    ) != 1:

        raise RuntimeError(
            "Case budget differs across six conditions."
        )


    if len(
        fg_counts
    ) != 1:

        raise RuntimeError(
            "Case FG count differs across six conditions."
        )


    if len(
        bg_hashes
    ) != 1:

        raise RuntimeError(
            "Frozen BG semantic hashes differ across six conditions."
        )


    if len(
        actual_bg_sets
    ) != 1:

        raise RuntimeError(
            "Actual BG coordinate sets differ across six conditions."
        )


    for coverage in COVERAGES:

        conditions = CONDITION_LOOKUP[
            coverage
        ]


        rows = [
            condition_lookup[
                (
                    case_id,
                    conditions[
                        geometry
                    ],
                )
            ]
            for geometry in GEOMETRIES
        ]


        selected_ids = [
            parse_semicolon_ints(
                row[
                    "selected_component_ids"
                ]
            )
            for row in rows
        ]


        quotas = [
            parse_semicolon_ints(
                row[
                    "selected_component_quotas"
                ]
            )
            for row in rows
        ]


        if not (
            selected_ids[
                0
            ]
            == selected_ids[
                1
            ]
            == selected_ids[
                2
            ]
        ):

            raise RuntimeError(
                "Selected component IDs differ across geometry."
            )


        if not (
            quotas[
                0
            ]
            == quotas[
                1
            ]
            == quotas[
                2
            ]
        ):

            raise RuntimeError(
                "Selected component quotas differ across geometry."
            )


        actual_groups = [
            sorted(
                artifact_data[
                    (
                        case_id,
                        conditions[
                            geometry
                        ],
                    )
                ][
                    "grouped_by_component"
                ].keys()
            )
            for geometry in GEOMETRIES
        ]


        if not (
            actual_groups[
                0
            ]
            == actual_groups[
                1
            ]
            == actual_groups[
                2
            ]
            == sorted(
                selected_ids[
                    0
                ]
            )
        ):

            raise RuntimeError(
                "Artifact component groups disagree with condition manifest."
            )


        causal_matching_rows.append(
            {
                "case_id":
                    case_id,

                "role":
                    role_lookup[
                        case_id
                    ],

                "coverage_fraction":
                    coverage,

                "selected_components":
                    len(
                        selected_ids[
                            0
                        ]
                    ),

                "budget":
                    int(
                        next(
                            iter(
                                budgets
                            )
                        )
                    ),

                "same_component_set":
                    True,

                "same_component_quota":
                    True,

                "same_background_all_six":
                    True,
            }
        )


print(
    "✓ Same FG budget across six cells      : 20 / 20 cases"
)

print(
    "✓ Same BG across six cells             : 20 / 20 cases"
)

print(
    "✓ Same component set across geometry   : 40 / 40 case-coverages"
)

print(
    "✓ Same component quota across geometry : 40 / 40 case-coverages"
)


# ==========================================================================================
# 8. COMPONENT-LEVEL OPERATIONALITY AUDIT
# ==========================================================================================

heading(
    "STEP 5/10 — COMPONENT-LEVEL GEOMETRY OPERATIONALITY"
)


component_rows = []

case_rows = []


for case_id in tqdm(
    cases,
    desc="Cases",
):

    role = role_lookup[
        case_id
    ]


    for coverage in COVERAGES:

        condition_ids = CONDITION_LOOKUP[
            coverage
        ]


        geometry_artifacts = {
            geometry:
                artifact_data[
                    (
                        case_id,
                        condition_ids[
                            geometry
                        ],
                    )
                ]
            for geometry in GEOMETRIES
        }


        component_ids = sorted(
            geometry_artifacts[
                "COH"
            ][
                "grouped_by_component"
            ].keys()
        )


        # ------------------------------------------------------------------
        # Whole-case FG geometry aliases.
        # ------------------------------------------------------------------

        whole_sets = {
            geometry:
                geometry_artifacts[
                    geometry
                ][
                    "direct_fg"
                ]
            for geometry in GEOMETRIES
        }


        case_coh_eq_dis = (
            whole_sets[
                "COH"
            ]
            == whole_sets[
                "DIS"
            ]
        )


        case_coh_eq_frg = (
            whole_sets[
                "COH"
            ]
            == whole_sets[
                "FRG"
            ]
        )


        case_dis_eq_frg = (
            whole_sets[
                "DIS"
            ]
            == whole_sets[
                "FRG"
            ]
        )


        case_alias_any = bool(
            case_coh_eq_dis
            or case_coh_eq_frg
            or case_dis_eq_frg
        )


        case_resolvable_aliases = (
            0
        )


        for component_id in component_ids:

            coords = {
                geometry:
                    geometry_artifacts[
                        geometry
                    ][
                        "grouped_by_component"
                    ][
                        component_id
                    ]
                for geometry in GEOMETRIES
            }


            sets = {
                geometry:
                    coord_set(
                        coords[
                            geometry
                        ]
                    )
                for geometry in GEOMETRIES
            }


            quotas = {
                geometry:
                    len(
                        sets[
                            geometry
                        ]
                    )
                for geometry in GEOMETRIES
            }


            if len(
                set(
                    quotas.values()
                )
            ) != 1:

                raise RuntimeError(
                    "Actual component quota differs across geometry."
                )


            quota = int(
                quotas[
                    "COH"
                ]
            )


            audit_rows = {
                geometry:
                    component_lookup[
                        (
                            case_id,
                            condition_ids[
                                geometry
                            ],
                            component_id,
                        )
                    ]
                for geometry in GEOMETRIES
            }


            classes = {
                str(
                    audit_rows[
                        geometry
                    ][
                        "global_resolution_class"
                    ]
                )
                for geometry in GEOMETRIES
            }


            if len(
                classes
            ) != 1:

                raise RuntimeError(
                    "Global resolution class differs across geometry."
                )


            resolution_class = next(
                iter(
                    classes
                )
            )


            for geometry in GEOMETRIES:

                audit_quota = int(
                    audit_rows[
                        geometry
                    ][
                        "quota"
                    ]
                )


                if audit_quota != quota:

                    raise RuntimeError(
                        "Component-audit quota differs from actual artifact quota."
                    )


            component_sizes = {
                geometry:
                    connected_component_sizes(
                        coords[
                            geometry
                        ]
                    )
                for geometry in GEOMETRIES
            }


            connected_sets = {
                geometry:
                    len(
                        component_sizes[
                            geometry
                        ]
                    )
                for geometry in GEOMETRIES
            }


            nn = {
                geometry:
                    mean_nearest_neighbor_mm(
                        coords[
                            geometry
                        ]
                    )
                for geometry in GEOMETRIES
            }


            coh_eq_dis = (
                sets[
                    "COH"
                ]
                == sets[
                    "DIS"
                ]
            )


            coh_eq_frg = (
                sets[
                    "COH"
                ]
                == sets[
                    "FRG"
                ]
            )


            dis_eq_frg = (
                sets[
                    "DIS"
                ]
                == sets[
                    "FRG"
                ]
            )


            pair_alias_any = bool(
                coh_eq_dis
                or coh_eq_frg
                or dis_eq_frg
            )


            j_coh_dis = jaccard(
                sets[
                    "COH"
                ],
                sets[
                    "DIS"
                ],
            )


            j_coh_frg = jaccard(
                sets[
                    "COH"
                ],
                sets[
                    "FRG"
                ],
            )


            j_dis_frg = jaccard(
                sets[
                    "DIS"
                ],
                sets[
                    "FRG"
                ],
            )


            claims = {
                geometry:
                    as_bool(
                        audit_rows[
                            geometry
                        ][
                            "geometry_claim_allowed"
                        ]
                    )
                for geometry in GEOMETRIES
            }


            if resolution_class == GLOBAL_LIMITED:

                limited_identity_pass = bool(
                    coh_eq_dis
                    and coh_eq_frg
                    and dis_eq_frg
                )


                geometry_claim_pass = bool(
                    not claims[
                        "COH"
                    ]
                    and not claims[
                        "DIS"
                    ]
                    and not claims[
                        "FRG"
                    ]
                )


                topology_pass = (
                    True
                )


                operational_distinct_pass = (
                    True
                )


            elif resolution_class == GLOBAL_RESOLVABLE:

                limited_identity_pass = (
                    True
                )


                geometry_claim_pass = bool(
                    claims[
                        "COH"
                    ]
                    and claims[
                        "DIS"
                    ]
                    and claims[
                        "FRG"
                    ]
                )


                topology_pass = bool(
                    connected_sets[
                        "COH"
                    ]
                    == 1
                    and connected_sets[
                        "DIS"
                    ]
                    >= 2
                    and connected_sets[
                        "FRG"
                    ]
                    >= 2
                )


                operational_distinct_pass = bool(
                    not pair_alias_any
                )


                if pair_alias_any:

                    case_resolvable_aliases += (
                        1
                    )


            else:

                raise RuntimeError(
                    "Unknown global resolution class: "
                    + resolution_class
                )


            component_pass = bool(
                limited_identity_pass
                and geometry_claim_pass
                and topology_pass
                and operational_distinct_pass
            )


            component_rows.append(
                {
                    "case_id":
                        case_id,

                    "role":
                        role,

                    "coverage_fraction":
                        coverage,

                    "component_native_id":
                        int(
                            component_id
                        ),

                    "resolution_class":
                        resolution_class,

                    "quota":
                        quota,

                    "coh_connected_sets":
                        connected_sets[
                            "COH"
                        ],

                    "dis_connected_sets":
                        connected_sets[
                            "DIS"
                        ],

                    "frg_connected_sets":
                        connected_sets[
                            "FRG"
                        ],

                    "coh_component_sizes":
                        ";".join(
                            str(
                                value
                            )
                            for value in component_sizes[
                                "COH"
                            ]
                        ),

                    "dis_component_sizes":
                        ";".join(
                            str(
                                value
                            )
                            for value in component_sizes[
                                "DIS"
                            ]
                        ),

                    "frg_component_sizes":
                        ";".join(
                            str(
                                value
                            )
                            for value in component_sizes[
                                "FRG"
                            ]
                        ),

                    "coh_mean_nn_mm":
                        nn[
                            "COH"
                        ],

                    "dis_mean_nn_mm":
                        nn[
                            "DIS"
                        ],

                    "frg_mean_nn_mm":
                        nn[
                            "FRG"
                        ],

                    "coh_equals_dis":
                        coh_eq_dis,

                    "coh_equals_frg":
                        coh_eq_frg,

                    "dis_equals_frg":
                        dis_eq_frg,

                    "any_pairwise_alias":
                        pair_alias_any,

                    "jaccard_coh_dis":
                        j_coh_dis,

                    "jaccard_coh_frg":
                        j_coh_frg,

                    "jaccard_dis_frg":
                        j_dis_frg,

                    "geometry_claim_COH":
                        claims[
                            "COH"
                        ],

                    "geometry_claim_DIS":
                        claims[
                            "DIS"
                        ],

                    "geometry_claim_FRG":
                        claims[
                            "FRG"
                        ],

                    "limited_identity_pass":
                        limited_identity_pass,

                    "geometry_claim_pass":
                        geometry_claim_pass,

                    "topology_pass":
                        topology_pass,

                    "operational_distinct_pass":
                        operational_distinct_pass,

                    "component_operationality_pass":
                        component_pass,

                    "common_capacity":
                        int(
                            audit_rows[
                                "COH"
                            ][
                                "common_capacity"
                            ]
                        ),
                }
            )


        case_rows.append(
            {
                "case_id":
                    case_id,

                "role":
                    role,

                "coverage_fraction":
                    coverage,

                "selected_components":
                    len(
                        component_ids
                    ),

                "case_coh_equals_dis":
                    case_coh_eq_dis,

                "case_coh_equals_frg":
                    case_coh_eq_frg,

                "case_dis_equals_frg":
                    case_dis_eq_frg,

                "whole_case_alias_any":
                    case_alias_any,

                "resolvable_components_with_any_alias":
                    case_resolvable_aliases,
            }
        )


component_df = pd.DataFrame(
    component_rows
)


case_df = pd.DataFrame(
    case_rows
)


resolvable_df = component_df[
    component_df[
        "resolution_class"
    ]
    == GLOBAL_RESOLVABLE
].copy()


limited_df = component_df[
    component_df[
        "resolution_class"
    ]
    == GLOBAL_LIMITED
].copy()


print(
    "✓ Component instances audited          :",
    len(
        component_df
    ),
)

print(
    "  Geometry-resolvable                  :",
    len(
        resolvable_df
    ),
)

print(
    "  Resolution-limited                   :",
    len(
        limited_df
    ),
)


# ==========================================================================================
# 9. SUMMARIZE EXACT GEOMETRY ALIASING
# ==========================================================================================

heading(
    "STEP 6/10 — EXACT GEOMETRY-ALIAS SUMMARY"
)


res_alias_coh_dis = int(
    resolvable_df[
        "coh_equals_dis"
    ].sum()
)


res_alias_coh_frg = int(
    resolvable_df[
        "coh_equals_frg"
    ].sum()
)


res_alias_dis_frg = int(
    resolvable_df[
        "dis_equals_frg"
    ].sum()
)


res_any_alias = int(
    resolvable_df[
        "any_pairwise_alias"
    ].sum()
)


whole_case_aliases = int(
    case_df[
        "whole_case_alias_any"
    ].sum()
)


limited_identity_failures = int(
    (
        ~limited_df[
            "limited_identity_pass"
        ]
    ).sum()
)


res_topology_failures = int(
    (
        ~resolvable_df[
            "topology_pass"
        ]
    ).sum()
)


claim_failures = int(
    (
        ~component_df[
            "geometry_claim_pass"
        ]
    ).sum()
)


quota_summary = (
    resolvable_df
    .groupby(
        [
            "quota",
        ],
        as_index=False,
    )
    .agg(
        resolvable_component_instances=(
            "component_native_id",
            "size",
        ),

        any_aliases=(
            "any_pairwise_alias",
            "sum",
        ),

        dis_equals_frg=(
            "dis_equals_frg",
            "sum",
        ),

        coh_equals_dis=(
            "coh_equals_dis",
            "sum",
        ),

        coh_equals_frg=(
            "coh_equals_frg",
            "sum",
        ),
    )
)


quota_summary[
    "alias_fraction"
] = (
    quota_summary[
        "any_aliases"
    ]
    / quota_summary[
        "resolvable_component_instances"
    ]
)


quota2 = resolvable_df[
    resolvable_df[
        "quota"
    ]
    == 2
]


quota2_count = int(
    len(
        quota2
    )
)


quota2_dis_frg_aliases = int(
    quota2[
        "dis_equals_frg"
    ].sum()
)


role_alias_summary = (
    resolvable_df
    .groupby(
        [
            "role",
            "coverage_fraction",
        ],
        as_index=False,
    )
    .agg(
        resolvable_component_instances=(
            "component_native_id",
            "size",
        ),

        any_aliases=(
            "any_pairwise_alias",
            "sum",
        ),

        dis_equals_frg=(
            "dis_equals_frg",
            "sum",
        ),
    )
)


print(
    "RESOLVABLE EXACT ALIASES"
)

print(
    "------------------------"
)

print(
    "COH == DIS                           :",
    res_alias_coh_dis,
)

print(
    "COH == FRG                           :",
    res_alias_coh_frg,
)

print(
    "DIS == FRG                           :",
    res_alias_dis_frg,
)

print(
    "Any pairwise alias                   :",
    res_any_alias,
    "/",
    len(
        resolvable_df
    ),
)

print()

print(
    "QUOTA = 2"
)

print(
    "---------"
)

print(
    "Resolvable q=2 instances             :",
    quota2_count,
)

print(
    "q=2 DIS == FRG                       :",
    quota2_dis_frg_aliases,
)

print()

print(
    "OTHER STRUCTURAL CHECKS"
)

print(
    "-----------------------"
)

print(
    "Whole-case geometry aliases          :",
    whole_case_aliases,
    "/",
    len(
        case_df
    ),
)

print(
    "Resolution-limited identity failures :",
    limited_identity_failures,
)

print(
    "Resolvable topology failures         :",
    res_topology_failures,
)

print(
    "Geometry-claim flag failures         :",
    claim_failures,
)

print()

print(
    "Quota-specific audit:"
)

print(
    quota_summary.to_string(
        index=False
    )
)

print()

print(
    "Role / coverage alias audit:"
)

print(
    role_alias_summary.to_string(
        index=False
    )
)


# ==========================================================================================
# 10. SPATIAL-DISTINCTION DESCRIPTIVES
# ==========================================================================================

heading(
    "STEP 7/10 — SPATIAL-DISTINCTION DESCRIPTIVES"
)


def safe_median(series):

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(
        dtype=np.float64
    )

    values = values[
        np.isfinite(
            values
        )
    ]


    if len(
        values
    ) == 0:

        return (
            np.nan
        )


    return float(
        np.median(
            values
        )
    )


jaccard_summary = {
    "COH_vs_DIS_median":
        safe_median(
            resolvable_df[
                "jaccard_coh_dis"
            ]
        ),

    "COH_vs_FRG_median":
        safe_median(
            resolvable_df[
                "jaccard_coh_frg"
            ]
        ),

    "DIS_vs_FRG_median":
        safe_median(
            resolvable_df[
                "jaccard_dis_frg"
            ]
        ),
}


nn_summary = {
    "COH_median_mm":
        safe_median(
            resolvable_df[
                "coh_mean_nn_mm"
            ]
        ),

    "DIS_median_mm":
        safe_median(
            resolvable_df[
                "dis_mean_nn_mm"
            ]
        ),

    "FRG_median_mm":
        safe_median(
            resolvable_df[
                "frg_mean_nn_mm"
            ]
        ),
}


print(
    "Pairwise coordinate-set Jaccard medians:"
)

for key, value in jaccard_summary.items():

    print(
        "  "
        + key.ljust(
            24
        )
        + ": "
        + (
            "nan"
            if not np.isfinite(
                value
            )
            else f"{value:.6f}"
        )
    )


print()

print(
    "Mean-nearest-neighbour distance medians:"
)

for key, value in nn_summary.items():

    print(
        "  "
        + key.ljust(
            24
        )
        + ": "
        + (
            "nan"
            if not np.isfinite(
                value
            )
            else f"{value:.6f} mm"
        )
    )


# ==========================================================================================
# 11. ADVISORY MINIMUM-QUOTA-3 FEASIBILITY SCREEN
# ==========================================================================================

heading(
    "STEP 8/10 — ADVISORY MINIMUM-QUOTA-3 FEASIBILITY SCREEN"
)


min3_rows = []


for case_id in cases:

    for coverage in COVERAGES:

        condition_id = CONDITION_LOOKUP[
            coverage
        ][
            "COH"
        ]


        subset = component_audit[
            (
                component_audit[
                    "case_id"
                ].astype(
                    str
                )
                == case_id
            )
            & (
                component_audit[
                    "condition_id"
                ].astype(
                    str
                )
                == condition_id
            )
        ].copy()


        if len(
            subset
        ) == 0:

            raise RuntimeError(
                "Missing component rows for min3 feasibility screen."
            )


        resolvable = subset[
            subset[
                "global_resolution_class"
            ]
            == GLOBAL_RESOLVABLE
        ]


        limited = subset[
            subset[
                "global_resolution_class"
            ]
            == GLOBAL_LIMITED
        ]


        min_required = int(
            3
            * len(
                resolvable
            )
            + 1
            * len(
                limited
            )
        )


        capacity_ok = bool(
            (
                resolvable[
                    "common_capacity"
                ].astype(
                    int
                )
                >= 3
            ).all()
        )


        budget = int(
            condition_lookup[
                (
                    case_id,
                    condition_id,
                )
            ][
                "train_positive_budget_B_i"
            ]
        )


        total_common_capacity = int(
            subset[
                "common_capacity"
            ].astype(
                int
            ).sum()
        )


        budget_lower_ok = bool(
            budget
            >= min_required
        )


        budget_upper_ok = bool(
            budget
            <= total_common_capacity
        )


        min3_feasible = bool(
            capacity_ok
            and budget_lower_ok
            and budget_upper_ok
        )


        low_capacity_components = (
            resolvable.loc[
                resolvable[
                    "common_capacity"
                ].astype(
                    int
                )
                < 3,
                "component_native_id",
            ]
            .astype(
                int
            )
            .tolist()
        )


        min3_rows.append(
            {
                "case_id":
                    case_id,

                "role":
                    role_lookup[
                        case_id
                    ],

                "coverage_fraction":
                    coverage,

                "resolvable_components":
                    len(
                        resolvable
                    ),

                "limited_components":
                    len(
                        limited
                    ),

                "current_budget":
                    budget,

                "min_required_if_resolvable_min3":
                    min_required,

                "total_common_capacity":
                    total_common_capacity,

                "all_resolvable_capacity_ge3":
                    capacity_ok,

                "budget_meets_new_minimum":
                    budget_lower_ok,

                "budget_within_common_capacity":
                    budget_upper_ok,

                "simple_min3_feasible":
                    min3_feasible,

                "resolvable_components_capacity_lt3":
                    ";".join(
                        str(
                            component_id
                        )
                        for component_id in low_capacity_components
                    ),
            }
        )


min3_df = pd.DataFrame(
    min3_rows
)


min3_feasible_count = int(
    min3_df[
        "simple_min3_feasible"
    ].sum()
)


min3_infeasible_count = int(
    (
        ~min3_df[
            "simple_min3_feasible"
        ]
    ).sum()
)


print(
    "Case-coverages simple min3 feasible    :",
    min3_feasible_count,
    "/",
    len(
        min3_df
    ),
)

print(
    "Case-coverages simple min3 infeasible  :",
    min3_infeasible_count,
)

print()

if min3_infeasible_count > 0:

    print(
        "Infeasible min3 screens:"
    )

    print(
        min3_df.loc[
            ~min3_df[
                "simple_min3_feasible"
            ]
        ].to_string(
            index=False
        )
    )


print()

print(
    "NOTE:"
)

print(
    "This is advisory only. No annotation coordinate or budget is changed "
    "in this diagnostic."
)


# ==========================================================================================
# 12. APPLY PRE-SPECIFIED STRUCTURAL DECISION
# ==========================================================================================

heading(
    "STEP 9/10 — APPLY PRE-SPECIFIED OPERATIONALITY DECISION"
)


strict_pass = bool(
    res_alias_coh_dis
    == 0
    and res_alias_coh_frg
    == 0
    and res_alias_dis_frg
    == 0
    and res_any_alias
    == 0
    and whole_case_aliases
    == 0
    and limited_identity_failures
    == 0
    and res_topology_failures
    == 0
    and claim_failures
    == 0
    and component_df[
        "component_operationality_pass"
    ].all()
)


if strict_pass:

    classification = (
        "PASS_STRICT_GEOMETRY_OPERATIONALITY"
    )

    next_block = (
        "09E-SANITY-EVAL"
    )

    next_action = (
        "Open only the four permanent-development dense lesion masks and "
        "evaluate the frozen R2 sanity checkpoint under the previously "
        "locked sanity gate."
    )

    dense_sanity_authorized = (
        True
    )

    annotation_ready = (
        True
    )

    coordinate_status = (
        "PASS_FINAL_COORDINATE_AND_OPERATIONALITY"
    )


else:

    classification = (
        "REQUIRES_PROSPECTIVE_GEOMETRY_AMENDMENT"
    )

    next_block = (
        "09E-GEOM-OP-REPAIR"
    )

    next_action = (
        "Keep all dense outcomes sealed. Diagnose and prospectively repair "
        "the frozen geometry intervention using only sparse annotation "
        "structure. Do not use segmentation outcomes."
    )

    dense_sanity_authorized = (
        False
    )

    annotation_ready = (
        False
    )

    coordinate_status = (
        "COORDINATE_FEASIBLE_BUT_OPERATIONAL_DISTINCTION_FAILED"
    )


print(
    "Geometry operationality classification :",
    classification,
)

print(
    "Resolvable component aliases            :",
    res_any_alias,
)

print(
    "Whole-case geometry aliases              :",
    whole_case_aliases,
)

print(
    "Resolution-limited identity failures    :",
    limited_identity_failures,
)

print(
    "Resolvable topology failures            :",
    res_topology_failures,
)

print(
    "Geometry-claim flag failures            :",
    claim_failures,
)

print()

print(
    "Dense sanity evaluation authorized      :",
    "YES"
    if dense_sanity_authorized
    else "NO",
)

print(
    "Factorial training authorized           : NO"
)

print(
    "Next block                              :",
    next_block,
)


# ==========================================================================================
# 13. SAVE TABLES
# ==========================================================================================

MANIFEST_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

AUDIT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

FIGURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


component_output_path = (
    MANIFEST_DIR
    / "cova3d_geometry_operationality_component_audit_v1_0.csv"
)


case_output_path = (
    MANIFEST_DIR
    / "cova3d_geometry_operationality_case_audit_v1_0.csv"
)


quota_output_path = (
    MANIFEST_DIR
    / "cova3d_geometry_operationality_quota_summary_v1_0.csv"
)


role_output_path = (
    MANIFEST_DIR
    / "cova3d_geometry_operationality_role_summary_v1_0.csv"
)


min3_output_path = (
    MANIFEST_DIR
    / "cova3d_geometry_min3_feasibility_screen_v1_0.csv"
)


causal_output_path = (
    MANIFEST_DIR
    / "cova3d_geometry_causal_matching_audit_v1_0.csv"
)


component_df.to_csv(
    component_output_path,
    index=False,
)


case_df.to_csv(
    case_output_path,
    index=False,
)


quota_summary.to_csv(
    quota_output_path,
    index=False,
)


role_alias_summary.to_csv(
    role_output_path,
    index=False,
)


min3_df.to_csv(
    min3_output_path,
    index=False,
)


pd.DataFrame(
    causal_matching_rows
).to_csv(
    causal_output_path,
    index=False,
)


# ==========================================================================================
# 14. PUBLICATION/AUDIT FIGURES
# ==========================================================================================

jaccard_long = pd.DataFrame(
    {
        "COH–DIS":
            resolvable_df[
                "jaccard_coh_dis"
            ].to_numpy(
                dtype=np.float64
            ),

        "COH–FRG":
            resolvable_df[
                "jaccard_coh_frg"
            ].to_numpy(
                dtype=np.float64
            ),

        "DIS–FRG":
            resolvable_df[
                "jaccard_dis_frg"
            ].to_numpy(
                dtype=np.float64
            ),
    }
)


fig, ax = plt.subplots(
    figsize=(
        9.5,
        6.0,
    )
)


ax.boxplot(
    [
        jaccard_long[
            "COH–DIS"
        ].dropna().to_numpy(),

        jaccard_long[
            "COH–FRG"
        ].dropna().to_numpy(),

        jaccard_long[
            "DIS–FRG"
        ].dropna().to_numpy(),
    ],
    labels=[
        "COH–DIS",
        "COH–FRG",
        "DIS–FRG",
    ],
    showfliers=True,
)


ax.axhline(
    1.0,
    linewidth=1.0,
    linestyle="--",
)


ax.set_ylim(
    -0.03,
    1.05,
)


ax.set_ylabel(
    "Coordinate-set Jaccard overlap"
)


ax.set_xlabel(
    "Geometry pair"
)


ax.set_title(
    "COVA-3D Geometry Operationality\n"
    "Resolvable Lesion Components Only"
)


ax.grid(
    axis="y",
    alpha=0.25,
)


fig.tight_layout()


jaccard_png = (
    FIGURE_DIR
    / "fig_cova3d_geometry_operationality_jaccard_v1_0.png"
)


jaccard_pdf = (
    FIGURE_DIR
    / "fig_cova3d_geometry_operationality_jaccard_v1_0.pdf"
)


fig.savefig(
    jaccard_png,
    dpi=600,
    bbox_inches="tight",
)


fig.savefig(
    jaccard_pdf,
    bbox_inches="tight",
)


plt.close(
    fig
)


# ------------------------------------------------------------------------------------------
# Alias rate by quota.
# ------------------------------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(
        10.0,
        5.8,
    )
)


if len(
    quota_summary
) > 0:

    ax.bar(
        quota_summary[
            "quota"
        ].astype(
            str
        ),
        quota_summary[
            "alias_fraction"
        ],
    )


ax.set_ylim(
    0.0,
    max(
        0.05,
        float(
            quota_summary[
                "alias_fraction"
            ].max()
        )
        * 1.15
        if len(
            quota_summary
        )
        else 0.05,
    ),
)


ax.set_xlabel(
    "Per-component foreground quota"
)


ax.set_ylabel(
    "Fraction with any exact geometry alias"
)


ax.set_title(
    "COVA-3D Exact Geometry Aliasing by Sparse Quota"
)


ax.grid(
    axis="y",
    alpha=0.25,
)


fig.tight_layout()


alias_png = (
    FIGURE_DIR
    / "fig_cova3d_geometry_alias_by_quota_v1_0.png"
)


alias_pdf = (
    FIGURE_DIR
    / "fig_cova3d_geometry_alias_by_quota_v1_0.pdf"
)


fig.savefig(
    alias_png,
    dpi=600,
    bbox_inches="tight",
)


fig.savefig(
    alias_pdf,
    bbox_inches="tight",
)


plt.close(
    fig
)


# ==========================================================================================
# 15. AUDIT JSON
# ==========================================================================================

audit_path = (
    AUDIT_DIR
    / "block09e_geometry_operationality_diagnostic.json"
)


audit_payload = {
    "project":
        "COVA-3D",

    "block":
        BLOCK,

    "parent_commit":
        head,

    "effective_protocol":
        EFFECTIVE_PROTOCOL,

    "timing":
        "AFTER_R2_TRAINING_BEFORE_ANY_DENSE_COVA_OUTCOME",

    "classification":
        classification,

    "decision_rule":
        DECISION_RULE,

    "population":
        {
            "cases":
                len(
                    cases
                ),

            "development_cases":
                len(
                    dev_cases
                ),

            "final_outer_cv_cases":
                len(
                    final_cases
                ),

            "sparse_artifacts":
                len(
                    artifact_manifest
                ),

            "component_instances":
                len(
                    component_df
                ),

            "resolvable_component_instances":
                len(
                    resolvable_df
                ),

            "limited_component_instances":
                len(
                    limited_df
                ),
        },

    "integrity":
        {
            "artifact_hashes":
                "PASS_120_OF_120",

            "same_budget_across_six_cells":
                True,

            "same_background_across_six_cells":
                True,

            "same_component_set_across_geometry":
                True,

            "same_component_quota_across_geometry":
                True,
        },

    "resolvable_aliases":
        {
            "COH_equals_DIS":
                res_alias_coh_dis,

            "COH_equals_FRG":
                res_alias_coh_frg,

            "DIS_equals_FRG":
                res_alias_dis_frg,

            "any_pairwise_alias":
                res_any_alias,

            "quota2_instances":
                quota2_count,

            "quota2_DIS_equals_FRG":
                quota2_dis_frg_aliases,
        },

    "case_level":
        {
            "case_coverage_units":
                len(
                    case_df
                ),

            "whole_case_geometry_aliases":
                whole_case_aliases,
        },

    "structural_checks":
        {
            "limited_identity_failures":
                limited_identity_failures,

            "resolvable_topology_failures":
                res_topology_failures,

            "geometry_claim_flag_failures":
                claim_failures,
        },

    "jaccard_medians":
        jaccard_summary,

    "mean_nearest_neighbor_medians_mm":
        nn_summary,

    "minimum_quota_3_screen":
        {
            "case_coverage_units":
                len(
                    min3_df
                ),

            "simple_feasible":
                min3_feasible_count,

            "simple_infeasible":
                min3_infeasible_count,

            "binding":
                False,
        },

    "R2":
        {
            "already_completed":
                True,

            "retained_optimizer_steps":
                1000,

            "final_model_sha256":
                EXPECTED_FINAL_MODEL_SHA,

            "model_loaded_in_this_diagnostic":
                False,

            "retraining_performed":
                False,
        },

    "firewall":
        {
            "CT_arrays_accessed":
                0,

            "dense_lesion_masks_accessed":
                0,

            "dense_lung_masks_accessed":
                0,

            "predictions_accessed":
                0,

            "checkpoints_accessed":
                0,

            "optimizer_steps":
                0,

            "final_outer_cv_outcome_access":
                0,
        },

    "authorization_after_diagnostic":
        {
            "dense_sanity_evaluation":
                dense_sanity_authorized,

            "factorial_training":
                False,

            "method_development":
                False,

            "final_outer_cv_outcomes":
                False,
        },

    "next_block":
        next_block,

    "generated_at_utc":
        NOW_ISO,
}


write_json(
    audit_path,
    audit_payload,
)


# ==========================================================================================
# 16. UPDATE PROJECT STATE
# ==========================================================================================

cova_state.update(
    {
        "last_completed_block":
            BLOCK,

        "current_stage":
            (
                "GEOMETRY_OPERATIONALITY_PASS_DENSE_SANITY_PENDING"
                if strict_pass
                else "GEOMETRY_OPERATIONALITY_REPAIR_REQUIRED"
            ),

        "geometry_operationality_status":
            classification,

        "geometry_operationality_resolvable_component_instances":
            len(
                resolvable_df
            ),

        "geometry_operationality_limited_component_instances":
            len(
                limited_df
            ),

        "geometry_operationality_COH_DIS_aliases":
            res_alias_coh_dis,

        "geometry_operationality_COH_FRG_aliases":
            res_alias_coh_frg,

        "geometry_operationality_DIS_FRG_aliases":
            res_alias_dis_frg,

        "geometry_operationality_any_resolvable_aliases":
            res_any_alias,

        "geometry_operationality_quota2_instances":
            quota2_count,

        "geometry_operationality_quota2_DIS_FRG_aliases":
            quota2_dis_frg_aliases,

        "geometry_operationality_whole_case_aliases":
            whole_case_aliases,

        "geometry_operationality_min3_simple_feasible_case_coverages":
            min3_feasible_count,

        "geometry_operationality_min3_simple_infeasible_case_coverages":
            min3_infeasible_count,

        "primary_dataset_coordinate_geometry_feasibility":
            coordinate_status,

        "annotation_methodology_structurally_ready":
            annotation_ready,

        "sanity_dense_evaluation_authorized_after_fit":
            dense_sanity_authorized,

        "sanity_gate_status":
            "EVALUATION_PENDING",

        "dense_outcomes_opened_in_cova3d":
            False,

        "factorial_training_authorized":
            False,

        "method_development_authorized":
            False,

        "final_outer_cv_outcomes_authorized":
            False,

        "final_outer_cv_access_in_cova3d":
            0,

        "next_block":
            next_block,

        "next_action":
            next_action,

        "updated_at_utc":
            NOW_ISO,
    }
)


write_json(
    COVA_STATE_PATH,
    cova_state,
)


project_state.update(
    {
        "last_completed_block":
            BLOCK,

        "last_completed_block_name":
            "cova3d_geometry_operationality_diagnostic",

        "current_stage":
            (
                "cova3d_dense_sanity_evaluation_pending"
                if strict_pass
                else "cova3d_geometry_operationality_repair_required"
            ),

        "cova3d_geometry_operationality":
            classification,

        "cova3d_dense_outcomes_opened":
            False,

        "cova3d_factorial_training_authorized":
            False,

        "cova3d_final_outer_cv_access":
            0,

        "next_action":
            next_action,

        "updated_at_utc":
            NOW_ISO,
    }
)


write_json(
    PROJECT_STATE_PATH,
    project_state,
)


# ==========================================================================================
# 17. REGRESSION TEST
# ==========================================================================================

test_path = (
    REPO
    / "tests/"
    "test_cova3d_geometry_operationality.py"
)


write_text(
    test_path,
    r'''
from pathlib import Path
import json

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_geometry_decision_rule_is_strict():

    cfg = yaml.safe_load(
        (
            ROOT
            / "configs/"
            "cova3d_geometry_operationality_decision_rule_v1_0.yaml"
        ).read_text(
            encoding="utf-8"
        )
    )

    req = cfg[
        "strict_pass_requirements"
    ]

    assert req[
        "resolvable_COH_equals_DIS"
    ] == 0

    assert req[
        "resolvable_COH_equals_FRG"
    ] == 0

    assert req[
        "resolvable_DIS_equals_FRG"
    ] == 0

    assert req[
        "whole_case_geometry_aliases"
    ] == 0


def test_geometry_audit_firewall():

    audit = json.loads(
        (
            ROOT
            / "experiments/audits/"
            "block09e_geometry_operationality_diagnostic.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    firewall = audit[
        "firewall"
    ]

    assert firewall[
        "CT_arrays_accessed"
    ] == 0

    assert firewall[
        "dense_lesion_masks_accessed"
    ] == 0

    assert firewall[
        "dense_lung_masks_accessed"
    ] == 0

    assert firewall[
        "predictions_accessed"
    ] == 0

    assert firewall[
        "checkpoints_accessed"
    ] == 0

    assert firewall[
        "optimizer_steps"
    ] == 0

    assert firewall[
        "final_outer_cv_outcome_access"
    ] == 0

    assert audit[
        "authorization_after_diagnostic"
    ][
        "factorial_training"
    ] is False


def test_geometry_component_table_consistent_with_classification():

    audit = json.loads(
        (
            ROOT
            / "experiments/audits/"
            "block09e_geometry_operationality_diagnostic.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_geometry_operationality_component_audit_v1_0.csv"
    )

    resolvable = frame[
        frame[
            "resolution_class"
        ]
        == "GLOBAL_GEOMETRY_RESOLVABLE"
    ]

    aliases = int(
        resolvable[
            "any_pairwise_alias"
        ].sum()
    )

    assert aliases == audit[
        "resolvable_aliases"
    ][
        "any_pairwise_alias"
    ]

    classification = audit[
        "classification"
    ]

    if classification == "PASS_STRICT_GEOMETRY_OPERATIONALITY":

        assert aliases == 0

        assert audit[
            "case_level"
        ][
            "whole_case_geometry_aliases"
        ] == 0

        assert audit[
            "structural_checks"
        ][
            "limited_identity_failures"
        ] == 0

        assert audit[
            "structural_checks"
        ][
            "resolvable_topology_failures"
        ] == 0

        assert audit[
            "authorization_after_diagnostic"
        ][
            "dense_sanity_evaluation"
        ] is True

        assert audit[
            "next_block"
        ] == "09E-SANITY-EVAL"

    else:

        assert (
            aliases > 0
            or audit[
                "case_level"
            ][
                "whole_case_geometry_aliases"
            ] > 0
            or audit[
                "structural_checks"
            ][
                "limited_identity_failures"
            ] > 0
            or audit[
                "structural_checks"
            ][
                "resolvable_topology_failures"
            ] > 0
            or audit[
                "structural_checks"
            ][
                "geometry_claim_flag_failures"
            ] > 0
        )

        assert audit[
            "authorization_after_diagnostic"
        ][
            "dense_sanity_evaluation"
        ] is False

        assert audit[
            "next_block"
        ] == "09E-GEOM-OP-REPAIR"


def test_state_matches_geometry_audit():

    audit = json.loads(
        (
            ROOT
            / "experiments/audits/"
            "block09e_geometry_operationality_diagnostic.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    state = json.loads(
        (
            ROOT
            / "COVA3D_STATE.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert state[
        "last_completed_block"
    ] == "09E-GEOM-OP-DIAG"

    assert state[
        "geometry_operationality_status"
    ] == audit[
        "classification"
    ]

    assert state[
        "dense_outcomes_opened_in_cova3d"
    ] is False

    assert state[
        "factorial_training_authorized"
    ] is False

    assert state[
        "final_outer_cv_access_in_cova3d"
    ] == 0
'''
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


tests = sh(
    [
        sys.executable,
        "-m",
        "pytest",

        "tests/test_cova3d_geometry_operationality.py",

        "-q",
        "-p",
        "no:cacheprovider",
    ],
    env=pytest_env,
    check=False,
)


print()
print(
    tests.stdout.rstrip()
)


if tests.stderr.strip():

    print()
    print(
        tests.stderr.rstrip()
    )


if tests.returncode != 0:

    raise RuntimeError(
        "Geometry operationality regression tests failed."
    )


print()
print(
    "✓ Geometry operationality tests        : PASS"
)


# ==========================================================================================
# 18. SOURCE CAPTURE
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
        "# COVA-3D — BLOCK 09E-GEOM-OP-DIAG"
        in cell
    ):

        source_path = (
            REPO
            / "scripts/code_blocks/"
            "block09e_geometry_operationality_diagnostic.py"
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


audit_payload[
    "source_capture"
] = (
    source_capture
)


write_json(
    audit_path,
    audit_payload,
)


# ==========================================================================================
# 19. NORMALIZE TEXT / MANIFEST
# ==========================================================================================

text_paths = [
    decision_rule_path,
    component_output_path,
    case_output_path,
    quota_output_path,
    role_output_path,
    min3_output_path,
    causal_output_path,
    audit_path,
    COVA_STATE_PATH,
    PROJECT_STATE_PATH,
    test_path,
]


source_path = (
    REPO
    / "scripts/code_blocks/"
    "block09e_geometry_operationality_diagnostic.py"
)


if source_path.exists():

    text_paths.append(
        source_path
    )


for path in text_paths:

    path = Path(
        path
    )

    content = path.read_text(
        encoding="utf-8"
    )


    path.write_text(
        content.rstrip()
        + "\n",
        encoding="utf-8",
    )


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
            and path
            != repo_manifest_path
            and not any(
                part in EXCLUDED_DIRS
                for part in path.parts
            )
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

        "geometry_operationality":
            classification,

        "resolvable_component_aliases":
            res_any_alias,

        "quota2_DIS_FRG_aliases":
            quota2_dis_frg_aliases,

        "whole_case_geometry_aliases":
            whole_case_aliases,

        "dense_outcomes_opened":
            False,

        "factorial_training_authorized":
            False,

        "next_block":
            next_block,

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
# 20. COMMIT / PUSH
# ==========================================================================================

heading(
    "STEP 10/10 — COMMIT PRE-OUTCOME STRUCTURAL DIAGNOSTIC"
)


git_paths = [
    "COVA3D_STATE.json",
    "PROJECT_STATE.json",
    "REPOSITORY_MANIFEST.json",

    "configs/"
    "cova3d_geometry_operationality_decision_rule_v1_0.yaml",

    "data/manifests/"
    "cova3d_geometry_operationality_component_audit_v1_0.csv",

    "data/manifests/"
    "cova3d_geometry_operationality_case_audit_v1_0.csv",

    "data/manifests/"
    "cova3d_geometry_operationality_quota_summary_v1_0.csv",

    "data/manifests/"
    "cova3d_geometry_operationality_role_summary_v1_0.csv",

    "data/manifests/"
    "cova3d_geometry_min3_feasibility_screen_v1_0.csv",

    "data/manifests/"
    "cova3d_geometry_causal_matching_audit_v1_0.csv",

    "experiments/audits/"
    "block09e_geometry_operationality_diagnostic.json",

    "figures/audit/"
    "fig_cova3d_geometry_operationality_jaccard_v1_0.png",

    "figures/audit/"
    "fig_cova3d_geometry_operationality_jaccard_v1_0.pdf",

    "figures/audit/"
    "fig_cova3d_geometry_alias_by_quota_v1_0.png",

    "figures/audit/"
    "fig_cova3d_geometry_alias_by_quota_v1_0.pdf",

    "tests/"
    "test_cova3d_geometry_operationality.py",
]


if source_path.exists():

    git_paths.append(
        "scripts/code_blocks/"
        "block09e_geometry_operationality_diagnostic.py"
    )


sh(
    [
        "git",
        "add",
        *git_paths,
    ]
)


diff_check = sh(
    [
        "git",
        "diff",
        "--cached",
        "--check",
    ],
    check=False,
)


if diff_check.returncode != 0:

    raise RuntimeError(
        "git diff --cached --check failed:\n"
        + (
            diff_check.stdout
            or ""
        )
        + (
            diff_check.stderr
            or ""
        )
    )


staged_files = sh(
    [
        "git",
        "diff",
        "--cached",
        "--name-only",
    ]
).stdout.splitlines()


if any(
    path.endswith(
        ".pt"
    )
    for path in staged_files
):

    raise RuntimeError(
        "A model/checkpoint was accidentally staged."
    )


unstaged = sh(
    [
        "git",
        "diff",
        "--name-only",
    ]
).stdout.strip()


if unstaged:

    raise RuntimeError(
        "Tracked unstaged files remain:\n"
        + unstaged
    )


print(
    "✓ git diff --cached --check            : PASS"
)

print(
    "✓ Model/checkpoint staged              : NO"
)


sh(
    [
        "git",
        "commit",
        "-m",
        "audit: close COVA-3D geometry operationality diagnostic",
    ]
)


final_commit = sh(
    [
        "git",
        "rev-parse",
        "HEAD",
    ]
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
    ]
).stdout.strip()


if final_status:

    raise RuntimeError(
        "Repository is not clean after geometry diagnostic:\n"
        + final_status
    )


# ==========================================================================================
# 21. FINAL REPORT
# ==========================================================================================

heading(
    "COVA-3D 09E-GEOM-OP-DIAG — FINAL REPORT"
)


print(
    "Starting commit                        :",
    head[:12],
)

print(
    "Diagnostic commit                      :",
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
    "FIREWALL"
)

print(
    "--------"
)

print(
    "CT arrays accessed                     : 0"
)

print(
    "Dense lesion masks accessed            : 0"
)

print(
    "Dense lung masks accessed              : 0"
)

print(
    "Predictions accessed                   : 0"
)

print(
    "Model/checkpoint accessed              : 0"
)

print(
    "optimizer.step() calls                 : 0"
)

print(
    "Final outer-CV outcome access          : 0"
)

print()

print(
    "ARTIFACT INTEGRITY"
)

print(
    "------------------"
)

print(
    "Final sparse artifacts                 : 120 / 120 SHA PASS"
)

print(
    "Cases                                  : 20"
)

print(
    "Case-coverages                         : 40"
)

print(
    "Same six-cell FG budget                : PASS"
)

print(
    "Same six-cell BG                       : PASS"
)

print(
    "Same component sets across geometry    : PASS"
)

print(
    "Same component quotas across geometry  : PASS"
)

print()

print(
    "COMPONENT OPERATIONALITY"
)

print(
    "------------------------"
)

print(
    "Component instances                    :",
    len(
        component_df
    ),
)

print(
    "Geometry-resolvable instances          :",
    len(
        resolvable_df
    ),
)

print(
    "Resolution-limited instances           :",
    len(
        limited_df
    ),
)

print()

print(
    "Resolvable COH == DIS                  :",
    res_alias_coh_dis,
)

print(
    "Resolvable COH == FRG                  :",
    res_alias_coh_frg,
)

print(
    "Resolvable DIS == FRG                  :",
    res_alias_dis_frg,
)

print(
    "Resolvable any pairwise alias          :",
    res_any_alias,
)

print()

print(
    "Resolvable quota=2 instances           :",
    quota2_count,
)

print(
    "quota=2 DIS == FRG                     :",
    quota2_dis_frg_aliases,
)

print()

print(
    "Whole-case geometry aliases            :",
    whole_case_aliases,
)

print(
    "Resolution-limited identity failures   :",
    limited_identity_failures,
)

print(
    "Resolvable topology failures           :",
    res_topology_failures,
)

print(
    "Geometry-claim flag failures           :",
    claim_failures,
)

print()

print(
    "SPATIAL DESCRIPTIVES"
)

print(
    "--------------------"
)

print(
    "Median Jaccard COH–DIS                 :",
    jaccard_summary[
        "COH_vs_DIS_median"
    ],
)

print(
    "Median Jaccard COH–FRG                 :",
    jaccard_summary[
        "COH_vs_FRG_median"
    ],
)

print(
    "Median Jaccard DIS–FRG                 :",
    jaccard_summary[
        "DIS_vs_FRG_median"
    ],
)

print(
    "Median NN COH (mm)                     :",
    nn_summary[
        "COH_median_mm"
    ],
)

print(
    "Median NN DIS (mm)                     :",
    nn_summary[
        "DIS_median_mm"
    ],
)

print(
    "Median NN FRG (mm)                     :",
    nn_summary[
        "FRG_median_mm"
    ],
)

print()

print(
    "MINIMUM-QUOTA-3 ADVISORY SCREEN"
)

print(
    "-------------------------------"
)

print(
    "Simple min3 feasible case-coverages    :",
    min3_feasible_count,
    "/",
    len(
        min3_df
    ),
)

print(
    "Simple min3 infeasible case-coverages  :",
    min3_infeasible_count,
)

print()

print(
    "DECISION"
)

print(
    "--------"
)

print(
    "Classification                         :",
    classification,
)

print(
    "Dense sanity evaluation authorized     :",
    "YES"
    if dense_sanity_authorized
    else "NO",
)

print(
    "Factorial training authorized           : NO"
)

print(
    "Method development authorized           : NO"
)

print(
    "Final outer-CV authorized               : NO"
)

print()

print(
    "Next block                             :",
    next_block,
)

print()

if strict_pass:

    print(
        "INTERPRETATION:"
    )

    print(
        "The frozen geometry interventions remain operationally distinct at "
        "the final training-grid coordinate level. Dense development sanity "
        "evaluation may now proceed."
    )

else:

    print(
        "INTERPRETATION:"
    )

    print(
        "At least one frozen geometry intervention is not operationally "
        "distinct under the strict pre-outcome rule. Dense outcomes remain "
        "sealed. A prospective sparse-annotation repair is required before "
        "any factorial comparison."
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
    "Send me this COMPLETE report."
)

print(
    "Do NOT open dense masks manually."
)

print(
    "Do NOT begin factorial training."
)

print(
    "=" * 132
)
