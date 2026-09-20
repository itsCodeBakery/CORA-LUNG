# ==========================================================================================
# COVA-3D — BLOCK 09E-GEOM-OP-REPAIR
# A1.3 Operational-Geometry Neutralization
#
# EXPECTED START COMMIT
# ---------------------
# 8dc72d01db2f
#
# TRIGGER
# -------
# Pre-outcome 09E-GEOM-OP-DIAG found:
#
#   resolvable COH == DIS : 0
#   resolvable COH == FRG : 0
#   resolvable DIS == FRG : 8
#
# All 8 aliases:
#   • quota = 3
#   • C100
#   • final_outer_cv
#
# quota=2 aliases:
#   • 0 / 20
#
# Therefore a global minimum-quota increase is NOT the repair.
#
# A1.3 PRINCIPLE
# --------------
# If a physical lesion component cannot maintain distinct geometry interventions
# at ANY selected coverage under the already-frozen A1.2 coordinates, then that
# entire physical component is removed from the geometry claim across coverages.
#
# It remains:
#   ✓ selected according to the same coverage factor
#   ✓ assigned the same per-component quota
#   ✓ included in the same patient FG budget
#
# But it becomes GEOMETRY-NEUTRAL:
#   ✓ same coordinates under COH / DIS / FRG
#   ✓ same deterministic neutral bank across coverage via prefixes
#   ✓ no geometry claim
#
# This mirrors the existing A1.2 treatment of resolution-limited lesions.
#
# IMPORTANT
# ---------
# A1.3 is driven ONLY by frozen sparse-coordinate structure.
# No CT, dense segmentation outcome, prediction, checkpoint or performance result
# is consulted.
#
# The repair is GLOBAL by physical component:
#   if alias occurs at one coverage -> component is neutral at BOTH coverages
#   whenever selected.
#
# EXPECTED CONSEQUENCE
# --------------------
# The 8 alias-triggering physical components should be promoted to
# GLOBAL_OPERATIONALLY_NEUTRAL_ALIAS.
#
# Existing A1.2 GLOBAL_RESOLUTION_LIMITED components remain neutral.
#
# Remaining components become GLOBAL_GEOMETRY_OPERATIONAL.
#
# Whole-case COH/DIS/FRG equality is permitted ONLY when that case-coverage
# contains zero geometry-operational components.
#
# VERSIONING
# ----------
# Historical v1.2 artifacts are preserved unchanged.
# New trainer artifacts are written as v1.3.
#
# R2 VALIDITY
# -----------
# All alias-triggering components were final_outer_cv.
# This block MUST prove all 24 permanent-development v1.3 artifacts are
# semantically identical to v1.2.
#
# If not, R2 cannot be reused and this block stops.
#
# FIREWALL
# --------
# NO CT
# NO dense lesion masks
# NO lung masks
# NO model/checkpoint
# NO predictions
# NO optimizer
# NO dense outcomes
# NO final-CV segmentation outcomes
#
# Factorial training remains LOCKED.
# ==========================================================================================


from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import hashlib
import io
import json
import math
import os
import shutil
import subprocess
import sys
import textwrap
import zipfile

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
    "8dc72d01db2f"
)

BLOCK = (
    "09E-GEOM-OP-REPAIR"
)

AMENDMENT_ID = (
    "A1.3_OPERATIONAL_GEOMETRY_NEUTRALIZATION"
)

PROTOCOL_BEFORE = (
    "COVA3D_1.0+A1+A1.1+A1.2+N1+N2"
)

PROTOCOL_AFTER = (
    "COVA3D_1.0+A1+A1.1+A1.2+A1.3+N1+N2"
)

OLD_ROOT = (
    REPO
    / "data/cova3d_training_grid_candidates_v1_2"
)

NEW_ROOT = (
    REPO
    / "data/cova3d_training_grid_candidates_v1_3"
)

OLD_ARTIFACT_MANIFEST = (
    REPO
    / "data/manifests/"
    "cova3d_training_grid_candidate_artifact_manifest_v1_2.csv"
)

OLD_CONDITION_MANIFEST = (
    REPO
    / "data/manifests/"
    "cova3d_training_grid_condition_manifest_v1_2.csv"
)

OLD_COMPONENT_AUDIT = (
    REPO
    / "data/manifests/"
    "cova3d_training_grid_component_audit_v1_2.csv"
)

GLOBAL_CLASS_PATH = (
    REPO
    / "data/manifests/"
    "cova3d_global_resolution_class_v1_0.csv"
)

GEOM_DIAG_COMPONENT_PATH = (
    REPO
    / "data/manifests/"
    "cova3d_geometry_operationality_component_audit_v1_0.csv"
)

GEOM_DIAG_CASE_PATH = (
    REPO
    / "data/manifests/"
    "cova3d_geometry_operationality_case_audit_v1_0.csv"
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

NEW_ARTIFACT_MANIFEST = (
    REPO
    / "data/manifests/"
    "cova3d_training_grid_candidate_artifact_manifest_v1_3.csv"
)

NEW_CONDITION_MANIFEST = (
    REPO
    / "data/manifests/"
    "cova3d_training_grid_condition_manifest_v1_3.csv"
)

NEW_COMPONENT_AUDIT = (
    REPO
    / "data/manifests/"
    "cova3d_training_grid_component_audit_v1_3.csv"
)

OPERATIONAL_CLASS_PATH = (
    REPO
    / "data/manifests/"
    "cova3d_operational_geometry_class_v1_0.csv"
)

PROMOTED_BANK_PATH = (
    REPO
    / "data/manifests/"
    "cova3d_A1_3_promoted_neutral_bank_v1_0.csv"
)

A13_CASE_AUDIT_PATH = (
    REPO
    / "data/manifests/"
    "cova3d_A1_3_case_operationality_audit_v1_0.csv"
)

A13_DEV_COMPAT_PATH = (
    REPO
    / "data/manifests/"
    "cova3d_A1_3_R2_development_compatibility_v1_0.csv"
)

A13_CONFIG_PATH = (
    REPO
    / "configs/"
    "cova3d_protocol_amendment_A1_3_operational_neutralization.yaml"
)

A13_AUDIT_PATH = (
    REPO
    / "experiments/audits/"
    "block09e_geometry_operationality_repair_A1_3.json"
)

TEST_PATH = (
    REPO
    / "tests/"
    "test_cova3d_geometry_operationality_A1_3.py"
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
        "COH",

    "C50_DIS":
        "DIS",

    "C50_FRG":
        "FRG",

    "C100_COH":
        "COH",

    "C100_DIS":
        "DIS",

    "C100_FRG":
        "FRG",
}

CONDITIONS_BY_COVERAGE = {
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

OLD_RESOLVABLE = (
    "GLOBAL_GEOMETRY_RESOLVABLE"
)

OLD_LIMITED = (
    "GLOBAL_RESOLUTION_LIMITED"
)

NEW_OPERATIONAL = (
    "GLOBAL_GEOMETRY_OPERATIONAL"
)

NEW_ALIAS_NEUTRAL = (
    "GLOBAL_OPERATIONALLY_NEUTRAL_ALIAS"
)

NEW_RESOLUTION_NEUTRAL = (
    "GLOBAL_RESOLUTION_NEUTRAL"
)

EXPECTED_ARTIFACTS = (
    120
)

EXPECTED_CASES = (
    20
)

EXPECTED_ALIAS_COMPONENTS = (
    8
)

EXPECTED_ALIAS_ROWS = (
    8
)

EXPECTED_ALIAS_QUOTA = (
    3
)

EXPECTED_R2_MODEL_SHA = (
    "fce56f781cbd4ffd3e733cccb04c81f3b5bb2a664aceed4b8c0688e06ac4891c"
)

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


def coords_from_set(values):

    if not values:

        return np.empty(
            (
                0,
                3,
            ),
            dtype=np.int32,
        )

    return np.asarray(
        sorted(
            values
        ),
        dtype=np.int32,
    )


def lexical_sort(coords):

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


def unique_lexical(coords):

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

    return lexical_sort(
        np.unique(
            coords,
            axis=0,
        )
    )


def deterministic_npz(
    path,
    **arrays,
):

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        ".npz.tmp"
    )

    with zipfile.ZipFile(
        temporary,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:

        for key in sorted(
            arrays.keys()
        ):

            array = np.asarray(
                arrays[
                    key
                ]
            )

            buffer = io.BytesIO()

            np.lib.format.write_array(
                buffer,
                array,
                allow_pickle=False,
            )

            info = zipfile.ZipInfo(
                filename=(
                    key
                    + ".npy"
                ),
                date_time=(
                    1980,
                    1,
                    1,
                    0,
                    0,
                    0,
                ),
            )

            info.compress_type = (
                zipfile.ZIP_DEFLATED
            )

            info.create_system = (
                0
            )

            archive.writestr(
                info,
                buffer.getvalue(),
                compress_type=zipfile.ZIP_DEFLATED,
                compresslevel=9,
            )

    os.replace(
        temporary,
        path,
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


def connected_set_count(coords):

    remaining = set(
        coord_set(
            coords
        )
    )

    count = (
        0
    )

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

        count += (
            1
        )

        while stack:

            z, y, x = stack.pop()

            for dz, dy, dx in OFFSETS_26:

                neighbour = (
                    z + dz,
                    y + dy,
                    x + dx,
                )

                if neighbour in remaining:

                    remaining.remove(
                        neighbour
                    )

                    stack.append(
                        neighbour
                    )

    return count


def mean_nn_mm(coords):

    coords = np.asarray(
        coords,
        dtype=np.float64,
    )

    if len(
        coords
    ) < 2:

        return np.nan

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


def parse_ids(text):

    value = str(
        text
    ).strip()

    if (
        not value
        or value.lower()
        == "nan"
    ):

        return []

    return [
        int(
            token
        )
        for token in value.split(
            ";"
        )
        if token
    ]


def load_sparse(path):

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        required = {
            "supervision_voxel_zyx",
            "supervision_label",
            "fg_membership_voxel_zyx",
            "fg_membership_group_id",
        }

        if not required.issubset(
            set(
                data.files
            )
        ):

            raise RuntimeError(
                "Unexpected sparse artifact schema:\n"
                + str(
                    path
                )
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

    fg = coord_set(
        supervision_coords[
            supervision_label
            == 1
        ]
    )

    bg = coord_set(
        supervision_coords[
            supervision_label
            == 0
        ]
    )

    grouped_fg = coord_set(
        membership_coords
    )

    if fg != grouped_fg:

        raise RuntimeError(
            "Direct FG differs from membership FG."
        )

    groups = {}

    for component_id in sorted(
        np.unique(
            membership_groups
        ).tolist()
    ):

        component_id = int(
            component_id
        )

        groups[
            component_id
        ] = unique_lexical(
            membership_coords[
                membership_groups
                == component_id
            ]
        )

    return {
        "supervision_coords":
            supervision_coords,

        "supervision_label":
            supervision_label,

        "membership_coords":
            membership_coords,

        "membership_groups":
            membership_groups,

        "fg":
            fg,

        "bg":
            bg,

        "groups":
            groups,
    }


def arrays_equal_sparse(
    first,
    second,
):

    return bool(
        np.array_equal(
            first[
                "supervision_coords"
            ],
            second[
                "supervision_coords"
            ],
        )
        and np.array_equal(
            first[
                "supervision_label"
            ],
            second[
                "supervision_label"
            ],
        )
        and np.array_equal(
            first[
                "membership_coords"
            ],
            second[
                "membership_coords"
            ],
        )
        and np.array_equal(
            first[
                "membership_groups"
            ],
            second[
                "membership_groups"
            ],
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
        "/tmp/cova3d_git_askpass_A1_3.sh"
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
# 2. VERIFY PRE-OUTCOME STATE
# ==========================================================================================

heading(
    "COVA-3D 09E-GEOM-OP-REPAIR — VERIFY PRE-OUTCOME STATE"
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
        "Repository must be clean before A1.3:\n"
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
) != "09E-GEOM-OP-DIAG":

    raise RuntimeError(
        "Expected geometry diagnostic as previous block."
    )


if cova_state.get(
    "next_block"
) != BLOCK:

    raise RuntimeError(
        "Project state does not authorize A1.3 repair."
    )


if cova_state.get(
    "geometry_operationality_status"
) != "REQUIRES_PROSPECTIVE_GEOMETRY_AMENDMENT":

    raise RuntimeError(
        "Geometry repair trigger is absent."
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
        "Final outer-CV outcome access is not zero."
    )


if cova_state.get(
    "factorial_training_authorized"
) is not False:

    raise RuntimeError(
        "Factorial training must remain locked."
    )


if str(
    cova_state.get(
        "sanity_fit_R2_final_model_sha256"
    )
) != EXPECTED_R2_MODEL_SHA:

    raise RuntimeError(
        "R2 model provenance changed."
    )


print(
    "✓ Starting commit                      :",
    head[:12],
)

print(
    "✓ Previous geometry decision           : REPAIR REQUIRED"
)

print(
    "✓ Dense outcomes                       : SEALED"
)

print(
    "✓ Final outer-CV outcomes              : SEALED"
)

print(
    "✓ Factorial training                   : LOCKED"
)

print(
    "✓ R2 model remains frozen              :",
    EXPECTED_R2_MODEL_SHA,
)


# ==========================================================================================
# 3. LOAD FROZEN MANIFESTS
# ==========================================================================================

heading(
    "STEP 1/11 — LOAD FROZEN A1.2 + DIAGNOSTIC ARTIFACTS"
)


for path in [
    OLD_ARTIFACT_MANIFEST,
    OLD_CONDITION_MANIFEST,
    OLD_COMPONENT_AUDIT,
    GLOBAL_CLASS_PATH,
    GEOM_DIAG_COMPONENT_PATH,
    GEOM_DIAG_CASE_PATH,
    SPLIT_PATH,
]:

    if not path.exists():

        raise RuntimeError(
            "Missing required frozen artifact:\n"
            + str(
                path
            )
        )


artifact_manifest = pd.read_csv(
    OLD_ARTIFACT_MANIFEST
)


condition_manifest = pd.read_csv(
    OLD_CONDITION_MANIFEST
)


component_audit_old = pd.read_csv(
    OLD_COMPONENT_AUDIT
)


global_class_df = pd.read_csv(
    GLOBAL_CLASS_PATH
)


geometry_diag = pd.read_csv(
    GEOM_DIAG_COMPONENT_PATH
)


split_df = pd.read_csv(
    SPLIT_PATH
)


if len(
    artifact_manifest
) != EXPECTED_ARTIFACTS:

    raise RuntimeError(
        "Expected 120 A1.2 sparse artifacts."
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
        "Expected 20 cases."
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


print(
    "✓ A1.2 sparse artifacts                : 120"
)

print(
    "✓ Cases                                : 20"
)

print(
    "✓ Dense arrays loaded                  : 0"
)

print(
    "✓ Model/checkpoint loaded              : 0"
)


# ==========================================================================================
# 4. DERIVE GLOBAL ALIAS-PROMOTION SET
# ==========================================================================================

heading(
    "STEP 2/11 — FREEZE A1.3 GLOBAL PROMOTION RULE"
)


alias_rows = geometry_diag[
    (
        geometry_diag[
            "resolution_class"
        ]
        == OLD_RESOLVABLE
    )
    & (
        geometry_diag[
            "any_pairwise_alias"
        ]
        == True
    )
].copy()


if len(
    alias_rows
) != EXPECTED_ALIAS_ROWS:

    raise RuntimeError(
        "Expected exactly 8 alias component-instances from frozen diagnostic."
    )


if not (
    alias_rows[
        "quota"
    ].astype(
        int
    )
    == EXPECTED_ALIAS_QUOTA
).all():

    raise RuntimeError(
        "Alias trigger is not restricted to quota=3 as expected."
    )


if not (
    alias_rows[
        "coverage_fraction"
    ].astype(
        float
    )
    == 1.0
).all():

    raise RuntimeError(
        "Alias trigger is not restricted to C100 as expected."
    )


if not (
    alias_rows[
        "role"
    ].astype(
        str
    )
    == "final_outer_cv"
).all():

    raise RuntimeError(
        "At least one alias-triggering component is a development component."
    )


promoted_keys = sorted(
    {
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
        for _, row in alias_rows.iterrows()
    }
)


if len(
    promoted_keys
) != EXPECTED_ALIAS_COMPONENTS:

    raise RuntimeError(
        "Expected exactly 8 unique physical components to promote."
    )


old_class_lookup = {
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
    for _, row in global_class_df.iterrows()
}


for key in promoted_keys:

    if old_class_lookup[
        key
    ] != OLD_RESOLVABLE:

        raise RuntimeError(
            "A1.3 attempted to promote a non-resolvable component."
        )


print(
    "✓ Alias rows                           : 8"
)

print(
    "✓ Unique promoted physical components  : 8"
)

print(
    "✓ Alias quota                          : 3 only"
)

print(
    "✓ Alias coverage                       : C100 only"
)

print(
    "✓ Alias role                           : FINAL OUTER-CV only"
)

print()

print(
    "Promoted components:"
)

for case_id, component_id in promoted_keys:

    print(
        "  "
        + case_id
        + " / component "
        + str(
            component_id
        )
    )


# ==========================================================================================
# 5. FREEZE A1.3 CONFIG BEFORE RECONSTRUCTION
# ==========================================================================================

A13_CONFIG = {
    "project":
        "COVA-3D",

    "amendment_id":
        AMENDMENT_ID,

    "status":
        "PROSPECTIVELY_FROZEN_BEFORE_DENSE_OUTCOME_OPENING",

    "trigger_block":
        "09E-GEOM-OP-DIAG",

    "trigger_commit":
        head,

    "protocol_before":
        PROTOCOL_BEFORE,

    "protocol_after":
        PROTOCOL_AFTER,

    "trigger_evidence":
        {
            "resolvable_COH_equals_DIS":
                0,

            "resolvable_COH_equals_FRG":
                0,

            "resolvable_DIS_equals_FRG":
                8,

            "quota2_DIS_equals_FRG":
                0,

            "alias_quota":
                3,

            "alias_rows":
                8,

            "unique_alias_physical_components":
                8,

            "all_aliases_final_outer_cv":
                True,

            "dense_outcomes_opened":
                False,
        },

    "classification_rule":
        {
            "GLOBAL_RESOLUTION_NEUTRAL":
                (
                    "A1.2 GLOBAL_RESOLUTION_LIMITED component. "
                    "No geometry claim."
                ),

            "GLOBAL_OPERATIONALLY_NEUTRAL_ALIAS":
                (
                    "Previously geometry-resolvable physical component with "
                    "at least one exact pairwise geometry-coordinate alias at "
                    "any selected coverage. It is neutralized globally across "
                    "all coverages where selected."
                ),

            "GLOBAL_GEOMETRY_OPERATIONAL":
                (
                    "Previously geometry-resolvable physical component with "
                    "no pairwise geometry-coordinate alias in any selected "
                    "coverage."
                ),
        },

    "neutralization_rule":
        {
            "coverage_selection_changed":
                False,

            "per_component_quota_changed":
                False,

            "patient_FG_budget_changed":
                False,

            "background_changed":
                False,

            "promoted_component_geometry_claim":
                False,

            "same_neutral_bank_across_geometry":
                True,

            "same_ordered_bank_across_coverage":
                True,

            "coverage_specific_coordinates":
                "PREFIX_OF_SAME_BANK",

            "bank_candidate_source":
                (
                    "Union of already validated sparse coordinates belonging "
                    "to the same physical component across all six A1.2 cells."
                ),

            "bank_forbidden_coordinates":
                (
                    "Any coordinate used by another lesion component in any "
                    "A1.2 cell for the same case, plus common background."
                ),

            "bank_order":
                (
                    "Ascending physical distance to pooled same-component "
                    "coordinate centroid, then z/y/x lexicographic tie-break."
                ),
        },

    "whole_case_alias_rule":
        {
            "geometry_informative_case_coverage":
                (
                    "At least one GLOBAL_GEOMETRY_OPERATIONAL component selected."
                ),

            "informative_case_coverage_alias_allowed":
                False,

            "zero_operational_component_case_coverage_alias_allowed":
                True,

            "reason":
                (
                    "A case-coverage containing only geometry-neutral lesions "
                    "has no geometry claim by construction."
                ),
        },

    "R2_compatibility_requirement":
        {
            "all_promoted_components_must_be_final_outer_cv":
                True,

            "all_24_development_artifacts_must_be_semantically_unchanged":
                True,

            "R2_retraining_if_requirement_passes":
                False,
        },

    "factorial_training_authorized":
        False,

    "dense_sanity_evaluation_authorized":
        False,

    "final_outer_cv_outcomes_authorized":
        False,

    "frozen_at_utc":
        NOW_ISO,
}


A13_CONFIG_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


A13_CONFIG_PATH.write_text(
    yaml.safe_dump(
        A13_CONFIG,
        sort_keys=False,
        allow_unicode=True,
    ),
    encoding="utf-8",
)


print()
print(
    "✓ A1.3 rule frozen                     : YES"
)

print(
    "✓ Coverage selection changes           : NO"
)

print(
    "✓ Quota changes                        : NO"
)

print(
    "✓ Budget changes                       : NO"
)

print(
    "✓ BG changes                           : NO"
)

print(
    "✓ Outcome information used             : NO"
)


# ==========================================================================================
# 6. VERIFY / LOAD ALL A1.2 SPARSE ARTIFACTS
# ==========================================================================================

heading(
    "STEP 3/11 — VERIFY 120 A1.2 ARTIFACTS"
)


old_data = {}


for _, row in tqdm(
    artifact_manifest.iterrows(),
    total=len(
        artifact_manifest
    ),
    desc="Verify A1.2",
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


    if sha256_file(
        path
    ) != str(
        row[
            "artifact_file_sha256"
        ]
    ):

        raise RuntimeError(
            "A1.2 sparse artifact SHA mismatch:\n"
            + str(
                path
            )
        )


    old_data[
        (
            case_id,
            condition_id,
        )
    ] = load_sparse(
        path
    )


print(
    "✓ SHA-verified A1.2 artifacts           : 120 / 120"
)

print(
    "✓ CT accessed                          : 0"
)

print(
    "✓ Dense mask accessed                  : 0"
)


# ==========================================================================================
# 7. BUILD NEW GLOBAL OPERATIONAL CLASSIFICATION
# ==========================================================================================

heading(
    "STEP 4/11 — BUILD GLOBAL OPERATIONAL-GEOMETRY CLASSIFICATION"
)


classification_rows = []


operational_class_lookup = {}


for _, row in global_class_df.iterrows():

    case_id = str(
        row[
            "case_id"
        ]
    )

    component_id = int(
        row[
            "component_native_id"
        ]
    )

    key = (
        case_id,
        component_id,
    )

    old_class = str(
        row[
            "global_resolution_class"
        ]
    )


    if old_class == OLD_LIMITED:

        new_class = (
            NEW_RESOLUTION_NEUTRAL
        )

        trigger = (
            "A1.2_resolution_limit"
        )

        geometry_claim = (
            False
        )


    elif key in promoted_keys:

        new_class = (
            NEW_ALIAS_NEUTRAL
        )

        trigger = (
            "A1.3_exact_geometry_alias"
        )

        geometry_claim = (
            False
        )


    elif old_class == OLD_RESOLVABLE:

        new_class = (
            NEW_OPERATIONAL
        )

        trigger = (
            "none"
        )

        geometry_claim = (
            True
        )


    else:

        raise RuntimeError(
            "Unexpected A1.2 global class."
        )


    operational_class_lookup[
        key
    ] = (
        new_class
    )


    record = row.to_dict()

    record[
        "A1_3_operational_geometry_class"
    ] = new_class

    record[
        "A1_3_trigger"
    ] = trigger

    record[
        "A1_3_geometry_claim_allowed"
    ] = geometry_claim

    classification_rows.append(
        record
    )


classification_df = pd.DataFrame(
    classification_rows
)


n_resolution_neutral = int(
    (
        classification_df[
            "A1_3_operational_geometry_class"
        ]
        == NEW_RESOLUTION_NEUTRAL
    ).sum()
)


n_alias_neutral = int(
    (
        classification_df[
            "A1_3_operational_geometry_class"
        ]
        == NEW_ALIAS_NEUTRAL
    ).sum()
)


n_operational = int(
    (
        classification_df[
            "A1_3_operational_geometry_class"
        ]
        == NEW_OPERATIONAL
    ).sum()
)


if n_alias_neutral != EXPECTED_ALIAS_COMPONENTS:

    raise RuntimeError(
        "A1.3 promoted-neutral count mismatch."
    )


if (
    n_resolution_neutral
    + n_alias_neutral
    + n_operational
) != len(
    classification_df
):

    raise RuntimeError(
        "Operational class partition mismatch."
    )


classification_df.to_csv(
    OPERATIONAL_CLASS_PATH,
    index=False,
)


print(
    "Geometry-operational physical components :",
    n_operational,
)

print(
    "Resolution-neutral physical components   :",
    n_resolution_neutral,
)

print(
    "Alias-promoted neutral components        :",
    n_alias_neutral,
)

print(
    "Total geometry-neutral components        :",
    n_resolution_neutral
    + n_alias_neutral,
)


# ==========================================================================================
# 8. BUILD GEOMETRY-NEUTRAL BANKS FOR 8 PROMOTED COMPONENTS
# ==========================================================================================

heading(
    "STEP 5/11 — BUILD A1.3 PROMOTED NEUTRAL BANKS"
)


promoted_banks = {}


bank_rows = []


for case_id, component_id in promoted_keys:

    own_union = set()

    other_union = set()

    background_union = set()

    quotas = []


    for condition_id in CONDITIONS:

        artifact = old_data[
            (
                case_id,
                condition_id,
            )
        ]


        background_union.update(
            artifact[
                "bg"
            ]
        )


        for observed_component_id, coords in artifact[
            "groups"
        ].items():

            coordinate_values = set(
                coord_set(
                    coords
                )
            )


            if observed_component_id == component_id:

                own_union.update(
                    coordinate_values
                )

                quotas.append(
                    len(
                        coordinate_values
                    )
                )


            else:

                other_union.update(
                    coordinate_values
                )


    if not quotas:

        raise RuntimeError(
            "Promoted component never appears in sparse artifacts."
        )


    max_quota = max(
        quotas
    )


    safe_set = (
        own_union
        - other_union
        - background_union
    )


    if len(
        safe_set
    ) < max_quota:

        raise RuntimeError(
            "A1.3 sparse-only neutral bank is insufficient for "
            + case_id
            + " component "
            + str(
                component_id
            )
            + "\n"
            + "Safe bank capacity="
            + str(
                len(
                    safe_set
                )
            )
            + " required="
            + str(
                max_quota
            )
            + "\n"
            + "No dense-mask fallback is allowed in this block."
        )


    safe_coords = coords_from_set(
        safe_set
    )


    own_coords = coords_from_set(
        own_union
    )


    own_physical = (
        own_coords.astype(
            np.float64
        )
        * TARGET_SPACING_ZYX[
            None,
            :
        ]
    )


    centroid = own_physical.mean(
        axis=0
    )


    safe_physical = (
        safe_coords.astype(
            np.float64
        )
        * TARGET_SPACING_ZYX[
            None,
            :
        ]
    )


    distances = np.sqrt(
        np.sum(
            (
                safe_physical
                - centroid[
                    None,
                    :
                ]
            )
            ** 2,
            axis=1,
        )
    )


    order = np.lexsort(
        (
            safe_coords[
                :,
                2
            ],
            safe_coords[
                :,
                1
            ],
            safe_coords[
                :,
                0
            ],
            distances,
        )
    )


    bank = safe_coords[
        order
    ]


    promoted_banks[
        (
            case_id,
            component_id,
        )
    ] = bank


    bank_rows.append(
        {
            "case_id":
                case_id,

            "component_native_id":
                component_id,

            "role":
                role_lookup[
                    case_id
                ],

            "trigger":
                "exact_DIS_FRG_alias_under_A1_2",

            "alias_quota":
                EXPECTED_ALIAS_QUOTA,

            "own_union_coordinates":
                len(
                    own_union
                ),

            "coordinates_excluded_other_components":
                len(
                    own_union
                    & other_union
                ),

            "coordinates_excluded_background":
                len(
                    own_union
                    & background_union
                ),

            "safe_neutral_bank_capacity":
                len(
                    bank
                ),

            "maximum_required_quota":
                max_quota,

            "capacity_pass":
                len(
                    bank
                )
                >= max_quota,

            "bank_semantic_sha256":
                semantic_hash(
                    bank.astype(
                        np.int32
                    )
                ),
        }
    )


bank_df = pd.DataFrame(
    bank_rows
)


if not bank_df[
    "capacity_pass"
].all():

    raise RuntimeError(
        "A1.3 promoted neutral-bank capacity failure."
    )


bank_df.to_csv(
    PROMOTED_BANK_PATH,
    index=False,
)


print(
    "✓ Promoted neutral banks               : 8 /8"
)

print(
    "✓ All bank capacities sufficient       : YES"
)

print()

print(
    bank_df[
        [
            "case_id",
            "component_native_id",
            "safe_neutral_bank_capacity",
            "maximum_required_quota",
        ]
    ].to_string(
        index=False
    )
)


# ==========================================================================================
# 9. WRITE V1.3 SPARSE ARTIFACTS
# ==========================================================================================

heading(
    "STEP 6/11 — CONSTRUCT A1.3 V1.3 SPARSE ARTIFACTS"
)


if NEW_ROOT.exists():

    shutil.rmtree(
        NEW_ROOT
    )


NEW_ROOT.mkdir(
    parents=True,
    exist_ok=True,
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
    for _, row in condition_manifest.iterrows()
}


new_artifact_rows = []

new_condition_rows = []

changed_artifacts = []

changed_component_instances = []


for case_id in tqdm(
    cases,
    desc="A1.3 cases",
):

    for condition_id in CONDITIONS:

        old = old_data[
            (
                case_id,
                condition_id,
            )
        ]


        groups = {
            component_id:
                coords.copy()
            for component_id, coords in old[
                "groups"
            ].items()
        }


        changed_components_here = []


        for component_id in sorted(
            groups.keys()
        ):

            key = (
                case_id,
                component_id,
            )


            if operational_class_lookup[
                key
            ] != NEW_ALIAS_NEUTRAL:

                continue


            quota = len(
                groups[
                    component_id
                ]
            )


            bank = promoted_banks[
                key
            ]


            if quota > len(
                bank
            ):

                raise RuntimeError(
                    "Promoted-neutral quota exceeds bank."
                )


            replacement = bank[
                :quota
            ].astype(
                np.int32
            )


            old_set = coord_set(
                groups[
                    component_id
                ]
            )


            new_set = coord_set(
                replacement
            )


            groups[
                component_id
            ] = replacement


            if old_set != new_set:

                changed_components_here.append(
                    component_id
                )

                changed_component_instances.append(
                    (
                        case_id,
                        condition_id,
                        component_id,
                    )
                )


        # ------------------------------------------------------------------
        # Reconstruct membership arrays.
        # ------------------------------------------------------------------

        membership_coords_parts = []

        membership_group_parts = []


        for component_id in sorted(
            groups.keys()
        ):

            component_coords = unique_lexical(
                groups[
                    component_id
                ]
            )


            if len(
                component_coords
            ) != len(
                groups[
                    component_id
                ]
            ):

                raise RuntimeError(
                    "A1.3 created duplicate component coordinates."
                )


            membership_coords_parts.append(
                component_coords
            )


            membership_group_parts.append(
                np.full(
                    len(
                        component_coords
                    ),
                    component_id,
                    dtype=np.int32,
                )
            )


        membership_coords = np.concatenate(
            membership_coords_parts,
            axis=0,
        )


        membership_groups = np.concatenate(
            membership_group_parts,
            axis=0,
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
                "A1.3 cross-component FG collision."
            )


        fg_coords = unique_lexical(
            membership_coords
        )


        bg_coords = coords_from_set(
            old[
                "bg"
            ]
        )


        if coord_set(
            fg_coords
        ) & coord_set(
            bg_coords
        ):

            raise RuntimeError(
                "A1.3 FG/BG collision."
            )


        old_fg_count = len(
            old[
                "fg"
            ]
        )


        if len(
            fg_coords
        ) != old_fg_count:

            raise RuntimeError(
                "A1.3 changed total FG budget."
            )


        supervision_coords = np.concatenate(
            [
                fg_coords,
                bg_coords,
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
                        bg_coords
                    ),
                    dtype=np.int8,
                ),
            ],
            axis=0,
        )


        membership_order = np.lexsort(
            (
                membership_coords[
                    :,
                    2
                ],
                membership_coords[
                    :,
                    1
                ],
                membership_coords[
                    :,
                    0
                ],
                membership_groups,
            )
        )


        membership_coords = membership_coords[
            membership_order
        ].astype(
            np.int32
        )


        membership_groups = membership_groups[
            membership_order
        ].astype(
            np.int32
        )


        output_path = (
            NEW_ROOT
            / case_id
            / (
                condition_id
                + ".npz"
            )
        )


        deterministic_npz(
            output_path,

            supervision_voxel_zyx=
                supervision_coords,

            supervision_label=
                supervision_labels,

            fg_membership_voxel_zyx=
                membership_coords,

            fg_membership_group_id=
                membership_groups,
        )


        annotation_sha = semantic_hash(
            supervision_coords,
            supervision_labels,
            membership_coords,
            membership_groups,
        )


        artifact_sha = sha256_file(
            output_path
        )


        old_condition = condition_lookup[
            (
                case_id,
                condition_id,
            )
        ]


        old_bg_sha = str(
            old_condition[
                "background_semantic_sha256"
            ]
        )


        new_bg_sha = semantic_hash(
            bg_coords
        )


        if new_bg_sha != old_bg_sha:

            raise RuntimeError(
                "A1.3 changed background supervision."
            )


        old_ids = parse_ids(
            old_condition[
                "selected_component_ids"
            ]
        )


        old_quotas = parse_ids(
            old_condition[
                "selected_component_quotas"
            ]
        )


        actual_quotas = [
            len(
                groups[
                    component_id
                ]
            )
            for component_id in old_ids
        ]


        if actual_quotas != old_quotas:

            raise RuntimeError(
                "A1.3 changed per-component quota."
            )


        selected_operational = sum(
            operational_class_lookup[
                (
                    case_id,
                    component_id,
                )
            ]
            == NEW_OPERATIONAL
            for component_id in old_ids
        )


        selected_resolution_neutral = sum(
            operational_class_lookup[
                (
                    case_id,
                    component_id,
                )
            ]
            == NEW_RESOLUTION_NEUTRAL
            for component_id in old_ids
        )


        selected_alias_neutral = sum(
            operational_class_lookup[
                (
                    case_id,
                    component_id,
                )
            ]
            == NEW_ALIAS_NEUTRAL
            for component_id in old_ids
        )


        record = old_condition.to_dict()


        record[
            "annotation_semantic_sha256"
        ] = annotation_sha

        record[
            "artifact_file"
        ] = str(
            output_path.relative_to(
                REPO
            )
        )

        record[
            "artifact_file_sha256"
        ] = artifact_sha

        record[
            "effective_protocol"
        ] = PROTOCOL_AFTER

        record[
            "selected_geometry_operational_components"
        ] = selected_operational

        record[
            "selected_resolution_neutral_components"
        ] = selected_resolution_neutral

        record[
            "selected_alias_neutral_components"
        ] = selected_alias_neutral

        record[
            "A1_3_changed_component_count"
        ] = len(
            changed_components_here
        )

        record[
            "coordinate_level_QA"
        ] = True

        record[
            "factorial_training_authorized"
        ] = False


        new_condition_rows.append(
            record
        )


        new_artifact_rows.append(
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
                    annotation_sha,

                "sparse_only":
                    True,

                "coordinate_level_QA":
                    True,

                "trainer_candidate":
                    True,

                "training_authorized":
                    False,

                "effective_protocol":
                    PROTOCOL_AFTER,
            }
        )


        if changed_components_here:

            changed_artifacts.append(
                (
                    case_id,
                    condition_id,
                    tuple(
                        changed_components_here
                    ),
                )
            )


new_condition_df = pd.DataFrame(
    new_condition_rows
)


new_artifact_df = pd.DataFrame(
    new_artifact_rows
)


new_condition_df.to_csv(
    NEW_CONDITION_MANIFEST,
    index=False,
)


new_artifact_df.to_csv(
    NEW_ARTIFACT_MANIFEST,
    index=False,
)


if len(
    new_artifact_df
) != EXPECTED_ARTIFACTS:

    raise RuntimeError(
        "A1.3 did not generate exactly 120 artifacts."
    )


print(
    "✓ V1.3 sparse artifacts                : 120"
)

print(
    "✓ Total FG budgets changed             : 0"
)

print(
    "✓ Background sets changed              : 0"
)

print(
    "✓ Component quotas changed             : 0"
)

print(
    "Artifacts with actual coordinate edits :",
    len(
        changed_artifacts
    ),
)

print(
    "Changed component-condition instances  :",
    len(
        changed_component_instances
    ),
)


# ==========================================================================================
# 10. BUILD V1.3 COMPONENT AUDIT
# ==========================================================================================

heading(
    "STEP 7/11 — BUILD V1.3 COMPONENT AUDIT"
)


new_data = {}


for _, row in new_artifact_df.iterrows():

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


    if sha256_file(
        path
    ) != str(
        row[
            "artifact_file_sha256"
        ]
    ):

        raise RuntimeError(
            "V1.3 artifact SHA mismatch."
        )


    new_data[
        (
            case_id,
            condition_id,
        )
    ] = load_sparse(
        path
    )


new_component_rows = []


for _, old_row in component_audit_old.iterrows():

    case_id = str(
        old_row[
            "case_id"
        ]
    )

    condition_id = str(
        old_row[
            "condition_id"
        ]
    )

    component_id = int(
        old_row[
            "component_native_id"
        ]
    )

    key = (
        case_id,
        component_id,
    )


    coords = new_data[
        (
            case_id,
            condition_id,
        )
    ][
        "groups"
    ][
        component_id
    ]


    operational_class = operational_class_lookup[
        key
    ]


    claim_allowed = (
        operational_class
        == NEW_OPERATIONAL
    )


    row = old_row.to_dict()


    row[
        "A1_3_operational_geometry_class"
    ] = operational_class

    row[
        "A1_3_geometry_claim_allowed"
    ] = claim_allowed

    row[
        "A1_3_promoted_alias_neutral"
    ] = (
        operational_class
        == NEW_ALIAS_NEUTRAL
    )

    row[
        "A1_3_effective_connected_sets"
    ] = connected_set_count(
        coords
    )

    row[
        "A1_3_effective_mean_nearest_neighbor_mm"
    ] = mean_nn_mm(
        coords
    )


    if operational_class == NEW_ALIAS_NEUTRAL:

        row[
            "A1_3_neutral_bank_capacity"
        ] = len(
            promoted_banks[
                key
            ]
        )

        row[
            "A1_3_representation"
        ] = (
            "operationally_neutral_geometry_bank"
        )


    elif operational_class == NEW_RESOLUTION_NEUTRAL:

        row[
            "A1_3_neutral_bank_capacity"
        ] = np.nan

        row[
            "A1_3_representation"
        ] = (
            "A1_2_resolution_neutral_bank"
        )


    else:

        row[
            "A1_3_neutral_bank_capacity"
        ] = np.nan

        row[
            "A1_3_representation"
        ] = str(
            old_row[
                "representation"
            ]
        )


    new_component_rows.append(
        row
    )


new_component_df = pd.DataFrame(
    new_component_rows
)


new_component_df.to_csv(
    NEW_COMPONENT_AUDIT,
    index=False,
)


print(
    "✓ V1.3 component audit rows            :",
    len(
        new_component_df
    ),
)


# ==========================================================================================
# 11. VERIFY A1.3 CAUSAL MATCHING + OPERATIONALITY
# ==========================================================================================

heading(
    "STEP 8/11 — VERIFY A1.3 OPERATIONALITY"
)


case_audit_rows = []


operational_alias_count = (
    0
)

neutral_identity_failures = (
    0
)

operational_topology_failures = (
    0
)

informative_whole_case_aliases = (
    0
)

neutral_only_case_coverages = (
    0
)

neutral_only_identity_failures = (
    0
)


for case_id in cases:

    # ------------------------------------------------------------------
    # Case-level six-cell budget/background checks.
    # ------------------------------------------------------------------

    case_rows = new_condition_df[
        new_condition_df[
            "case_id"
        ].astype(
            str
        )
        == case_id
    ]


    if len(
        case_rows
    ) != 6:

        raise RuntimeError(
            "A1.3 case does not contain six cells."
        )


    if len(
        set(
            case_rows[
                "train_positive_budget_B_i"
            ].astype(
                int
            ).tolist()
        )
    ) != 1:

        raise RuntimeError(
            "A1.3 patient budget differs across cells."
        )


    bg_sets = {
        new_data[
            (
                case_id,
                condition_id,
            )
        ][
            "bg"
        ]
        for condition_id in CONDITIONS
    }


    if len(
        bg_sets
    ) != 1:

        raise RuntimeError(
            "A1.3 background differs across cells."
        )


    for coverage in [
        0.50,
        1.00,
    ]:

        condition_ids = CONDITIONS_BY_COVERAGE[
            coverage
        ]


        geometry_groups = {
            geometry:
                new_data[
                    (
                        case_id,
                        condition_ids[
                            geometry
                        ],
                    )
                ][
                    "groups"
                ]
            for geometry in [
                "COH",
                "DIS",
                "FRG",
            ]
        }


        component_ids = sorted(
            geometry_groups[
                "COH"
            ].keys()
        )


        if not (
            component_ids
            == sorted(
                geometry_groups[
                    "DIS"
                ].keys()
            )
            == sorted(
                geometry_groups[
                    "FRG"
                ].keys()
            )
        ):

            raise RuntimeError(
                "A1.3 component set differs across geometry."
            )


        operational_selected = (
            0
        )


        for component_id in component_ids:

            key = (
                case_id,
                component_id,
            )


            sets = {
                geometry:
                    coord_set(
                        geometry_groups[
                            geometry
                        ][
                            component_id
                        ]
                    )
                for geometry in [
                    "COH",
                    "DIS",
                    "FRG",
                ]
            }


            quotas = {
                len(
                    sets[
                        geometry
                    ]
                )
                for geometry in sets
            }


            if len(
                quotas
            ) != 1:

                raise RuntimeError(
                    "A1.3 component quota differs across geometry."
                )


            component_class = operational_class_lookup[
                key
            ]


            if component_class == NEW_OPERATIONAL:

                operational_selected += (
                    1
                )


                any_alias = bool(
                    sets[
                        "COH"
                    ]
                    == sets[
                        "DIS"
                    ]
                    or sets[
                        "COH"
                    ]
                    == sets[
                        "FRG"
                    ]
                    or sets[
                        "DIS"
                    ]
                    == sets[
                        "FRG"
                    ]
                )


                if any_alias:

                    operational_alias_count += (
                        1
                    )


                coh_sets = connected_set_count(
                    geometry_groups[
                        "COH"
                    ][
                        component_id
                    ]
                )

                dis_sets = connected_set_count(
                    geometry_groups[
                        "DIS"
                    ][
                        component_id
                    ]
                )

                frg_sets = connected_set_count(
                    geometry_groups[
                        "FRG"
                    ][
                        component_id
                    ]
                )


                if not (
                    coh_sets
                    == 1
                    and dis_sets
                    >= 2
                    and frg_sets
                    >= 2
                ):

                    operational_topology_failures += (
                        1
                    )


            else:

                if not (
                    sets[
                        "COH"
                    ]
                    == sets[
                        "DIS"
                    ]
                    == sets[
                        "FRG"
                    ]
                ):

                    neutral_identity_failures += (
                        1
                    )


        total_sets = {
            geometry:
                frozenset().union(
                    *[
                        coord_set(
                            geometry_groups[
                                geometry
                            ][
                                component_id
                            ]
                        )
                        for component_id in component_ids
                    ]
                )
            for geometry in [
                "COH",
                "DIS",
                "FRG",
            ]
        }


        whole_alias = bool(
            total_sets[
                "COH"
            ]
            == total_sets[
                "DIS"
            ]
            or total_sets[
                "COH"
            ]
            == total_sets[
                "FRG"
            ]
            or total_sets[
                "DIS"
            ]
            == total_sets[
                "FRG"
            ]
        )


        if operational_selected > 0:

            if whole_alias:

                informative_whole_case_aliases += (
                    1
                )


        else:

            neutral_only_case_coverages += (
                1
            )


            if not (
                total_sets[
                    "COH"
                ]
                == total_sets[
                    "DIS"
                ]
                == total_sets[
                    "FRG"
                ]
            ):

                neutral_only_identity_failures += (
                    1
                )


        case_audit_rows.append(
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
                        component_ids
                    ),

                "selected_geometry_operational_components":
                    operational_selected,

                "geometry_informative":
                    operational_selected
                    > 0,

                "whole_case_geometry_alias":
                    whole_alias,

                "whole_case_alias_allowed":
                    operational_selected
                    == 0,

                "whole_case_operationality_pass":
                    (
                        not whole_alias
                        if operational_selected
                        > 0
                        else (
                            total_sets[
                                "COH"
                            ]
                            == total_sets[
                                "DIS"
                            ]
                            == total_sets[
                                "FRG"
                            ]
                        )
                    ),
            }
        )


case_audit_df = pd.DataFrame(
    case_audit_rows
)


case_audit_df.to_csv(
    A13_CASE_AUDIT_PATH,
    index=False,
)


strict_operationality_pass = bool(
    operational_alias_count
    == 0
    and neutral_identity_failures
    == 0
    and operational_topology_failures
    == 0
    and informative_whole_case_aliases
    == 0
    and neutral_only_identity_failures
    == 0
    and case_audit_df[
        "whole_case_operationality_pass"
    ].all()
)


print(
    "Operational component aliases          :",
    operational_alias_count,
)

print(
    "Neutral identity failures              :",
    neutral_identity_failures,
)

print(
    "Operational topology failures          :",
    operational_topology_failures,
)

print(
    "Informative whole-case aliases         :",
    informative_whole_case_aliases,
)

print(
    "Neutral-only case-coverages            :",
    neutral_only_case_coverages,
)

print(
    "Neutral-only identity failures         :",
    neutral_only_identity_failures,
)

print()

print(
    "A1.3 strict operationality             :",
    "PASS"
    if strict_operationality_pass
    else "FAIL",
)


if not strict_operationality_pass:

    raise RuntimeError(
        "A1.3 did not resolve geometry operationality."
    )


# ==========================================================================================
# 12. VERIFY R2 DEVELOPMENT COMPATIBILITY
# ==========================================================================================

heading(
    "STEP 9/11 — VERIFY R2 DEVELOPMENT ANNOTATIONS ARE UNCHANGED"
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


compatibility_rows = []


for case_id in dev_cases:

    for condition_id in CONDITIONS:

        old = old_data[
            (
                case_id,
                condition_id,
            )
        ]


        new = new_data[
            (
                case_id,
                condition_id,
            )
        ]


        exact_arrays = arrays_equal_sparse(
            old,
            new,
        )


        semantic_same = bool(
            old[
                "fg"
            ]
            == new[
                "fg"
            ]
            and old[
                "bg"
            ]
            == new[
                "bg"
            ]
        )


        compatibility_rows.append(
            {
                "case_id":
                    case_id,

                "condition_id":
                    condition_id,

                "exact_sparse_arrays_equal":
                    exact_arrays,

                "FG_semantic_equal":
                    old[
                        "fg"
                    ]
                    == new[
                        "fg"
                    ],

                "BG_semantic_equal":
                    old[
                        "bg"
                    ]
                    == new[
                        "bg"
                    ],

                "semantic_equal":
                    semantic_same,
            }
        )


compatibility_df = pd.DataFrame(
    compatibility_rows
)


if len(
    compatibility_df
) != 24:

    raise RuntimeError(
        "Expected 24 development condition compatibility checks."
    )


if not compatibility_df[
    "exact_sparse_arrays_equal"
].all():

    raise RuntimeError(
        "A1.3 altered at least one development sparse artifact. "
        "R2 cannot be reused."
    )


if not compatibility_df[
    "semantic_equal"
].all():

    raise RuntimeError(
        "A1.3 altered development annotation semantics."
    )


compatibility_df.to_csv(
    A13_DEV_COMPAT_PATH,
    index=False,
)


print(
    "✓ Permanent-development cases          : 4"
)

print(
    "✓ Development condition artifacts      : 24"
)

print(
    "✓ Exact sparse arrays unchanged        : 24 /24"
)

print(
    "✓ R2 retraining required               : NO"
)

print(
    "✓ R2 checkpoint remains valid          : YES"
)


# ==========================================================================================
# 13. VERIFY DETERMINISTIC ARTIFACT RELOAD / HASHES
# ==========================================================================================

heading(
    "STEP 10/11 — VERIFY V1.3 ARTIFACT INTEGRITY"
)


for _, row in tqdm(
    new_artifact_df.iterrows(),
    total=len(
        new_artifact_df
    ),
    desc="Verify v1.3",
):

    path = (
        REPO
        / str(
            row[
                "artifact_file"
            ]
        )
    )


    if sha256_file(
        path
    ) != str(
        row[
            "artifact_file_sha256"
        ]
    ):

        raise RuntimeError(
            "V1.3 file hash mismatch."
        )


    artifact = load_sparse(
        path
    )


    recomputed_semantic = semantic_hash(
        artifact[
            "supervision_coords"
        ],
        artifact[
            "supervision_label"
        ],
        artifact[
            "membership_coords"
        ],
        artifact[
            "membership_groups"
        ],
    )


    if recomputed_semantic != str(
        row[
            "annotation_semantic_sha256"
        ]
    ):

        raise RuntimeError(
            "V1.3 semantic hash mismatch."
        )


print(
    "✓ V1.3 file hashes                     : 120 /120"
)

print(
    "✓ V1.3 semantic hashes                 : 120 /120"
)


# ==========================================================================================
# 14. WRITE A1.3 AUDIT
# ==========================================================================================

audit_payload = {
    "project":
        "COVA-3D",

    "block":
        BLOCK,

    "amendment":
        AMENDMENT_ID,

    "parent_commit":
        head,

    "protocol_before":
        PROTOCOL_BEFORE,

    "protocol_after":
        PROTOCOL_AFTER,

    "status":
        "PASS",

    "trigger":
        {
            "resolvable_DIS_FRG_alias_rows":
                len(
                    alias_rows
                ),

            "unique_promoted_physical_components":
                len(
                    promoted_keys
                ),

            "alias_quota":
                3,

            "quota2_aliases":
                0,

            "all_aliases_final_outer_cv":
                True,
        },

    "operational_classes":
        {
            "geometry_operational":
                n_operational,

            "resolution_neutral":
                n_resolution_neutral,

            "alias_promoted_neutral":
                n_alias_neutral,

            "total_neutral":
                n_resolution_neutral
                + n_alias_neutral,
        },

    "repair":
        {
            "coverage_selection_changed":
                False,

            "component_quotas_changed":
                False,

            "patient_FG_budgets_changed":
                False,

            "background_changed":
                False,

            "promoted_neutral_banks":
                len(
                    promoted_banks
                ),

            "artifacts_with_coordinate_changes":
                len(
                    changed_artifacts
                ),

            "changed_component_condition_instances":
                len(
                    changed_component_instances
                ),

            "new_artifact_version":
                "v1.3",
        },

    "post_repair_operationality":
        {
            "operational_component_aliases":
                operational_alias_count,

            "neutral_identity_failures":
                neutral_identity_failures,

            "operational_topology_failures":
                operational_topology_failures,

            "informative_whole_case_aliases":
                informative_whole_case_aliases,

            "neutral_only_case_coverages":
                neutral_only_case_coverages,

            "neutral_only_identity_failures":
                neutral_only_identity_failures,

            "strict_pass":
                strict_operationality_pass,
        },

    "R2_compatibility":
        {
            "development_artifacts_checked":
                len(
                    compatibility_df
                ),

            "exact_sparse_arrays_unchanged":
                int(
                    compatibility_df[
                        "exact_sparse_arrays_equal"
                    ].sum()
                ),

            "semantic_unchanged":
                int(
                    compatibility_df[
                        "semantic_equal"
                    ].sum()
                ),

            "R2_retraining_required":
                False,

            "R2_model_sha256":
                EXPECTED_R2_MODEL_SHA,
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

            "model_checkpoints_accessed":
                0,

            "optimizer_steps":
                0,

            "final_outer_cv_segmentation_outcomes_accessed":
                0,
        },

    "authorization":
        {
            "dense_sanity_evaluation":
                True,

            "factorial_training":
                False,

            "method_development":
                False,

            "final_outer_cv_outcomes":
                False,
        },

    "next_block":
        "09E-SANITY-EVAL",

    "generated_at_utc":
        NOW_ISO,
}


A13_AUDIT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


write_json(
    A13_AUDIT_PATH,
    audit_payload,
)


# ==========================================================================================
# 15. UPDATE STATE
# ==========================================================================================

cova_state.update(
    {
        "last_completed_block":
            BLOCK,

        "current_stage":
            "A1_3_OPERATIONAL_GEOMETRY_PASS_DENSE_SANITY_PENDING",

        "effective_protocol":
            PROTOCOL_AFTER,

        "protocol_amendment":
            "A1+A1.1+A1.2+A1.3",

        "A1_3_status":
            "FROZEN_PASS",

        "A1_3_amendment_id":
            AMENDMENT_ID,

        "A1_3_alias_trigger_rows":
            len(
                alias_rows
            ),

        "A1_3_promoted_physical_components":
            len(
                promoted_keys
            ),

        "A1_3_geometry_operational_components":
            n_operational,

        "A1_3_resolution_neutral_components":
            n_resolution_neutral,

        "A1_3_alias_neutral_components":
            n_alias_neutral,

        "A1_3_total_geometry_neutral_components":
            n_resolution_neutral
            + n_alias_neutral,

        "A1_3_v1_3_sparse_artifacts":
            120,

        "A1_3_operational_component_aliases_after_repair":
            operational_alias_count,

        "A1_3_neutral_identity_failures":
            neutral_identity_failures,

        "A1_3_informative_whole_case_aliases":
            informative_whole_case_aliases,

        "A1_3_neutral_only_case_coverages":
            neutral_only_case_coverages,

        "A1_3_R2_development_artifacts_unchanged":
            24,

        "A1_3_R2_compatibility":
            "PASS",

        "primary_dataset_coordinate_geometry_feasibility":
            "PASS_A1_3_OPERATIONAL_GEOMETRY",

        "annotation_methodology_structurally_ready":
            True,

        "trainer_grid_candidate_version":
            "v1.3",

        "trainer_grid_candidate_artifacts":
            120,

        "sanity_fit_R2_status":
            "COMPLETE",

        "sanity_fit_R2_final_model_sha256":
            EXPECTED_R2_MODEL_SHA,

        "sanity_fit_R2_retraining_required_after_A1_3":
            False,

        "sanity_dense_evaluation_authorized_after_fit":
            True,

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
            "09E-SANITY-EVAL",

        "next_action":
            (
                "Evaluate the already-frozen R2 checkpoint once on only the "
                "four permanent-development dense lesion masks under the "
                "prospectively locked sanity gate. Use no final outer-CV "
                "segmentation outcome."
            ),

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
            "cova3d_A1_3_operational_geometry_neutralization",

        "current_stage":
            "cova3d_dense_sanity_evaluation_pending",

        "cova3d_effective_protocol":
            PROTOCOL_AFTER,

        "cova3d_geometry_operationality":
            "PASS_AFTER_A1_3",

        "cova3d_annotation_artifact_version":
            "v1.3",

        "cova3d_R2_retraining_required":
            False,

        "cova3d_dense_outcomes_opened":
            False,

        "cova3d_factorial_training_authorized":
            False,

        "cova3d_final_outer_cv_access":
            0,

        "next_action":
            "Run 09E-SANITY-EVAL only.",

        "updated_at_utc":
            NOW_ISO,
    }
)


write_json(
    PROJECT_STATE_PATH,
    project_state,
)


# ==========================================================================================
# 16. REGRESSION TEST
# ==========================================================================================

heading(
    "STEP 11/11 — REGRESSION TESTS"
)


write_text(
    TEST_PATH,
    r'''
from pathlib import Path
import json

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_A1_3_protocol():

    cfg = yaml.safe_load(
        (
            ROOT
            / "configs/"
            "cova3d_protocol_amendment_A1_3_operational_neutralization.yaml"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert cfg[
        "status"
    ] == "PROSPECTIVELY_FROZEN_BEFORE_DENSE_OUTCOME_OPENING"

    assert cfg[
        "amendment_id"
    ] == "A1.3_OPERATIONAL_GEOMETRY_NEUTRALIZATION"

    assert cfg[
        "trigger_evidence"
    ][
        "unique_alias_physical_components"
    ] == 8

    assert cfg[
        "trigger_evidence"
    ][
        "quota2_DIS_equals_FRG"
    ] == 0

    assert cfg[
        "neutralization_rule"
    ][
        "coverage_selection_changed"
    ] is False

    assert cfg[
        "neutralization_rule"
    ][
        "per_component_quota_changed"
    ] is False

    assert cfg[
        "neutralization_rule"
    ][
        "patient_FG_budget_changed"
    ] is False

    assert cfg[
        "neutralization_rule"
    ][
        "background_changed"
    ] is False


def test_A1_3_audit_pass():

    audit = json.loads(
        (
            ROOT
            / "experiments/audits/"
            "block09e_geometry_operationality_repair_A1_3.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert audit[
        "status"
    ] == "PASS"

    assert audit[
        "post_repair_operationality"
    ][
        "operational_component_aliases"
    ] == 0

    assert audit[
        "post_repair_operationality"
    ][
        "neutral_identity_failures"
    ] == 0

    assert audit[
        "post_repair_operationality"
    ][
        "operational_topology_failures"
    ] == 0

    assert audit[
        "post_repair_operationality"
    ][
        "informative_whole_case_aliases"
    ] == 0

    assert audit[
        "post_repair_operationality"
    ][
        "strict_pass"
    ] is True

    assert audit[
        "authorization"
    ][
        "factorial_training"
    ] is False

    assert audit[
        "authorization"
    ][
        "dense_sanity_evaluation"
    ] is True


def test_A1_3_R2_compatibility():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_A1_3_R2_development_compatibility_v1_0.csv"
    )

    assert len(frame) == 24

    assert frame[
        "exact_sparse_arrays_equal"
    ].all()

    assert frame[
        "semantic_equal"
    ].all()


def test_A1_3_artifact_manifest():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_training_grid_candidate_artifact_manifest_v1_3.csv"
    )

    assert len(frame) == 120

    assert frame[
        "sparse_only"
    ].all()

    assert frame[
        "coordinate_level_QA"
    ].all()

    assert frame[
        "trainer_candidate"
    ].all()

    assert (
        ~frame[
            "training_authorized"
        ]
    ).all()


def test_A1_3_operational_classes():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_operational_geometry_class_v1_0.csv"
    )

    counts = (
        frame[
            "A1_3_operational_geometry_class"
        ]
        .value_counts()
        .to_dict()
    )

    assert counts[
        "GLOBAL_OPERATIONALLY_NEUTRAL_ALIAS"
    ] == 8

    assert (
        sum(
            counts.values()
        )
        == len(frame)
    )


def test_A1_3_state():

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
    ] == "09E-GEOM-OP-REPAIR"

    assert state[
        "A1_3_status"
    ] == "FROZEN_PASS"

    assert state[
        "annotation_methodology_structurally_ready"
    ] is True

    assert state[
        "A1_3_R2_compatibility"
    ] == "PASS"

    assert state[
        "sanity_fit_R2_retraining_required_after_A1_3"
    ] is False

    assert state[
        "dense_outcomes_opened_in_cova3d"
    ] is False

    assert state[
        "factorial_training_authorized"
    ] is False

    assert state[
        "next_block"
    ] == "09E-SANITY-EVAL"
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

        "tests/test_cova3d_geometry_operationality_A1_3.py",

        "-q",
        "-p",
        "no:cacheprovider",
    ],
    env=pytest_env,
    check=False,
)


print(
    tests.stdout.rstrip()
)


if tests.stderr.strip():

    print(
        tests.stderr.rstrip()
    )


if tests.returncode != 0:

    raise RuntimeError(
        "A1.3 regression tests failed."
    )


print(
    "✓ A1.3 regression tests                : PASS"
)


# ==========================================================================================
# 17. SOURCE CAPTURE
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
        "# COVA-3D — BLOCK 09E-GEOM-OP-REPAIR"
        in cell
    ):

        source_path = (
            REPO
            / "scripts/code_blocks/"
            "block09e_geometry_operationality_repair_A1_3.py"
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
] = source_capture


write_json(
    A13_AUDIT_PATH,
    audit_payload,
)


# ==========================================================================================
# 18. GITIGNORE V1.3 NPZ EXCEPTION
# ==========================================================================================

gitignore_path = (
    REPO
    / ".gitignore"
)


gitignore_text = gitignore_path.read_text(
    encoding="utf-8"
)


exception_text = """
# COVA-3D A1.3 frozen sparse trainer artifacts
!data/cova3d_training_grid_candidates_v1_3/
!data/cova3d_training_grid_candidates_v1_3/**/*.npz
""".strip()


if (
    "cova3d_training_grid_candidates_v1_3"
    not in gitignore_text
):

    gitignore_path.write_text(
        gitignore_text.rstrip()
        + "\n\n"
        + exception_text
        + "\n",
        encoding="utf-8",
    )


# ==========================================================================================
# 19. NORMALIZE TEXT
# ==========================================================================================

text_paths = [
    gitignore_path,
    A13_CONFIG_PATH,
    OPERATIONAL_CLASS_PATH,
    PROMOTED_BANK_PATH,
    NEW_ARTIFACT_MANIFEST,
    NEW_CONDITION_MANIFEST,
    NEW_COMPONENT_AUDIT,
    A13_CASE_AUDIT_PATH,
    A13_DEV_COMPAT_PATH,
    A13_AUDIT_PATH,
    COVA_STATE_PATH,
    PROJECT_STATE_PATH,
    TEST_PATH,
]


source_path = (
    REPO
    / "scripts/code_blocks/"
    "block09e_geometry_operationality_repair_A1_3.py"
)


if source_path.exists():

    text_paths.append(
        source_path
    )


for path in text_paths:

    content = Path(
        path
    ).read_text(
        encoding="utf-8"
    )

    Path(
        path
    ).write_text(
        content.rstrip()
        + "\n",
        encoding="utf-8",
    )


# ==========================================================================================
# 20. REPOSITORY MANIFEST — LAST
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
            PROTOCOL_AFTER,

        "annotation_artifact_version":
            "v1.3",

        "A1_3_status":
            "PASS",

        "alias_promoted_neutral_components":
            n_alias_neutral,

        "geometry_operational_components":
            n_operational,

        "resolution_neutral_components":
            n_resolution_neutral,

        "post_repair_operational_aliases":
            operational_alias_count,

        "post_repair_informative_whole_case_aliases":
            informative_whole_case_aliases,

        "R2_development_artifacts_unchanged":
            24,

        "R2_retraining_required":
            False,

        "dense_outcomes_opened":
            False,

        "factorial_training_authorized":
            False,

        "next_block":
            "09E-SANITY-EVAL",

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
# 21. COMMIT + PUSH
# ==========================================================================================

heading(
    "COMMIT A1.3 PRE-OUTCOME REPAIR"
)


git_paths = [
    ".gitignore",

    "COVA3D_STATE.json",
    "PROJECT_STATE.json",
    "REPOSITORY_MANIFEST.json",

    "configs/"
    "cova3d_protocol_amendment_A1_3_operational_neutralization.yaml",

    "data/manifests/"
    "cova3d_operational_geometry_class_v1_0.csv",

    "data/manifests/"
    "cova3d_A1_3_promoted_neutral_bank_v1_0.csv",

    "data/manifests/"
    "cova3d_training_grid_candidate_artifact_manifest_v1_3.csv",

    "data/manifests/"
    "cova3d_training_grid_condition_manifest_v1_3.csv",

    "data/manifests/"
    "cova3d_training_grid_component_audit_v1_3.csv",

    "data/manifests/"
    "cova3d_A1_3_case_operationality_audit_v1_0.csv",

    "data/manifests/"
    "cova3d_A1_3_R2_development_compatibility_v1_0.csv",

    "experiments/audits/"
    "block09e_geometry_operationality_repair_A1_3.json",

    "tests/"
    "test_cova3d_geometry_operationality_A1_3.py",

    "data/cova3d_training_grid_candidates_v1_3",
]


if source_path.exists():

    git_paths.append(
        "scripts/code_blocks/"
        "block09e_geometry_operationality_repair_A1_3.py"
    )


sh(
    [
        "git",
        "add",
        "-f",
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


staged_v13_npz = [
    path
    for path in staged_files
    if (
        path.startswith(
            "data/cova3d_training_grid_candidates_v1_3/"
        )
        and path.endswith(
            ".npz"
        )
    )
]


if len(
    staged_v13_npz
) != 120:

    raise RuntimeError(
        "Expected exactly 120 staged v1.3 NPZ artifacts; observed "
        + str(
            len(
                staged_v13_npz
            )
        )
    )


if any(
    path.endswith(
        ".pt"
    )
    for path in staged_files
):

    raise RuntimeError(
        "Model/checkpoint accidentally staged."
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
    "✓ v1.3 NPZ artifacts staged            : 120 /120"
)

print(
    "✓ Model/checkpoint staged              : NO"
)


sh(
    [
        "git",
        "commit",
        "-m",
        "protocol: freeze COVA-3D operational geometry amendment A1.3",
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
        "Repository is not clean after A1.3:\n"
        + final_status
    )


# ==========================================================================================
# 22. FINAL REPORT
# ==========================================================================================

heading(
    "COVA-3D 09E-GEOM-OP-REPAIR — FINAL REPORT"
)


print(
    "Starting commit                        :",
    head[:12],
)

print(
    "A1.3 commit                            :",
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
    "TRIGGER"
)

print(
    "-------"
)

print(
    "A1.2 resolvable DIS == FRG aliases     : 8"
)

print(
    "Alias quota                            : 3"
)

print(
    "Quota=2 DIS == FRG aliases             : 0"
)

print(
    "Alias-triggering physical components   : 8"
)

print(
    "Alias-triggering development components: 0"
)

print()

print(
    "A1.3 OPERATIONAL CLASSES"
)

print(
    "------------------------"
)

print(
    "Geometry-operational components        :",
    n_operational,
)

print(
    "Resolution-neutral components          :",
    n_resolution_neutral,
)

print(
    "Alias-promoted neutral components      :",
    n_alias_neutral,
)

print(
    "Total geometry-neutral components      :",
    n_resolution_neutral
    + n_alias_neutral,
)

print()

print(
    "REPAIR"
)

print(
    "------"
)

print(
    "Coverage selections changed            : NO"
)

print(
    "Per-component quotas changed           : NO"
)

print(
    "Patient FG budgets changed             : NO"
)

print(
    "Background supervision changed         : NO"
)

print(
    "Promoted neutral banks                 :",
    len(
        promoted_banks
    ),
)

print(
    "Artifacts with coordinate edits        :",
    len(
        changed_artifacts
    ),
)

print(
    "New sparse artifact version            : v1.3"
)

print()

print(
    "POST-REPAIR OPERATIONALITY"
)

print(
    "---------------------------"
)

print(
    "Operational component aliases          :",
    operational_alias_count,
)

print(
    "Neutral identity failures              :",
    neutral_identity_failures,
)

print(
    "Operational topology failures          :",
    operational_topology_failures,
)

print(
    "Informative whole-case aliases         :",
    informative_whole_case_aliases,
)

print(
    "Neutral-only case-coverages            :",
    neutral_only_case_coverages,
)

print(
    "Neutral-only identity failures         :",
    neutral_only_identity_failures,
)

print(
    "Strict A1.3 operationality             : PASS"
)

print()

print(
    "R2 COMPATIBILITY"
)

print(
    "----------------"
)

print(
    "Development artifacts checked          : 24"
)

print(
    "Development sparse arrays unchanged    : 24 /24"
)

print(
    "R2 retraining required                 : NO"
)

print(
    "R2 model SHA                           :",
    EXPECTED_R2_MODEL_SHA,
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
    "Final outer-CV segmentation outcomes   : 0"
)

print()

print(
    "AUTHORIZATION"
)

print(
    "-------------"
)

print(
    "Effective protocol                     :",
    PROTOCOL_AFTER,
)

print(
    "Annotation methodology structurally ready : YES"
)

print(
    "Dense sanity evaluation authorized     : YES"
)

print(
    "Factorial training authorized          : NO"
)

print(
    "Method development authorized          : NO"
)

print(
    "Final outer-CV outcomes authorized     : NO"
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
    "Do NOT retrain R2."
)

print(
    "Do NOT begin factorial training."
)

print(
    "If this block passes, the next block is 09E-SANITY-EVAL."
)

print(
    "=" * 132
)
