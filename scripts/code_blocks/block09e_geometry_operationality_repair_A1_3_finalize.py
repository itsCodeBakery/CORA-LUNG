# ==========================================================================================
# COVA-3D — BLOCK 09E-GEOM-OP-REPAIR-FINALIZE
# Finalize A1.3 After Confirmed Direct-Array Ordering Bug
#
# EXPECTED COMMITTED HEAD
# -----------------------
# 8dc72d01db2f
#
# CONFIRMED DIAGNOSIS
# -------------------
# Failed-attempt A1.3 development artifacts:
#
#   FG semantic equality                 24/24
#   BG semantic equality                 24/24
#   grouped membership semantic equality 24/24
#   membership coordinate arrays exact   24/24
#   membership group arrays exact        24/24
#
# Raw direct arrays exact:
#   0/24
#
# After applying ORIGINAL B2 direct-array lexicographic ordering:
#   direct coordinates exact             24/24
#   direct labels exact                  24/24
#   all four sparse arrays exact         24/24
#
# ROOT CAUSE
# ----------
# The failed A1.3 writer omitted the ORIGINAL 09D-B2 global direct-supervision
# lexicographic sort:
#
#     direct_order = np.lexsort((x, y, z))
#
# This changed representation order only.
# Development supervision semantics DID NOT CHANGE.
#
# ACTION
# ------
# 1. Archive a pre-fix ordering diagnostic.
# 2. Canonicalize direct arrays in ALL 120 already-generated v1.3 NPZs using
#    the exact original B2 rule.
# 3. Recompute semantic hashes + file hashes.
# 4. Refresh v1.3 condition/artifact manifests.
# 5. Verify all 24 permanent-development v1.3 artifacts are now EXACTLY equal
#    to v1.2 across all four sparse arrays.
# 6. Revalidate A1.3 operationality.
# 7. Verify budgets/background/component quotas remain unchanged.
# 8. Future-proof historical R2 and geometry-diagnostic tests so that they test
#    immutable historical artifacts rather than mutable CURRENT state.
# 9. Freeze A1.3 audit/state.
# 10. Commit + push.
#
# THIS BLOCK DOES NOT
# -------------------
# ✗ retrain R2
# ✗ call optimizer.step()
# ✗ load CT
# ✗ load dense lesion masks
# ✗ load lung masks
# ✗ load predictions
# ✗ load model/checkpoint
# ✗ access final-CV segmentation outcomes
#
# AFTER PASS
# ----------
# Next block = 09E-SANITY-EVAL
#
# Factorial training remains LOCKED.
# ==========================================================================================


from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import hashlib
import io
import json
import os
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

EXPECTED_HEAD = (
    "8dc72d01db2f"
)

BLOCK = (
    "09E-GEOM-OP-REPAIR-FINALIZE"
)

SCIENTIFIC_BLOCK = (
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

ORDERING_DIAG_PATH = (
    REPO
    / "data/manifests/"
    "cova3d_A1_3_direct_array_ordering_diagnostic_v1_0.csv"
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

ORDERING_AUDIT_PATH = (
    REPO
    / "experiments/audits/"
    "block09e_A1_3_direct_array_ordering_fix.json"
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

R2_TEST_PATH = (
    REPO
    / "tests/"
    "test_cova3d_sanity_fit_R2.py"
)

GEOM_DIAG_TEST_PATH = (
    REPO
    / "tests/"
    "test_cova3d_geometry_operationality.py"
)

A13_TEST_PATH = (
    REPO
    / "tests/"
    "test_cova3d_geometry_operationality_A1_3.py"
)

CONDITIONS = [
    "C50_COH",
    "C50_DIS",
    "C50_FRG",
    "C100_COH",
    "C100_DIS",
    "C100_FRG",
]

CONDITIONS_BY_COVERAGE = {
    0.50: {
        "COH":
            "C50_COH",

        "DIS":
            "C50_DIS",

        "FRG":
            "C50_FRG",
    },

    1.00: {
        "COH":
            "C100_COH",

        "DIS":
            "C100_DIS",

        "FRG":
            "C100_FRG",
    },
}

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

EXPECTED_DEV_ARTIFACTS = (
    24
)

EXPECTED_PROMOTED = (
    8
)

EXPECTED_R2_SHA = (
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
                    8 * 1024 * 1024
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

    return frozenset(
        tuple(
            int(
                value
            )
            for value in row
        )
        for row in np.asarray(
            coords,
            dtype=np.int32,
        )
    )


def membership_set(
    coords,
    groups,
):

    coords = np.asarray(
        coords,
        dtype=np.int32,
    )

    groups = np.asarray(
        groups,
        dtype=np.int32,
    )

    return frozenset(
        (
            int(
                group
            ),
            int(
                coord[
                    0
                ]
            ),
            int(
                coord[
                    1
                ]
            ),
            int(
                coord[
                    2
                ]
            ),
        )
        for coord, group in zip(
            coords,
            groups,
        )
    )


def direct_sort_order(coords):

    coords = np.asarray(
        coords,
        dtype=np.int32,
    )

    # EXACT original B2 order:
    #
    # np.lexsort((x, y, z))
    #
    # Final primary key is z, then y, then x.

    return np.lexsort(
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


def canonicalize_direct(
    coords,
    labels,
):

    order = direct_sort_order(
        coords
    )

    return (
        np.asarray(
            coords,
            dtype=np.int32,
        )[
            order
        ],

        np.asarray(
            labels,
            dtype=np.int8,
        )[
            order
        ],
    )


def deterministic_npz(
    path,
    **arrays,
):

    path = Path(
        path
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

        if set(
            data.files
        ) != required:

            raise RuntimeError(
                "Unexpected sparse artifact schema:\n"
                + str(
                    path
                )
                + "\nObserved: "
                + str(
                    data.files
                )
            )

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

    return {
        "coords":
            coords,

        "labels":
            labels,

        "membership_coords":
            membership_coords,

        "membership_groups":
            membership_groups,

        "fg":
            coord_set(
                coords[
                    labels
                    == 1
                ]
            ),

        "bg":
            coord_set(
                coords[
                    labels
                    == 0
                ]
            ),

        "membership_semantic":
            membership_set(
                membership_coords,
                membership_groups,
            ),
    }


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


def connected_count(coords):

    remaining = set(
        coord_set(
            coords
        )
    )

    total = (
        0
    )

    while remaining:

        start = min(
            remaining
        )

        remaining.remove(
            start
        )

        stack = [
            start
        ]

        total += (
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

    return total


def groups_from_sparse(artifact):

    groups = {}

    ids = sorted(
        np.unique(
            artifact[
                "membership_groups"
            ]
        ).tolist()
    )

    for component_id in ids:

        component_id = int(
            component_id
        )

        groups[
            component_id
        ] = np.asarray(
            artifact[
                "membership_coords"
            ][
                artifact[
                    "membership_groups"
                ]
                == component_id
            ],
            dtype=np.int32,
        )

    return groups


def make_git_auth():

    token = UserSecretsClient().get_secret(
        "pushCora"
    )

    if not token:

        raise RuntimeError(
            "Kaggle secret pushCora unavailable."
        )

    token = token.strip()

    askpass = Path(
        "/tmp/cova3d_git_askpass_A1_3_finalize.sh"
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

    return token, askpass, env


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
# 2. VERIFY FAILED-ATTEMPT STATE
# ==========================================================================================

heading(
    "COVA-3D 09E-GEOM-OP-REPAIR-FINALIZE — VERIFY STATE"
)


head = sh(
    [
        "git",
        "rev-parse",
        "HEAD",
    ]
).stdout.strip()


if not head.startswith(
    EXPECTED_HEAD
):

    raise RuntimeError(
        "Unexpected committed HEAD.\n"
        + "Expected: "
        + EXPECTED_HEAD
        + "\nObserved: "
        + head
    )


print(
    "✓ Committed HEAD                       :",
    head[:12],
)


working_tree_before = sh(
    [
        "git",
        "status",
        "--short",
    ]
).stdout


print()
print(
    "Failed-attempt working tree:"
)

print(
    working_tree_before.rstrip()
)


required_failed_attempt_paths = [
    NEW_ROOT,
    NEW_ARTIFACT_MANIFEST,
    NEW_CONDITION_MANIFEST,
    NEW_COMPONENT_AUDIT,
    OPERATIONAL_CLASS_PATH,
    PROMOTED_BANK_PATH,
    A13_CASE_AUDIT_PATH,
    A13_CONFIG_PATH,
]


missing = [
    str(
        path
    )
    for path in required_failed_attempt_paths
    if not path.exists()
]


if missing:

    raise RuntimeError(
        "Expected failed-attempt A1.3 files missing:\n"
        + "\n".join(
            missing
        )
    )


state = json.loads(
    COVA_STATE_PATH.read_text(
        encoding="utf-8"
    )
)


if state.get(
    "last_completed_block"
) != "09E-GEOM-OP-DIAG":

    raise RuntimeError(
        "Committed/current state should still be at 09E-GEOM-OP-DIAG."
    )


if state.get(
    "dense_outcomes_opened_in_cova3d"
) is not False:

    raise RuntimeError(
        "Dense outcomes unexpectedly opened."
    )


if state.get(
    "factorial_training_authorized"
) is not False:

    raise RuntimeError(
        "Factorial training unexpectedly authorized."
    )


if str(
    state.get(
        "sanity_fit_R2_final_model_sha256"
    )
) != EXPECTED_R2_SHA:

    raise RuntimeError(
        "R2 model provenance mismatch."
    )


print()
print(
    "✓ Dense outcomes                       : SEALED"
)

print(
    "✓ Final outer-CV outcomes              : SEALED"
)

print(
    "✓ R2 frozen SHA                        :",
    EXPECTED_R2_SHA,
)

print(
    "✓ R2 retraining performed here         : NO"
)


# ==========================================================================================
# 3. LOAD MANIFESTS
# ==========================================================================================

heading(
    "STEP 1/10 — LOAD OLD + FAILED-ATTEMPT V1.3 MANIFESTS"
)


old_artifact_df = pd.read_csv(
    OLD_ARTIFACT_MANIFEST
)

old_condition_df = pd.read_csv(
    OLD_CONDITION_MANIFEST
)

new_artifact_df = pd.read_csv(
    NEW_ARTIFACT_MANIFEST
)

new_condition_df = pd.read_csv(
    NEW_CONDITION_MANIFEST
)

operational_class_df = pd.read_csv(
    OPERATIONAL_CLASS_PATH
)

split_df = pd.read_csv(
    SPLIT_PATH
)


if len(
    old_artifact_df
) != EXPECTED_ARTIFACTS:

    raise RuntimeError(
        "Old artifact manifest does not contain 120 rows."
    )


if len(
    new_artifact_df
) != EXPECTED_ARTIFACTS:

    raise RuntimeError(
        "Failed-attempt v1.3 manifest does not contain 120 rows."
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
    split_df.loc[
        split_df[
            "role"
        ]
        == "permanent_development",
        "case_id",
    ]
    .astype(
        str
    )
    .tolist()
)


if len(
    dev_cases
) != 4:

    raise RuntimeError(
        "Expected four permanent-development cases."
    )


class_lookup = {
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
                "A1_3_operational_geometry_class"
            ]
        )
    for _, row in operational_class_df.iterrows()
}


promoted_keys = sorted(
    [
        key
        for key, value in class_lookup.items()
        if value
        == NEW_ALIAS_NEUTRAL
    ]
)


if len(
    promoted_keys
) != EXPECTED_PROMOTED:

    raise RuntimeError(
        "Expected eight A1.3 alias-neutral physical components."
    )


if any(
    role_lookup[
        case_id
    ]
    != "final_outer_cv"
    for case_id, _ in promoted_keys
):

    raise RuntimeError(
        "A promoted A1.3 component belongs to permanent development."
    )


print(
    "✓ Old v1.2 artifact rows               : 120"
)

print(
    "✓ Failed-attempt v1.3 artifact rows    : 120"
)

print(
    "✓ Promoted alias-neutral components    : 8"
)

print(
    "✓ Promoted development components      : 0"
)


# ==========================================================================================
# 4. REPRODUCE / ARCHIVE PRE-FIX ORDERING BUG
# ==========================================================================================

heading(
    "STEP 2/10 — ARCHIVE CONFIRMED PRE-FIX ORDERING BUG"
)


pre_fix_rows = []


for case_id in dev_cases:

    for condition_id in CONDITIONS:

        old_path = (
            OLD_ROOT
            / case_id
            / (
                condition_id
                + ".npz"
            )
        )

        new_path = (
            NEW_ROOT
            / case_id
            / (
                condition_id
                + ".npz"
            )
        )


        old = load_sparse(
            old_path
        )

        new = load_sparse(
            new_path
        )


        membership_coords_exact = np.array_equal(
            old[
                "membership_coords"
            ],
            new[
                "membership_coords"
            ],
        )


        membership_groups_exact = np.array_equal(
            old[
                "membership_groups"
            ],
            new[
                "membership_groups"
            ],
        )


        fg_semantic_equal = (
            old[
                "fg"
            ]
            == new[
                "fg"
            ]
        )


        bg_semantic_equal = (
            old[
                "bg"
            ]
            == new[
                "bg"
            ]
        )


        grouped_semantic_equal = (
            old[
                "membership_semantic"
            ]
            == new[
                "membership_semantic"
            ]
        )


        new_canonical_coords, new_canonical_labels = canonicalize_direct(
            new[
                "coords"
            ],
            new[
                "labels"
            ],
        )


        canonical_full_equal = bool(
            np.array_equal(
                old[
                    "coords"
                ],
                new_canonical_coords,
            )
            and np.array_equal(
                old[
                    "labels"
                ],
                new_canonical_labels,
            )
            and membership_coords_exact
            and membership_groups_exact
        )


        old_sorted = np.array_equal(
            direct_sort_order(
                old[
                    "coords"
                ]
            ),
            np.arange(
                len(
                    old[
                        "coords"
                    ]
                )
            ),
        )


        new_sorted = np.array_equal(
            direct_sort_order(
                new[
                    "coords"
                ]
            ),
            np.arange(
                len(
                    new[
                        "coords"
                    ]
                )
            ),
        )


        pre_fix_rows.append(
            {
                "case_id":
                    case_id,

                "condition_id":
                    condition_id,

                "raw_direct_coords_exact":
                    np.array_equal(
                        old[
                            "coords"
                        ],
                        new[
                            "coords"
                        ],
                    ),

                "raw_direct_labels_exact":
                    np.array_equal(
                        old[
                            "labels"
                        ],
                        new[
                            "labels"
                        ],
                    ),

                "membership_coords_exact":
                    membership_coords_exact,

                "membership_groups_exact":
                    membership_groups_exact,

                "FG_semantic_equal":
                    fg_semantic_equal,

                "BG_semantic_equal":
                    bg_semantic_equal,

                "grouped_membership_semantic_equal":
                    grouped_semantic_equal,

                "old_B2_sorted":
                    old_sorted,

                "failed_v1_3_B2_sorted":
                    new_sorted,

                "canonical_full_array_equal":
                    canonical_full_equal,
            }
        )


pre_fix_df = pd.DataFrame(
    pre_fix_rows
)


if len(
    pre_fix_df
) != EXPECTED_DEV_ARTIFACTS:

    raise RuntimeError(
        "Expected 24 development compatibility rows."
    )


if not pre_fix_df[
    "FG_semantic_equal"
].all():

    raise RuntimeError(
        "Unexpected true FG semantic difference. Stop."
    )


if not pre_fix_df[
    "BG_semantic_equal"
].all():

    raise RuntimeError(
        "Unexpected true BG semantic difference. Stop."
    )


if not pre_fix_df[
    "grouped_membership_semantic_equal"
].all():

    raise RuntimeError(
        "Unexpected grouped-membership semantic difference. Stop."
    )


if not pre_fix_df[
    "membership_coords_exact"
].all():

    raise RuntimeError(
        "Membership coordinate arrays differ. Stop."
    )


if not pre_fix_df[
    "membership_groups_exact"
].all():

    raise RuntimeError(
        "Membership group arrays differ. Stop."
    )


if not pre_fix_df[
    "canonical_full_array_equal"
].all():

    raise RuntimeError(
        "Canonical B2 ordering does not restore exact development arrays."
    )


pre_fix_df.to_csv(
    ORDERING_DIAG_PATH,
    index=False,
)


print(
    "✓ Development artifacts checked        : 24"
)

print(
    "✓ FG semantics equal                   : 24 /24"
)

print(
    "✓ BG semantics equal                   : 24 /24"
)

print(
    "✓ Membership arrays exact              : 24 /24"
)

print(
    "✓ Canonical full arrays exact          : 24 /24"
)

print(
    "✓ Diagnosis                            : ORDERING BUG ONLY"
)


# ==========================================================================================
# 5. CANONICALIZE ALL 120 V1.3 DIRECT ARRAYS
# ==========================================================================================

heading(
    "STEP 3/10 — RESTORE EXACT ORIGINAL B2 DIRECT-ARRAY ORDERING"
)


artifact_index_lookup = {
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
        index
    for index, row in new_artifact_df.iterrows()
}


condition_index_lookup = {
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
        index
    for index, row in new_condition_df.iterrows()
}


rewritten = (
    0
)


for _, row in tqdm(
    new_artifact_df.iterrows(),
    total=len(
        new_artifact_df
    ),
    desc="Canonicalize v1.3",
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


    artifact = load_sparse(
        path
    )


    canonical_coords, canonical_labels = canonicalize_direct(
        artifact[
            "coords"
        ],
        artifact[
            "labels"
        ],
    )


    deterministic_npz(
        path,

        supervision_voxel_zyx=
            canonical_coords.astype(
                np.int32
            ),

        supervision_label=
            canonical_labels.astype(
                np.int8
            ),

        fg_membership_voxel_zyx=
            artifact[
                "membership_coords"
            ].astype(
                np.int32
            ),

        fg_membership_group_id=
            artifact[
                "membership_groups"
            ].astype(
                np.int32
            ),
    )


    rewritten += (
        1
    )


    reloaded = load_sparse(
        path
    )


    if not np.array_equal(
        direct_sort_order(
            reloaded[
                "coords"
            ]
        ),
        np.arange(
            len(
                reloaded[
                    "coords"
                ]
            )
        ),
    ):

        raise RuntimeError(
            "Canonical direct ordering failed:\n"
            + str(
                path
            )
        )


    annotation_sha = semantic_hash(
        reloaded[
            "coords"
        ].astype(
            np.int32
        ),
        reloaded[
            "labels"
        ].astype(
            np.int8
        ),
        reloaded[
            "membership_coords"
        ].astype(
            np.int32
        ),
        reloaded[
            "membership_groups"
        ].astype(
            np.int32
        ),
    )


    file_sha = sha256_file(
        path
    )


    artifact_index = artifact_index_lookup[
        (
            case_id,
            condition_id,
        )
    ]


    condition_index = condition_index_lookup[
        (
            case_id,
            condition_id,
        )
    ]


    new_artifact_df.loc[
        artifact_index,
        "annotation_semantic_sha256",
    ] = annotation_sha


    new_artifact_df.loc[
        artifact_index,
        "artifact_file_sha256",
    ] = file_sha


    new_condition_df.loc[
        condition_index,
        "annotation_semantic_sha256",
    ] = annotation_sha


    new_condition_df.loc[
        condition_index,
        "artifact_file_sha256",
    ] = file_sha


new_artifact_df.to_csv(
    NEW_ARTIFACT_MANIFEST,
    index=False,
)


new_condition_df.to_csv(
    NEW_CONDITION_MANIFEST,
    index=False,
)


if rewritten != EXPECTED_ARTIFACTS:

    raise RuntimeError(
        "Expected to canonicalize 120 artifacts."
    )


print(
    "✓ V1.3 artifacts canonicalized         : 120 /120"
)

print(
    "✓ Original B2 direct ordering restored : YES"
)

print(
    "✓ Artifact hashes refreshed            : 120 /120"
)

print(
    "✓ Semantic hashes refreshed            : 120 /120"
)


# ==========================================================================================
# 6. FULL A1.3 STRUCTURAL CONSISTENCY VS V1.2
# ==========================================================================================

heading(
    "STEP 4/10 — VERIFY BUDGET / BG / QUOTA CONSISTENCY"
)


old_artifact_lookup = {
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
    for _, row in old_artifact_df.iterrows()
}


new_data = {}

old_data = {}


consistency_rows = []


for _, row in tqdm(
    new_artifact_df.iterrows(),
    total=len(
        new_artifact_df
    ),
    desc="Structural comparison",
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


    old_path = (
        REPO
        / str(
            old_artifact_lookup[
                (
                    case_id,
                    condition_id,
                )
            ][
                "artifact_file"
            ]
        )
    )


    new_path = (
        REPO
        / str(
            row[
                "artifact_file"
            ]
        )
    )


    old = load_sparse(
        old_path
    )

    new = load_sparse(
        new_path
    )


    old_data[
        (
            case_id,
            condition_id,
        )
    ] = old


    new_data[
        (
            case_id,
            condition_id,
        )
    ] = new


    old_groups = groups_from_sparse(
        old
    )

    new_groups = groups_from_sparse(
        new
    )


    component_ids_equal = (
        sorted(
            old_groups.keys()
        )
        == sorted(
            new_groups.keys()
        )
    )


    if not component_ids_equal:

        raise RuntimeError(
            "A1.3 changed selected component set."
        )


    quotas_equal = all(
        len(
            old_groups[
                component_id
            ]
        )
        == len(
            new_groups[
                component_id
            ]
        )
        for component_id in old_groups
    )


    if not quotas_equal:

        raise RuntimeError(
            "A1.3 changed a component quota."
        )


    bg_equal = (
        old[
            "bg"
        ]
        == new[
            "bg"
        ]
    )


    if not bg_equal:

        raise RuntimeError(
            "A1.3 changed background supervision."
        )


    fg_budget_equal = (
        len(
            old[
                "fg"
            ]
        )
        == len(
            new[
                "fg"
            ]
        )
    )


    if not fg_budget_equal:

        raise RuntimeError(
            "A1.3 changed total FG budget."
        )


    consistency_rows.append(
        {
            "case_id":
                case_id,

            "condition_id":
                condition_id,

            "role":
                role_lookup[
                    case_id
                ],

            "selected_component_set_equal":
                component_ids_equal,

            "per_component_quotas_equal":
                quotas_equal,

            "FG_budget_equal":
                fg_budget_equal,

            "BG_semantic_equal":
                bg_equal,

            "full_sparse_arrays_exact":
                bool(
                    np.array_equal(
                        old[
                            "coords"
                        ],
                        new[
                            "coords"
                        ],
                    )
                    and np.array_equal(
                        old[
                            "labels"
                        ],
                        new[
                            "labels"
                        ],
                    )
                    and np.array_equal(
                        old[
                            "membership_coords"
                        ],
                        new[
                            "membership_coords"
                        ],
                    )
                    and np.array_equal(
                        old[
                            "membership_groups"
                        ],
                        new[
                            "membership_groups"
                        ],
                    )
                ),
        }
    )


consistency_df = pd.DataFrame(
    consistency_rows
)


print(
    "✓ Selected-component sets unchanged    : 120 /120"
)

print(
    "✓ Component quotas unchanged           : 120 /120"
)

print(
    "✓ FG budgets unchanged                 : 120 /120"
)

print(
    "✓ BG supervision unchanged             : 120 /120"
)


# ==========================================================================================
# 7. PROVE R2 DEVELOPMENT COMPATIBILITY EXACTLY
# ==========================================================================================

heading(
    "STEP 5/10 — PROVE R2 DEVELOPMENT COMPATIBILITY"
)


dev_compat = consistency_df[
    consistency_df[
        "role"
    ]
    == "permanent_development"
].copy()


if len(
    dev_compat
) != EXPECTED_DEV_ARTIFACTS:

    raise RuntimeError(
        "Expected 24 development compatibility rows."
    )


if not dev_compat[
    "full_sparse_arrays_exact"
].all():

    failed = dev_compat.loc[
        ~dev_compat[
            "full_sparse_arrays_exact"
        ]
    ]

    raise RuntimeError(
        "Development arrays still differ after ordering repair:\n"
        + failed.to_string(
            index=False
        )
    )


old_condition_lookup = {
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
    for _, row in old_condition_df.iterrows()
}


new_condition_lookup = {
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
    for _, row in new_condition_df.iterrows()
}


compat_rows = []


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


        old_semantic_hash = str(
            old_condition_lookup[
                (
                    case_id,
                    condition_id,
                )
            ][
                "annotation_semantic_sha256"
            ]
        )


        new_semantic_hash = str(
            new_condition_lookup[
                (
                    case_id,
                    condition_id,
                )
            ][
                "annotation_semantic_sha256"
            ]
        )


        exact_arrays = bool(
            np.array_equal(
                old[
                    "coords"
                ],
                new[
                    "coords"
                ],
            )
            and np.array_equal(
                old[
                    "labels"
                ],
                new[
                    "labels"
                ],
            )
            and np.array_equal(
                old[
                    "membership_coords"
                ],
                new[
                    "membership_coords"
                ],
            )
            and np.array_equal(
                old[
                    "membership_groups"
                ],
                new[
                    "membership_groups"
                ],
            )
        )


        compat_rows.append(
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

                "grouped_membership_equal":
                    old[
                        "membership_semantic"
                    ]
                    == new[
                        "membership_semantic"
                    ],

                "annotation_semantic_hash_equal":
                    old_semantic_hash
                    == new_semantic_hash,
            }
        )


compat_df = pd.DataFrame(
    compat_rows
)


if not compat_df[
    "exact_sparse_arrays_equal"
].all():

    raise RuntimeError(
        "Development sparse arrays are not exactly identical."
    )


if not compat_df[
    "annotation_semantic_hash_equal"
].all():

    raise RuntimeError(
        "Development semantic hashes are not identical."
    )


compat_df.to_csv(
    A13_DEV_COMPAT_PATH,
    index=False,
)


print(
    "✓ Development artifacts                : 24"
)

print(
    "✓ Exact sparse-array equality          : 24 /24"
)

print(
    "✓ FG semantics identical               : 24 /24"
)

print(
    "✓ BG semantics identical               : 24 /24"
)

print(
    "✓ Membership semantics identical       : 24 /24"
)

print(
    "✓ Annotation semantic hashes identical : 24 /24"
)

print(
    "✓ R2 retraining required               : NO"
)


# ==========================================================================================
# 8. REVALIDATE A1.3 OPERATIONALITY
# ==========================================================================================

heading(
    "STEP 6/10 — REVALIDATE A1.3 OPERATIONALITY"
)


cases = sorted(
    new_artifact_df[
        "case_id"
    ].astype(
        str
    ).unique().tolist()
)


operational_aliases = (
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


case_rows = []


for case_id in cases:

    for coverage in [
        0.50,
        1.00,
    ]:

        condition_ids = CONDITIONS_BY_COVERAGE[
            coverage
        ]


        geometry_groups = {
            geometry:
                groups_from_sparse(
                    new_data[
                        (
                            case_id,
                            condition_ids[
                                geometry
                            ],
                        )
                    ]
                )
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
                "Component sets differ across geometry after A1.3."
            )


        operational_selected = (
            0
        )


        for component_id in component_ids:

            key = (
                case_id,
                component_id,
            )


            component_class = class_lookup[
                key
            ]


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


            if component_class == NEW_OPERATIONAL:

                operational_selected += (
                    1
                )


                alias = bool(
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


                if alias:

                    operational_aliases += (
                        1
                    )


                coh_cc = connected_count(
                    geometry_groups[
                        "COH"
                    ][
                        component_id
                    ]
                )

                dis_cc = connected_count(
                    geometry_groups[
                        "DIS"
                    ][
                        component_id
                    ]
                )

                frg_cc = connected_count(
                    geometry_groups[
                        "FRG"
                    ][
                        component_id
                    ]
                )


                if not (
                    coh_cc
                    == 1
                    and dis_cc
                    >= 2
                    and frg_cc
                    >= 2
                ):

                    operational_topology_failures += (
                        1
                    )


            elif component_class in {
                NEW_ALIAS_NEUTRAL,
                NEW_RESOLUTION_NEUTRAL,
            }:

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


            else:

                raise RuntimeError(
                    "Unknown A1.3 operational class."
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


        case_rows.append(
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


case_df = pd.DataFrame(
    case_rows
)


case_df.to_csv(
    A13_CASE_AUDIT_PATH,
    index=False,
)


strict_operationality_pass = bool(
    operational_aliases
    == 0
    and neutral_identity_failures
    == 0
    and operational_topology_failures
    == 0
    and informative_whole_case_aliases
    == 0
    and neutral_only_identity_failures
    == 0
    and case_df[
        "whole_case_operationality_pass"
    ].all()
)


if not strict_operationality_pass:

    raise RuntimeError(
        "A1.3 operationality no longer passes after canonicalization."
    )


print(
    "✓ Operational component aliases        :",
    operational_aliases,
)

print(
    "✓ Neutral identity failures            :",
    neutral_identity_failures,
)

print(
    "✓ Operational topology failures        :",
    operational_topology_failures,
)

print(
    "✓ Informative whole-case aliases       :",
    informative_whole_case_aliases,
)

print(
    "✓ Neutral-only case-coverages          :",
    neutral_only_case_coverages,
)

print(
    "✓ Neutral-only identity failures       :",
    neutral_only_identity_failures,
)

print(
    "✓ Strict A1.3 operationality           : PASS"
)


# ==========================================================================================
# 9. VERIFY ALL 120 FINAL V1.3 HASHES
# ==========================================================================================

heading(
    "STEP 7/10 — VERIFY FINAL V1.3 HASHES"
)


for _, row in tqdm(
    new_artifact_df.iterrows(),
    total=len(
        new_artifact_df
    ),
    desc="Verify final v1.3",
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
            "Final v1.3 file SHA mismatch."
        )


    artifact = load_sparse(
        path
    )


    observed_semantic = semantic_hash(
        artifact[
            "coords"
        ].astype(
            np.int32
        ),
        artifact[
            "labels"
        ].astype(
            np.int8
        ),
        artifact[
            "membership_coords"
        ].astype(
            np.int32
        ),
        artifact[
            "membership_groups"
        ].astype(
            np.int32
        ),
    )


    if observed_semantic != str(
        row[
            "annotation_semantic_sha256"
        ]
    ):

        raise RuntimeError(
            "Final v1.3 semantic SHA mismatch."
        )


    order = direct_sort_order(
        artifact[
            "coords"
        ]
    )


    if not np.array_equal(
        order,
        np.arange(
            len(
                order
            )
        ),
    ):

        raise RuntimeError(
            "Final v1.3 direct array is not B2-canonical."
        )


print(
    "✓ File SHA verification                : 120 /120"
)

print(
    "✓ Semantic SHA verification            : 120 /120"
)

print(
    "✓ B2 canonical direct ordering         : 120 /120"
)


# ==========================================================================================
# 10. FUTURE-PROOF HISTORICAL REGRESSION TESTS
# ==========================================================================================

heading(
    "STEP 8/10 — FUTURE-PROOF HISTORICAL TESTS"
)


# ------------------------------------------------------------------------------------------
# R2 historical test: test immutable R2 artifacts, NOT mutable CURRENT state.
# ------------------------------------------------------------------------------------------

write_text(
    R2_TEST_PATH,
    r'''
from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_R2_epoch_contract():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_sanity_fit_R2_epoch_log_v1_0.csv"
    )

    assert len(frame) == 20

    assert (
        frame[
            "logical_microbatches"
        ]
        == 100
    ).all()

    assert (
        frame[
            "logical_optimizer_steps"
        ]
        == 50
    ).all()

    assert int(
        frame.iloc[
            -1
        ][
            "successful_optimizer_steps_total"
        ]
    ) == 1000

    assert (
        frame[
            [
                "coronacases_004_samples",
                "coronacases_008_samples",
                "radiopaedia_14_85914_0_samples",
                "radiopaedia_27_86410_0_samples",
            ]
        ]
        == 25
    ).all().all()

    assert (
        frame[
            "foreground_centered"
        ]
        + frame[
            "background_centered"
        ]
        + frame[
            "uniform_crop"
        ]
        == 100
    ).all()

    assert (
        frame[
            "amp_scale_min"
        ]
        >= 1.0
    ).all()

    assert (
        frame[
            "amp_scale_max"
        ]
        <= 4096.0
    ).all()

    assert np.isfinite(
        frame[
            [
                "mean_loss",
                "mean_partial_bce",
                "mean_partial_dice_fp32",
                "mean_preclip_grad_norm",
                "max_preclip_grad_norm",
            ]
        ].to_numpy()
    ).all()


def test_R2_historical_audit():

    audit = json.loads(
        (
            ROOT
            / "experiments/audits/"
            "block09e_sanity_fit_R2.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert audit[
        "status"
    ] == "FIT_COMPLETE_EVALUATION_PENDING"

    assert audit[
        "R1_updates_retained"
    ] == 0

    assert audit[
        "successful_optimizer_steps"
    ] == 1000

    assert audit[
        "firewall"
    ][
        "dense_lesion_masks_accessed"
    ] == 0

    assert audit[
        "firewall"
    ][
        "dense_lung_masks_accessed"
    ] == 0

    assert audit[
        "firewall"
    ][
        "final_outer_cv_access"
    ] == 0

    assert audit[
        "sanity_gate_applied"
    ] is False

    assert audit[
        "factorial_training_authorized"
    ] is False
'''
)


# ------------------------------------------------------------------------------------------
# Historical geometry diagnostic: test immutable diagnostic artifacts only.
# ------------------------------------------------------------------------------------------

write_text(
    GEOM_DIAG_TEST_PATH,
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


def test_geometry_historical_audit_firewall():

    audit = json.loads(
        (
            ROOT
            / "experiments/audits/"
            "block09e_geometry_operationality_diagnostic.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert audit[
        "classification"
    ] == "REQUIRES_PROSPECTIVE_GEOMETRY_AMENDMENT"

    assert audit[
        "resolvable_aliases"
    ][
        "DIS_equals_FRG"
    ] == 8

    assert audit[
        "resolvable_aliases"
    ][
        "quota2_DIS_equals_FRG"
    ] == 0

    assert audit[
        "case_level"
    ][
        "whole_case_geometry_aliases"
    ] == 2

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


def test_geometry_historical_component_table():

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

    assert int(
        resolvable[
            "dis_equals_frg"
        ].sum()
    ) == 8

    assert int(
        resolvable[
            "any_pairwise_alias"
        ].sum()
    ) == audit[
        "resolvable_aliases"
    ][
        "any_pairwise_alias"
    ]
'''
)


print(
    "✓ Historical R2 current-state test     : REMOVED"
)

print(
    "✓ Historical R2 immutable audit test   : ACTIVE"
)

print(
    "✓ Geometry diagnostic current-state test: REMOVED"
)

print(
    "✓ Geometry diagnostic immutable audit  : ACTIVE"
)


# ==========================================================================================
# 11. WRITE FINAL A1.3 AUDITS
# ==========================================================================================

heading(
    "STEP 9/10 — FREEZE A1.3 AUDIT / STATE"
)


ordering_audit = {
    "project":
        "COVA-3D",

    "block":
        BLOCK,

    "status":
        "PASS",

    "defect":
        "A1_3_DIRECT_ARRAY_ORDERING_BUG_ONLY",

    "root_cause":
        (
            "Failed-attempt A1.3 reconstructed FG and BG semantics correctly "
            "but omitted the original 09D-B2 global lexicographic sort of "
            "supervision_voxel_zyx and supervision_label."
        ),

    "original_B2_order":
        "np.lexsort((x, y, z))",

    "development_pre_fix":
        {
            "artifacts":
                24,

            "raw_direct_arrays_exact":
                int(
                    pre_fix_df[
                        "raw_direct_coords_exact"
                    ].sum()
                ),

            "membership_arrays_exact":
                int(
                    (
                        pre_fix_df[
                            "membership_coords_exact"
                        ]
                        & pre_fix_df[
                            "membership_groups_exact"
                        ]
                    ).sum()
                ),

            "FG_semantic_equal":
                int(
                    pre_fix_df[
                        "FG_semantic_equal"
                    ].sum()
                ),

            "BG_semantic_equal":
                int(
                    pre_fix_df[
                        "BG_semantic_equal"
                    ].sum()
                ),

            "canonical_full_arrays_equal":
                int(
                    pre_fix_df[
                        "canonical_full_array_equal"
                    ].sum()
                ),
        },

    "fix":
        {
            "scientific_annotation_changed":
                False,

            "representation_order_only":
                True,

            "v1_3_artifacts_canonicalized":
                120,

            "development_exact_arrays_after_fix":
                int(
                    compat_df[
                        "exact_sparse_arrays_equal"
                    ].sum()
                ),
        },

    "R2":
        {
            "retraining_required":
                False,

            "final_model_sha256":
                EXPECTED_R2_SHA,
        },

    "firewall":
        {
            "CT_arrays_accessed":
                0,

            "dense_masks_accessed":
                0,

            "predictions_accessed":
                0,

            "model_checkpoint_accessed":
                0,

            "optimizer_steps":
                0,

            "final_outer_cv_outcomes_accessed":
                0,
        },

    "generated_at_utc":
        NOW_ISO,
}


write_json(
    ORDERING_AUDIT_PATH,
    ordering_audit,
)


n_operational = int(
    (
        operational_class_df[
            "A1_3_operational_geometry_class"
        ]
        == NEW_OPERATIONAL
    ).sum()
)


n_alias_neutral = int(
    (
        operational_class_df[
            "A1_3_operational_geometry_class"
        ]
        == NEW_ALIAS_NEUTRAL
    ).sum()
)


n_resolution_neutral = int(
    (
        operational_class_df[
            "A1_3_operational_geometry_class"
        ]
        == NEW_RESOLUTION_NEUTRAL
    ).sum()
)


a13_audit = {
    "project":
        "COVA-3D",

    "block":
        SCIENTIFIC_BLOCK,

    "finalization_block":
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
            "A1_2_DIS_FRG_aliases":
                8,

            "quota2_DIS_FRG_aliases":
                0,

            "unique_promoted_physical_components":
                n_alias_neutral,

            "all_promoted_components_final_outer_cv":
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

            "total_geometry_neutral":
                n_resolution_neutral
                + n_alias_neutral,
        },

    "causal_matching_preserved":
        {
            "selected_component_sets":
                bool(
                    consistency_df[
                        "selected_component_set_equal"
                    ].all()
                ),

            "per_component_quotas":
                bool(
                    consistency_df[
                        "per_component_quotas_equal"
                    ].all()
                ),

            "FG_budgets":
                bool(
                    consistency_df[
                        "FG_budget_equal"
                    ].all()
                ),

            "background":
                bool(
                    consistency_df[
                        "BG_semantic_equal"
                    ].all()
                ),
        },

    "post_repair_operationality":
        {
            "operational_component_aliases":
                operational_aliases,

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
                    compat_df
                ),

            "exact_sparse_arrays_unchanged":
                int(
                    compat_df[
                        "exact_sparse_arrays_equal"
                    ].sum()
                ),

            "annotation_semantic_hashes_unchanged":
                int(
                    compat_df[
                        "annotation_semantic_hash_equal"
                    ].sum()
                ),

            "R2_retraining_required":
                False,

            "R2_model_sha256":
                EXPECTED_R2_SHA,
        },

    "ordering_fix":
        {
            "required":
                True,

            "scientific_change":
                False,

            "audit_file":
                str(
                    ORDERING_AUDIT_PATH.relative_to(
                        REPO
                    )
                ),
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


write_json(
    A13_AUDIT_PATH,
    a13_audit,
)


# ==========================================================================================
# 12. UPDATE CURRENT STATE
# ==========================================================================================

state = json.loads(
    COVA_STATE_PATH.read_text(
        encoding="utf-8"
    )
)


state.update(
    {
        "last_completed_block":
            SCIENTIFIC_BLOCK,

        "last_finalization_block":
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

        "A1_3_direct_array_ordering_bug":
            "FIXED_REPRESENTATION_ONLY",

        "A1_3_alias_neutral_components":
            n_alias_neutral,

        "A1_3_resolution_neutral_components":
            n_resolution_neutral,

        "A1_3_geometry_operational_components":
            n_operational,

        "A1_3_operational_component_aliases_after_repair":
            operational_aliases,

        "A1_3_neutral_identity_failures":
            neutral_identity_failures,

        "A1_3_informative_whole_case_aliases":
            informative_whole_case_aliases,

        "A1_3_v1_3_sparse_artifacts":
            120,

        "A1_3_R2_development_artifacts_exact":
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
            EXPECTED_R2_SHA,

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
                "Evaluate the frozen R2 checkpoint once using only the four "
                "permanent-development dense lesion masks under the locked "
                "sanity gate. Final outer-CV outcomes remain sealed."
            ),

        "updated_at_utc":
            NOW_ISO,
    }
)


write_json(
    COVA_STATE_PATH,
    state,
)


project_state = json.loads(
    PROJECT_STATE_PATH.read_text(
        encoding="utf-8"
    )
)


project_state.update(
    {
        "last_completed_block":
            SCIENTIFIC_BLOCK,

        "last_finalization_block":
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
# 13. A1.3 REGRESSION TEST
# ==========================================================================================

write_text(
    A13_TEST_PATH,
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


def test_A1_3_audit():

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
        "dense_sanity_evaluation"
    ] is True

    assert audit[
        "authorization"
    ][
        "factorial_training"
    ] is False


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
        "annotation_semantic_hash_equal"
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


def test_A1_3_current_state():

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


# ==========================================================================================
# 14. RUN FULL RELEVANT REGRESSION SUITE
# ==========================================================================================

heading(
    "RUN FULL RELEVANT REGRESSION SUITE"
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

        "tests/test_partial_losses.py",
        "tests/test_cova3d_baseline_lock.py",
        "tests/test_cova3d_numerical_amendment_n1.py",
        "tests/test_cova3d_numerical_amendment_n2.py",
        "tests/test_cova3d_sanity_fit_R2.py",
        "tests/test_cova3d_geometry_operationality.py",
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

    print()
    print(
        tests.stderr.rstrip()
    )


if tests.returncode != 0:

    raise RuntimeError(
        "A1.3 final regression suite failed."
    )


print()
print(
    "✓ Full relevant regression suite       : PASS"
)


# ==========================================================================================
# 15. SOURCE CAPTURE
# ==========================================================================================

heading(
    "SOURCE CAPTURE"
)


scripts_dir = (
    REPO
    / "scripts/code_blocks"
)


scripts_dir.mkdir(
    parents=True,
    exist_ok=True,
)


failed_source_path = (
    scripts_dir
    / "block09e_geometry_operationality_repair_A1_3_failed_ordering.py"
)


finalize_source_path = (
    scripts_dir
    / "block09e_geometry_operationality_repair_A1_3_finalize.py"
)


failed_source_capture = (
    "NOT_AVAILABLE"
)


finalize_source_capture = (
    "NOT_AVAILABLE"
)


try:

    ip = get_ipython()

    history = list(
        ip.history_manager.input_hist_raw
    )


    prior = (
        history[
            :-1
        ]
        if history
        else []
    )


    for cell in reversed(
        prior
    ):

        if (
            "# COVA-3D — BLOCK 09E-GEOM-OP-REPAIR"
            in cell
            and "A1.3 Operational-Geometry Neutralization"
            in cell
        ):

            failed_source_path.write_text(
                cell.rstrip()
                + "\n",
                encoding="utf-8",
            )

            failed_source_capture = (
                "PASS"
            )

            break


    current = history[
        -1
    ]


    if (
        "# COVA-3D — BLOCK 09E-GEOM-OP-REPAIR-FINALIZE"
        in current
    ):

        finalize_source_path.write_text(
            current.rstrip()
            + "\n",
            encoding="utf-8",
        )

        finalize_source_capture = (
            "PASS"
        )


except Exception:

    pass


a13_audit[
    "source_capture"
] = {
    "failed_A1_3_block":
        failed_source_capture,

    "finalization_block":
        finalize_source_capture,
}


write_json(
    A13_AUDIT_PATH,
    a13_audit,
)


print(
    "Failed A1.3 source capture             :",
    failed_source_capture,
)

print(
    "Finalization source capture            :",
    finalize_source_capture,
)


# ==========================================================================================
# 16. GITIGNORE
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
# 17. NORMALIZE TEXT BEFORE REPOSITORY MANIFEST
# ==========================================================================================

text_paths = [
    gitignore_path,
    A13_CONFIG_PATH,
    NEW_ARTIFACT_MANIFEST,
    NEW_CONDITION_MANIFEST,
    NEW_COMPONENT_AUDIT,
    OPERATIONAL_CLASS_PATH,
    PROMOTED_BANK_PATH,
    A13_CASE_AUDIT_PATH,
    A13_DEV_COMPAT_PATH,
    ORDERING_DIAG_PATH,
    A13_AUDIT_PATH,
    ORDERING_AUDIT_PATH,
    COVA_STATE_PATH,
    PROJECT_STATE_PATH,
    R2_TEST_PATH,
    GEOM_DIAG_TEST_PATH,
    A13_TEST_PATH,
]


for optional in [
    failed_source_path,
    finalize_source_path,
]:

    if optional.exists():

        text_paths.append(
            optional
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


# ==========================================================================================
# 18. REPOSITORY MANIFEST — GENERATE LAST
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

        "completed_scientific_block":
            SCIENTIFIC_BLOCK,

        "finalization_block":
            BLOCK,

        "active_track":
            "COVA3D",

        "effective_protocol":
            PROTOCOL_AFTER,

        "annotation_artifact_version":
            "v1.3",

        "A1_3_status":
            "FROZEN_PASS",

        "A1_3_ordering_bug":
            "FIXED_REPRESENTATION_ONLY",

        "A1_3_alias_neutral_components":
            n_alias_neutral,

        "A1_3_geometry_operational_components":
            n_operational,

        "A1_3_resolution_neutral_components":
            n_resolution_neutral,

        "post_repair_operational_aliases":
            operational_aliases,

        "post_repair_informative_whole_case_aliases":
            informative_whole_case_aliases,

        "R2_development_exact_artifacts":
            24,

        "R2_retraining_required":
            False,

        "R2_model_sha256":
            EXPECTED_R2_SHA,

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
# 19. COMMIT / PUSH
# ==========================================================================================

heading(
    "STEP 10/10 — COMMIT FINAL A1.3"
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

    "data/manifests/"
    "cova3d_A1_3_direct_array_ordering_diagnostic_v1_0.csv",

    "experiments/audits/"
    "block09e_geometry_operationality_repair_A1_3.json",

    "experiments/audits/"
    "block09e_A1_3_direct_array_ordering_fix.json",

    "tests/"
    "test_cova3d_sanity_fit_R2.py",

    "tests/"
    "test_cova3d_geometry_operationality.py",

    "tests/"
    "test_cova3d_geometry_operationality_A1_3.py",

    "data/cova3d_training_grid_candidates_v1_3",
]


for optional_relative in [
    "scripts/code_blocks/"
    "block09e_geometry_operationality_repair_A1_3_failed_ordering.py",

    "scripts/code_blocks/"
    "block09e_geometry_operationality_repair_A1_3_finalize.py",
]:

    if (
        REPO
        / optional_relative
    ).exists():

        git_paths.append(
            optional_relative
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


staged = sh(
    [
        "git",
        "diff",
        "--cached",
        "--name-only",
    ]
).stdout.splitlines()


staged_npz = [
    path
    for path in staged
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
    staged_npz
) != 120:

    raise RuntimeError(
        "Expected 120 staged v1.3 NPZ artifacts; observed "
        + str(
            len(
                staged_npz
            )
        )
    )


if any(
    path.endswith(
        ".pt"
    )
    for path in staged
):

    raise RuntimeError(
        "Model/checkpoint accidentally staged."
    )


unstaged_tracked = sh(
    [
        "git",
        "diff",
        "--name-only",
    ]
).stdout.strip()


if unstaged_tracked:

    raise RuntimeError(
        "Tracked unstaged changes remain:\n"
        + unstaged_tracked
    )


# Check for unexpected remaining untracked files.
remaining_status = sh(
    [
        "git",
        "status",
        "--porcelain",
    ]
).stdout.splitlines()


remaining_untracked = [
    line
    for line in remaining_status
    if line.startswith(
        "?? "
    )
]


if remaining_untracked:

    raise RuntimeError(
        "Unexpected untracked files remain before commit:\n"
        + "\n".join(
            remaining_untracked
        )
    )


print(
    "✓ git diff --cached --check            : PASS"
)

print(
    "✓ v1.3 NPZs staged                     : 120 /120"
)

print(
    "✓ Model/checkpoint staged              : NO"
)

print(
    "✓ Unexpected untracked files           : NONE"
)


sh(
    [
        "git",
        "commit",
        "-m",
        "protocol: finalize COVA-3D operational geometry amendment A1.3",
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
        "Repository is not clean after A1.3 commit:\n"
        + final_status
    )


# ==========================================================================================
# 20. FINAL REPORT
# ==========================================================================================

heading(
    "COVA-3D 09E-GEOM-OP-REPAIR-FINALIZE — FINAL REPORT"
)


print(
    "Starting commit                        :",
    head[:12],
)

print(
    "A1.3 final commit                      :",
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
    "ORDERING BUG"
)

print(
    "------------"
)

print(
    "Root cause                             : missing B2 direct-array sort"
)

print(
    "Scientific annotation change           : NO"
)

print(
    "V1.3 artifacts canonicalized           : 120 /120"
)

print(
    "Final B2 direct ordering               : 120 /120"
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
    "Exact four-array equality              : 24 /24"
)

print(
    "Semantic-hash equality                 : 24 /24"
)

print(
    "R2 retraining required                 : NO"
)

print(
    "R2 model SHA                           :",
    EXPECTED_R2_SHA,
)

print()

print(
    "A1.3 CAUSAL MATCHING"
)

print(
    "--------------------"
)

print(
    "Selected-component sets changed        : NO"
)

print(
    "Per-component quotas changed           : NO"
)

print(
    "FG budgets changed                     : NO"
)

print(
    "Background supervision changed         : NO"
)

print()

print(
    "A1.3 OPERATIONALITY"
)

print(
    "--------------------"
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
    "Alias-neutral components               :",
    n_alias_neutral,
)

print(
    "Operational component aliases          :",
    operational_aliases,
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
    "Strict A1.3 operationality             : PASS"
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
    "Final-CV segmentation outcomes         : 0"
)

print()

print(
    "REGRESSION"
)

print(
    "----------"
)

print(
    "Historical R2 test future-proofed      : YES"
)

print(
    "Historical geometry test future-proofed: YES"
)

print(
    "Full relevant regression suite         : PASS"
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
    "NEXT:"
)

print(
    "09E-SANITY-EVAL"
)

print()

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
    "=" * 132
)
