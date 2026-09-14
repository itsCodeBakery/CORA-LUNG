# ==========================================================================================
# COVA-3D — CODE BLOCK 09B
# Primary Dataset Structural Suitability + Factorial Feasibility Audit
#
# PURPOSE
#
# This block answers:
#
#   1. Can the existing 20-volume lung cohort structurally support the
#      prospectively frozen 50% x 100% coverage experiment?
#
#   2. Do all development and sealed final cases contain >=2 eligible
#      lesion components?
#
#   3. How many components would be selected under the frozen
#      ceil(0.5*K_i) rule?
#
#   4. Does source-subject grouping remain independent across
#      development versus final roles?
#
#   5. Are there low-component cases where nominal 50% coverage differs
#      materially from exactly 0.50 because of discreteness?
#
#   6. Which second dataset is prospectively nominated for cross-organ
#      replication?
#
# IMPORTANT
#
# THIS BLOCK READS ONLY:
#
#   • committed CSV/JSON/YAML metadata
#
# IT DOES NOT READ:
#
#   • CT voxel arrays
#   • lesion-mask voxel arrays
#   • lung-mask voxel arrays
#   • model predictions
#   • checkpoints
#
# IT DOES NOT:
#
#   • simulate COVA annotations
#   • determine final B_i
#   • train anything
#   • instantiate an optimizer
#   • access final-CV model outcomes
#
# EXACT COORDINATE-LEVEL FEASIBILITY REMAINS PENDING BLOCK 09C.
# ==========================================================================================


from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import subprocess
import hashlib
import json
import os
import sys
import textwrap

import numpy as np
import pandas as pd
import yaml

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
    "94438c0ea5b3"
)

BLOCK = (
    "09B"
)

TRACK_ID = (
    "COVA3D"
)

PROTOCOL_VERSION = (
    "1.0"
)

PRIMARY_DATASET_ID = (
    "COVID19_CT_Seg_20"
)

SECONDARY_DATASET_ID = (
    "MSD_Task03_Liver"
)

SECONDARY_OFFICIAL_NAME = (
    "Medical Segmentation Decathlon Task03 Liver — Liver Tumours"
)

SECONDARY_OFFICIAL_URL = (
    "https://medicaldecathlon.com/"
)

SECONDARY_DOWNLOAD_URL = (
    "https://medicaldecathlon.com/dataaws/"
)

SECONDARY_MODALITY = (
    "Portal venous phase CT"
)

SECONDARY_TARGET = (
    "Liver and tumour"
)

SECONDARY_TOTAL_VOLUMES = (
    201
)

SECONDARY_TRAINING_VOLUMES = (
    131
)

SECONDARY_TESTING_VOLUMES = (
    70
)

SECONDARY_LICENSE = (
    "CC-BY-SA 4.0"
)

SECONDARY_ACQUISITION_POLICY = (
    "Prefer original MSD NIfTI source. "
    "Do not make a pre-normalized NPY conversion the canonical dataset."
)

MIN_COMPONENTS_PRIMARY_FACTORIAL = (
    2
)

NOMINAL_LOW_COVERAGE = (
    0.50
)

DISCRETIZATION_WARNING_ABS = (
    0.10
)

EXPECTED_PRIMARY_VOLUMES = (
    20
)

EXPECTED_DEVELOPMENT_VOLUMES = (
    4
)

EXPECTED_FINAL_VOLUMES = (
    16
)

EXPECTED_SOURCE_GROUPS = (
    18
)

EXPECTED_ELIGIBLE_COMPONENTS = (
    308
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
        + "=" * 124
    )

    print(
        text
    )

    print(
        "=" * 124
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
        "/tmp/cova3d_git_askpass_09b.sh"
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


# ==========================================================================================
# 2. RESTORE / VERIFY REPOSITORY
# ==========================================================================================

heading(
    "COVA-3D BLOCK 09B — DATASET SUITABILITY + FACTORIAL FEASIBILITY"
)


token, askpass, git_env = make_git_auth()


if not (
    REPO
    / ".git"
).exists():

    if REPO.exists():

        raise RuntimeError(
            "Non-git directory already exists at repository path."
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
        "Repository must be clean before Block 09B.\n"
        + dirty
    )


print(
    "✓ Starting commit                    :",
    starting_commit[:12],
)

print(
    "✓ Repository state                   : CLEAN"
)


# ==========================================================================================
# 3. VERIFY FROZEN 09A LINEAGE
# ==========================================================================================

heading(
    "STEP 1/8 — VERIFY COVA-3D PROTOCOL FREEZE"
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


registry_path = (
    REPO
    / "data/manifests/"
    "cova3d_factorial_register_v1_0.csv"
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
) != "09A":

    raise RuntimeError(
        "Expected 09A as last completed COVA block."
    )


if cova_state.get(
    "protocol_status"
) != "FROZEN":

    raise RuntimeError(
        "COVA protocol is not frozen."
    )


if cova_state.get(
    "factorial_training_authorized"
) is not False:

    raise RuntimeError(
        "Factorial training must remain unauthorized."
    )


if cova_state.get(
    "method_development_authorized"
) is not False:

    raise RuntimeError(
        "Method development must remain unauthorized."
    )


if cova_state.get(
    "final_outer_cv_outcomes_authorized"
) is not False:

    raise RuntimeError(
        "Final outcome access must remain unauthorized."
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


if sha256_file(
    protocol_path
) != cova_state[
    "protocol_config_sha256"
]:

    raise RuntimeError(
        "Frozen protocol SHA mismatch."
    )


if sha256_file(
    registry_path
) != cova_state[
    "factorial_registry_sha256"
]:

    raise RuntimeError(
        "Frozen factorial registry SHA mismatch."
    )


if project_state.get(
    "gate_b"
) != "NO_GO":

    raise RuntimeError(
        "Historical CORA Gate B result changed."
    )


if project_state.get(
    "gate_c"
) != "NOT_RUN":

    raise RuntimeError(
        "Historical CORA Gate C status changed."
    )


print(
    "✓ COVA protocol                      : FROZEN"
)

print(
    "✓ Factorial training                 : NOT AUTHORIZED"
)

print(
    "✓ COVA optimizer steps               : 0"
)

print(
    "✓ CORA Gate B                        : NO_GO / PRESERVED"
)

print(
    "✓ CORA Gate C                        : NOT RUN"
)


# ==========================================================================================
# 4. LOAD PRE-EXISTING AUDIT METADATA ONLY
# ==========================================================================================

heading(
    "STEP 2/8 — LOAD COMMITTED COMPONENT/SPLIT METADATA ONLY"
)


component_path = (
    REPO
    / "data/manifests/"
    "primary_case_component_summary.csv"
)


split_path = (
    REPO
    / "data/splits/"
    "primary_volume_split_registry.csv"
)


component_df = pd.read_csv(
    component_path
)


split_df = pd.read_csv(
    split_path
)


required_component_columns = {
    "case_id",
    "source_subject_key",
    "eligible_components_26conn_ge0p1ml",
    "multifocal_eligible",
}


required_split_columns = {
    "case_id",
    "source_subject_key",
    "source_origin",
    "role",
    "outer_fold",
    "independence_unit",
}


if not required_component_columns.issubset(
    component_df.columns
):

    raise RuntimeError(
        "Component-summary schema mismatch."
    )


if not required_split_columns.issubset(
    split_df.columns
):

    raise RuntimeError(
        "Split-registry schema mismatch."
    )


if len(
    component_df
) != EXPECTED_PRIMARY_VOLUMES:

    raise RuntimeError(
        "Unexpected primary component-summary row count."
    )


if len(
    split_df
) != EXPECTED_PRIMARY_VOLUMES:

    raise RuntimeError(
        "Unexpected split-registry row count."
    )


if component_df[
    "case_id"
].astype(
    str
).duplicated().any():

    raise RuntimeError(
        "Duplicate case IDs in component summary."
    )


if split_df[
    "case_id"
].astype(
    str
).duplicated().any():

    raise RuntimeError(
        "Duplicate case IDs in split registry."
    )


print(
    "✓ Component metadata rows             :",
    len(
        component_df
    ),
)

print(
    "✓ Split-registry rows                 :",
    len(
        split_df
    ),
)

print(
    "✓ CT arrays accessed                  : 0"
)

print(
    "✓ Dense lesion arrays accessed        : 0"
)

print(
    "✓ Prediction arrays accessed          : 0"
)


# ==========================================================================================
# 5. MERGE AND AUDIT STRUCTURAL FACTORIAL ELIGIBILITY
# ==========================================================================================

heading(
    "STEP 3/8 — AUDIT 50% / 100% COVERAGE STRUCTURAL FEASIBILITY"
)


component_df[
    "case_id"
] = component_df[
    "case_id"
].astype(
    str
)


split_df[
    "case_id"
] = split_df[
    "case_id"
].astype(
    str
)


merged = component_df[
    [
        "case_id",
        "source_subject_key",
        "eligible_components_26conn_ge0p1ml",
        "multifocal_eligible",
    ]
].merge(
    split_df[
        [
            "case_id",
            "source_subject_key",
            "source_origin",
            "role",
            "outer_fold",
            "independence_unit",
        ]
    ],
    on="case_id",
    how="inner",
    suffixes=(
        "_component",
        "_split",
    ),
)


if len(
    merged
) != EXPECTED_PRIMARY_VOLUMES:

    raise RuntimeError(
        "Component/split merge lost cases."
    )


if not np.array_equal(
    merged[
        "source_subject_key_component"
    ].astype(
        str
    ).to_numpy(),
    merged[
        "source_subject_key_split"
    ].astype(
        str
    ).to_numpy(),
):

    raise RuntimeError(
        "source_subject_key mismatch across manifests."
    )


merged[
    "source_subject_key"
] = merged[
    "source_subject_key_split"
].astype(
    str
)


merged[
    "eligible_components"
] = merged[
    "eligible_components_26conn_ge0p1ml"
].astype(
    int
)


merged[
    "selected_components_50"
] = np.ceil(
    NOMINAL_LOW_COVERAGE
    * merged[
        "eligible_components"
    ]
).astype(
    int
)


merged[
    "selected_components_100"
] = merged[
    "eligible_components"
].astype(
    int
)


merged[
    "omitted_components_50"
] = (
    merged[
        "eligible_components"
    ]
    - merged[
        "selected_components_50"
    ]
)


merged[
    "realized_coverage_50"
] = (
    merged[
        "selected_components_50"
    ]
    / merged[
        "eligible_components"
    ]
)


merged[
    "coverage_discretization_abs"
] = np.abs(
    merged[
        "realized_coverage_50"
    ]
    - NOMINAL_LOW_COVERAGE
)


merged[
    "coverage_discretization_flag"
] = (
    merged[
        "coverage_discretization_abs"
    ]
    > DISCRETIZATION_WARNING_ABS
    + 1e-12
)


merged[
    "primary_factorial_structural_eligible"
] = (
    merged[
        "eligible_components"
    ]
    >= MIN_COMPONENTS_PRIMARY_FACTORIAL
)


if not merged[
    "primary_factorial_structural_eligible"
].all():

    failed = merged.loc[
        ~merged[
            "primary_factorial_structural_eligible"
        ],
        [
            "case_id",
            "eligible_components",
        ],
    ]

    raise RuntimeError(
        "One or more primary cases cannot support "
        "50% versus 100% component coverage:\n"
        + failed.to_string(
            index=False
        )
    )


if not merged[
    "multifocal_eligible"
].astype(
    bool
).all():

    raise RuntimeError(
        "Existing multifocal audit and new structural audit disagree."
    )


total_components = int(
    merged[
        "eligible_components"
    ].sum()
)


if total_components != EXPECTED_ELIGIBLE_COMPONENTS:

    raise RuntimeError(
        "Eligible-component total differs from frozen audit.\n"
        + "Observed: "
        + str(
            total_components
        )
        + "\nExpected: "
        + str(
            EXPECTED_ELIGIBLE_COMPONENTS
        )
    )


total_selected_50 = int(
    merged[
        "selected_components_50"
    ].sum()
)


total_selected_100 = int(
    merged[
        "selected_components_100"
    ].sum()
)


print(
    merged[
        [
            "case_id",
            "role",
            "source_origin",
            "eligible_components",
            "selected_components_50",
            "selected_components_100",
            "realized_coverage_50",
            "coverage_discretization_flag",
        ]
    ]
    .sort_values(
        [
            "role",
            "case_id",
        ]
    )
    .to_string(
        index=False
    )
)


print()

print(
    "✓ Structurally eligible cases         : 20 / 20"
)

print(
    "✓ Eligible components                 :",
    total_components,
)

print(
    "✓ Components selected at nominal 50% :",
    total_selected_50,
)

print(
    "✓ Components selected at 100%        :",
    total_selected_100,
)

print(
    "✓ Coordinate-level geometry feasible : PENDING BLOCK 09C"
)


# ==========================================================================================
# 6. ROLE / INDEPENDENCE AUDIT
# ==========================================================================================

heading(
    "STEP 4/8 — VERIFY DEVELOPMENT / FINAL INDEPENDENCE"
)


role_counts = (
    merged[
        "role"
    ]
    .value_counts()
    .to_dict()
)


development_count = int(
    role_counts.get(
        "permanent_development",
        0,
    )
)


final_count = int(
    role_counts.get(
        "final_outer_cv",
        0,
    )
)


if development_count != EXPECTED_DEVELOPMENT_VOLUMES:

    raise RuntimeError(
        "Unexpected permanent-development count."
    )


if final_count != EXPECTED_FINAL_VOLUMES:

    raise RuntimeError(
        "Unexpected final outer-CV count."
    )


source_group_count = int(
    merged[
        "source_subject_key"
    ].nunique()
)


if source_group_count != EXPECTED_SOURCE_GROUPS:

    raise RuntimeError(
        "Unexpected conservative source-group count."
    )


role_per_source_group = (
    merged.groupby(
        "source_subject_key"
    )[
        "role"
    ]
    .nunique()
)


cross_role_groups = role_per_source_group[
    role_per_source_group
    > 1
]


if len(
    cross_role_groups
):

    raise RuntimeError(
        "A conservative source-subject group crosses "
        "development/final roles:\n"
        + cross_role_groups.to_string()
    )


if set(
    merged[
        "independence_unit"
    ].astype(
        str
    )
) != {
    "source_subject_key"
}:

    raise RuntimeError(
        "Unexpected independence unit."
    )


development_df = merged[
    merged[
        "role"
    ]
    == "permanent_development"
].copy()


final_df = merged[
    merged[
        "role"
    ]
    == "final_outer_cv"
].copy()


development_components = int(
    development_df[
        "eligible_components"
    ].sum()
)


development_selected_50 = int(
    development_df[
        "selected_components_50"
    ].sum()
)


final_components = int(
    final_df[
        "eligible_components"
    ].sum()
)


final_selected_50 = int(
    final_df[
        "selected_components_50"
    ].sum()
)


print(
    "✓ Permanent development volumes       :",
    development_count,
)

print(
    "✓ Final outer-CV volumes              :",
    final_count,
)

print(
    "✓ Conservative source groups          :",
    source_group_count,
)

print(
    "✓ Groups crossing roles               : 0"
)

print(
    "✓ Development eligible components     :",
    development_components,
)

print(
    "✓ Development selected @ nominal 50% :",
    development_selected_50,
)

print(
    "✓ Final eligible components           :",
    final_components,
)

print(
    "✓ Final selected @ nominal 50%       :",
    final_selected_50,
)


# ==========================================================================================
# 7. DISCRETE-COVERAGE AUDIT
# ==========================================================================================

heading(
    "STEP 5/8 — AUDIT NOMINAL 50% COVERAGE DISCRETIZATION"
)


discretization_flagged = merged[
    merged[
        "coverage_discretization_flag"
    ]
].copy()


max_realized_coverage = float(
    merged[
        "realized_coverage_50"
    ].max()
)


min_realized_coverage = float(
    merged[
        "realized_coverage_50"
    ].min()
)


max_abs_deviation = float(
    merged[
        "coverage_discretization_abs"
    ].max()
)


print(
    "Realized nominal-50% coverage range   : "
    + "{:.3f} – {:.3f}".format(
        min_realized_coverage,
        max_realized_coverage,
    )
)

print(
    "Maximum |realized - 0.50|             :",
    "{:.3f}".format(
        max_abs_deviation
    ),
)

print(
    "Cases >0.10 away from nominal 0.50    :",
    len(
        discretization_flagged
    ),
)


if len(
    discretization_flagged
):

    print()

    print(
        discretization_flagged[
            [
                "case_id",
                "eligible_components",
                "selected_components_50",
                "realized_coverage_50",
            ]
        ].to_string(
            index=False
        )
    )


# Important:
# We do NOT exclude these cases now.
#
# The primary protocol already froze ceil(0.5*K).
# Instead, we prospectively register a secondary sensitivity analysis
# restricted to cases with K >= 4.
merged[
    "coverage_discretization_sensitivity_K_ge_4"
] = (
    merged[
        "eligible_components"
    ]
    >= 4
)


sensitivity_count = int(
    merged[
        "coverage_discretization_sensitivity_K_ge_4"
    ].sum()
)


print()
print(
    "✓ Primary eligibility changed         : NO"
)

print(
    "✓ Planned K>=4 sensitivity cases      :",
    sensitivity_count,
    "/20"
)


# ==========================================================================================
# 8. FREEZE PRIMARY FEASIBILITY TABLES
# ==========================================================================================

heading(
    "STEP 6/8 — FREEZE DATASET FEASIBILITY REGISTERS"
)


manifest_dir = (
    REPO
    / "data/manifests"
)


manifest_dir.mkdir(
    parents=True,
    exist_ok=True,
)


primary_feasibility_path = (
    manifest_dir
    / "cova3d_primary_factorial_structural_feasibility_v1_0.csv"
)


primary_output = merged[
    [
        "case_id",
        "source_subject_key",
        "source_origin",
        "role",
        "outer_fold",
        "eligible_components",
        "selected_components_50",
        "selected_components_100",
        "omitted_components_50",
        "realized_coverage_50",
        "coverage_discretization_abs",
        "coverage_discretization_flag",
        "coverage_discretization_sensitivity_K_ge_4",
        "primary_factorial_structural_eligible",
    ]
].sort_values(
    [
        "role",
        "case_id",
    ]
)


primary_output.to_csv(
    primary_feasibility_path,
    index=False,
)


role_summary = (
    primary_output.groupby(
        "role",
        as_index=False,
    )
    .agg(
        volumes=(
            "case_id",
            "count",
        ),

        source_groups=(
            "source_subject_key",
            "nunique",
        ),

        eligible_components=(
            "eligible_components",
            "sum",
        ),

        selected_components_50=(
            "selected_components_50",
            "sum",
        ),

        selected_components_100=(
            "selected_components_100",
            "sum",
        ),

        mean_realized_coverage_50=(
            "realized_coverage_50",
            "mean",
        ),

        max_coverage_discretization_abs=(
            "coverage_discretization_abs",
            "max",
        ),
    )
)


role_summary_path = (
    manifest_dir
    / "cova3d_primary_factorial_role_summary_v1_0.csv"
)


role_summary.to_csv(
    role_summary_path,
    index=False,
)


dataset_register = pd.DataFrame(
    [
        {
            "dataset_id":
                PRIMARY_DATASET_ID,

            "dataset_name":
                "COVID-19 CT Lung and Infection Segmentation",

            "planned_role":
                "primary_factorial_dataset",

            "modality":
                "CT",

            "target":
                "lung infection / lesion components",

            "known_volumes":
                EXPECTED_PRIMARY_VOLUMES,

            "known_dense_training_labels":
                EXPECTED_PRIMARY_VOLUMES,

            "structural_factorial_status":
                "PASS",

            "coordinate_level_geometry_status":
                "PENDING_BLOCK_09C",

            "multifocal_component_audit":
                "PASS",

            "factorial_training_authorized":
                False,

            "canonical_source_policy":
                "existing frozen primary NIfTI dataset",

            "cross_domain_role":
                "primary lung cohort",

            "notes":
                (
                    "All 20 volumes have >=2 eligible 26-connected "
                    "lesion components >=0.1mL."
                ),
        },

        {
            "dataset_id":
                SECONDARY_DATASET_ID,

            "dataset_name":
                SECONDARY_OFFICIAL_NAME,

            "planned_role":
                "cross_organ_replication_candidate",

            "modality":
                SECONDARY_MODALITY,

            "target":
                SECONDARY_TARGET,

            "known_volumes":
                SECONDARY_TOTAL_VOLUMES,

            "known_dense_training_labels":
                SECONDARY_TRAINING_VOLUMES,

            "structural_factorial_status":
                "NOT_YET_AUDITED",

            "coordinate_level_geometry_status":
                "NOT_YET_AUDITED",

            "multifocal_component_audit":
                "REQUIRED_BEFORE_INCLUSION",

            "factorial_training_authorized":
                False,

            "canonical_source_policy":
                SECONDARY_ACQUISITION_POLICY,

            "cross_domain_role":
                "cross-organ tumour replication",

            "notes":
                (
                    "Official MSD Task03 candidate. Inclusion requires "
                    "local NIfTI integrity audit, tumour-component audit, "
                    "and exact six-cell feasibility."
                ),
        },
    ]
)


dataset_register_path = (
    manifest_dir
    / "cova3d_dataset_register_v1_0.csv"
)


dataset_register.to_csv(
    dataset_register_path,
    index=False,
)


secondary_plan = {
    "dataset_id":
        SECONDARY_DATASET_ID,

    "official_name":
        SECONDARY_OFFICIAL_NAME,

    "selection_status":
        "PRIORITY_CANDIDATE_PENDING_LOCAL_AUDIT",

    "scientific_role":
        (
            "cross-organ replication of coverage x annotation-geometry "
            "effects in another multifocal tumour segmentation domain"
        ),

    "official_url":
        SECONDARY_OFFICIAL_URL,

    "official_download_url":
        SECONDARY_DOWNLOAD_URL,

    "modality":
        SECONDARY_MODALITY,

    "target":
        SECONDARY_TARGET,

    "official_total_volumes":
        SECONDARY_TOTAL_VOLUMES,

    "official_training_volumes":
        SECONDARY_TRAINING_VOLUMES,

    "official_testing_volumes":
        SECONDARY_TESTING_VOLUMES,

    "license":
        SECONDARY_LICENSE,

    "canonical_acquisition_policy":
        SECONDARY_ACQUISITION_POLICY,

    "pre_inclusion_requirements": [
        "original geometry/header audit",
        "label semantics audit",
        "tumour component count with 26-connectivity",
        "physical volume threshold audit",
        "multifocal prevalence audit",
        "50pct/100pct structural feasibility",
        "COVA geometry feasibility",
        "source_subject independence audit",
    ],

    "important_constraint":
        (
            "Published dataset size/description does not prove multifocal "
            "factorial suitability. No cross-organ claim is allowed until "
            "the local component audit passes."
        ),

    "frozen_at_utc":
        NOW_ISO,
}


secondary_plan_path = (
    REPO
    / "configs/"
    "cova3d_secondary_dataset_candidate_v1_0.yaml"
)


secondary_plan_path.write_text(
    yaml.safe_dump(
        secondary_plan,
        sort_keys=False,
        allow_unicode=True,
    ),
    encoding="utf-8",
)


print(
    "✓ Primary feasibility table           : FROZEN"
)

print(
    "✓ Role summary                        : FROZEN"
)

print(
    "✓ Dataset register                    : FROZEN"
)

print(
    "✓ Priority second-dataset candidate   : MSD Task03 Liver"
)

print(
    "✓ Secondary dataset included yet      : NO"
)


# ==========================================================================================
# 9. WRITE SCIENTIFIC FEASIBILITY NOTE
# ==========================================================================================

feasibility_note = f"""
# COVA-3D Block 09B — Dataset Suitability and Structural Factorial Feasibility

## Primary cohort

The existing lung CT cohort contains {EXPECTED_PRIMARY_VOLUMES} volumes.

Using the already frozen lesion definition:

- 26-connectivity;
- lesion component volume >= 0.1 mL;

all {EXPECTED_PRIMARY_VOLUMES}/{EXPECTED_PRIMARY_VOLUMES} volumes contain at
least two eligible lesion components and therefore satisfy the structural
minimum for comparing nominal 50% versus 100% lesion-instance coverage.

The current cohort contains {total_components} eligible components at the
per-volume level.

Under the prospectively frozen count rule:

`K_50(i) = ceil(0.5 K_i)`

the cohort contributes:

- {total_selected_50} selected lesion components at nominal 50% coverage;
- {total_selected_100} selected lesion components at 100% coverage.

The four permanent-development cases contain:

- {development_components} eligible components at 100% coverage;
- {development_selected_50} selected components under the nominal 50% rule.

No source-subject group crosses development and final roles.

## Coverage discreteness

Nominal 50% coverage is a component-count factor.

Because lesion counts are discrete, `ceil(0.5 K)` is not exactly 0.50 for
every odd-valued K.

The primary protocol is not changed after this audit.

A pre-outcome sensitivity analysis restricted to cases with `K >= 4` is now
registered to examine whether extremely low component counts influence the
interpretation of the nominal coverage factor.

This sensitivity analysis cannot replace the primary analysis.

## Coordinate-level annotation feasibility

Block 09B establishes structural feasibility only.

It does **not** establish that coherent, dispersed and fragmented geometry can
all realize an identical unique-foreground budget `B_i`.

That question is reserved for Block 09C, where the three annotation algorithms
will be implemented using dense masks offline and the largest common feasible
budget will be determined prospectively.

No COVA training is authorized by this block.

## Second dataset

The priority cross-organ candidate is:

**{SECONDARY_OFFICIAL_NAME}**

Official metadata:

- modality: {SECONDARY_MODALITY};
- target: {SECONDARY_TARGET};
- total volumes: {SECONDARY_TOTAL_VOLUMES};
- labelled training volumes: {SECONDARY_TRAINING_VOLUMES};
- test volumes: {SECONDARY_TESTING_VOLUMES};
- license: {SECONDARY_LICENSE}.

It is only a candidate.

The published dataset description does not establish that enough training
cases contain multiple tumour components for the COVA factorial experiment.
A local NIfTI component audit is mandatory before the dataset can be admitted
to the final study.

The original NIfTI/physical-geometry representation is preferred over
pre-normalized convenience conversions.

## Status

Primary structural feasibility: **PASS**

Primary coordinate-level six-cell feasibility: **PENDING 09C**

Secondary dataset inclusion: **PENDING LOCAL AUDIT**

COVA training authorization: **NO**
"""


feasibility_note_path = (
    REPO
    / "docs/"
    "cova3d_dataset_feasibility_v1_0.md"
)


write_text(
    feasibility_note_path,
    feasibility_note,
)


# ==========================================================================================
# 10. REGRESSION TESTS
# ==========================================================================================

heading(
    "STEP 7/8 — RUN DATASET-FEASIBILITY REGRESSION TESTS"
)


test_source = r'''
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_primary_structural_factorial_feasibility():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_primary_factorial_structural_feasibility_v1_0.csv"
    )

    assert len(frame) == 20

    assert (
        frame[
            "eligible_components"
        ]
        >= 2
    ).all()

    assert (
        frame[
            "selected_components_50"
        ]
        >= 1
    ).all()

    assert (
        frame[
            "selected_components_50"
        ]
        <= frame[
            "selected_components_100"
        ]
    ).all()

    assert frame[
        "primary_factorial_structural_eligible"
    ].astype(bool).all()


def test_primary_roles_remain_frozen():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_primary_factorial_structural_feasibility_v1_0.csv"
    )

    counts = frame[
        "role"
    ].value_counts().to_dict()

    assert counts[
        "permanent_development"
    ] == 4

    assert counts[
        "final_outer_cv"
    ] == 16

    role_count = (
        frame.groupby(
            "source_subject_key"
        )[
            "role"
        ]
        .nunique()
    )

    assert int(
        role_count.max()
    ) == 1


def test_secondary_dataset_is_candidate_only():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_dataset_register_v1_0.csv"
    )

    secondary = frame[
        frame[
            "dataset_id"
        ]
        == "MSD_Task03_Liver"
    ]

    assert len(
        secondary
    ) == 1

    row = secondary.iloc[
        0
    ]

    assert row[
        "structural_factorial_status"
    ] == "NOT_YET_AUDITED"

    assert not bool(
        row[
            "factorial_training_authorized"
        ]
    )


def test_training_remains_unauthorized():

    state = json.loads(
        (
            ROOT
            / "COVA3D_STATE.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert state[
        "factorial_training_authorized"
    ] is False

    assert state[
        "method_development_authorized"
    ] is False

    assert state[
        "final_outer_cv_outcomes_authorized"
    ] is False

    assert state[
        "optimizer_steps_in_cova3d"
    ] == 0
'''


test_path = (
    REPO
    / "tests/"
    "test_cova3d_dataset_feasibility.py"
)


write_text(
    test_path,
    test_source,
)


# ==========================================================================================
# 11. UPDATE STATES
# ==========================================================================================

cova_state.update(
    {
        "last_completed_block":
            BLOCK,

        "current_stage":
            "PRIMARY_DATASET_STRUCTURAL_FEASIBILITY_PASS",

        "next_block":
            "09C",

        "primary_dataset_id":
            PRIMARY_DATASET_ID,

        "primary_dataset_structural_feasibility":
            "PASS",

        "primary_dataset_coordinate_geometry_feasibility":
            "PENDING_09C",

        "primary_factorial_structurally_eligible_cases":
            EXPECTED_PRIMARY_VOLUMES,

        "development_factorial_structurally_eligible_cases":
            EXPECTED_DEVELOPMENT_VOLUMES,

        "final_factorial_structurally_eligible_cases":
            EXPECTED_FINAL_VOLUMES,

        "primary_total_eligible_components":
            total_components,

        "primary_total_selected_components_nominal_50":
            total_selected_50,

        "development_total_eligible_components":
            development_components,

        "development_selected_components_nominal_50":
            development_selected_50,

        "secondary_dataset_candidate":
            SECONDARY_DATASET_ID,

        "secondary_dataset_status":
            "PRIORITY_CANDIDATE_PENDING_LOCAL_AUDIT",

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

        "next_action":
            (
                "Implement the three annotation-geometry generators "
                "and determine exact common B_i feasibility in Block 09C. "
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
            "cova3d_dataset_structural_feasibility",

        "active_research_track":
            TRACK_ID,

        "current_stage":
            "cova3d_primary_dataset_structural_feasibility_pass",

        "current_gate":
            "COVA_DATASET_FEASIBILITY",

        "cova3d_primary_dataset_structural_feasibility":
            "PASS",

        "cova3d_coordinate_geometry_feasibility":
            "PENDING_09C",

        "cova3d_primary_eligible_cases":
            EXPECTED_PRIMARY_VOLUMES,

        "cova3d_secondary_dataset_candidate":
            SECONDARY_DATASET_ID,

        "cova3d_secondary_dataset_status":
            "PENDING_LOCAL_AUDIT",

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

        # Historical predecessor remains immutable.
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
                "Run Block 09C annotation simulator v2 and exact "
                "coordinate-level feasibility audit. Do not train."
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
# 12. RUN TESTS
# ==========================================================================================

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
        "Block-09B regression tests failed."
    )


print(
    "✓ Dataset-feasibility regression tests: PASS"
)


# ==========================================================================================
# 13. CAPTURE SOURCE
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
        "COVA-3D — CODE BLOCK 09B"
        in cell
    ):

        source_path = (
            REPO
            / "scripts/code_blocks/"
            "block09b_cova3d_dataset_feasibility.py"
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


# ==========================================================================================
# 14. BLOCK AUDIT
# ==========================================================================================

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

    "purpose":
        (
            "primary dataset structural factorial feasibility and "
            "prospective second-dataset nomination"
        ),

    "primary_dataset":
        PRIMARY_DATASET_ID,

    "primary_volumes":
        EXPECTED_PRIMARY_VOLUMES,

    "primary_source_groups":
        source_group_count,

    "primary_structurally_eligible_cases":
        int(
            merged[
                "primary_factorial_structural_eligible"
            ].sum()
        ),

    "primary_total_eligible_components":
        total_components,

    "primary_selected_components_nominal_50":
        total_selected_50,

    "primary_selected_components_100":
        total_selected_100,

    "development_volumes":
        development_count,

    "development_eligible_components":
        development_components,

    "development_selected_components_nominal_50":
        development_selected_50,

    "final_volumes":
        final_count,

    "final_eligible_components":
        final_components,

    "final_selected_components_nominal_50":
        final_selected_50,

    "source_groups_crossing_roles":
        0,

    "coverage_discretization": {
        "nominal":
            NOMINAL_LOW_COVERAGE,

        "minimum_realized":
            min_realized_coverage,

        "maximum_realized":
            max_realized_coverage,

        "maximum_absolute_deviation":
            max_abs_deviation,

        "warning_threshold":
            DISCRETIZATION_WARNING_ABS,

        "flagged_cases":
            int(
                len(
                    discretization_flagged
                )
            ),

        "planned_K_ge_4_sensitivity_cases":
            sensitivity_count,

        "primary_protocol_changed":
            False,
    },

    "coordinate_level_geometry_feasibility":
        "PENDING_BLOCK_09C",

    "common_positive_budget_B_i":
        "NOT_YET_FROZEN",

    "secondary_dataset": {
        "dataset_id":
            SECONDARY_DATASET_ID,

        "official_name":
            SECONDARY_OFFICIAL_NAME,

        "status":
            "PRIORITY_CANDIDATE_PENDING_LOCAL_AUDIT",

        "official_total_volumes":
            SECONDARY_TOTAL_VOLUMES,

        "official_training_volumes":
            SECONDARY_TRAINING_VOLUMES,

        "official_testing_volumes":
            SECONDARY_TESTING_VOLUMES,

        "license":
            SECONDARY_LICENSE,

        "included_in_current_analysis":
            False,
    },

    "ct_arrays_accessed":
        0,

    "dense_lesion_arrays_accessed":
        0,

    "dense_lung_arrays_accessed":
        0,

    "prediction_arrays_accessed":
        0,

    "development_model_outcomes_accessed":
        0,

    "final_model_outcomes_accessed":
        0,

    "new_training_performed":
        False,

    "optimizer_steps":
        0,

    "factorial_training_authorized":
        False,

    "source_capture":
        source_capture,

    "next_block":
        "09C",

    "next_action":
        (
            "Implement coherent/dispersed/fragmented annotation "
            "geometry and determine exact common unique-positive B_i "
            "without model training."
        ),
}


audit_path = (
    REPO
    / "experiments/audits/"
    "block09b_cova3d_dataset_feasibility.json"
)


write_json(
    audit_path,
    audit_payload,
)


# ==========================================================================================
# 15. REPOSITORY MANIFEST
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

        "cova3d_protocol":
            "FROZEN",

        "primary_structural_feasibility":
            "PASS",

        "coordinate_geometry_feasibility":
            "PENDING_09C",

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
# 16. COMMIT + PUSH
# ==========================================================================================

heading(
    "STEP 8/8 — COMMIT DATASET SUITABILITY AUDIT"
)


git_paths = [
    "COVA3D_STATE.json",
    "PROJECT_STATE.json",
    "REPOSITORY_MANIFEST.json",
    "configs/cova3d_secondary_dataset_candidate_v1_0.yaml",
    "data/manifests/"
    "cova3d_primary_factorial_structural_feasibility_v1_0.csv",
    "data/manifests/"
    "cova3d_primary_factorial_role_summary_v1_0.csv",
    "data/manifests/"
    "cova3d_dataset_register_v1_0.csv",
    "docs/cova3d_dataset_feasibility_v1_0.md",
    "experiments/audits/"
    "block09b_cova3d_dataset_feasibility.json",
    "tests/test_cova3d_dataset_feasibility.py",
]


source_path = (
    REPO
    / "scripts/code_blocks/"
    "block09b_cova3d_dataset_feasibility.py"
)


if source_path.exists():

    git_paths.append(
        "scripts/code_blocks/"
        "block09b_cova3d_dataset_feasibility.py"
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
        "No Block-09B artifacts available to commit."
    )


print(
    status
)


sh(
    [
        "git",
        "commit",
        "-m",
        "audit: freeze COVA-3D dataset structural feasibility",
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
# 17. FINAL REPORT
# ==========================================================================================

print(
    "\n"
    + "=" * 124
)

print(
    "COVA-3D CODE BLOCK 09B — FINAL DATASET FEASIBILITY REPORT"
)

print(
    "=" * 124
)


print(
    "Primary dataset                      :",
    PRIMARY_DATASET_ID,
)

print(
    "Primary volumes                      :",
    EXPECTED_PRIMARY_VOLUMES,
)

print(
    "Conservative source groups           :",
    source_group_count,
)

print(
    "Structurally eligible volumes        :",
    int(
        merged[
            "primary_factorial_structural_eligible"
        ].sum()
    ),
    "/20",
)

print(
    "Total eligible lesion components     :",
    total_components,
)

print(
    "Selected components @ nominal 50%    :",
    total_selected_50,
)

print(
    "Selected components @ 100%           :",
    total_selected_100,
)

print()

print(
    "Development volumes                  :",
    development_count,
)

print(
    "Development eligible components      :",
    development_components,
)

print(
    "Development selected @ nominal 50%   :",
    development_selected_50,
)

print()

print(
    "Final outer-CV volumes               :",
    final_count,
)

print(
    "Final eligible components            :",
    final_components,
)

print(
    "Final selected @ nominal 50%         :",
    final_selected_50,
)

print(
    "Source groups crossing roles         : 0"
)

print()

print(
    "Realized nominal-50% coverage range  : "
    + "{:.3f} – {:.3f}".format(
        min_realized_coverage,
        max_realized_coverage,
    )
)

print(
    "Maximum coverage discretization      :",
    "{:.3f}".format(
        max_abs_deviation
    ),
)

print(
    "Discretization-warning cases         :",
    len(
        discretization_flagged
    ),
)

print(
    "Planned K>=4 sensitivity cases       :",
    sensitivity_count,
)

print(
    "Primary eligibility changed          : NO"
)

print()

print(
    "Coordinate-level six-cell feasibility: PENDING 09C"
)

print(
    "Common positive budget B_i frozen    : NO"
)

print()

print(
    "Priority second-dataset candidate    :",
    SECONDARY_DATASET_ID,
)

print(
    "Second dataset role                  : CROSS-ORGAN REPLICATION"
)

print(
    "Official labelled training volumes   :",
    SECONDARY_TRAINING_VOLUMES,
)

print(
    "Second dataset locally audited       : NO"
)

print(
    "Second dataset admitted to study     : NO"
)

print()

print(
    "CT arrays accessed                   : 0"
)

print(
    "Dense lesion arrays accessed         : 0"
)

print(
    "Prediction arrays accessed           : 0"
)

print(
    "Model outcomes accessed              : 0"
)

print(
    "New optimizer steps                  : 0"
)

print(
    "Factorial training authorized        : NO"
)

print(
    "Method development authorized        : NO"
)

print()

print(
    "Regression tests                     : PASS"
)

print(
    "Exact Block-09B source captured      :",
    source_capture,
)

print(
    "Starting commit                      :",
    starting_commit[:12],
)

print(
    "Final audit commit                   :",
    final_commit[:12],
)

print(
    "GitHub synchronization               : PASS"
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
    "Next block: 09C — implement coherent, dispersed and fragmented "
    "annotation geometry and freeze exact common B_i feasibility."
)

print(
    "=" * 124
)