# ==========================================================================================
# COVA-3D — BLOCK 09D-A2-LOCK
# Cross-Coverage-Stable Resolution Classification
#
# WHY THIS CLARIFICATION IS REQUIRED
# -----------------------------------
# Amendment A1 identified resolution-limited components separately within the
# 50% and 100% annotation realizations.
#
# Post-freeze review of the archived 09D-A diagnostic revealed that some
# components change resolution class between C50 and C100.
#
# That is undesirable for the primary coverage x geometry interaction:
#
#     C50: geometry-neutral
#     C100: geometry-manipulated
#
# for the SAME lesion would make the geometry intervention partly dependent
# on the coverage factor.
#
# A1.1 therefore freezes a coverage-INVARIANT resolution class:
#
#     GLOBAL_RESOLUTION_LIMITED
#         if the lesion is resolution-limited in ANY coverage realization
#         in which it appears.
#
#     GLOBAL_GEOMETRY_RESOLVABLE
#         only if every available coverage realization is geometry-resolvable.
#
# Coverage-lost remains a hard STOP.
#
# For GLOBAL_RESOLUTION_LIMITED lesions:
#
#     • quota = 1
#     • one identical anchor is reused in every factorial cell in which
#       the lesion is selected
#     • anchor selection in 09D-B will be derived from the physical lesion
#       component itself, not from condition-specific sparse candidates
#     • no geometry effect is claimed for these lesions
#
# THIS BLOCK:
#     ✓ reads only already-committed diagnostic metadata
#     ✓ performs no CT access
#     ✓ performs no dense-mask access
#     ✓ performs no model training
#     ✓ performs no outcome access
#     ✓ freezes A1.1 before 09D-B
#
# Expected starting commit:
#     6d194763fe9b
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

EXPECTED_START_COMMIT = (
    "6d194763fe9b"
)

BLOCK = (
    "09D-A2-LOCK"
)

PARENT_AMENDMENT = (
    "A1_RESOLUTION_AWARE_GEOMETRY"
)

AMENDMENT_ID = (
    "A1_1_CROSS_COVERAGE_STABLE_RESOLUTION_CLASS"
)

EFFECTIVE_PROTOCOL = (
    "COVA3D_1.0+A1+A1.1"
)

EXPECTED_UNIQUE_COMPONENTS = (
    308
)

EXPECTED_C50_LIMITED_ROWS = (
    20
)

EXPECTED_C100_LIMITED_ROWS = (
    18
)

EXPECTED_GLOBAL_LIMITED_COMPONENTS = (
    28
)

EXPECTED_GLOBAL_LIMITED_CASES = (
    13
)

EXPECTED_CROSS_COVERAGE_DISCORDANT = (
    10
)

EXPECTED_COVERAGE_LOST = (
    0
)

DIAGNOSTIC_PATH = (
    REPO
    / "experiments/audits/"
    "block09d_a_diagnostics/"
    "component_training_grid_feasibility.csv"
)

GLOBAL_CLASS_PATH = (
    REPO
    / "data/manifests/"
    "cova3d_global_resolution_class_v1_0.csv"
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
                    8 * 1024 * 1024
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
        "/tmp/cova3d_git_askpass_09da2.sh"
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
# 2. VERIFY REPOSITORY + A1
# ==========================================================================================

heading(
    "COVA-3D BLOCK 09D-A2-LOCK — CROSS-COVERAGE-STABLE RESOLUTION CLASS"
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
        "Repository must be clean before A1.1.\n"
        + dirty
    )


cova_state_path = (
    REPO
    / "COVA3D_STATE.json"
)


project_state_path = (
    REPO
    / "PROJECT_STATE.json"
)


a1_path = (
    REPO
    / "configs/"
    "cova3d_resolution_aware_amendment_A1.yaml"
)


a1_audit_path = (
    REPO
    / "experiments/audits/"
    "block09d_a_resolution_aware_amendment.json"
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


a1 = yaml.safe_load(
    a1_path.read_text(
        encoding="utf-8"
    )
)


a1_audit = json.loads(
    a1_audit_path.read_text(
        encoding="utf-8"
    )
)


if cova_state.get(
    "last_completed_block"
) != "09D-A-LOCK":

    raise RuntimeError(
        "Expected 09D-A-LOCK as last completed block."
    )


if cova_state.get(
    "resolution_aware_amendment_status"
) != "FROZEN":

    raise RuntimeError(
        "A1 is not frozen."
    )


if a1.get(
    "amendment_id"
) != PARENT_AMENDMENT:

    raise RuntimeError(
        "Unexpected A1 amendment identity."
    )


if a1_audit.get(
    "status"
) != "PASS":

    raise RuntimeError(
        "A1 audit is not PASS."
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
        "Training was unexpectedly authorized."
    )


print(
    "✓ Starting commit                     :",
    head[:12],
)

print(
    "✓ Parent amendment                    : A1 / FROZEN"
)

print(
    "✓ Optimizer steps                     : 0"
)

print(
    "✓ Factorial training                  : NOT AUTHORIZED"
)


# ==========================================================================================
# 3. LOAD ARCHIVED DIAGNOSTIC
# ==========================================================================================

heading(
    "STEP 1/6 — AUDIT CROSS-COVERAGE RESOLUTION CLASS CONSISTENCY"
)


if not DIAGNOSTIC_PATH.exists():

    raise RuntimeError(
        "Archived component diagnostic is missing."
    )


diag = pd.read_csv(
    DIAGNOSTIC_PATH
)


required_columns = {
    "case_id",
    "coverage",
    "component_native_id",
    "component_volume_ml",
    "common_raw_capacity",
    "common_geometry_capacity",
    "resolution_class",
}


if not required_columns.issubset(
    set(
        diag.columns
    )
):

    raise RuntimeError(
        "Diagnostic schema mismatch."
    )


c50_limited_rows = int(
    (
        (
            np.isclose(
                diag[
                    "coverage"
                ],
                0.5,
            )
        )
        & (
            diag[
                "resolution_class"
            ]
            == "RESOLUTION_LIMITED"
        )
    ).sum()
)


c100_limited_rows = int(
    (
        (
            np.isclose(
                diag[
                    "coverage"
                ],
                1.0,
            )
        )
        & (
            diag[
                "resolution_class"
            ]
            == "RESOLUTION_LIMITED"
        )
    ).sum()
)


coverage_lost_rows = int(
    (
        diag[
            "resolution_class"
        ]
        == "COVERAGE_LOST"
    ).sum()
)


if c50_limited_rows != EXPECTED_C50_LIMITED_ROWS:

    raise RuntimeError(
        "Unexpected C50 resolution-limited count."
    )


if c100_limited_rows != EXPECTED_C100_LIMITED_ROWS:

    raise RuntimeError(
        "Unexpected C100 resolution-limited count."
    )


if coverage_lost_rows != EXPECTED_COVERAGE_LOST:

    raise RuntimeError(
        "Coverage-lost components detected."
    )


print(
    "✓ C50 limited rows                    :",
    c50_limited_rows,
)

print(
    "✓ C100 limited rows                   :",
    c100_limited_rows,
)

print(
    "✓ Coverage-lost rows                  :",
    coverage_lost_rows,
)


# ==========================================================================================
# 4. BUILD COVERAGE-INVARIANT COMPONENT CLASS
# ==========================================================================================

heading(
    "STEP 2/6 — FREEZE GLOBAL CASE-COMPONENT RESOLUTION CLASS"
)


global_rows = []

discordant_rows = []


grouped = diag.groupby(
    [
        "case_id",
        "component_native_id",
    ],
    sort=True,
)


for (
    case_id,
    component_id,
), group in grouped:

    classes = set(
        group[
            "resolution_class"
        ].astype(
            str
        ).tolist()
    )


    if "COVERAGE_LOST" in classes:

        global_class = (
            "COVERAGE_LOST"
        )


    elif "RESOLUTION_LIMITED" in classes:

        global_class = (
            "GLOBAL_RESOLUTION_LIMITED"
        )


    else:

        global_class = (
            "GLOBAL_GEOMETRY_RESOLVABLE"
        )


    has_c50 = bool(
        np.isclose(
            group[
                "coverage"
            ],
            0.5,
        ).any()
    )


    has_c100 = bool(
        np.isclose(
            group[
                "coverage"
            ],
            1.0,
        ).any()
    )


    c50_class = (
        str(
            group.loc[
                np.isclose(
                    group[
                        "coverage"
                    ],
                    0.5,
                ),
                "resolution_class",
            ].iloc[
                0
            ]
        )
        if has_c50
        else ""
    )


    c100_class = (
        str(
            group.loc[
                np.isclose(
                    group[
                        "coverage"
                    ],
                    1.0,
                ),
                "resolution_class",
            ].iloc[
                0
            ]
        )
        if has_c100
        else ""
    )


    discordant = bool(
        has_c50
        and has_c100
        and c50_class
        != c100_class
    )


    if discordant:

        discordant_rows.append(
            {
                "case_id":
                    str(
                        case_id
                    ),

                "component_native_id":
                    int(
                        component_id
                    ),

                "c50_class":
                    c50_class,

                "c100_class":
                    c100_class,
            }
        )


    minimum_common_raw = int(
        group[
            "common_raw_capacity"
        ].min()
    )


    minimum_common_geometry = int(
        group[
            "common_geometry_capacity"
        ].min()
    )


    component_volume_ml = float(
        group[
            "component_volume_ml"
        ].iloc[
            0
        ]
    )


    global_rows.append(
        {
            "case_id":
                str(
                    case_id
                ),

            "component_native_id":
                int(
                    component_id
                ),

            "component_volume_ml":
                component_volume_ml,

            "selected_in_C50":
                has_c50,

            "selected_in_C100":
                has_c100,

            "C50_resolution_class":
                c50_class,

            "C100_resolution_class":
                c100_class,

            "cross_coverage_discordant":
                discordant,

            "minimum_common_raw_capacity_across_available_coverages":
                minimum_common_raw,

            "minimum_common_geometry_capacity_across_available_coverages":
                minimum_common_geometry,

            "global_resolution_class":
                global_class,

            "minimum_train_quota":
                (
                    1
                    if global_class
                    == "GLOBAL_RESOLUTION_LIMITED"
                    else 2
                ),

            "geometry_claim_allowed":
                bool(
                    global_class
                    == "GLOBAL_GEOMETRY_RESOLVABLE"
                ),

            "shared_anchor_required":
                bool(
                    global_class
                    == "GLOBAL_RESOLUTION_LIMITED"
                ),
        }
    )


global_df = pd.DataFrame(
    global_rows
).sort_values(
    [
        "case_id",
        "component_native_id",
    ]
).reset_index(
    drop=True
)


discordant_df = pd.DataFrame(
    discordant_rows
)


if len(
    global_df
) != EXPECTED_UNIQUE_COMPONENTS:

    raise RuntimeError(
        "Expected 308 unique case-component rows."
    )


if (
    global_df[
        "global_resolution_class"
    ]
    == "COVERAGE_LOST"
).any():

    raise RuntimeError(
        "Coverage-lost component found."
    )


global_limited = global_df[
    global_df[
        "global_resolution_class"
    ]
    == "GLOBAL_RESOLUTION_LIMITED"
].copy()


global_limited_count = int(
    len(
        global_limited
    )
)


global_limited_cases = int(
    global_limited[
        "case_id"
    ].nunique()
)


discordant_count = int(
    len(
        discordant_df
    )
)


if global_limited_count != EXPECTED_GLOBAL_LIMITED_COMPONENTS:

    raise RuntimeError(
        "Unexpected global resolution-limited component count.\n"
        + "Expected: "
        + str(
            EXPECTED_GLOBAL_LIMITED_COMPONENTS
        )
        + "\nObserved: "
        + str(
            global_limited_count
        )
    )


if global_limited_cases != EXPECTED_GLOBAL_LIMITED_CASES:

    raise RuntimeError(
        "Unexpected number of cases with globally limited lesions."
    )


if discordant_count != EXPECTED_CROSS_COVERAGE_DISCORDANT:

    raise RuntimeError(
        "Unexpected cross-coverage discordance count."
    )


global_median_volume = float(
    global_limited[
        "component_volume_ml"
    ].median()
)


GLOBAL_CLASS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


global_df.to_csv(
    GLOBAL_CLASS_PATH,
    index=False,
)


print(
    "✓ Unique lesion components            :",
    len(
        global_df
    ),
)

print(
    "✓ Cross-coverage discordant lesions   :",
    discordant_count,
)

print(
    "✓ Globally resolution-limited lesions :",
    global_limited_count,
)

print(
    "✓ Cases with global limitation        :",
    global_limited_cases,
    "/20",
)

print(
    "✓ Global limited median volume        :",
    "{:.4f} mL".format(
        global_median_volume
    ),
)

print(
    "✓ Coverage-lost components            : 0"
)


print()
print(
    "Cross-coverage discordant components:"
)


print(
    discordant_df.to_string(
        index=False
    )
)


# ==========================================================================================
# 5. FREEZE A1.1
# ==========================================================================================

heading(
    "STEP 3/6 — FREEZE A1.1 FACTORIAL-CONSISTENCY RULE"
)


a1_1 = {
    "project":
        "COVA-3D",

    "parent_amendment":
        PARENT_AMENDMENT,

    "amendment_id":
        AMENDMENT_ID,

    "effective_protocol":
        EFFECTIVE_PROTOCOL,

    "status":
        "PROSPECTIVELY_FROZEN_BEFORE_COVA_TRAINING",

    "reason":
        (
            "Resolution classification must not itself vary with the "
            "coverage factor because doing so would make geometry-treatment "
            "availability coverage dependent and could confound the primary "
            "coverage-by-geometry interaction."
        ),

    "observed_preoutcome_diagnostic": {
        "C50_resolution_limited_rows":
            c50_limited_rows,

        "C100_resolution_limited_rows":
            c100_limited_rows,

        "cross_coverage_discordant_components":
            discordant_count,

        "global_resolution_limited_components":
            global_limited_count,

        "cases_with_global_resolution_limitation":
            global_limited_cases,

        "coverage_lost_components":
            0,

        "median_global_limited_volume_ml":
            global_median_volume,
    },

    "global_classification_rule": {
        "scope":
            "case-component, invariant across coverage levels",

        "GLOBAL_GEOMETRY_RESOLVABLE":
            (
                "geometry-resolvable in every available coverage "
                "realization for that case-component"
            ),

        "GLOBAL_RESOLUTION_LIMITED":
            (
                "resolution-limited in at least one available coverage "
                "realization"
            ),

        "COVERAGE_LOST":
            "coverage-lost in any available realization; hard stop",
    },

    "annotation_rule": {
        "GLOBAL_GEOMETRY_RESOLVABLE":
            {
                "minimum_quota":
                    2,

                "geometry_manipulation":
                    True,
            },

        "GLOBAL_RESOLUTION_LIMITED":
            {
                "quota":
                    1,

                "geometry_manipulation":
                    False,

                "anchor_reuse":
                    (
                        "same anchor across every geometry and coverage "
                        "cell in which the component is selected"
                    ),

                "anchor_source_in_09D_B":
                    (
                        "physical lesion component mapped to the frozen "
                        "training grid, independent of condition-specific "
                        "sparse candidate realization"
                    ),

                "anchor_selection":
                    (
                        "mapped component voxel nearest the native physical "
                        "component centroid; deterministic lexicographic "
                        "tie-break"
                    ),
            },
    },

    "preserved_constraints": {
        "C50_component_selection":
            "unchanged",

        "C100_component_selection":
            "unchanged",

        "C50_nested_in_C100":
            True,

        "selected_lesions_dropped":
            False,

        "same_component_set_across_geometry":
            True,

        "same_per_component_quota_across_geometry":
            True,

        "equal_total_FG_across_six_cells":
            True,

        "identical_BG_across_six_cells":
            True,

        "primary_endpoint_changed":
            False,
    },

    "outcome_independence": {
        "model_predictions_used":
            False,

        "model_checkpoints_used":
            False,

        "segmentation_metrics_used":
            False,

        "optimizer_steps_before_freeze":
            0,
    },

    "next_block":
        "09D-B",

    "frozen_at_utc":
        NOW_ISO,
}


a1_1_path = (
    REPO
    / "configs/"
    "cova3d_resolution_aware_amendment_A1_1.yaml"
)


a1_1_path.write_text(
    yaml.safe_dump(
        a1_1,
        sort_keys=False,
        allow_unicode=True,
    ),
    encoding="utf-8",
)


doc_path = (
    REPO
    / "docs/"
    "cova3d_protocol_amendment_A1_1_cross_coverage_stability.md"
)


write_text(
    doc_path,
    f"""
    # COVA-3D Protocol Amendment A1.1

    ## Cross-coverage-stable resolution classification

    Amendment A1 introduced a resolution-aware rule before any COVA-3D
    training.

    A subsequent audit of the already archived feasibility table showed
    {discordant_count} lesion components whose resolution class differed
    between their 50% and 100% coverage realizations.

    Allowing that difference would make the availability of the geometry
    manipulation partly dependent on the coverage factor itself.

    Therefore A1.1 freezes a case-component resolution class that is invariant
    across coverage.

    A lesion is globally geometry-resolvable only when every available
    coverage realization is geometry-resolvable.

    A lesion is globally resolution-limited when any available realization is
    resolution-limited.

    This conservative rule produces {global_limited_count}/{EXPECTED_UNIQUE_COMPONENTS}
    globally resolution-limited lesions across {global_limited_cases}/20 cases.

    No lesion is coverage-lost.

    Globally resolution-limited lesions receive a single geometry-neutral
    anchor in every factorial cell in which they are selected. The exact same
    anchor will be reused across geometry and coverage.

    Block 09D-B will derive this anchor from the physical lesion component on
    the frozen training grid rather than from a condition-specific candidate
    annotation. This prevents the anchor location itself from becoming a
    coverage-dependent intervention.

    Geometry-resolvable lesions retain the coherent, dispersed and fragmented
    manipulation.

    No model outcomes, predictions or optimizer steps were used to define this
    clarification.

    Training remains prohibited until Block 09D-B proves exact six-cell
    feasibility.
    """
)


print(
    "✓ A1.1 amendment                      : FROZEN"
)

print(
    "✓ Resolution class scope              : CASE-COMPONENT"
)

print(
    "✓ Resolution class coverage-invariant : YES"
)

print(
    "✓ Limited anchor coverage-invariant   : REQUIRED"
)

print(
    "✓ Primary endpoint changed            : NO"
)

print(
    "✓ Training authorized                 : NO"
)


# ==========================================================================================
# 6. UPDATE STATE
# ==========================================================================================

heading(
    "STEP 4/6 — UPDATE STATE WITHOUT AUTHORIZING TRAINING"
)


cova_state.update(
    {
        "last_completed_block":
            BLOCK,

        "current_stage":
            "CROSS_COVERAGE_STABLE_RESOLUTION_CLASS_FROZEN",

        "effective_protocol":
            EFFECTIVE_PROTOCOL,

        "protocol_amendment":
            "A1+A1.1",

        "resolution_aware_amendment_status":
            "FROZEN_A1_1",

        "global_resolution_class_status":
            "FROZEN",

        "global_resolution_limited_components":
            global_limited_count,

        "global_resolution_limited_cases":
            global_limited_cases,

        "cross_coverage_resolution_discordant_components":
            discordant_count,

        "coverage_lost_components":
            0,

        "trainer_grid_budget_status":
            "PENDING_09D_B",

        "factorial_training_authorized":
            False,

        "method_development_authorized":
            False,

        "final_outer_cv_outcomes_authorized":
            False,

        "optimizer_steps_in_cova3d":
            0,

        "next_block":
            "09D-B",

        "next_action":
            (
                "Run final resolution-aware training-grid construction "
                "using globally stable resolution classes and shared "
                "component-derived anchors. Do not train."
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
            "cova3d_cross_coverage_stable_resolution_class",

        "current_stage":
            "cova3d_A1_1_frozen",

        "current_gate":
            "COVA_GRID_RESOLUTION_STABILITY",

        "cova3d_effective_protocol":
            EFFECTIVE_PROTOCOL,

        "cova3d_global_resolution_class_status":
            "FROZEN",

        "cova3d_global_resolution_limited_components":
            global_limited_count,

        "cova3d_global_resolution_limited_cases":
            global_limited_cases,

        "cova3d_cross_coverage_resolution_discordance":
            discordant_count,

        "cova3d_coverage_lost_components":
            0,

        "cova3d_trainer_grid_feasibility":
            "PENDING_09D_B",

        "cova3d_factorial_training_authorized":
            False,

        "cova3d_method_development_authorized":
            False,

        "cova3d_optimizer_steps":
            0,

        "training_authorized":
            False,

        "pilot_training_authorized":
            False,

        "next_action":
            (
                "Run Block 09D-B final resolution-aware transfer and "
                "budget freeze. Do not train."
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
# 7. AUDIT
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

    "parent_amendment":
        PARENT_AMENDMENT,

    "amendment_id":
        AMENDMENT_ID,

    "effective_protocol":
        EFFECTIVE_PROTOCOL,

    "diagnostic": {
        "unique_components":
            len(
                global_df
            ),

        "C50_limited_rows":
            c50_limited_rows,

        "C100_limited_rows":
            c100_limited_rows,

        "cross_coverage_discordant_components":
            discordant_count,

        "global_resolution_limited_components":
            global_limited_count,

        "global_resolution_limited_cases":
            global_limited_cases,

        "coverage_lost_components":
            0,

        "median_global_limited_volume_ml":
            global_median_volume,
    },

    "frozen_rule": {
        "classification_scope":
            "case-component",

        "coverage_invariant":
            True,

        "limited_if_any_available_coverage_is_limited":
            True,

        "limited_quota":
            1,

        "limited_anchor_shared_across_geometry":
            True,

        "limited_anchor_shared_across_coverage":
            True,

        "limited_anchor_source":
            "physical lesion component",

        "geometry_resolvable_minimum_quota":
            2,
    },

    "firewall": {
        "ct_arrays_accessed":
            0,

        "dense_mask_arrays_accessed":
            0,

        "prediction_arrays_accessed":
            0,

        "checkpoint_arrays_accessed":
            0,

        "model_outcomes_accessed":
            0,

        "optimizer_steps":
            0,
    },

    "factorial_training_authorized":
        False,

    "global_class_manifest":
        str(
            GLOBAL_CLASS_PATH.relative_to(
                REPO
            )
        ),

    "global_class_manifest_sha256":
        sha256_file(
            GLOBAL_CLASS_PATH
        ),

    "next_block":
        "09D-B",
}


audit_path = (
    REPO
    / "experiments/audits/"
    "block09d_a2_cross_coverage_stability.json"
)


write_json(
    audit_path,
    audit,
)


# ==========================================================================================
# 8. SOURCE CAPTURE
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
        "COVA-3D — BLOCK 09D-A2-LOCK"
        in cell
    ):

        source_path = (
            REPO
            / "scripts/code_blocks/"
            "block09d_a2_lock_cross_coverage_stability.py"
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


audit[
    "source_capture"
] = source_capture


write_json(
    audit_path,
    audit,
)


# ==========================================================================================
# 9. COMMIT
# ==========================================================================================

heading(
    "STEP 5/6 — COMMIT A1.1"
)


git_paths = [
    "COVA3D_STATE.json",
    "PROJECT_STATE.json",
    "configs/cova3d_resolution_aware_amendment_A1_1.yaml",
    "data/manifests/cova3d_global_resolution_class_v1_0.csv",
    "docs/cova3d_protocol_amendment_A1_1_cross_coverage_stability.md",
    "experiments/audits/block09d_a2_cross_coverage_stability.json",
]


source_path = (
    REPO
    / "scripts/code_blocks/"
    "block09d_a2_lock_cross_coverage_stability.py"
)


if source_path.exists():

    git_paths.append(
        "scripts/code_blocks/"
        "block09d_a2_lock_cross_coverage_stability.py"
    )


# Normalize text EOF.
for relative in git_paths:

    path = (
        REPO
        / relative
    )

    if path.exists():

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
    "✓ git diff --cached --check            : PASS"
)


sh(
    [
        "git",
        "commit",
        "-m",
        "protocol: stabilize COVA-3D resolution class across coverage",
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
        "Repository not clean after A1.1 commit:\n"
        + final_status
    )


# ==========================================================================================
# 10. FINAL REPORT
# ==========================================================================================

heading(
    "STEP 6/6 — FINAL A1.1 REPORT"
)


print(
    "COVA-3D BLOCK 09D-A2-LOCK — FINAL REPORT"
)

print(
    "-" * 126
)

print(
    "Unique lesion components             :",
    len(
        global_df
    ),
)

print(
    "C50 resolution-limited rows          :",
    c50_limited_rows,
)

print(
    "C100 resolution-limited rows         :",
    c100_limited_rows,
)

print(
    "Cross-coverage discordant components :",
    discordant_count,
)

print(
    "Global resolution-limited components :",
    global_limited_count,
)

print(
    "Cases with global limitation         :",
    global_limited_cases,
    "/20",
)

print(
    "Global limited median volume         :",
    "{:.4f} mL".format(
        global_median_volume
    ),
)

print(
    "Coverage-lost components             : 0"
)

print()

print(
    "A1.1"
)

print(
    "----"
)

print(
    "Effective protocol                   :",
    EFFECTIVE_PROTOCOL,
)

print(
    "Resolution class coverage-invariant  : YES"
)

print(
    "Limited lesion quota                 : 1"
)

print(
    "Shared across geometry               : YES"
)

print(
    "Shared across coverage               : YES"
)

print(
    "Anchor source                        : PHYSICAL LESION COMPONENT"
)

print(
    "Resolvable minimum quota             : 2"
)

print(
    "Lesions dropped                      : 0"
)

print(
    "Primary endpoint changed             : NO"
)

print()

print(
    "TRAINING / FIREWALL"
)

print(
    "-------------------"
)

print(
    "CT arrays accessed                   : 0"
)

print(
    "Dense masks accessed                 : 0"
)

print(
    "Predictions accessed                 : 0"
)

print(
    "Checkpoints accessed                 : 0"
)

print(
    "Model outcomes accessed              : 0"
)

print(
    "Optimizer steps                      : 0"
)

print(
    "Factorial training authorized        : NO"
)

print()

print(
    "Source capture                       :",
    source_capture,
)

print(
    "Starting commit                      :",
    head[:12],
)

print(
    "A1.1 commit                          :",
    final_commit[:12],
)

print(
    "GitHub synchronization               : PASS"
)

print(
    "Repository state                     : CLEAN"
)

print()

print(
    "NEXT:"
)

print(
    "Send me this COMPLETE report."
)

print(
    "Do NOT train yet."
)

print(
    "Next block will be 09D-B — final resolution-aware grid construction, "
    "shared-anchor verification, exact six-cell causal matching, and "
    "B_i_train freeze."
)

print(
    "=" * 126
)
