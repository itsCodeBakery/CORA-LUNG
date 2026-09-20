# ==========================================================================================
# COVA-3D — BLOCK 09D-A-LOCK
# Prospectively Freeze Resolution-Aware Geometry Amendment
#
# SCIENTIFIC REASON
# -----------------
# Block 09D-A established:
#
#   • 308 total C100 lesion components
#   • 0 components lost completely after training-grid transfer
#   • 18 components are geometry-degenerate at the 3.0 x 1.5 x 1.5 mm grid
#   • 11/20 cases contain >=1 geometry-degenerate component
#   • strict original grid rule passes only 7/20 cases
#
# The problem is therefore not lesion disappearance.
#
# The problem is that very small lesions cannot always represent a meaningful
# 2+ voxel coherent/dispersed/fragmented annotation after resampling.
#
# PROSPECTIVE AMENDMENT
# ---------------------
#
# For each selected lesion component:
#
#   RESOLUTION-RESOLVABLE:
#       common geometry capacity >= 2
#
#       -> retain the original geometry manipulation
#       -> minimum per-component quota = 2
#
#   RESOLUTION-LIMITED:
#       common raw mapped capacity >= 1
#       AND common geometry capacity < 2
#
#       -> retain lesion coverage
#       -> use exactly ONE identical geometry-neutral anchor
#          in COH / DIS / FRG
#       -> geometry manipulation is NOT claimed for this component
#
# IMPORTANT
# ---------
#   • No component is dropped.
#   • 100% coverage remains 100% lesion-instance coverage.
#   • Equal total FG budget across all six cells remains mandatory.
#   • Same BG realization remains mandatory.
#   • Same component set and quota across geometry remain mandatory.
#   • Primary R@1 remains unchanged.
#   • A resolution-stratified secondary analysis is added.
#   • No model outcomes influence this amendment.
#
# BLOCK 09D-B WILL THEN:
#   1. verify a common shared anchor exists for every resolution-limited lesion;
#   2. construct the amended six-cell grid annotations;
#   3. freeze B_i_train;
#   4. re-run causal-matching QA.
#
# THIS BLOCK DOES NOT:
#   • train
#   • instantiate optimizer
#   • access predictions
#   • access checkpoints
#   • access dense masks
#   • authorize factorial training
# ==========================================================================================


from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import subprocess
import hashlib
import json
import os
import shutil
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
    "04d0ecf00786"
)

BLOCK = (
    "09D-A-LOCK"
)

AMENDMENT_ID = (
    "A1_RESOLUTION_AWARE_GEOMETRY"
)

AMENDMENT_VERSION = (
    "1.0"
)

EFFECTIVE_PROTOCOL = (
    "COVA3D_1.0+A1"
)

DIAGNOSTIC_DIR = Path(
    "/kaggle/working/cova3d_09d_diagnostic"
)

CASE_DIAGNOSTIC_SOURCE = (
    DIAGNOSTIC_DIR
    / "case_training_grid_feasibility.csv"
)

COMPONENT_DIAGNOSTIC_SOURCE = (
    DIAGNOSTIC_DIR
    / "component_training_grid_feasibility.csv"
)

EXPECTED_CASES = (
    20
)

EXPECTED_C100_COMPONENTS = (
    308
)

EXPECTED_STRICT_PASS_CASES = (
    7
)

EXPECTED_STRICT_FAIL_CASES = (
    13
)

EXPECTED_ZERO_CAPACITY_COMPONENTS = (
    0
)

EXPECTED_GEOMETRY_DEGENERATE_COMPONENTS = (
    18
)

EXPECTED_CASES_WITH_DEGENERATION = (
    11
)

EXPECTED_PROBLEM_MEDIAN_VOLUME_ML = (
    0.1409
)

RESOLVABLE_MIN_CAPACITY = (
    2
)

RESOLUTION_LIMITED_QUOTA = (
    1
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
        "/tmp/cova3d_git_askpass_09da_lock.sh"
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


# ==========================================================================================
# 2. VERIFY REPOSITORY
# ==========================================================================================

heading(
    "COVA-3D BLOCK 09D-A-LOCK — RESOLUTION-AWARE AMENDMENT FREEZE"
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
        "Repository must be clean before protocol amendment.\n"
        + dirty
    )


print(
    "✓ Starting commit                    :",
    head[:12],
)

print(
    "✓ Repository state                   : CLEAN"
)


# ==========================================================================================
# 3. VERIFY FAILED 09D LEFT NO PARTIAL SPARSE ARTIFACTS
# ==========================================================================================

heading(
    "STEP 1/7 — VERIFY FAILED 09D DID NOT LEAVE PARTIAL SCIENTIFIC OUTPUT"
)


partial_grid_root = (
    REPO
    / "data/cova3d_training_grid_candidates_v1_0"
)


partial_grid_files = (
    list(
        partial_grid_root.rglob(
            "*.npz"
        )
    )
    if partial_grid_root.exists()
    else []
)


if partial_grid_files:

    raise RuntimeError(
        "Failed 09D left partial training-grid NPZ files.\n"
        "Do not proceed automatically.\n"
        "Found: "
        + str(
            len(
                partial_grid_files
            )
        )
    )


print(
    "✓ Partial 09D candidate artifacts     : 0"
)

print(
    "✓ Failed 09D scientific state         : NOT COMMITTED"
)


# ==========================================================================================
# 4. LOAD + VERIFY DIAGNOSTIC OUTPUT
# ==========================================================================================

heading(
    "STEP 2/7 — VERIFY 09D-A DIAGNOSTIC"
)


if not CASE_DIAGNOSTIC_SOURCE.exists():

    raise RuntimeError(
        "Case diagnostic CSV is missing:\n"
        + str(
            CASE_DIAGNOSTIC_SOURCE
        )
    )


if not COMPONENT_DIAGNOSTIC_SOURCE.exists():

    raise RuntimeError(
        "Component diagnostic CSV is missing:\n"
        + str(
            COMPONENT_DIAGNOSTIC_SOURCE
        )
    )


case_df = pd.read_csv(
    CASE_DIAGNOSTIC_SOURCE
)


component_df = pd.read_csv(
    COMPONENT_DIAGNOSTIC_SOURCE
)


if len(
    case_df
) != EXPECTED_CASES:

    raise RuntimeError(
        "Expected 20 diagnostic case rows."
    )


strict_pass = int(
    case_df[
        "strict_feasible"
    ].astype(
        bool
    ).sum()
)


strict_fail = int(
    EXPECTED_CASES
    - strict_pass
)


c100_df = component_df[
    np.isclose(
        component_df[
            "coverage"
        ],
        1.0,
    )
].copy()


if len(
    c100_df
) != EXPECTED_C100_COMPONENTS:

    raise RuntimeError(
        "Expected 308 C100 component rows."
    )


zero_capacity = int(
    (
        c100_df[
            "common_raw_capacity"
        ]
        == 0
    ).sum()
)


degenerate_df = c100_df[
    (
        c100_df[
            "common_raw_capacity"
        ]
        >= 1
    )
    & (
        c100_df[
            "common_geometry_capacity"
        ]
        < RESOLVABLE_MIN_CAPACITY
    )
].copy()


degenerate_count = int(
    len(
        degenerate_df
    )
)


cases_with_degeneration = int(
    degenerate_df[
        "case_id"
    ].nunique()
)


problem_median_volume = float(
    degenerate_df[
        "component_volume_ml"
    ].median()
)


if strict_pass != EXPECTED_STRICT_PASS_CASES:

    raise RuntimeError(
        "Diagnostic strict-pass count mismatch."
    )


if strict_fail != EXPECTED_STRICT_FAIL_CASES:

    raise RuntimeError(
        "Diagnostic strict-fail count mismatch."
    )


if zero_capacity != EXPECTED_ZERO_CAPACITY_COMPONENTS:

    raise RuntimeError(
        "Unexpected completely lost lesion components."
    )


if degenerate_count != EXPECTED_GEOMETRY_DEGENERATE_COMPONENTS:

    raise RuntimeError(
        "Geometry-degenerate component count mismatch."
    )


if cases_with_degeneration != EXPECTED_CASES_WITH_DEGENERATION:

    raise RuntimeError(
        "Geometry-degenerate case count mismatch."
    )


if not np.isclose(
    problem_median_volume,
    EXPECTED_PROBLEM_MEDIAN_VOLUME_ML,
    atol=5e-5,
):

    raise RuntimeError(
        "Problem-component median volume does not reproduce."
    )


print(
    "✓ Strict rule passing cases           :",
    strict_pass,
    "/20",
)

print(
    "✓ Strict rule failing cases           :",
    strict_fail,
    "/20",
)

print(
    "✓ C100 lesion components              :",
    len(
        c100_df
    ),
)

print(
    "✓ Completely lost components          :",
    zero_capacity,
)

print(
    "✓ Resolution-limited components       :",
    degenerate_count,
)

print(
    "✓ Cases with resolution limitation    :",
    cases_with_degeneration,
    "/20",
)

print(
    "✓ Problem median lesion volume        :",
    "{:.4f} mL".format(
        problem_median_volume
    ),
)


# ==========================================================================================
# 5. CLASSIFY ALL COMPONENTS PROSPECTIVELY
# ==========================================================================================

heading(
    "STEP 3/7 — FREEZE RESOLUTION CLASSIFICATION RULE"
)


component_df[
    "resolution_class"
] = np.where(
    component_df[
        "common_geometry_capacity"
    ]
    >= RESOLVABLE_MIN_CAPACITY,

    "GEOMETRY_RESOLVABLE",

    np.where(
        component_df[
            "common_raw_capacity"
        ]
        >= 1,

        "RESOLUTION_LIMITED",

        "COVERAGE_LOST",
    ),
)


if (
    component_df[
        "resolution_class"
    ]
    == "COVERAGE_LOST"
).any():

    raise RuntimeError(
        "At least one lesion is completely lost. "
        "The proposed one-anchor amendment is not sufficient."
    )


component_df[
    "minimum_train_quota_amended"
] = np.where(
    component_df[
        "resolution_class"
    ]
    == "GEOMETRY_RESOLVABLE",

    2,

    1,
).astype(
    int
)


resolution_counts = (
    c100_df.assign(
        resolution_class=np.where(
            c100_df[
                "common_geometry_capacity"
            ]
            >= 2,

            "GEOMETRY_RESOLVABLE",

            "RESOLUTION_LIMITED",
        )
    )[
        "resolution_class"
    ]
    .value_counts()
    .to_dict()
)


print(
    "✓ Classification rule                : FROZEN"
)

print(
    "✓ Geometry-resolvable C100 lesions    :",
    int(
        resolution_counts.get(
            "GEOMETRY_RESOLVABLE",
            0,
        )
    ),
)

print(
    "✓ Resolution-limited C100 lesions     :",
    int(
        resolution_counts.get(
            "RESOLUTION_LIMITED",
            0,
        )
    ),
)

print(
    "✓ Coverage-lost C100 lesions          : 0"
)


# ==========================================================================================
# 6. COPY DIAGNOSTIC PROVENANCE INTO REPOSITORY
# ==========================================================================================

heading(
    "STEP 4/7 — PRESERVE DIAGNOSTIC PROVENANCE"
)


diagnostic_repo_dir = (
    REPO
    / "experiments/audits/"
    "block09d_a_diagnostics"
)


diagnostic_repo_dir.mkdir(
    parents=True,
    exist_ok=True,
)


case_diag_repo = (
    diagnostic_repo_dir
    / "case_training_grid_feasibility.csv"
)


component_diag_repo = (
    diagnostic_repo_dir
    / "component_training_grid_feasibility.csv"
)


shutil.copy2(
    CASE_DIAGNOSTIC_SOURCE,
    case_diag_repo,
)


component_df.to_csv(
    component_diag_repo,
    index=False,
)


print(
    "✓ Case diagnostic archived            : YES"
)

print(
    "✓ Component diagnostic archived       : YES"
)


# ==========================================================================================
# 7. FREEZE AMENDMENT
# ==========================================================================================

heading(
    "STEP 5/7 — FREEZE RESOLUTION-AWARE PROTOCOL AMENDMENT"
)


amendment = {
    "project":
        "COVA-3D",

    "amendment_id":
        AMENDMENT_ID,

    "amendment_version":
        AMENDMENT_VERSION,

    "effective_protocol":
        EFFECTIVE_PROTOCOL,

    "status":
        "PROSPECTIVELY_FROZEN_BEFORE_COVA_TRAINING",

    "trigger":
        {
            "block":
                "09D-A",

            "strict_grid_feasible_cases":
                strict_pass,

            "strict_grid_failed_cases":
                strict_fail,

            "c100_components":
                EXPECTED_C100_COMPONENTS,

            "completely_lost_components":
                zero_capacity,

            "resolution_limited_components":
                degenerate_count,

            "cases_with_resolution_limited_components":
                cases_with_degeneration,

            "median_volume_resolution_limited_ml":
                problem_median_volume,
        },

    "scientific_interpretation":
        (
            "The original >=2-voxel per-component geometry constraint is "
            "not physically representable for a small subset of lesion "
            "instances after mapping to the fixed training resolution. "
            "Lesion coverage remains representable because no component "
            "has zero mapped foreground support."
        ),

    "resolution_classification":
        {
            "geometry_resolvable":
                (
                    "common mapped geometry capacity across coherent, "
                    "dispersed and fragmented conditions >= 2"
                ),

            "resolution_limited":
                (
                    "common mapped raw support >=1 but common geometry "
                    "capacity <2"
                ),

            "coverage_lost":
                "common mapped raw support == 0",

            "coverage_lost_policy":
                "STOP_AND_REVIEW",

            "observed_coverage_lost_components":
                0,
        },

    "amended_annotation_rule":
        {
            "geometry_resolvable_component":
                {
                    "minimum_quota":
                        2,

                    "coherent":
                        "retain connected geometry",

                    "dispersed":
                        "retain spatially separated geometry",

                    "fragmented":
                        "retain multi-fragment geometry",
                },

            "resolution_limited_component":
                {
                    "quota":
                        1,

                    "geometry":
                        (
                            "one identical geometry-neutral shared anchor "
                            "used in coherent, dispersed and fragmented cells"
                        ),

                    "anchor_rule":
                        (
                            "deterministic shared mapped voxel from the "
                            "intersection of the three geometry candidate "
                            "supports; exact deterministic tie-break frozen "
                            "and verified in Block 09D-B"
                        ),

                    "claim":
                        (
                            "no within-instance geometry manipulation is "
                            "claimed for resolution-limited lesions"
                        ),
                },
        },

    "preserved_factorial_constraints":
        {
            "50pct_component_selection":
                "unchanged",

            "100pct_component_selection":
                "unchanged",

            "all_selected_components_receive_foreground_supervision":
                True,

            "C50_nested_in_C100":
                True,

            "same_component_set_across_geometry":
                True,

            "same_per_component_quota_across_geometry":
                True,

            "same_total_foreground_budget_across_six_cells":
                True,

            "identical_background_coordinates_across_six_cells":
                True,
        },

    "endpoint_policy":
        {
            "primary_endpoint":
                "macro patient lesion recall at <=1 FP/patient",

            "primary_endpoint_changed":
                False,

            "new_secondary_analysis":
                [
                    "recall_geometry_resolvable_lesions",
                    "recall_resolution_limited_lesions",
                    "effect_by_lesion_volume",
                ],
        },

    "hypothesis_policy":
        {
            "coverage_main_effect":
                "unchanged",

            "geometry_main_effect":
                (
                    "interpreted as geometry manipulation on "
                    "resolution-resolvable lesion annotations"
                ),

            "coverage_by_geometry_interaction":
                "unchanged",

            "directional_hypothesis":
                False,
        },

    "methodological_guardrails":
        {
            "lesions_dropped_due_resolution":
                False,

            "resolution_limited_anchor_selected_using_model_outcomes":
                False,

            "budget_selected_using_model_outcomes":
                False,

            "factorial_training_authorized":
                False,

            "method_development_authorized":
                False,
        },

    "required_next_block":
        {
            "block":
                "09D-B",

            "requirements":
                [
                    "shared-anchor intersection audit",
                    "all-20-case amended feasibility",
                    "exact six-cell FG equality",
                    "exact six-cell BG equality",
                    "per-component quota equality",
                    "geometry preservation on resolvable components",
                    "freeze final B_i_train",
                ],
        },

    "frozen_at_utc":
        NOW_ISO,
}


amendment_path = (
    REPO
    / "configs/"
    "cova3d_resolution_aware_amendment_A1.yaml"
)


amendment_path.write_text(
    yaml.safe_dump(
        amendment,
        sort_keys=False,
        allow_unicode=True,
    ),
    encoding="utf-8",
)


amendment_doc = f"""
# COVA-3D Protocol Amendment A1

## Resolution-aware sparse annotation geometry

Block 09D-A was performed before any COVA-3D optimizer step.

The diagnostic showed that the original training-grid rule was feasible for
only {strict_pass}/20 cases.

No lesion component disappeared completely after grid transfer.

However, {degenerate_count}/{EXPECTED_C100_COMPONENTS} C100 lesion components
could not retain a meaningful two-or-more-voxel geometry representation at
the frozen 3.0 x 1.5 x 1.5 mm training resolution. These components occurred
in {cases_with_degeneration}/20 cases. Their median physical lesion volume was
{problem_median_volume:.4f} mL.

The original rule would therefore remove many otherwise valid cases because
of the representation limit of the training grid rather than because the
lesions themselves were absent.

## Amended rule

A selected lesion is classified as **geometry-resolvable** when its common
mapped geometry capacity is at least two voxels.

These lesions retain the original coherent, dispersed and fragmented
annotation manipulation.

A selected lesion is classified as **resolution-limited** when it retains at
least one mapped foreground voxel but cannot represent all three geometry
conditions with at least two distinct voxels.

A resolution-limited lesion remains part of the coverage factor. It receives
one identical geometry-neutral anchor in all three geometry conditions.

Thus the amendment preserves lesion-instance coverage while refusing to claim
a geometry manipulation that cannot physically exist at the frozen training
resolution.

No selected lesion is discarded.

## Primary analysis

The primary endpoint remains macro patient lesion recall at <=1 false-positive
component per patient.

The coverage factor remains unchanged.

The geometry factor is interpreted as manipulation of annotation geometry
where that geometry is physically resolvable.

Resolution-limited lesions remain in the overall lesion-recovery endpoint.

Secondary results will also stratify lesion recovery by resolution class and
lesion volume.

## Next required test

Block 09D-B must demonstrate that every resolution-limited component has a
deterministic common anchor shared across all three geometry conditions.

It must then prove exact six-cell foreground/background matching for all
20 cases and freeze the final B_i_train.

Training remains prohibited until that audit passes.
"""


amendment_doc_path = (
    REPO
    / "docs/"
    "cova3d_protocol_amendment_A1_resolution_aware_geometry.md"
)


write_text(
    amendment_doc_path,
    amendment_doc,
)


print(
    "✓ Amendment ID                       :",
    AMENDMENT_ID,
)

print(
    "✓ Effective protocol                  :",
    EFFECTIVE_PROTOCOL,
)

print(
    "✓ Lesions dropped                     : NO"
)

print(
    "✓ Resolution-limited quota            : 1 shared anchor"
)

print(
    "✓ Primary endpoint changed            : NO"
)

print(
    "✓ Factorial training authorized       : NO"
)


# ==========================================================================================
# 8. UPDATE PROJECT STATES
# ==========================================================================================

cova_state_path = (
    REPO
    / "COVA3D_STATE.json"
)


project_state_path = (
    REPO
    / "PROJECT_STATE.json"
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


if cova_state.get(
    "last_completed_block"
) != "09C":

    raise RuntimeError(
        "Unexpected COVA state before amendment."
    )


if int(
    cova_state.get(
        "optimizer_steps_in_cova3d",
        -1,
    )
) != 0:

    raise RuntimeError(
        "COVA optimizer steps already exist."
    )


cova_state.update(
    {
        "last_completed_block":
            BLOCK,

        "current_stage":
            "RESOLUTION_AWARE_AMENDMENT_FROZEN",

        "effective_protocol":
            EFFECTIVE_PROTOCOL,

        "protocol_amendment":
            AMENDMENT_ID,

        "resolution_aware_amendment_status":
            "FROZEN",

        "strict_training_grid_feasible_cases":
            strict_pass,

        "strict_training_grid_failed_cases":
            strict_fail,

        "resolution_limited_c100_components":
            degenerate_count,

        "cases_with_resolution_limited_components":
            cases_with_degeneration,

        "coverage_lost_components":
            zero_capacity,

        "trainer_grid_budget_status":
            "PENDING_09D_B",

        "trainer_grid_collision_audit":
            "DIAGNOSTIC_COMPLETE_AMENDMENT_REQUIRED",

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
                "Run resolution-aware training-grid feasibility and "
                "shared-anchor audit. Do not train."
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
            "cova3d_resolution_aware_protocol_amendment",

        "active_research_track":
            "COVA3D",

        "current_stage":
            "cova3d_resolution_aware_amendment_frozen",

        "current_gate":
            "COVA_GRID_RESOLUTION_AMENDMENT",

        "cova3d_effective_protocol":
            EFFECTIVE_PROTOCOL,

        "cova3d_resolution_aware_amendment":
            "FROZEN",

        "cova3d_resolution_limited_components":
            degenerate_count,

        "cova3d_coverage_lost_components":
            zero_capacity,

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

        "gate_b":
            "NO_GO",

        "gate_c":
            "NOT_RUN",

        "next_action":
            (
                "Run Block 09D-B resolution-aware transfer audit. "
                "Do not train."
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
# 9. WRITE AUDIT
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

    "amendment_id":
        AMENDMENT_ID,

    "effective_protocol":
        EFFECTIVE_PROTOCOL,

    "diagnostic": {
        "cases":
            EXPECTED_CASES,

        "strict_pass_cases":
            strict_pass,

        "strict_fail_cases":
            strict_fail,

        "c100_components":
            EXPECTED_C100_COMPONENTS,

        "zero_mapped_components":
            zero_capacity,

        "resolution_limited_components":
            degenerate_count,

        "cases_with_resolution_limitation":
            cases_with_degeneration,

        "median_problem_volume_ml":
            problem_median_volume,
    },

    "amendment": {
        "geometry_resolvable_minimum_quota":
            2,

        "resolution_limited_quota":
            1,

        "resolution_limited_policy":
            "shared_geometry_neutral_anchor",

        "lesions_dropped":
            0,

        "primary_endpoint_changed":
            False,

        "coverage_factor_changed":
            False,
    },

    "firewall": {
        "ct_arrays_accessed_in_this_block":
            0,

        "dense_lesion_arrays_accessed":
            0,

        "dense_lung_arrays_accessed":
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

    "method_development_authorized":
        False,

    "diagnostic_case_csv_sha256":
        sha256_file(
            case_diag_repo
        ),

    "diagnostic_component_csv_sha256":
        sha256_file(
            component_diag_repo
        ),

    "next_block":
        "09D-B",
}


audit_path = (
    REPO
    / "experiments/audits/"
    "block09d_a_resolution_aware_amendment.json"
)


write_json(
    audit_path,
    audit,
)


# ==========================================================================================
# 10. CAPTURE SOURCE
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
        "COVA-3D — BLOCK 09D-A-LOCK"
        in cell
    ):

        source_path = (
            REPO
            / "scripts/code_blocks/"
            "block09d_a_lock_resolution_aware_amendment.py"
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
# 11. GIT COMMIT + PUSH
# ==========================================================================================

heading(
    "STEP 6/7 — COMMIT PROSPECTIVE AMENDMENT"
)


git_paths = [
    "COVA3D_STATE.json",
    "PROJECT_STATE.json",
    "configs/cova3d_resolution_aware_amendment_A1.yaml",
    "docs/cova3d_protocol_amendment_A1_resolution_aware_geometry.md",
    "experiments/audits/block09d_a_resolution_aware_amendment.json",
    "experiments/audits/block09d_a_diagnostics/"
    "case_training_grid_feasibility.csv",
    "experiments/audits/block09d_a_diagnostics/"
    "component_training_grid_feasibility.csv",
]


source_path = (
    REPO
    / "scripts/code_blocks/"
    "block09d_a_lock_resolution_aware_amendment.py"
)


if source_path.exists():

    git_paths.append(
        "scripts/code_blocks/"
        "block09d_a_lock_resolution_aware_amendment.py"
    )


# Normalize EOF of text files.
for relative in git_paths:

    path = (
        REPO
        / relative
    )

    if (
        path.exists()
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
    "✓ git diff --cached --check           : PASS"
)


sh(
    [
        "git",
        "commit",
        "-m",
        "protocol: freeze resolution-aware COVA-3D geometry amendment",
    ],
    cwd=REPO,
)


amendment_commit = sh(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    cwd=REPO,
).stdout.strip()


token, askpass, git_env = make_git_auth()


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
        push.stderr
        or ""
    ).replace(
        token,
        "***TOKEN_REDACTED***",
    )

    raise RuntimeError(
        "GitHub push failed:\n"
        + safe_error
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
        "Repository is not clean after amendment commit:\n"
        + final_status
    )


# ==========================================================================================
# 12. FINAL REPORT
# ==========================================================================================

heading(
    "STEP 7/7 — FINAL RESOLUTION-AWARE AMENDMENT REPORT"
)


print(
    "COVA-3D BLOCK 09D-A-LOCK — FINAL REPORT"
)

print(
    "-" * 124
)

print(
    "Diagnostic strict-rule pass          :",
    strict_pass,
    "/20",
)

print(
    "Diagnostic strict-rule fail          :",
    strict_fail,
    "/20",
)

print(
    "Total C100 lesion components         :",
    EXPECTED_C100_COMPONENTS,
)

print(
    "Completely lost components           :",
    zero_capacity,
)

print(
    "Resolution-limited components        :",
    degenerate_count,
)

print(
    "Cases with resolution limitation     :",
    cases_with_degeneration,
    "/20",
)

print(
    "Median limited-lesion volume         :",
    "{:.4f} mL".format(
        problem_median_volume
    ),
)

print()

print(
    "AMENDMENT"
)

print(
    "---------"
)

print(
    "Amendment ID                         :",
    AMENDMENT_ID,
)

print(
    "Effective protocol                   :",
    EFFECTIVE_PROTOCOL,
)

print(
    "Resolvable lesion minimum quota      : 2"
)

print(
    "Resolution-limited lesion quota      : 1"
)

print(
    "Limited-lesion treatment             : IDENTICAL SHARED ANCHOR"
)

print(
    "Lesions dropped                      : 0"
)

print(
    "100% coverage semantics preserved    : YES"
)

print(
    "50% component selection preserved    : YES"
)

print(
    "Equal-six-cell FG requirement        : YES"
)

print(
    "Identical BG requirement             : YES"
)

print(
    "Primary R@1 endpoint changed         : NO"
)

print(
    "Resolution-stratified analysis added : YES"
)

print()

print(
    "TRAINING / FIREWALL"
)

print(
    "-------------------"
)

print(
    "Dense masks accessed                 : 0"
)

print(
    "CT arrays accessed                   : 0"
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

print(
    "Method development authorized        : NO"
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
    "Amendment commit                     :",
    amendment_commit[:12],
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
    "Next block will be 09D-B: verify shared anchors, rebuild the "
    "resolution-aware six-cell training-grid annotations, and freeze "
    "the final B_i_train for all 20 cases."
)

print(
    "=" * 124
)
