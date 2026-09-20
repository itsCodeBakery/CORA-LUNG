# ==========================================================================================
# COVA-3D — BLOCK 09D-A3-LOCK
# Freeze Amendment A1.2: Geometry-Neutral Physical Anchor Banks
#
# EXPECTED START COMMIT:
#       8690900f10d4
#
# WHY A1.2 IS REQUIRED
# ---------------------
# A1.1 froze GLOBAL_RESOLUTION_LIMITED lesions to exactly one shared anchor.
#
# Block 09D-B then revealed a mathematical conflict:
#
#   radiopaedia_29_86490_1
#       C50 selects 1 limited lesion  -> maximum FG under fixed quota=1 = 1
#       C100 selects 2 limited lesions -> minimum FG under fixed quota=1 = 2
#
# Since the factorial experiment requires the SAME total FG budget across
# C50 and C100, no common B_i could exist.
#
# The post-failure, pre-outcome diagnostic tested a geometry-neutral
# PHYSICAL ANCHOR BANK:
#
#   GLOBAL_RESOLUTION_LIMITED:
#       minimum quota = 1
#       maximum capacity = all unique training-grid voxels occupied by
#                          that physical lesion component
#
#       The SAME deterministic ordered bank is used in COH / DIS / FRG.
#       Coverage levels may use different PREFIX LENGTHS because annotation
#       budget is redistributed across different numbers of selected lesions.
#
# RESULT:
#       20 / 20 cases aggregate-feasible
#       28 / 28 limited lesions have >=2 physical-grid support voxels
#       support range = 17–101 voxels
#       support median = 39 voxels
#
# A1.2 freezes this rule BEFORE any COVA optimizer step.
#
# IMPORTANT:
# This block does NOT yet claim final 09D-B coordinate feasibility.
# The next block must still verify:
#
#   • no cross-component voxel collisions
#   • identical BG coordinates across all six cells
#   • exact FG equality across all six cells
#   • geometry preservation for resolvable lesions
#   • nested neutral-bank prefixes across coverage
#
# NO TRAINING IS PERFORMED.
# ==========================================================================================

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import subprocess
import hashlib
import json
import os
import sys
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
    "8690900f10d4"
)

BLOCK = (
    "09D-A3-LOCK"
)

PARENT_PROTOCOL = (
    "COVA3D_1.0+A1+A1.1"
)

AMENDMENT_ID = (
    "A1_2_GEOMETRY_NEUTRAL_PHYSICAL_ANCHOR_BANK"
)

EFFECTIVE_PROTOCOL = (
    "COVA3D_1.0+A1+A1.1+A1.2"
)

EXPECTED_CASES = (
    20
)

EXPECTED_LIMITED_COMPONENTS = (
    28
)

EXPECTED_LIMITED_CASES = (
    13
)

EXPECTED_SUPPORT_MIN = (
    17
)

EXPECTED_SUPPORT_MEDIAN = (
    39.0
)

EXPECTED_SUPPORT_MAX = (
    101
)

EXPECTED_SUPPORT_GE2 = (
    28
)

FAILED_09DB_OUTPUT = (
    REPO
    / "data/cova3d_training_grid_candidates_v1_0"
)

DIAGNOSTIC_ROOT = Path(
    "/kaggle/working/cova3d_09db_neutral_bank_diagnostic"
)

CASE_DIAGNOSTIC_SOURCE = (
    DIAGNOSTIC_ROOT
    / "case_neutral_bank_feasibility.csv"
)

LIMITED_DIAGNOSTIC_SOURCE = (
    DIAGNOSTIC_ROOT
    / "limited_component_physical_capacity.csv"
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
        + "=" * 128
    )

    print(
        text
    )

    print(
        "=" * 128
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
        "/tmp/cova3d_git_askpass_09da3.sh"
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
# 2. VERIFY REPOSITORY + CLEAN FAILED 09D-B RUNTIME OUTPUT ONLY
# ==========================================================================================

heading(
    "COVA-3D BLOCK 09D-A3-LOCK — FREEZE GEOMETRY-NEUTRAL ANCHOR BANKS"
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


status_before = sh(
    [
        "git",
        "status",
        "--porcelain",
    ],
    cwd=REPO,
).stdout.strip()


if status_before:

    allowed_prefix = (
        "data/cova3d_training_grid_candidates_v1_0/"
    )

    unexpected = []

    for line in status_before.splitlines():

        path = line[
            3:
        ].strip()

        if not (
            path.startswith(
                allowed_prefix
            )
            or path
            == "data/cova3d_training_grid_candidates_v1_0/"
        ):

            unexpected.append(
                line
            )

    if unexpected:

        raise RuntimeError(
            "Unexpected repository modifications exist.\n"
            + "\n".join(
                unexpected
            )
        )


if FAILED_09DB_OUTPUT.exists():

    partial_files = list(
        FAILED_09DB_OUTPUT.rglob(
            "*.npz"
        )
    )

    print(
        "✓ Failed 09D-B partial NPZ files found :",
        len(
            partial_files
        )
    )

    shutil.rmtree(
        FAILED_09DB_OUTPUT
    )

else:

    print(
        "✓ Failed 09D-B partial NPZ files found : 0"
    )


status_after_cleanup = sh(
    [
        "git",
        "status",
        "--porcelain",
    ],
    cwd=REPO,
).stdout.strip()


if status_after_cleanup:

    raise RuntimeError(
        "Repository is not clean after removing only failed runtime outputs:\n"
        + status_after_cleanup
    )


print(
    "✓ Starting commit                      :",
    head[:12],
)

print(
    "✓ Failed 09D-B committed state         : NONE"
)

print(
    "✓ Repository state                     : CLEAN"
)


# ==========================================================================================
# 3. VERIFY A1.1 PRECONDITIONS
# ==========================================================================================

heading(
    "STEP 1/7 — VERIFY A1.1 LINEAGE"
)


cova_state_path = (
    REPO
    / "COVA3D_STATE.json"
)

project_state_path = (
    REPO
    / "PROJECT_STATE.json"
)

a11_path = (
    REPO
    / "configs/"
    "cova3d_resolution_aware_amendment_A1_1.yaml"
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

a11 = yaml.safe_load(
    a11_path.read_text(
        encoding="utf-8"
    )
)


if cova_state.get(
    "last_completed_block"
) != "09D-A2-LOCK":

    raise RuntimeError(
        "Expected 09D-A2-LOCK as the previous completed block."
    )


if cova_state.get(
    "effective_protocol"
) != PARENT_PROTOCOL:

    raise RuntimeError(
        "Unexpected effective protocol."
    )


if a11.get(
    "status"
) != "PROSPECTIVELY_FROZEN_BEFORE_COVA_TRAINING":

    raise RuntimeError(
        "A1.1 is not prospectively frozen."
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


if cova_state.get(
    "factorial_training_authorized"
) is not False:

    raise RuntimeError(
        "Factorial training was unexpectedly authorized."
    )


print(
    "✓ Parent protocol                     :",
    PARENT_PROTOCOL,
)

print(
    "✓ A1.1                                : FROZEN"
)

print(
    "✓ COVA optimizer steps                : 0"
)

print(
    "✓ Factorial training                  : NOT AUTHORIZED"
)


# ==========================================================================================
# 4. VERIFY NEUTRAL-BANK DIAGNOSTIC
# ==========================================================================================

heading(
    "STEP 2/7 — VERIFY PRE-OUTCOME NEUTRAL-BANK DIAGNOSTIC"
)


if not CASE_DIAGNOSTIC_SOURCE.exists():

    raise RuntimeError(
        "Neutral-bank case diagnostic is missing:\n"
        + str(
            CASE_DIAGNOSTIC_SOURCE
        )
    )


if not LIMITED_DIAGNOSTIC_SOURCE.exists():

    raise RuntimeError(
        "Neutral-bank component diagnostic is missing:\n"
        + str(
            LIMITED_DIAGNOSTIC_SOURCE
        )
    )


case_df = pd.read_csv(
    CASE_DIAGNOSTIC_SOURCE
)


limited_df = pd.read_csv(
    LIMITED_DIAGNOSTIC_SOURCE
)


if len(
    case_df
) != EXPECTED_CASES:

    raise RuntimeError(
        "Expected 20 case diagnostic rows."
    )


if len(
    limited_df
) != EXPECTED_LIMITED_COMPONENTS:

    raise RuntimeError(
        "Expected 28 limited-component rows."
    )


feasible_cases = int(
    case_df[
        "neutral_bank_aggregate_feasible"
    ].astype(
        bool
    ).sum()
)


limited_cases = int(
    limited_df[
        "case_id"
    ].nunique()
)


support_min = int(
    limited_df[
        "mapped_physical_support_capacity"
    ].min()
)


support_median = float(
    limited_df[
        "mapped_physical_support_capacity"
    ].median()
)


support_max = int(
    limited_df[
        "mapped_physical_support_capacity"
    ].max()
)


support_ge2 = int(
    (
        limited_df[
            "mapped_physical_support_capacity"
        ]
        >= 2
    ).sum()
)


if feasible_cases != EXPECTED_CASES:

    raise RuntimeError(
        "Neutral-bank diagnostic did not reproduce 20/20 feasibility."
    )


if limited_cases != EXPECTED_LIMITED_CASES:

    raise RuntimeError(
        "Expected globally limited lesions in 13 cases."
    )


if support_min != EXPECTED_SUPPORT_MIN:

    raise RuntimeError(
        "Unexpected minimum physical support."
    )


if not np.isclose(
    support_median,
    EXPECTED_SUPPORT_MEDIAN,
):

    raise RuntimeError(
        "Unexpected median physical support."
    )


if support_max != EXPECTED_SUPPORT_MAX:

    raise RuntimeError(
        "Unexpected maximum physical support."
    )


if support_ge2 != EXPECTED_SUPPORT_GE2:

    raise RuntimeError(
        "Not all limited lesions support >=2 physical anchors."
    )


failure_case = case_df[
    case_df[
        "case_id"
    ]
    == "radiopaedia_29_86490_1"
]


if len(
    failure_case
) != 1:

    raise RuntimeError(
        "Missing diagnostic row for radiopaedia_29_86490_1."
    )


failure_case = failure_case.iloc[
    0
]


if not bool(
    failure_case[
        "neutral_bank_aggregate_feasible"
    ]
):

    raise RuntimeError(
        "Previously failing case remains infeasible."
    )


if int(
    failure_case[
        "common_minimum"
    ]
) != 2:

    raise RuntimeError(
        "Unexpected common minimum for radiopaedia_29_86490_1."
    )


if int(
    failure_case[
        "common_maximum"
    ]
) != 20:

    raise RuntimeError(
        "Unexpected common maximum for radiopaedia_29_86490_1."
    )


print(
    "✓ Neutral-bank aggregate feasibility   : 20/20"
)

print(
    "✓ Globally limited components          : 28"
)

print(
    "✓ Cases with limited components        : 13/20"
)

print(
    "✓ Physical support range               :",
    support_min,
    "–",
    support_max,
    "voxels",
)

print(
    "✓ Physical support median              :",
    "{:.1f}".format(
        support_median
    ),
    "voxels",
)

print(
    "✓ Limited lesions with >=2 support     :",
    support_ge2,
    "/28",
)

print(
    "✓ radiopaedia_29_86490_1              : FEASIBLE [2,20]"
)


# ==========================================================================================
# 5. ARCHIVE THE DIAGNOSTIC AS SCIENTIFIC PROVENANCE
# ==========================================================================================

heading(
    "STEP 3/7 — ARCHIVE PRE-OUTCOME DIAGNOSTIC"
)


archive_dir = (
    REPO
    / "experiments/audits/"
    "block09d_b_neutral_bank_diagnostic"
)


archive_dir.mkdir(
    parents=True,
    exist_ok=True,
)


case_archive = (
    archive_dir
    / "case_neutral_bank_feasibility.csv"
)


limited_archive = (
    archive_dir
    / "limited_component_physical_capacity.csv"
)


shutil.copy2(
    CASE_DIAGNOSTIC_SOURCE,
    case_archive,
)


shutil.copy2(
    LIMITED_DIAGNOSTIC_SOURCE,
    limited_archive,
)


capacity_manifest_path = (
    REPO
    / "data/manifests/"
    "cova3d_resolution_limited_neutral_bank_capacity_v1_0.csv"
)


limited_df.sort_values(
    [
        "case_id",
        "component_native_id",
    ]
).to_csv(
    capacity_manifest_path,
    index=False,
)


print(
    "✓ Case diagnostic archived             : YES"
)

print(
    "✓ Limited-component diagnostic archived: YES"
)

print(
    "✓ Neutral-bank capacity manifest       : 28 rows"
)


# ==========================================================================================
# 6. FREEZE AMENDMENT A1.2
# ==========================================================================================

heading(
    "STEP 4/7 — FREEZE A1.2"
)


a12 = {
    "project":
        "COVA-3D",

    "parent_protocol":
        PARENT_PROTOCOL,

    "amendment_id":
        AMENDMENT_ID,

    "effective_protocol":
        EFFECTIVE_PROTOCOL,

    "status":
        "PROSPECTIVELY_FROZEN_BEFORE_COVA_TRAINING",

    "trigger":
        {
            "failed_block":
                "09D-B",

            "failure_case":
                "radiopaedia_29_86490_1",

            "failure_type":
                "FIXED_LIMITED_QUOTA_PREVENTED_EQUAL_COVERAGE_BUDGET",

            "C50_fixed_quota_capacity":
                1,

            "C100_fixed_quota_minimum":
                2,
        },

    "diagnostic_evidence":
        {
            "cases_tested":
                20,

            "aggregate_feasible_cases":
                feasible_cases,

            "globally_resolution_limited_components":
                len(
                    limited_df
                ),

            "cases_with_limited_components":
                limited_cases,

            "physical_support_min_voxels":
                support_min,

            "physical_support_median_voxels":
                support_median,

            "physical_support_max_voxels":
                support_max,

            "limited_components_with_support_ge_2":
                support_ge2,

            "model_outcomes_used":
                False,

            "optimizer_steps":
                0,
        },

    "supersedes_only":
        (
            "A1.1 rule that GLOBAL_RESOLUTION_LIMITED quota is fixed "
            "at exactly one voxel"
        ),

    "unchanged_from_A1_1":
        {
            "global_resolution_class":
                "coverage-invariant",

            "coverage_selection":
                "unchanged",

            "geometry_claim_for_limited_lesions":
                False,

            "geometry_manipulation_for_resolvable_lesions":
                True,

            "primary_endpoint":
                "macro patient lesion recall at <=1 FP/patient",

            "primary_endpoint_changed":
                False,
        },

    "GLOBAL_RESOLUTION_LIMITED":
        {
            "minimum_quota":
                1,

            "maximum_quota":
                "mapped physical lesion support capacity",

            "geometry_manipulation":
                False,

            "representation":
                "geometry-neutral deterministic physical anchor bank",

            "bank_source":
                (
                    "all unique training-grid voxels occupied by the "
                    "physical native lesion component"
                ),

            "bank_order":
                (
                    "ascending Euclidean world-space distance from the "
                    "native physical lesion-component centroid; ties "
                    "resolved lexicographically by crop-grid z, y, x"
                ),

            "condition_use":
                (
                    "for assigned quota q, use the first q voxels of the "
                    "same frozen ordered bank"
                ),

            "shared_across_geometry":
                True,

            "shared_across_coverage":
                (
                    "same ordered bank; coverage-specific quota may use "
                    "different nested prefix lengths"
                ),

            "cross_coverage_prefix_property":
                True,

            "geometry_claim_allowed":
                False,
        },

    "GLOBAL_GEOMETRY_RESOLVABLE":
        {
            "minimum_quota":
                2,

            "maximum_quota":
                (
                    "minimum geometry-preserving capacity across COH, "
                    "DIS and FRG for that coverage realization"
                ),

            "geometry_manipulation":
                True,
        },

    "budget_allocation":
        {
            "target":
                "largest exact common B_i_train feasible across all six cells",

            "upper_bound":
                [
                    "native frozen B_i",
                    "C50 total component capacity",
                    "C100 total component capacity",
                    "mapped common BG capacity",
                ],

            "lower_bound":
                (
                    "maximum of the C50 and C100 sums of component-specific "
                    "minimum quotas"
                ),

            "candidate_search":
                "descending from upper bound to lower bound",

            "component_allocation":
                (
                    "initialize every component at its frozen minimum; "
                    "distribute remaining quota one voxel per component "
                    "in frozen selected-component order while capacity remains"
                ),

            "allocation_outcome_adaptive":
                False,

            "model_outcomes_used":
                False,
        },

    "coordinate_level_requirements_for_09D_B2":
        {
            "cross_component_FG_collision":
                "forbidden",

            "FG_BG_collision":
                "forbidden",

            "same_total_FG_across_six":
                True,

            "same_BG_coordinates_across_six":
                True,

            "same_component_set_across_geometry":
                True,

            "same_component_quota_across_geometry":
                True,

            "C50_component_set_nested_in_C100":
                True,

            "limited_bank_prefix_nested_across_coverage":
                True,

            "resolvable_geometry_preservation":
                True,

            "selected_lesions_dropped":
                False,
        },

    "failure_policy":
        (
            "If exact coordinate-level construction fails for any case, "
            "stop. Do not alter A1.2 based on model outcomes."
        ),

    "factorial_training_authorized":
        False,

    "method_development_authorized":
        False,

    "next_block":
        "09D-B2",

    "frozen_at_utc":
        NOW_ISO,
}


a12_path = (
    REPO
    / "configs/"
    "cova3d_resolution_aware_amendment_A1_2.yaml"
)


a12_path.write_text(
    yaml.safe_dump(
        a12,
        sort_keys=False,
        allow_unicode=True,
    ),
    encoding="utf-8",
)


write_text(
    REPO
    / "docs/"
    "cova3d_protocol_amendment_A1_2_neutral_anchor_banks.md",
    f"""
    # COVA-3D Protocol Amendment A1.2

    ## Geometry-neutral physical anchor banks

    Amendment A1.1 used exactly one foreground anchor for every globally
    resolution-limited lesion.

    This created a mathematical contradiction in
    `radiopaedia_29_86490_1`.

    The 50% condition selects one lesion while the 100% condition selects two.
    With one fixed anchor per lesion, C50 could contain at most one positive
    supervision voxel while C100 required at least two. An equal foreground
    annotation budget was therefore impossible.

    This failure occurred before any COVA-3D training.

    A pre-outcome diagnostic then measured the physical training-grid support
    of all {len(limited_df)} globally resolution-limited lesions.

    Every limited lesion retained at least {support_min} unique grid voxels.
    The median support was {support_median:.1f} voxels and the maximum was
    {support_max} voxels.

    Allowing these lesions to use a geometry-neutral ordered physical anchor
    bank made the aggregate budget interval feasible in all {feasible_cases}
    primary cases.

    ## Frozen A1.2 rule

    Globally resolution-limited lesions still have no coherent, dispersed or
    fragmented geometry claim.

    They instead have a deterministic physical anchor bank.

    The bank consists of unique training-grid voxels occupied by the physical
    lesion component. Voxels are ordered by increasing world-space distance
    from the physical native-component centroid. Ties are resolved
    lexicographically by crop-grid z, y and x.

    The minimum quota is one voxel.

    The maximum quota is the mapped physical support capacity.

    For quota q, the first q voxels of the ordered bank are used.

    The same ordered bank is used for all geometry conditions and all coverage
    conditions in which the lesion appears.

    Different coverage conditions may use different prefix lengths because
    fixed total annotation budget is redistributed across different numbers of
    lesion instances. The prefixes remain nested.

    Geometry-resolvable lesions retain the original geometry manipulation and
    require at least two annotation voxels.

    ## Budget rule

    For each patient, the final B_i_train is the largest common foreground
    budget that can be constructed in all six factorial cells.

    Component allocation begins at the frozen minimum quotas. Remaining quota
    is assigned deterministically in selected-component order, one voxel at a
    time while capacity remains.

    The final construction must also pass coordinate-level collision tests.
    Therefore the 20/20 result obtained before A1.2 is an aggregate capacity
    result, not yet the final annotation-feasibility result.

    Block 09D-B2 must verify the complete coordinate-level construction.

    Training remains unauthorized.
    """
)


print(
    "✓ Amendment ID                         :",
    AMENDMENT_ID,
)

print(
    "✓ Effective protocol                   :",
    EFFECTIVE_PROTOCOL,
)

print(
    "✓ Limited minimum quota                : 1"
)

print(
    "✓ Limited fixed maximum quota          : REMOVED"
)

print(
    "✓ Limited maximum capacity             : PHYSICAL GRID SUPPORT"
)

print(
    "✓ Same neutral bank across geometry    : REQUIRED"
)

print(
    "✓ Same neutral bank across coverage    : REQUIRED"
)

print(
    "✓ Cross-coverage bank prefixes         : NESTED"
)

print(
    "✓ Training authorized                  : NO"
)


# ==========================================================================================
# 7. UPDATE STATE
# ==========================================================================================

heading(
    "STEP 5/7 — UPDATE STATE WITHOUT AUTHORIZING TRAINING"
)


cova_state.update(
    {
        "last_completed_block":
            BLOCK,

        "current_stage":
            "NEUTRAL_ANCHOR_BANK_AMENDMENT_FROZEN",

        "effective_protocol":
            EFFECTIVE_PROTOCOL,

        "protocol_amendment":
            "A1+A1.1+A1.2",

        "resolution_aware_amendment_status":
            "FROZEN_A1_2",

        "neutral_anchor_bank_status":
            "FROZEN",

        "neutral_anchor_bank_aggregate_feasibility":
            "PASS_20_OF_20",

        "neutral_anchor_bank_limited_components":
            28,

        "neutral_anchor_bank_support_min":
            support_min,

        "neutral_anchor_bank_support_median":
            support_median,

        "neutral_anchor_bank_support_max":
            support_max,

        "trainer_grid_budget_status":
            "PENDING_09D_B2",

        "trainer_grid_collision_audit":
            "PENDING_09D_B2",

        "factorial_training_authorized":
            False,

        "method_development_authorized":
            False,

        "final_outer_cv_outcomes_authorized":
            False,

        "optimizer_steps_in_cova3d":
            0,

        "next_block":
            "09D-B2",

        "next_action":
            (
                "Construct final neutral-bank training-grid annotations and "
                "verify exact coordinate-level six-cell feasibility. "
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
            "cova3d_neutral_anchor_bank_amendment",

        "current_stage":
            "cova3d_A1_2_frozen",

        "current_gate":
            "COVA_NEUTRAL_BANK_COORDINATE_FEASIBILITY",

        "cova3d_effective_protocol":
            EFFECTIVE_PROTOCOL,

        "cova3d_neutral_anchor_bank_status":
            "FROZEN",

        "cova3d_neutral_bank_aggregate_feasibility":
            "PASS_20_OF_20",

        "cova3d_trainer_grid_feasibility":
            "PENDING_09D_B2",

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
                "Run Block 09D-B2 final coordinate-level neutral-bank "
                "construction. Do not train."
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
# 8. REGRESSION TEST
# ==========================================================================================

test_source = r'''
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_A1_2_is_frozen():

    config = yaml.safe_load(
        (
            ROOT
            / "configs/"
            "cova3d_resolution_aware_amendment_A1_2.yaml"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        config[
            "status"
        ]
        == "PROSPECTIVELY_FROZEN_BEFORE_COVA_TRAINING"
    )

    assert (
        config[
            "GLOBAL_RESOLUTION_LIMITED"
        ][
            "minimum_quota"
        ]
        == 1
    )

    assert (
        config[
            "GLOBAL_RESOLUTION_LIMITED"
        ][
            "geometry_manipulation"
        ]
        is False
    )

    assert (
        config[
            "GLOBAL_RESOLUTION_LIMITED"
        ][
            "cross_coverage_prefix_property"
        ]
        is True
    )

    assert (
        config[
            "factorial_training_authorized"
        ]
        is False
    )


def test_neutral_bank_capacity_manifest():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_resolution_limited_neutral_bank_capacity_v1_0.csv"
    )

    assert len(
        frame
    ) == 28

    assert frame[
        "case_id"
    ].nunique() == 13

    assert (
        frame[
            "mapped_physical_support_capacity"
        ]
        >= 2
    ).all()


def test_neutral_bank_case_diagnostic():

    frame = pd.read_csv(
        ROOT
        / "experiments/audits/"
        "block09d_b_neutral_bank_diagnostic/"
        "case_neutral_bank_feasibility.csv"
    )

    assert len(
        frame
    ) == 20

    assert frame[
        "neutral_bank_aggregate_feasible"
    ].astype(
        bool
    ).all()

    special = frame[
        frame[
            "case_id"
        ]
        == "radiopaedia_29_86490_1"
    ].iloc[
        0
    ]

    assert int(
        special[
            "common_minimum"
        ]
    ) == 2

    assert int(
        special[
            "common_maximum"
        ]
    ) == 20
'''


test_path = (
    REPO
    / "tests/"
    "test_cova3d_amendment_a1_2.py"
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
        "tests/test_cova3d_amendment_a1_2.py",
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
        "A1.2 regression tests FAILED."
    )


print(
    "✓ A1.2 regression tests                : PASS"
)


# ==========================================================================================
# 9. AUDIT
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

    "parent_protocol":
        PARENT_PROTOCOL,

    "amendment_id":
        AMENDMENT_ID,

    "effective_protocol":
        EFFECTIVE_PROTOCOL,

    "failed_09D_B": {
        "committed":
            False,

        "failure_case":
            "radiopaedia_29_86490_1",

        "failure_reason":
            "FIXED_QUOTA_1_PREVENTED_COMMON_C50_C100_BUDGET",
    },

    "neutral_bank_diagnostic": {
        "cases":
            20,

        "aggregate_feasible_cases":
            feasible_cases,

        "limited_components":
            28,

        "limited_cases":
            limited_cases,

        "support_min":
            support_min,

        "support_median":
            support_median,

        "support_max":
            support_max,

        "support_ge2":
            support_ge2,

        "radiopaedia_29_86490_1_common_minimum":
            int(
                failure_case[
                    "common_minimum"
                ]
            ),

        "radiopaedia_29_86490_1_common_maximum":
            int(
                failure_case[
                    "common_maximum"
                ]
            ),
    },

    "frozen_rule": {
        "limited_minimum_quota":
            1,

        "limited_capacity":
            "mapped_physical_component_support",

        "limited_geometry_manipulation":
            False,

        "same_ordered_bank_across_geometry":
            True,

        "same_ordered_bank_across_coverage":
            True,

        "coverage_uses_nested_bank_prefix":
            True,

        "resolvable_minimum_quota":
            2,

        "final_coordinate_level_feasibility":
            "PENDING_09D_B2",
    },

    "firewall": {
        "CT_arrays_accessed_in_this_lock":
            0,

        "dense_masks_accessed_in_this_lock":
            0,

        "predictions_accessed":
            0,

        "checkpoints_accessed":
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

    "next_block":
        "09D-B2",
}


audit_path = (
    REPO
    / "experiments/audits/"
    "block09d_a3_neutral_anchor_bank_amendment.json"
)


write_json(
    audit_path,
    audit,
)


# ==========================================================================================
# 10. SOURCE CAPTURE
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
        "COVA-3D — BLOCK 09D-A3-LOCK"
        in cell
    ):

        source_path = (
            REPO
            / "scripts/code_blocks/"
            "block09d_a3_lock_neutral_anchor_bank.py"
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
# 11. REFRESH REPOSITORY MANIFEST
# ==========================================================================================

repo_manifest_path = (
    REPO
    / "REPOSITORY_MANIFEST.json"
)


EXCLUDED = {
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
                part in EXCLUDED
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

        "neutral_anchor_bank_amendment":
            "FROZEN",

        "neutral_bank_aggregate_feasibility":
            "PASS_20_OF_20",

        "final_coordinate_grid_feasibility":
            "PENDING_09D_B2",

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
# 12. COMMIT + PUSH
# ==========================================================================================

heading(
    "STEP 6/7 — COMMIT A1.2"
)


git_paths = [
    "COVA3D_STATE.json",
    "PROJECT_STATE.json",
    "REPOSITORY_MANIFEST.json",

    "configs/"
    "cova3d_resolution_aware_amendment_A1_2.yaml",

    "data/manifests/"
    "cova3d_resolution_limited_neutral_bank_capacity_v1_0.csv",

    "docs/"
    "cova3d_protocol_amendment_A1_2_neutral_anchor_banks.md",

    "experiments/audits/"
    "block09d_a3_neutral_anchor_bank_amendment.json",

    "experiments/audits/"
    "block09d_b_neutral_bank_diagnostic/"
    "case_neutral_bank_feasibility.csv",

    "experiments/audits/"
    "block09d_b_neutral_bank_diagnostic/"
    "limited_component_physical_capacity.csv",

    "tests/"
    "test_cova3d_amendment_a1_2.py",
]


source_path = (
    REPO
    / "scripts/code_blocks/"
    "block09d_a3_lock_neutral_anchor_bank.py"
)


if source_path.exists():

    git_paths.append(
        "scripts/code_blocks/"
        "block09d_a3_lock_neutral_anchor_bank.py"
    )


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
    "✓ git diff --cached --check             : PASS"
)


sh(
    [
        "git",
        "commit",
        "-m",
        "protocol: freeze COVA-3D neutral physical anchor banks",
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
        "Repository is not clean after A1.2 commit:\n"
        + final_status
    )


# ==========================================================================================
# 13. FINAL REPORT
# ==========================================================================================

heading(
    "STEP 7/7 — FINAL A1.2 REPORT"
)


print(
    "COVA-3D BLOCK 09D-A3-LOCK — FINAL REPORT"
)

print(
    "-" * 128
)

print(
    "Failed 09D-B committed                : NO"
)

print(
    "Failure source                        : FIXED QUOTA=1"
)

print(
    "Previously failing case               : radiopaedia_29_86490_1"
)

print()

print(
    "NEUTRAL-BANK DIAGNOSTIC"
)

print(
    "-----------------------"
)

print(
    "Aggregate feasible cases              :",
    feasible_cases,
    "/20",
)

print(
    "Globally limited lesions              :",
    len(
        limited_df
    ),
)

print(
    "Cases with limited lesions            :",
    limited_cases,
    "/20",
)

print(
    "Physical support range                :",
    support_min,
    "–",
    support_max,
    "voxels",
)

print(
    "Physical support median               :",
    "{:.1f}".format(
        support_median
    ),
    "voxels",
)

print(
    "Limited lesions with >=2 support      :",
    support_ge2,
    "/28",
)

print()

print(
    "A1.2"
)

print(
    "----"
)

print(
    "Effective protocol                    :",
    EFFECTIVE_PROTOCOL,
)

print(
    "Limited minimum quota                 : 1"
)

print(
    "Limited maximum capacity              : PHYSICAL SUPPORT"
)

print(
    "Fixed quota=1                         : RETIRED"
)

print(
    "Limited geometry manipulation         : NO"
)

print(
    "Same ordered bank across geometry     : YES"
)

print(
    "Same ordered bank across coverage     : YES"
)

print(
    "Coverage-specific use                 : NESTED PREFIX"
)

print(
    "Resolvable minimum quota              : 2"
)

print(
    "Equal-six-cell FG requirement         : UNCHANGED"
)

print(
    "Identical BG requirement              : UNCHANGED"
)

print(
    "Selected lesions dropped              : 0"
)

print(
    "Primary endpoint changed              : NO"
)

print()

print(
    "STATUS"
)

print(
    "------"
)

print(
    "Aggregate feasibility                 : PASS 20/20"
)

print(
    "Coordinate-level feasibility          : PENDING 09D-B2"
)

print(
    "Model outcomes accessed               : 0"
)

print(
    "Optimizer steps                       : 0"
)

print(
    "Factorial training authorized         : NO"
)

print(
    "Method development authorized         : NO"
)

print()

print(
    "Regression tests                      : PASS"
)

print(
    "Source capture                        :",
    source_capture,
)

print(
    "Starting commit                       :",
    head[:12],
)

print(
    "A1.2 commit                           :",
    final_commit[:12],
)

print(
    "GitHub synchronization                : PASS"
)

print(
    "Repository state                      : CLEAN"
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
    "Next block will be 09D-B2: final coordinate-level construction using "
    "the frozen neutral-bank prefixes."
)

print(
    "That block must pass 20/20 before the annotation methodology is "
    "declared structurally ready."
)

print(
    "=" * 128
)
