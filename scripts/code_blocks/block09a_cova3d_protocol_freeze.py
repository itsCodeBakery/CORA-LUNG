# ==========================================================================================
# COVA-3D — CODE BLOCK 09A
# Prospective Protocol Freeze + CORA-Lineage Closure
#
# WORKING TITLE
#
# COVA-3D:
# Coverage–Coherence Trade-offs in Fixed-Budget Sparse Supervision
# for 3-D Multi-Lesion Segmentation
#
# PURPOSE
#
# 1. Preserve the completed CORA-Lung NO-GO lineage without alteration.
# 2. Start a NEW prospective scientific track in the same repository.
# 3. Freeze the new research question BEFORE any new labels are simulated,
#    any new model is trained, or any new outcome is inspected.
# 4. Register the 2 x 3 factorial design:
#
#       lesion-instance coverage:
#           50%
#           100%
#
#       within-instance annotation geometry:
#           coherent
#           dispersed
#           fragmented
#
# 5. Freeze exact causal-matching principles:
#
#       - same image
#       - same patient
#       - same positive annotation budget B_i
#       - identical background supervision
#       - same selected component set across geometry conditions
#       - same per-component positive quota across geometry conditions
#       - 50% component set nested inside 100% component set
#
# 6. Freeze the primary scientific estimands.
#
# THIS BLOCK DOES NOT:
#
#   - open CT volumes
#   - open lesion masks
#   - open lung masks
#   - regenerate weak labels
#   - access development outcomes
#   - access final outer-CV outcomes
#   - train a model
#   - instantiate an optimizer
#   - authorize factorial training
#   - authorize method development
#
# CORA-LUNG STATUS AFTER THIS BLOCK:
#
#   Gate A = PASS
#   Gate B = NO_GO
#   Gate C = NOT RUN
#   Original CORA replay hypothesis = CLOSED / RETIRED
#
# COVA-3D STATUS AFTER THIS BLOCK:
#
#   Protocol v1.0 = FROZEN
#   Training = NOT AUTHORIZED
#   Next = Block 09B dataset suitability + factorial feasibility audit
# ==========================================================================================


from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from itertools import product

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
# 0. FROZEN CONSTANTS
# ==========================================================================================

REPO = Path(
    "/kaggle/working/CORA-LUNG"
)

REPOSITORY_FULL_NAME = (
    "itsCodeBakery/CORA-LUNG"
)

REPOSITORY_URL = (
    "https://github.com/itsCodeBakery/CORA-LUNG.git"
)

EXPECTED_PARENT_COMMIT = (
    "e8ee13f3270e"
)

BLOCK = (
    "09A"
)

TRACK_ID = (
    "COVA3D"
)

PROTOCOL_VERSION = (
    "1.0"
)

WORKING_TITLE = (
    "COVA-3D: Coverage–Coherence Trade-offs in Fixed-Budget "
    "Sparse Supervision for 3-D Multi-Lesion Segmentation"
)

RESEARCH_QUESTION = (
    "Under an equal sparse-annotation budget, how do lesion-instance "
    "coverage and within-instance annotation coherence independently "
    "and jointly affect 3-D multi-lesion segmentation performance and "
    "recovery of unannotated lesions?"
)

COVERAGE_LEVELS = [
    0.50,
    1.00,
]

GEOMETRY_LEVELS = [
    "coherent",
    "dispersed",
    "fragmented",
]

GEOMETRY_IDS = {
    "coherent":
        "COH",

    "dispersed":
        "DIS",

    "fragmented":
        "FRG",
}

CONDITION_IDS = [
    "C50_COH",
    "C50_DIS",
    "C50_FRG",
    "C100_COH",
    "C100_DIS",
    "C100_FRG",
]

COMPONENT_SELECTION_MASTER_SEED = (
    20260914
)

DEVELOPMENT_MODEL_SEEDS = [
    17,
]

FINAL_MODEL_SEEDS = [
    17,
    29,
    43,
]

BOOTSTRAP_SEED = (
    20260914
)

BOOTSTRAP_REPLICATES = (
    10000
)

PRIMARY_REFERENCE_CONNECTIVITY = (
    26
)

PRIMARY_REFERENCE_MIN_VOLUME_ML = (
    0.10
)

PRIMARY_MATCH_IOU = (
    0.10
)

PRIMARY_FROC_BUDGET = (
    1.0
)

SECONDARY_FROC_BUDGETS = [
    0.5,
    2.0,
    4.0,
]

# New-track FROC will recompute connected components at every probability
# threshold rather than fixing topology at p=0.5.
#
# This threshold grid is frozen BEFORE any COVA-3D training outcome exists.
DYNAMIC_FROC_THRESHOLDS = [
    round(
        float(x),
        4,
    )
    for x in np.arange(
        0.05,
        0.951,
        0.05,
    )
] + [
    0.975,
    0.990,
    0.995,
    1.000001,
]

INTERACTION_SIGNAL_MARGIN = (
    0.05
)

INTERACTION_CASE_CONSISTENCY_MIN = (
    3
)

NOW = datetime.now(
    timezone.utc
)

NOW_ISO = NOW.strftime(
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
            "Kaggle secret 'pushCora' is unavailable."
        )

    token = token.strip()

    askpass = Path(
        "/tmp/cova3d_git_askpass_09a.sh"
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
    "COVA-3D BLOCK 09A — PROSPECTIVE PROTOCOL FREEZE"
)


token, askpass, git_env = make_git_auth()


if not (
    REPO
    / ".git"
).exists():

    print(
        "Repository not present locally."
    )

    print(
        "Restoring repository from GitHub..."
    )

    if REPO.exists():

        raise RuntimeError(
            "A non-git directory already exists at "
            + str(
                REPO
            )
        )

    clone_result = sh(
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

    if clone_result.returncode != 0:

        safe_error = (
            clone_result.stderr
            or ""
        ).replace(
            token,
            "***TOKEN_REDACTED***",
        )

        raise RuntimeError(
            "Repository clone failed:\n"
            + safe_error
        )

    print(
        "✓ Repository restoration             : PASS"
    )


if not (
    REPO
    / ".git"
).exists():

    raise RuntimeError(
        "Repository is unavailable."
    )


# Local git identity is required on fresh Kaggle sessions.
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


current_branch = sh(
    [
        "git",
        "branch",
        "--show-current",
    ],
    cwd=REPO,
).stdout.strip()


if current_branch != "main":

    raise RuntimeError(
        "Expected main branch; observed "
        + repr(
            current_branch
        )
    )


current_commit = sh(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    cwd=REPO,
).stdout.strip()


if not current_commit.startswith(
    EXPECTED_PARENT_COMMIT
):

    raise RuntimeError(
        "Unexpected starting commit.\n"
        + "Expected prefix: "
        + EXPECTED_PARENT_COMMIT
        + "\nObserved: "
        + current_commit
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
        "Repository must be clean before Block 09A.\n"
        + dirty
    )


print(
    "✓ Starting commit                    :",
    current_commit[:12],
)

print(
    "✓ Repository state                   : CLEAN"
)


# ==========================================================================================
# 3. VERIFY IMMUTABLE CORA CLOSURE STATE
# ==========================================================================================

heading(
    "STEP 1/7 — VERIFY CORA-LUNG NO-GO LINEAGE"
)


project_state_path = (
    REPO
    / "PROJECT_STATE.json"
)


project_state = json.loads(
    project_state_path.read_text(
        encoding="utf-8"
    )
)


required_old_state = {
    "last_completed_block":
        "08G",

    "gate_a":
        "PASS",

    "gate_b":
        "NO_GO",

    "gate_c":
        "NOT_RUN",

    "gate_b_dynamic_topology_audit":
        "PASS",

    "gate_b_dynamic_topology_interpretation":
        "PRIMARY_PREMISE_REMAINS_UNSUPPORTED",
}


for key, expected in required_old_state.items():

    observed = project_state.get(
        key
    )

    if observed != expected:

        raise RuntimeError(
            "CORA closure prerequisite failed:\n"
            + str(
                key
            )
            + "\nExpected: "
            + repr(
                expected
            )
            + "\nObserved: "
            + repr(
                observed
            )
        )


if int(
    project_state.get(
        "final_outer_cv_dynamic_audit_access",
        -1,
    )
) != 0:

    raise RuntimeError(
        "Final outer-CV access occurred during the old topology audit."
    )


if int(
    project_state.get(
        "final_outer_cv_failure_audit_access",
        -1,
    )
) != 0:

    raise RuntimeError(
        "Final outer-CV access occurred during the old failure audit."
    )


old_optimizer_steps = int(
    project_state.get(
        "optimizer_steps_performed",
        -1,
    )
)


if old_optimizer_steps != 13500:

    raise RuntimeError(
        "Unexpected historical optimizer-step lineage."
    )


print(
    "✓ CORA Gate A                        : PASS"
)

print(
    "✓ CORA Gate B                        : NO_GO"
)

print(
    "✓ CORA Gate C                        : NOT RUN"
)

print(
    "✓ Dynamic-topology interpretation    : PRIMARY_PREMISE_REMAINS_UNSUPPORTED"
)

print(
    "✓ Historical optimizer steps         :",
    old_optimizer_steps,
)

print(
    "✓ Final outer-CV audit access         : 0"
)


# ==========================================================================================
# 4. BUILD FACTORIAL REGISTER
# ==========================================================================================

heading(
    "STEP 2/7 — REGISTER 2 x 3 PRIMARY FACTORIAL DESIGN"
)


registry_rows = []


for coverage, geometry in product(
    COVERAGE_LEVELS,
    GEOMETRY_LEVELS,
):

    coverage_percent = int(
        round(
            coverage
            * 100
        )
    )

    condition_id = (
        "C"
        + str(
            coverage_percent
        )
        + "_"
        + GEOMETRY_IDS[
            geometry
        ]
    )

    registry_rows.append(
        {
            "condition_id":
                condition_id,

            "coverage_fraction":
                float(
                    coverage
                ),

            "coverage_percent":
                coverage_percent,

            "geometry":
                geometry,

            "geometry_id":
                GEOMETRY_IDS[
                    geometry
                ],

            "primary_factorial_cell":
                True,

            "positive_budget_rule":
                "exact_same_B_i_within_case_across_all_six_cells",

            "background_rule":
                "identical_BG_coordinates_within_case_across_all_six_cells",

            "component_selection_rule":
                (
                    "deterministic_seeded_nested_selection;"
                    "50pct_subset_of_100pct"
                ),

            "geometry_pairing_rule":
                (
                    "same_selected_components_and_same_per_component_"
                    "positive_quota_across_geometry_levels"
                ),

            "dense_masks_allowed_in_generator":
                True,

            "dense_masks_allowed_in_trainer":
                False,

            "training_authorized":
                False,

            "outcome_status":
                "UNOPENED_NEW_TRACK",
        }
    )


registry_df = pd.DataFrame(
    registry_rows
).sort_values(
    [
        "coverage_fraction",
        "geometry",
    ]
).reset_index(
    drop=True
)


if len(
    registry_df
) != 6:

    raise RuntimeError(
        "Expected exactly six factorial cells."
    )


if set(
    registry_df[
        "condition_id"
    ]
) != set(
    CONDITION_IDS
):

    raise RuntimeError(
        "Factorial condition registry mismatch."
    )


if registry_df[
    [
        "coverage_fraction",
        "geometry",
    ]
].duplicated().any():

    raise RuntimeError(
        "Duplicate factorial cell detected."
    )


for coverage in COVERAGE_LEVELS:

    observed_geometries = set(
        registry_df.loc[
            np.isclose(
                registry_df[
                    "coverage_fraction"
                ],
                coverage,
            ),
            "geometry",
        ]
    )

    if observed_geometries != set(
        GEOMETRY_LEVELS
    ):

        raise RuntimeError(
            "Incomplete geometry cross-product."
        )


print(
    registry_df[
        [
            "condition_id",
            "coverage_percent",
            "geometry",
        ]
    ].to_string(
        index=False
    )
)


print()
print(
    "✓ Primary factorial cells             : 6"
)

print(
    "✓ Coverage levels                     : 50%, 100%"
)

print(
    "✓ Geometry levels                     : coherent / dispersed / fragmented"
)

print(
    "✓ Factorial cross-product             : PASS"
)


# ==========================================================================================
# 5. FREEZE PROTOCOL CONFIG
# ==========================================================================================

heading(
    "STEP 3/7 — FREEZE COVA-3D PROTOCOL v1.0"
)


protocol_config = {
    "project": {
        "track_id":
            TRACK_ID,

        "working_title":
            WORKING_TITLE,

        "protocol_version":
            PROTOCOL_VERSION,

        "status":
            "PROSPECTIVE_FROZEN",

        "parent_repository":
            REPOSITORY_FULL_NAME,

        "parent_commit":
            current_commit,

        "cora_predecessor_status":
            "CLOSED_AFTER_GATE_B_NO_GO",
    },

    "research_question":
        RESEARCH_QUESTION,

    "scope": {
        "task":
            "3D_multi_lesion_segmentation",

        "supervision":
            "fixed_budget_sparse_foreground_and_background",

        "primary_scientific_goal":
            (
                "estimate independent and joint effects of lesion-instance "
                "coverage and within-instance annotation geometry"
            ),

        "new_method_development":
            "PROHIBITED_UNTIL_FACTORIAL_SIGNAL_REVIEW",
    },

    "factors": {
        "coverage": {
            "levels":
                COVERAGE_LEVELS,

            "definition":
                (
                    "fraction of eligible lesion components receiving at "
                    "least one foreground annotation"
                ),

            "count_rule":
                (
                    "100pct uses all eligible components; "
                    "50pct uses ceil(0.5*K_i), minimum 1"
                ),

            "nesting":
                "50pct_selected_components_must_be_subset_of_100pct",

            "selection":
                "uniform_seeded_without_replacement",

            "selection_master_seed":
                COMPONENT_SELECTION_MASTER_SEED,
        },

        "geometry": {
            "levels":
                GEOMETRY_LEVELS,

            "coherent":
                (
                    "connected interior stroke/path within each selected "
                    "component; exact coordinate algorithm frozen in Block 09C"
                ),

            "dispersed":
                (
                    "spatially separated points within each selected component "
                    "using deterministic farthest-point-style allocation; "
                    "exact algorithm frozen in Block 09C"
                ),

            "fragmented":
                (
                    "multiple spatially separated pieces derived from an "
                    "interior path while preserving the same per-component "
                    "positive quota; exact algorithm frozen in Block 09C"
                ),
        },
    },

    "causal_matching": {
        "within_case_total_positive_budget":
            "EXACTLY_EQUAL_ACROSS_ALL_SIX_CELLS",

        "budget_symbol":
            "B_i",

        "budget_value_status":
            "TO_BE_FROZEN_PRETRAINING_AFTER_09B_09C_FEASIBILITY",

        "budget_selection_rule":
            (
                "largest_common_feasible_unique_positive_count across all "
                "six cells for that case, determined before any COVA training "
                "or COVA outcome inspection"
            ),

        "same_selected_component_set_within_coverage_across_geometries":
            True,

        "same_per_component_positive_quota_within_coverage_across_geometries":
            True,

        "same_background_coordinates_across_all_six_cells":
            True,

        "same_background_positive_negative_balance_policy":
            True,

        "same_image_preprocessing":
            True,

        "same_crop":
            True,

        "same_training_schedule_within_model_seed":
            True,

        "same_architecture_within_factorial_experiment":
            True,

        "same_optimizer_within_factorial_experiment":
            True,

        "same_augmentation_realization_within_paired_model_seed":
            True,
    },

    "eligibility": {
        "reference_connectivity":
            PRIMARY_REFERENCE_CONNECTIVITY,

        "reference_component_minimum_volume_ml":
            PRIMARY_REFERENCE_MIN_VOLUME_ML,

        "minimum_components_for_primary_factorial_case":
            2,

        "reason":
            (
                "a 50pct versus 100pct coverage comparison requires at "
                "least two eligible lesion components"
            ),
    },

    "primary_endpoint": {
        "name":
            "macro_patient_lesion_recall_at_1_fp_per_patient",

        "short_name":
            "R@1",

        "fp_budget_per_patient":
            PRIMARY_FROC_BUDGET,

        "eligible_reference_minimum_volume_ml":
            PRIMARY_REFERENCE_MIN_VOLUME_ML,

        "reference_connectivity":
            PRIMARY_REFERENCE_CONNECTIVITY,

        "prediction_connectivity":
            26,

        "matching_iou":
            PRIMARY_MATCH_IOU,

        "matching":
            "maximum_cardinality_then_maximum_total_iou",

        "prediction_size_filtering":
            False,

        "small_reference_policy":
            (
                "predictions matched only to references below 0.1mL "
                "are neutral rather than false positives"
            ),

        "froc_topology":
            (
                "recompute_prediction_components_at_each_voxel_probability_threshold"
            ),

        "voxel_probability_thresholds":
            DYNAMIC_FROC_THRESHOLDS,
    },

    "secondary_endpoints": [
        "R_at_0p5_FP_per_patient",
        "R_at_2_FP_per_patient",
        "R_at_4_FP_per_patient",
        "Dice",
        "IoU",
        "any_overlap_lesion_recall",
        "IoU_0p25_lesion_recall",
        "false_positive_volume_ml",
        "prediction_component_count",
        "component_to_reference_fragmentation_ratio",
        "lesion_size_stratified_recall",
    ],

    "primary_estimands": {
        "coverage_effect_within_geometry":
            (
                "R(100%,g) - R(50%,g) for each geometry g"
            ),

        "geometry_effect_disperse_vs_coherent_within_coverage":
            (
                "R(c,dispersed) - R(c,coherent)"
            ),

        "geometry_effect_fragmented_vs_coherent_within_coverage":
            (
                "R(c,fragmented) - R(c,coherent)"
            ),

        "interaction_disperse":
            (
                "[R(100%,dispersed)-R(100%,coherent)] - "
                "[R(50%,dispersed)-R(50%,coherent)]"
            ),

        "interaction_fragmented":
            (
                "[R(100%,fragmented)-R(100%,coherent)] - "
                "[R(50%,fragmented)-R(50%,coherent)]"
            ),

        "interaction_direction":
            "TWO_SIDED_NO_DIRECTIONAL_HYPOTHESIS",
    },

    "hypotheses": {
        "primary_null":
            (
                "coverage and within-instance annotation geometry do not "
                "interact with respect to lesion-recovery performance"
            ),

        "primary_alternative":
            (
                "at least one geometry contrast changes as lesion-instance "
                "coverage changes"
            ),

        "directional_claim_predeclared":
            False,

        "reason":
            (
                "the predecessor pilot motivated the question after observing "
                "an unexpected ordering; the new prospective study therefore "
                "uses a non-directional interaction hypothesis"
            ),
    },

    "development_stage": {
        "existing_permanent_development_volumes":
            4,

        "existing_final_outer_cv_volumes":
            16,

        "development_model_seeds":
            DEVELOPMENT_MODEL_SEEDS,

        "factorial_training_authorized":
            False,

        "baseline_sanity_gate_required":
            True,

        "baseline_sanity_gate_block":
            "09E",

        "factorial_fit_after_sanity_gate":
            "Block_10",

        "interaction_signal_review_margin":
            INTERACTION_SIGNAL_MARGIN,

        "interaction_case_consistency_min":
            INTERACTION_CASE_CONSISTENCY_MIN,

        "interaction_signal_definition":
            (
                "absolute interaction contrast >=0.05 in macro R@1 and "
                "same-sign case-level interaction in at least 3 of 4 "
                "development cases"
            ),

        "note":
            (
                "development signal is a method-development trigger only; "
                "it is not confirmatory evidence"
            ),
    },

    "final_stage": {
        "model_seeds":
            FINAL_MODEL_SEEDS,

        "current_primary_final_outer_cv_status":
            "SEALED",

        "additional_dataset_required":
            True,

        "external_or_second_dataset_required_for_general_claim":
            True,

        "cluster_unit":
            "source_subject_key",

        "bootstrap_replicates":
            BOOTSTRAP_REPLICATES,

        "bootstrap_seed":
            BOOTSTRAP_SEED,

        "interaction_reporting":
            (
                "paired effect estimates with 95% cluster-bootstrap "
                "confidence intervals"
            ),

        "multiple_interaction_contrasts":
            2,

        "multiple_comparison_policy":
            "Holm_correction_for_two_primary_interaction_contrasts",
    },

    "baseline_policy": {
        "current_cora_custom_unet_can_be_primary_baseline":
            False,

        "strong_3d_baseline_required":
            True,

        "target_family":
            "3D_nnU_Net_family_sparse_supervision_baseline",

        "exact_implementation_status":
            "TO_BE_FROZEN_IN_BLOCK_09E_BEFORE_FACTORIAL_OUTCOMES",

        "dense_masks_allowed_for_training":
            False,

        "sparse_partial_loss_required":
            True,

        "baseline_must_pass_fragmentation_and_recovery_sanity_gate":
            True,
    },

    "firewall": {
        "dense_masks_allowed_in_annotation_generator":
            True,

        "dense_masks_allowed_in_training_cache":
            False,

        "dense_masks_allowed_in_optimizer_batch":
            False,

        "final_outer_cv_dense_outcomes_before_final_protocol":
            False,

        "existing_dense_checksum_denylist_reused":
            True,
    },

    "seeds": {
        "component_selection_master_seed":
            COMPONENT_SELECTION_MASTER_SEED,

        "development_model_seeds":
            DEVELOPMENT_MODEL_SEEDS,

        "final_model_seeds":
            FINAL_MODEL_SEEDS,

        "bootstrap_seed":
            BOOTSTRAP_SEED,
    },

    "stop_go": {
        "before_factorial_training":
            [
                "09B_dataset_suitability_PASS",
                "09C_annotation_simulator_PASS",
                "09D_exact_causal_matching_QA_PASS",
                "09E_strong_baseline_sanity_PASS",
            ],

        "if_no_meaningful_factorial_effect":
            (
                "stop method development; report characterization result "
                "or retire track"
            ),

        "if_main_effect_without_interaction":
            (
                "method development may target the supported main effect, "
                "but interaction-specific method claims are prohibited"
            ),

        "if_reproducible_interaction":
            (
                "new adaptive method may be designed only after interaction "
                "result is frozen and reviewed"
            ),
    },

    "prohibited_actions_before_authorization": [
        "COVA3D_factorial_training",
        "COVA3D_method_training",
        "final_outer_cv_outcome_access",
        "checkpoint_selection_using_dense_final_labels",
        "changing_factor_levels_after_COVA3D_outcomes",
        "changing_B_i_based_on_model_performance",
    ],
}


config_path = (
    REPO
    / "configs/"
    "cova3d_protocol_v1_0.yaml"
)


config_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)


config_path.write_text(
    yaml.safe_dump(
        protocol_config,
        sort_keys=False,
        allow_unicode=True,
    ),
    encoding="utf-8",
)


print(
    "✓ Research question                   : FROZEN"
)

print(
    "✓ Factorial design                    : 2 x 3"
)

print(
    "✓ Primary endpoint                    : macro patient R@1"
)

print(
    "✓ FROC topology                       : DYNAMIC PER VOXEL THRESHOLD"
)

print(
    "✓ Directional interaction hypothesis  : NO"
)

print(
    "✓ Training authorization              : NO"
)


# ==========================================================================================
# 6. WRITE PROSPECTIVE PROTOCOL DOCUMENT
# ==========================================================================================

heading(
    "STEP 4/7 — WRITE PROSPECTIVE SCIENTIFIC SPECIFICATION"
)


protocol_markdown = f"""
# {WORKING_TITLE}

**Track:** {TRACK_ID}  
**Protocol version:** {PROTOCOL_VERSION}  
**Frozen:** {NOW_ISO}  
**Predecessor:** CORA-Lung  
**Predecessor outcome:** Gate B = NO-GO; Gate C was never run.

---

## 1. Scientific lineage

COVA-3D is a new prospective research track motivated by, but analytically
separate from, the completed CORA-Lung pilot.

The predecessor experiment tested whether omission of whole lesion components
caused a larger lesion-recovery deficit than matched random-pixel thinning.
That directional premise was not supported. The locked Gate-B result was
NO-GO, and a subsequent dynamic-topology diagnostic also failed to reveal the
hypothesized deficit.

Those results are preserved as historical pilot evidence. They are not reused
as confirmatory evidence for COVA-3D.

No CORA checkpoint is automatically eligible as a COVA-3D model.

---

## 2. Research question

> **{RESEARCH_QUESTION}**

Only one primary research question is registered.

---

## 3. Scientific factors

### Factor A — lesion-instance coverage

Two levels are registered:

1. **50% coverage:** annotate `ceil(0.5 K_i)` eligible lesion components,
   with a minimum of one.
2. **100% coverage:** annotate all `K_i` eligible lesion components.

The 50% component set must be a deterministic nested subset of the 100% set.

Selection is uniform without replacement from a seeded permutation and must
not use lesion difficulty, model prediction, model confidence, or future
outcomes.

### Factor B — within-instance annotation geometry

Three levels are registered:

1. **Coherent:** connected interior stroke/path.
2. **Dispersed:** deterministic spatially separated points.
3. **Fragmented:** multiple separated pieces derived from a coherent interior
   path.

The exact coordinate algorithms will be implemented and frozen in Block 09C
before training.

---

## 4. Primary factorial cells

| Condition | Coverage | Geometry |
|---|---:|---|
| C50_COH | 50% | Coherent |
| C50_DIS | 50% | Dispersed |
| C50_FRG | 50% | Fragmented |
| C100_COH | 100% | Coherent |
| C100_DIS | 100% | Dispersed |
| C100_FRG | 100% | Fragmented |

These six cells are the primary experiment.

---

## 5. Equal-budget causal matching

For patient/case `i`, define `B_i` as the number of unique foreground
annotation voxels.

The primary requirement is:

`N_FG(i, condition) = B_i`

for all six conditions.

`B_i` will be selected before any COVA-3D training as the largest common
feasible unique-positive budget that can be instantiated under every primary
cell for that case.

If no common feasible budget exists, the case is not repaired using outcomes.
Its eligibility is handled prospectively in the feasibility audit.

Within a fixed coverage level:

- selected lesion components are identical across geometries;
- per-component positive-voxel quotas are identical across geometries;
- only annotation spatial organization changes.

Across all six cells:

- background supervision coordinates are identical;
- CT image and preprocessing are identical;
- crop is identical;
- model architecture is identical;
- optimizer is identical;
- model-seed-specific patient and augmentation schedules are paired.

This design is intended to isolate coverage and geometry rather than annotation
quantity.

---

## 6. Component eligibility

Primary lesion references use:

- 26-connectivity;
- physical lesion volume >= {PRIMARY_REFERENCE_MIN_VOLUME_ML:.2f} mL.

A case requires at least two eligible lesion components to enter the primary
coverage factorial experiment.

---

## 7. Primary endpoint

The primary endpoint is:

**macro patient lesion recall at <= {PRIMARY_FROC_BUDGET:.0f} false-positive
component per patient (R@1).**

Prediction components are recomputed at each frozen voxel-probability
threshold. This is deliberately different from the historical CORA Gate-B
fixed-p=0.5 topology.

Primary matching uses:

- 26-connected prediction components;
- no prediction-size removal;
- one-to-one maximum-cardinality matching;
- maximum-total-IoU tie break;
- admissible match IoU >= {PRIMARY_MATCH_IOU:.2f};
- references < {PRIMARY_REFERENCE_MIN_VOLUME_ML:.2f} mL excluded from the
  primary reference set;
- predictions matching only such small excluded references are neutral rather
  than counted as false positives.

Secondary FROC budgets are 0.5, 2 and 4 FP/patient.

---

## 8. Primary estimands

Let `R(c,g)` denote macro R@1 for coverage `c` and geometry `g`.

### Coverage effects

`R(100%, g) - R(50%, g)` for every geometry.

### Geometry effects

At each coverage level:

`R(c, dispersed) - R(c, coherent)`

and

`R(c, fragmented) - R(c, coherent)`.

### Interaction contrasts

Dispersed interaction:

`[R(100%,dispersed)-R(100%,coherent)] -
 [R(50%,dispersed)-R(50%,coherent)]`

Fragmented interaction:

`[R(100%,fragmented)-R(100%,coherent)] -
 [R(50%,fragmented)-R(50%,coherent)]`

No interaction direction is hypothesized prospectively.

---

## 9. Primary hypothesis

### Null

Coverage and within-instance annotation geometry do not interact with respect
to lesion-recovery performance.

### Alternative

At least one geometry contrast changes as lesion-instance coverage changes.

The hypothesis is two-sided.

The previous CORA pilot observed an unexpected ordering and therefore cannot be
used to justify a directional COVA-3D hypothesis.

---

## 10. Development stage

The existing four permanent development volumes remain development-only.

Initially:

- one model seed: 17;
- no final outer-CV outcome access;
- no method invention;
- no architecture comparison inside the factorial experiment.

A strong 3-D sparse-supervision baseline must first pass a separate sanity
gate.

The current CORA custom U-Net is not automatically accepted as the primary
COVA-3D baseline.

---

## 11. Baseline requirement

A strong 3-D nnU-Net-family sparse-supervision baseline will be frozen before
factorial fitting.

The baseline must demonstrate non-pathological dense segmentation on the
development cohort before the six-condition causal experiment is interpreted.

The exact baseline implementation and numerical sanity criteria are frozen in
Block 09E before factorial outcomes are generated.

---

## 12. Development signal rule

For method-development purposes only, a prospective interaction signal is:

- absolute interaction contrast >= {INTERACTION_SIGNAL_MARGIN:.2f} in macro
  R@1; and
- same-sign case-level interaction in at least
  {INTERACTION_CASE_CONSISTENCY_MIN}/4 development cases.

This is not final confirmatory evidence.

If no interaction exists but a reproducible main effect exists, a later method
may target that main effect but may not claim an interaction mechanism.

If neither meaningful interaction nor meaningful main effects emerge, method
development stops.

---

## 13. Final-stage requirements

The final stage will require:

- model seeds 17, 29 and 43;
- the currently sealed outer-CV cohort;
- source-subject-level independence;
- cluster bootstrap confidence intervals;
- {BOOTSTRAP_REPLICATES:,} bootstrap replicates;
- bootstrap seed {BOOTSTRAP_SEED};
- Holm correction for the two primary interaction contrasts;
- at least one additional 3-D multifocal lesion dataset before a broad
  cross-domain claim.

The current 20-volume lung cohort alone is not sufficient for a broad general
claim about sparse annotation design.

---

## 14. Annotation simulator firewall

Dense masks may be used offline to simulate annotations.

Dense masks must never enter:

- training cache;
- optimizer-bound batches;
- model inputs;
- model-selection logic on the final set.

The existing dense-checksum denylist/firewall remains applicable.

---

## 15. Frozen execution sequence

1. **09A — prospective protocol freeze** — this block.
2. **09B — dataset suitability and factorial feasibility audit.**
3. **09C — annotation simulator v2 implementation.**
4. **09D — exact causal-matching QA.**
5. **09E — strong sparse baseline + baseline sanity gate.**
6. **Block 10 — six-condition development factorial experiment.**
7. Review main effects and interactions.
8. Only then decide whether any adaptive method should be invented.
9. Final outer-CV and second-dataset confirmation only after the method and
   final protocol are frozen.

---

## 16. Prohibited actions at protocol freeze

Until explicitly authorized:

- no COVA-3D model training;
- no six-cell factorial fitting;
- no COVA-3D method development;
- no final outer-CV dense outcome access;
- no budget selection from model outcomes;
- no factor-level changes after seeing COVA-3D results;
- no reinterpretation of the historical CORA Gate-B result.

---

## 17. Status after Block 09A

**CORA-Lung:** closed as a NO-GO pilot.  
**COVA-3D protocol:** frozen prospectively.  
**COVA-3D training:** not authorized.  
**Next action:** Block 09B.
"""


protocol_doc_path = (
    REPO
    / "docs/"
    "cova3d_prospective_protocol_v1_0.md"
)


write_text(
    protocol_doc_path,
    protocol_markdown,
)


closure_text = """
# CORA-Lung Track Closure

## Final scientific status

The original CORA-Lung component-omission-replay track is closed after its
predefined problem-validation gate did not pass.

Historical state:

- Gate A: PASS
- Gate B: NO-GO
- Gate C: NOT RUN
- Replay efficacy experiment: never authorized
- Final outer-CV outcomes: remained sealed during the Gate-B failure and
  dynamic-topology audits
- Locked Gate-B result: retained without reinterpretation

The CORA commits, checkpoints, audits, figures and NO-GO result remain part of
the permanent repository history.

## Relationship to COVA-3D

COVA-3D is not a renamed positive continuation of CORA.

It is a new prospective research question motivated by the unexpected pilot
observation that annotation organization may interact with lesion-instance
coverage.

The CORA observations may be cited as pilot motivation only. They are not
confirmatory COVA-3D evidence.

No unfavorable CORA result is deleted or overwritten.
"""


closure_path = (
    REPO
    / "docs/"
    "cora_track_closure.md"
)


write_text(
    closure_path,
    closure_text,
)


print(
    "✓ Prospective protocol document       : WRITTEN"
)

print(
    "✓ CORA closure document               : WRITTEN"
)


# ==========================================================================================
# 7. WRITE FACTORIAL REGISTER
# ==========================================================================================

heading(
    "STEP 5/7 — FREEZE FACTORIAL CONDITION REGISTER"
)


registry_path = (
    REPO
    / "data/manifests/"
    "cova3d_factorial_register_v1_0.csv"
)


registry_path.parent.mkdir(
    parents=True,
    exist_ok=True,
)


registry_df.to_csv(
    registry_path,
    index=False,
)


# Validate from disk, not only in memory.
registry_check = pd.read_csv(
    registry_path
)


if len(
    registry_check
) != 6:

    raise RuntimeError(
        "On-disk factorial registry is incomplete."
    )


if set(
    registry_check[
        "condition_id"
    ]
) != set(
    CONDITION_IDS
):

    raise RuntimeError(
        "On-disk factorial registry IDs are incorrect."
    )


if registry_check[
    "training_authorized"
].astype(
    bool
).any():

    raise RuntimeError(
        "Factorial training was accidentally authorized."
    )


print(
    "✓ Factorial register                  : PASS"
)

print(
    "✓ Six cells                           : FROZEN"
)

print(
    "✓ All cells training_authorized       : FALSE"
)


# ==========================================================================================
# 8. LIGHTWEIGHT PROTOCOL REGRESSION TEST
# ==========================================================================================

test_source = r'''
import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_cova3d_factorial_registry_is_complete():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_factorial_register_v1_0.csv"
    )

    assert len(frame) == 6

    assert set(
        frame[
            "coverage_percent"
        ].astype(int)
    ) == {
        50,
        100,
    }

    assert set(
        frame[
            "geometry"
        ]
    ) == {
        "coherent",
        "dispersed",
        "fragmented",
    }

    assert not frame[
        "training_authorized"
    ].astype(bool).any()


def test_cova3d_equal_budget_contract_is_frozen():

    config = yaml.safe_load(
        (
            ROOT
            / "configs/"
            "cova3d_protocol_v1_0.yaml"
        ).read_text(
            encoding="utf-8"
        )
    )

    matching = config[
        "causal_matching"
    ]

    assert (
        matching[
            "within_case_total_positive_budget"
        ]
        == "EXACTLY_EQUAL_ACROSS_ALL_SIX_CELLS"
    )

    assert (
        matching[
            "same_selected_component_set_within_coverage_across_geometries"
        ]
        is True
    )

    assert (
        matching[
            "same_per_component_positive_quota_within_coverage_across_geometries"
        ]
        is True
    )

    assert (
        matching[
            "same_background_coordinates_across_all_six_cells"
        ]
        is True
    )


def test_cora_no_go_is_preserved():

    state = json.loads(
        (
            ROOT
            / "PROJECT_STATE.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert state[
        "gate_b"
    ] == "NO_GO"

    assert state[
        "gate_c"
    ] == "NOT_RUN"
'''


test_path = (
    REPO
    / "tests/"
    "test_cova3d_protocol_registry.py"
)


write_text(
    test_path,
    test_source,
)


# ==========================================================================================
# 9. CREATE COVA-3D STATE + UPDATE REPOSITORY STATE
# ==========================================================================================

heading(
    "STEP 6/7 — FREEZE NEW TRACK STATE WITHOUT ALTERING CORA RESULT"
)


config_sha = sha256_file(
    config_path
)


registry_sha = sha256_file(
    registry_path
)


protocol_doc_sha = sha256_file(
    protocol_doc_path
)


closure_sha = sha256_file(
    closure_path
)


cova_state = {
    "project":
        "COVA-3D",

    "working_title":
        WORKING_TITLE,

    "track_id":
        TRACK_ID,

    "protocol_version":
        PROTOCOL_VERSION,

    "protocol_status":
        "FROZEN",

    "created_at_utc":
        NOW_ISO,

    "parent_repository":
        REPOSITORY_FULL_NAME,

    "parent_commit":
        current_commit,

    "predecessor_track":
        "CORA-Lung",

    "predecessor_status":
        "CLOSED_AFTER_GATE_B_NO_GO",

    "predecessor_gate_b":
        "NO_GO",

    "predecessor_gate_c":
        "NOT_RUN",

    "research_question":
        RESEARCH_QUESTION,

    "primary_factorial_cells":
        CONDITION_IDS,

    "factorial_training_authorized":
        False,

    "method_development_authorized":
        False,

    "final_outer_cv_outcomes_authorized":
        False,

    "current_stage":
        "PROTOCOL_FROZEN",

    "last_completed_block":
        BLOCK,

    "next_block":
        "09B",

    "next_action":
        (
            "Dataset suitability and factorial feasibility audit; "
            "no training."
        ),

    "protocol_config_sha256":
        config_sha,

    "factorial_registry_sha256":
        registry_sha,

    "prospective_protocol_sha256":
        protocol_doc_sha,

    "cora_closure_sha256":
        closure_sha,

    "optimizer_steps_in_cova3d":
        0,

    "dense_outcomes_opened_in_cova3d":
        False,

    "final_outer_cv_access_in_cova3d":
        0,
}


cova_state_path = (
    REPO
    / "COVA3D_STATE.json"
)


write_json(
    cova_state_path,
    cova_state,
)


# Preserve ALL old CORA fields and add new-track metadata.
project_state.update(
    {
        "last_attempted_block":
            BLOCK,

        "last_completed_block":
            BLOCK,

        "last_completed_block_name":
            "cova3d_prospective_protocol_freeze",

        "active_research_track":
            TRACK_ID,

        "active_working_title":
            WORKING_TITLE,

        "current_stage":
            "cova3d_protocol_frozen",

        "current_gate":
            "COVA_PROTOCOL",

        "cora_track_status":
            "CLOSED_AFTER_GATE_B_NO_GO",

        # Historical CORA facts are deliberately unchanged.
        "gate_b":
            "NO_GO",

        "gate_c":
            "NOT_RUN",

        "cova3d_protocol_version":
            PROTOCOL_VERSION,

        "cova3d_protocol_status":
            "FROZEN",

        "cova3d_factorial_cells":
            6,

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

        "next_action":
            (
                "Run Block 09B dataset suitability and factorial "
                "feasibility audit. Do not train."
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
# 10. RUN REGRESSION TEST
# ==========================================================================================

pytest_env = os.environ.copy()


src_path = (
    REPO
    / "src"
)


pytest_env[
    "PYTHONPATH"
] = (
    str(
        src_path
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
        "COVA-3D protocol regression test failed."
    )


print(
    "✓ Protocol regression tests           : PASS"
)


# ==========================================================================================
# 11. CAPTURE EXACT BLOCK SOURCE
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
        "COVA-3D — CODE BLOCK 09A"
        in cell
    ):

        source_path = (
            REPO
            / "scripts/code_blocks/"
            "block09a_cova3d_protocol_freeze.py"
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
# 12. WRITE BLOCK AUDIT
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
        current_commit,

    "purpose":
        "prospective protocol freeze and predecessor-track closure",

    "working_title":
        WORKING_TITLE,

    "protocol_version":
        PROTOCOL_VERSION,

    "research_question":
        RESEARCH_QUESTION,

    "factorial_design":
        {
            "coverage_levels":
                COVERAGE_LEVELS,

            "geometry_levels":
                GEOMETRY_LEVELS,

            "cells":
                CONDITION_IDS,

            "cell_count":
                6,
        },

    "causal_matching":
        {
            "same_positive_budget_all_cells":
                True,

            "same_background_all_cells":
                True,

            "same_selected_components_across_geometry_within_coverage":
                True,

            "same_per_component_quota_across_geometry_within_coverage":
                True,

            "nested_50pct_in_100pct":
                True,
        },

    "primary_endpoint":
        "macro_patient_lesion_recall_at_1_fp_per_patient",

    "primary_interaction_directional":
        False,

    "dynamic_topology_froc":
        True,

    "new_training_performed":
        False,

    "optimizer_steps":
        0,

    "ct_arrays_accessed":
        0,

    "dense_lesion_arrays_accessed":
        0,

    "dense_lung_arrays_accessed":
        0,

    "development_outcomes_accessed":
        0,

    "final_outer_cv_outcomes_accessed":
        0,

    "factorial_training_authorized":
        False,

    "method_development_authorized":
        False,

    "cora_lineage": {
        "gate_a":
            "PASS",

        "gate_b":
            "NO_GO",

        "gate_c":
            "NOT_RUN",

        "status":
            "CLOSED_AFTER_GATE_B_NO_GO",

        "historical_optimizer_steps":
            old_optimizer_steps,

        "result_modified":
            False,
    },

    "artifacts": {
        "config":
            str(
                config_path.relative_to(
                    REPO
                )
            ),

        "config_sha256":
            config_sha,

        "registry":
            str(
                registry_path.relative_to(
                    REPO
                )
            ),

        "registry_sha256":
            registry_sha,

        "prospective_protocol":
            str(
                protocol_doc_path.relative_to(
                    REPO
                )
            ),

        "prospective_protocol_sha256":
            protocol_doc_sha,

        "closure_document":
            str(
                closure_path.relative_to(
                    REPO
                )
            ),

        "closure_document_sha256":
            closure_sha,
    },

    "source_capture":
        source_capture,

    "next_block":
        "09B",

    "next_action":
        (
            "Audit dataset suitability and exact factorial-budget "
            "feasibility. Do not train."
        ),
}


audit_path = (
    REPO
    / "experiments/audits/"
    "block09a_cova3d_protocol_freeze.json"
)


write_json(
    audit_path,
    audit_payload,
)


# ==========================================================================================
# 13. REFRESH REPOSITORY MANIFEST
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
# 14. COMMIT + PUSH
# ==========================================================================================

heading(
    "STEP 7/7 — COMMIT PROSPECTIVE COVA-3D LINEAGE"
)


git_paths = [
    "PROJECT_STATE.json",
    "COVA3D_STATE.json",
    "REPOSITORY_MANIFEST.json",
    "configs/cova3d_protocol_v1_0.yaml",
    "data/manifests/cova3d_factorial_register_v1_0.csv",
    "docs/cova3d_prospective_protocol_v1_0.md",
    "docs/cora_track_closure.md",
    "experiments/audits/block09a_cova3d_protocol_freeze.json",
    "tests/test_cova3d_protocol_registry.py",
]


source_path = (
    REPO
    / "scripts/code_blocks/"
    "block09a_cova3d_protocol_freeze.py"
)


if source_path.exists():

    git_paths.append(
        "scripts/code_blocks/"
        "block09a_cova3d_protocol_freeze.py"
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
        "No Block-09A artifacts available to commit."
    )


print(
    status
)


sh(
    [
        "git",
        "commit",
        "-m",
        "protocol: freeze prospective COVA-3D factorial study",
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


# Cleanup credential helper.
try:

    askpass.unlink(
        missing_ok=True
    )

except Exception:

    pass


# ==========================================================================================
# 15. FINAL REPORT
# ==========================================================================================

print(
    "\n"
    + "=" * 124
)

print(
    "COVA-3D CODE BLOCK 09A — FINAL PROSPECTIVE PROTOCOL REPORT"
)

print(
    "=" * 124
)


print(
    "Predecessor track                    : CORA-Lung"
)

print(
    "CORA Gate A                          : PASS"
)

print(
    "CORA Gate B                          : NO_GO"
)

print(
    "CORA Gate C                          : NOT RUN"
)

print(
    "CORA lineage                         : CLOSED / PRESERVED"
)

print()

print(
    "Active track                         : COVA-3D"
)

print(
    "Protocol version                     : 1.0"
)

print(
    "Protocol status                      : FROZEN"
)

print(
    "Working title                        :"
)

print(
    "  "
    + WORKING_TITLE
)

print()

print(
    "Primary research question            :"
)

print(
    "  "
    + RESEARCH_QUESTION
)

print()

print(
    "Coverage levels                      : 50%, 100%"
)

print(
    "Geometry levels                      : coherent / dispersed / fragmented"
)

print(
    "Primary factorial cells              : 6"
)

for condition_id in CONDITION_IDS:

    print(
        "  •",
        condition_id,
    )


print()

print(
    "Same FG budget across all cells      : REQUIRED"
)

print(
    "Identical BG coordinates             : REQUIRED"
)

print(
    "Same selected components by geometry : REQUIRED"
)

print(
    "Same per-component quota by geometry : REQUIRED"
)

print(
    "50% subset nested in 100%            : REQUIRED"
)

print()

print(
    "Primary endpoint                     : macro patient R@1"
)

print(
    "Primary FROC topology                : dynamic per voxel threshold"
)

print(
    "Primary hypothesis                   : coverage × geometry interaction"
)

print(
    "Directional hypothesis               : NO"
)

print(
    "Interaction signal margin (dev only) : |ΔΔR@1| >= 0.05"
)

print(
    "Case-consistency requirement         : >= 3/4 development cases"
)

print()

print(
    "Development model seeds              :",
    DEVELOPMENT_MODEL_SEEDS,
)

print(
    "Final model seeds                    :",
    FINAL_MODEL_SEEDS,
)

print(
    "Bootstrap replicates                 :",
    BOOTSTRAP_REPLICATES,
)

print(
    "Second 3-D lesion dataset required   : YES"
)

print(
    "Strong nnU-Net-family baseline req.  : YES"
)

print()

print(
    "New optimizer steps                  : 0"
)

print(
    "CT arrays accessed                   : 0"
)

print(
    "Dense lesion arrays accessed         : 0"
)

print(
    "Development outcomes accessed        : 0"
)

print(
    "Final outer-CV outcomes accessed     : 0"
)

print(
    "COVA factorial training authorized   : NO"
)

print(
    "COVA method development authorized   : NO"
)

print()

print(
    "Protocol regression tests            : PASS"
)

print(
    "Exact Block-09A source captured      :",
    source_capture,
)

print(
    "Starting commit                      :",
    current_commit[:12],
)

print(
    "Final protocol commit                :",
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
    "Next block will be 09B: dataset suitability + factorial feasibility audit."
)

print(
    "=" * 124
)