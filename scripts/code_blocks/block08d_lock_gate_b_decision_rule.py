# ==========================================================================================
# CORA-LUNG — CODE BLOCK 08D-LOCK
# Freeze Gate-B Numerical Decision Rule BEFORE Dense Development Outcomes Are Opened
#
# PURPOSE
#   The locked pilot protocol states that the numerical Gate-B separation margin
#   must be fixed after the component/precision audit and before model outcomes.
#
# THIS BLOCK:
#   • reads ONLY already-committed component-count audit metadata
#   • reads the already-frozen Block-08D training audit
#   • DOES NOT open CTs or dense masks
#   • DOES NOT evaluate any model
#   • DOES NOT load model checkpoints
#   • DOES NOT reveal Gate-B outcomes
#
# PRIMARY GATE-B QUESTION
#
#   Does natural whole-component omission produce a reproducible lesion-recovery
#   deficit beyond equal-count random-pixel thinning?
#
# PRIMARY CONTRAST
#
#   Delta_B =
#       R@1(pixel_dropout_matched_50)
#       -
#       R@1(component_natural_50)
#
# PREDECLARED DECISION
#
#   PASS:
#       Delta_B >= 0.05
#       AND case-level Delta >= 0 in >= 3/4 development cases
#       AND case-level Delta > 0 in >= 2/4 development cases
#
#   BORDERLINE:
#       Delta_B > 0 but complete PASS rule is not satisfied
#
#   NO_GO:
#       Delta_B <= 0
#
# Supporting Dice / FP / sensitivity analyses CANNOT override the primary rule.
# ==========================================================================================

from pathlib import Path
from datetime import datetime, timezone

import subprocess
import hashlib
import json
import os
import textwrap

import numpy as np
import pandas as pd

from kaggle_secrets import UserSecretsClient


# ==========================================================================================
# 0. CONSTANTS
# ==========================================================================================

REPO = Path(
    "/kaggle/working/CORA-LUNG"
)

EXPECTED_START_COMMIT = (
    "23e2ec082dfa"
)

PRIMARY_CONTROL = (
    "pixel_dropout_matched_50"
)

PRIMARY_OMISSION = (
    "component_natural_50"
)

PRIMARY_MARGIN = 0.05

MIN_NONNEGATIVE_CASES = 3

MIN_STRICTLY_POSITIVE_CASES = 2

FROC_PRIMARY_FP_BUDGET = 1.0

COMPONENT_MATCH_IOU = 0.10

REFERENCE_MIN_VOLUME_ML = 0.10

PREDICTION_MASK_THRESHOLD = 0.50

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
        + "=" * 118
    )

    print(
        text
    )

    print(
        "=" * 118
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


# ==========================================================================================
# 2. PRECONDITIONS
# ==========================================================================================

heading(
    "CORA-LUNG BLOCK 08D-LOCK — FREEZE GATE-B DECISION RULE"
)


if not (
    REPO
    / ".git"
).exists():

    raise RuntimeError(
        "CORA-LUNG repository missing."
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
        "Unexpected starting commit."
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
        "Repository must be clean before locking Gate B."
    )


state_path = (
    REPO
    / "PROJECT_STATE.json"
)


state = json.loads(
    state_path.read_text(
        encoding="utf-8"
    )
)


if (
    state.get(
        "last_completed_block"
    )
    != "08D"
):

    raise RuntimeError(
        "Expected Block 08D to be completed."
    )


if (
    state.get(
        "gate_b_corrected_weak_only_fits"
    )
    != "PASS"
):

    raise RuntimeError(
        "Corrected Gate-B fits are not PASS."
    )


if (
    state.get(
        "dense_development_evaluation"
    )
    != "NOT_RUN"
):

    raise RuntimeError(
        "Dense development outcomes have already been evaluated."
    )


if (
    state.get(
        "gate_b_outcomes_opened"
    )
    is not False
):

    raise RuntimeError(
        "Gate-B outcomes appear to have been opened."
    )


print(
    "✓ Starting commit                 :",
    head[:12],
)

print(
    "✓ Corrected Gate-B fits           : PASS"
)

print(
    "✓ Dense development evaluation    : NOT RUN"
)

print(
    "✓ Gate-B outcomes opened          : NO"
)


# ==========================================================================================
# 3. VERIFY BLOCK-08D CHECKPOINT FREEZE
# ==========================================================================================

heading(
    "STEP 1/5 — VERIFY CORRECTED CHECKPOINT FREEZE"
)


fit_audit_path = (
    REPO
    / "experiments/audits/"
    "block08d_corrected_gate_b_weak_only_fits.json"
)


fit_audit = json.loads(
    fit_audit_path.read_text(
        encoding="utf-8"
    )
)


if (
    fit_audit.get(
        "status"
    )
    != "PASS"
):

    raise RuntimeError(
        "Block-08D audit is not PASS."
    )


if (
    fit_audit.get(
        "gate_b_outcomes_opened"
    )
    is not False
):

    raise RuntimeError(
        "Block-08D audit indicates opened outcomes."
    )


if (
    fit_audit.get(
        "dense_mask_arrays_accessed"
    )
    != 0
):

    raise RuntimeError(
        "Unexpected dense-mask access during Block 08D."
    )


model_hashes = (
    fit_audit[
        "final_model_hashes"
    ]
)


required_conditions = {
    "complete",
    "pixel_dropout_matched_50",
    "component_natural_50",
    "component_fixed_50",
}


if set(
    model_hashes.keys()
) != required_conditions:

    raise RuntimeError(
        "Unexpected corrected checkpoint set."
    )


print(
    "✓ Corrected final models          : 4/4"
)

print(
    "✓ Model hashes already frozen     : PASS"
)

print(
    "✓ Checkpoint selection            : FINAL EPOCH ONLY"
)


# ==========================================================================================
# 4. PRECISION / DISCRETE-RESOLUTION AUDIT
# ==========================================================================================

heading(
    "STEP 2/5 — COMPUTE DEVELOPMENT-COHORT RECALL RESOLUTION"
)


component_summary = pd.read_csv(
    REPO
    / "data/manifests/"
    "primary_case_component_summary.csv"
)


split_df = pd.read_csv(
    REPO
    / "data/splits/"
    "primary_volume_split_registry.csv"
)


development_cases = sorted(
    split_df.loc[
        split_df[
            "role"
        ]
        == "permanent_development",
        "case_id",
    ].astype(
        str
    ).tolist()
)


if len(
    development_cases
) != 4:

    raise RuntimeError(
        "Expected exactly four permanent development cases."
    )


development_components = (
    component_summary[
        component_summary[
            "case_id"
        ].astype(
            str
        ).isin(
            development_cases
        )
    ][
        [
            "case_id",
            "eligible_components_26conn_ge0p1ml",
        ]
    ]
    .copy()
)


development_components[
    "case_id"
] = development_components[
    "case_id"
].astype(
    str
)


development_components = (
    development_components
    .set_index(
        "case_id"
    )
    .loc[
        development_cases
    ]
    .reset_index()
)


if len(
    development_components
) != 4:

    raise RuntimeError(
        "Development component-count audit is incomplete."
    )


if (
    development_components[
        "eligible_components_26conn_ge0p1ml"
    ]
    <= 0
).any():

    raise RuntimeError(
        "Every development case must contain eligible components."
    )


development_components[
    "one_component_case_recall_step"
] = (
    1.0
    / development_components[
        "eligible_components_26conn_ge0p1ml"
    ].astype(
        float
    )
)


macro_one_component_each_case_step = float(
    development_components[
        "one_component_case_recall_step"
    ].mean()
)


margin_as_fraction_of_one_component_each_case = (
    PRIMARY_MARGIN
    / macro_one_component_each_case_step
)


total_eligible_components = int(
    development_components[
        "eligible_components_26conn_ge0p1ml"
    ].sum()
)


print(
    development_components.to_string(
        index=False
    )
)


print()

print(
    "Development eligible components    :",
    total_eligible_components,
)

print(
    "Macro step for +1 lesion/case       :",
    "{:.6f}".format(
        macro_one_component_each_case_step
    ),
)

print(
    "Locked practical margin             :",
    "{:.3f}".format(
        PRIMARY_MARGIN
    ),
)

print(
    "Margin / one-lesion-per-case step   :",
    "{:.3f}".format(
        margin_as_fraction_of_one_component_each_case
    ),
)


# ==========================================================================================
# 5. FREEZE EXACT EVALUATION / DECISION CONTRACT
# ==========================================================================================

heading(
    "STEP 3/5 — FREEZE GATE-B DECISION CONTRACT"
)


decision_contract = {
    "project":
        "CORA-Lung",

    "block":
        "08D-LOCK",

    "status":
        "LOCKED_BEFORE_DENSE_OUTCOMES",

    "locked_at_utc":
        NOW_ISO,

    "parent_commit":
        head,

    "gate":
        "B",

    "purpose":
        (
            "Problem validation: determine whether whole-component "
            "annotation omission produces a lesion-recovery deficit "
            "beyond equal-count random-pixel sparsity."
        ),

    "evaluation_population":
        "four_permanent_development_cases",

    "development_cases":
        development_cases,

    "eligible_components_per_case": {
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
        for _, row
        in development_components.iterrows()
    },

    "total_eligible_components":
        total_eligible_components,

    "precision_resolution": {
        "macro_recall_step_if_one_additional_component_recovered_per_case":
            macro_one_component_each_case_step,

        "locked_practical_margin":
            PRIMARY_MARGIN,

        "margin_fraction_of_one_component_each_case_step":
            margin_as_fraction_of_one_component_each_case,
    },

    "primary_contrast": {
        "control":
            PRIMARY_CONTROL,

        "omission":
            PRIMARY_OMISSION,

        "effect_definition":
            (
                "R@1(control) - R@1(omission)"
            ),
    },

    "component_evaluation": {
        "native_grid":
            True,

        "prediction_mask_threshold":
            PREDICTION_MASK_THRESHOLD,

        "prediction_connectivity":
            26,

        "prediction_size_removal":
            False,

        "candidate_score":
            "mean_probability_within_fixed_prediction_component",

        "eligible_reference_min_volume_ml":
            REFERENCE_MIN_VOLUME_ML,

        "primary_matching_iou":
            COMPONENT_MATCH_IOU,

        "matching":
            (
                "maximum-cardinality one-to-one, "
                "maximum-total-IoU tie break"
            ),

        "primary_froc_budget_fp_per_patient":
            FROC_PRIMARY_FP_BUDGET,

        "secondary_froc_budgets":
            [
                0.5,
                2.0,
                4.0,
            ],

        "sensitivity_matching":
            [
                "any_overlap",
                "iou_ge_0.25",
            ],
    },

    "primary_gate_rule": {
        "pass": {
            "macro_delta_r_at_1_min":
                PRIMARY_MARGIN,

            "case_level_nonnegative_count_min":
                MIN_NONNEGATIVE_CASES,

            "case_level_strict_positive_count_min":
                MIN_STRICTLY_POSITIVE_CASES,
        },

        "borderline":
            (
                "macro_delta_r_at_1 > 0 but one or more PASS "
                "requirements are not satisfied"
            ),

        "no_go":
            "macro_delta_r_at_1 <= 0",
    },

    "supporting_metrics": [
        "Dice_at_probability_0.5",
        "IoU_at_probability_0.5",
        "FP_components_at_probability_0.5",
        "FP_volume_ml_at_probability_0.5",
        "any_overlap_component_recall",
        "IoU_0.25_component_recall",
        "pooled_eligible_component_recall",
    ],

    "supporting_metrics_can_override_primary_gate":
        False,

    "complementary_contrast": {
        "reference":
            "complete",

        "omission":
            "component_fixed_50",

        "role":
            "supporting_only_not_primary_gate",
    },

    "dense_outcomes_opened_at_lock_time":
        False,

    "block08a_models_eligible":
        False,

    "replay_signal_evaluated_here":
        False,

    "note":
        (
            "Gate B validates the benchmark premise only. "
            "Replay efficacy is tested separately at Gate C."
        ),
}


audit_dir = (
    REPO
    / "experiments/audits"
)


write_json(
    audit_dir
    / "block08d_gate_b_decision_lock.json",
    decision_contract,
)


config_text = """
gate_b:
  role: problem_validation
  population: four_permanent_development_cases

primary_contrast:
  control: pixel_dropout_matched_50
  omission: component_natural_50
  statistic: control_R_at_1_minus_omission_R_at_1

evaluation:
  prediction_probability_threshold: 0.5
  prediction_connectivity: 26
  prediction_size_removal: false
  component_score: mean_probability
  reference_min_volume_ml: 0.1
  admissible_match_iou: 0.10
  matching: maximum_cardinality_then_maximum_total_iou
  primary_froc_budget_fp_per_patient: 1.0
  secondary_froc_budgets: [0.5, 2.0, 4.0]
  sensitivity_matching:
    - any_overlap
    - iou_ge_0.25

decision:
  meaningful_macro_delta_r_at_1: 0.05
  minimum_nonnegative_case_differences: 3
  minimum_strictly_positive_case_differences: 2

  PASS:
    - macro_delta_r_at_1 >= 0.05
    - nonnegative_case_differences >= 3
    - strictly_positive_case_differences >= 2

  BORDERLINE:
    - macro_delta_r_at_1 > 0
    - PASS_not_fully_satisfied

  NO_GO:
    - macro_delta_r_at_1 <= 0

supporting_metrics:
  - dice_at_0p5
  - iou_at_0p5
  - fp_components_at_0p5
  - fp_volume_ml_at_0p5
  - any_overlap_recall
  - iou_0p25_recall
  - pooled_component_recall

supporting_metrics_override_primary_gate: false

complementary_contrast:
  reference: complete
  omission: component_fixed_50
  role: supporting_only

scientific_scope:
  replay_effect_tested: false
  replay_gate: C
  independent_generalization_claim: prohibited
  block08a_checkpoints_eligible: false
"""


write_text(
    REPO
    / "configs/gate_b_decision_rule.yaml",
    config_text,
)


print(
    "✓ Primary effect                   : "
    "R@1(pixel) - R@1(component)"
)

print(
    "✓ PASS margin                      : >= 0.050"
)

print(
    "✓ Nonnegative patient differences  : >= 3/4"
)

print(
    "✓ Strictly positive differences    : >= 2/4"
)

print(
    "✓ Supporting metrics override      : NO"
)


# ==========================================================================================
# 6. CAPTURE SOURCE / UPDATE STATE
# ==========================================================================================

heading(
    "STEP 4/5 — FREEZE PRE-OUTCOME LINEAGE"
)


source_capture = "NOT_AVAILABLE"


try:

    ip = get_ipython()

    cell = (
        ip.history_manager
        .input_hist_raw[
            -1
        ]
    )


    if (
        "CORA-LUNG — CODE BLOCK 08D-LOCK"
        in cell
    ):

        source_path = (
            REPO
            / "scripts/code_blocks/"
            "block08d_lock_gate_b_decision_rule.py"
        )


        source_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )


        source_path.write_text(
            cell,
            encoding="utf-8",
        )


        source_capture = "PASS"


except Exception:

    pass


state.update(
    {
        "last_attempted_block":
            "08D-LOCK",

        "last_completed_block":
            "08D-LOCK",

        "last_completed_block_name":
            "gate_b_decision_rule_lock",

        "current_stage":
            "gate_b_dense_development_evaluation_ready",

        "current_gate":
            "B",

        "gate_b":
            "IN_PROGRESS",

        "gate_b_decision_rule":
            "LOCKED",

        "gate_b_primary_margin_r_at_1":
            PRIMARY_MARGIN,

        "gate_b_case_nonnegative_min":
            MIN_NONNEGATIVE_CASES,

        "gate_b_case_positive_min":
            MIN_STRICTLY_POSITIVE_CASES,

        "dense_development_evaluation":
            "NOT_RUN",

        "gate_b_outcomes_opened":
            False,

        "next_action":
            (
                "Run Block 08E frozen-checkpoint dense development "
                "evaluation and apply the locked Gate-B rule."
            ),

        "updated_at_utc":
            NOW_ISO,
    }
)


write_json(
    state_path,
    state,
)


# ==========================================================================================
# 7. REFRESH REPOSITORY MANIFEST
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
            "08D-LOCK",

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
# 8. COMMIT + PUSH
# ==========================================================================================

heading(
    "STEP 5/5 — COMMIT GATE-B DECISION LOCK"
)


token = UserSecretsClient().get_secret(
    "pushCora"
)


if not token:

    raise RuntimeError(
        "Kaggle secret 'pushCora' unavailable."
    )


token = token.strip()


askpass = Path(
    "/tmp/cora_git_askpass_gateb_lock.sh"
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


git_env = os.environ.copy()


git_env[
    "GITHUB_TOKEN"
] = token


git_env[
    "GIT_ASKPASS"
] = str(
    askpass
)


git_env[
    "GIT_TERMINAL_PROMPT"
] = "0"


git_paths = [
    "PROJECT_STATE.json",
    "REPOSITORY_MANIFEST.json",
    "configs/gate_b_decision_rule.yaml",
    "experiments/audits/block08d_gate_b_decision_lock.json",
]


source_path = (
    REPO
    / "scripts/code_blocks/"
    "block08d_lock_gate_b_decision_rule.py"
)


if source_path.exists():

    git_paths.append(
        "scripts/code_blocks/"
        "block08d_lock_gate_b_decision_rule.py"
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
        "No Gate-B lock changes available to commit."
    )


print(
    status
)


sh(
    [
        "git",
        "commit",
        "-m",
        "audit: lock Gate-B decision margin before dense outcomes",
    ],
    cwd=REPO,
)


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


current_commit = sh(
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
# 9. FINAL REPORT
# ==========================================================================================

print(
    "\n"
    + "=" * 118
)

print(
    "CORA-LUNG CODE BLOCK 08D-LOCK — FINAL PRE-OUTCOME REPORT"
)

print(
    "=" * 118
)


print(
    "Development cases                    : 4"
)

print(
    "Eligible components                  :",
    total_eligible_components,
)

print(
    "Components / case                    :",
    development_components[
        "eligible_components_26conn_ge0p1ml"
    ].astype(
        int
    ).tolist(),
)

print(
    "Macro +1-component-per-case step     :",
    "{:.6f}".format(
        macro_one_component_each_case_step
    ),
)

print(
    "Primary contrast                     : "
    "R@1(pixel matched) - R@1(component omission)"
)

print(
    "Locked meaningful margin             : 0.050"
)

print(
    "Required nonnegative cases           : >= 3 / 4"
)

print(
    "Required strictly positive cases     : >= 2 / 4"
)

print(
    "PASS                                : "
    "Delta >= 0.05 + consistency rule"
)

print(
    "BORDERLINE                          : "
    "Delta > 0 but PASS rule incomplete"
)

print(
    "NO-GO                               : "
    "Delta <= 0"
)

print(
    "Primary matching IoU                 : 0.10"
)

print(
    "Primary FROC budget                  : 1 FP / patient"
)

print(
    "Prediction component mask threshold  : 0.50"
)

print(
    "Supporting metrics override gate     : NO"
)

print(
    "Replay evaluated in Gate B           : NO"
)

print(
    "Replay efficacy gate                 : C"
)

print(
    "Dense lesion arrays accessed         : 0"
)

print(
    "Dense Gate-B outcomes opened         : NO"
)

print(
    "Exact source captured                :",
    source_capture,
)

print(
    "Starting commit                      :",
    head[:12],
)

print(
    "Current commit                       :",
    current_commit[:12],
)

print(
    "GitHub synchronization               : PASS"
)

print()
print(
    "NEXT: Send me this COMPLETE report. "
    "Then Block 08E may open the four development dense masks."
)

print(
    "=" * 118
)