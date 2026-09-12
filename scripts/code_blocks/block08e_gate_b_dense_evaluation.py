# ==========================================================================================
# CORA-LUNG — CODE BLOCK 08E
# FIRST DENSE DEVELOPMENT EVALUATION
# Frozen-checkpoint Gate-B Problem-Validation Decision
#
# IMPORTANT:
#   THIS IS THE FIRST BLOCK ALLOWED TO OPEN DENSE DEVELOPMENT LESION MASKS.
#
# PRE-OUTCOME SAFEGUARDS INSIDE THIS BLOCK:
#
#   1. Verify locked Gate-B decision rule.
#   2. Verify all four final-model SHA-256 hashes.
#   3. Freeze evaluator implementation + synthetic unit tests in Git.
#   4. Freeze inference protocol in Git.
#   5. Run all 16 image-only predictions.
#   6. Hash/freeze those predictions in Git BEFORE opening dense masks.
#   7. Only then load the four permanent-development infection masks.
#   8. Apply the already-locked Gate-B rule automatically.
#
# PRIMARY GATE-B CONTRAST
#
#   control  = pixel_dropout_matched_50
#   omission = component_natural_50
#
#   Delta_B =
#       macro R@1(control)
#       -
#       macro R@1(omission)
#
# LOCKED PASS RULE
#
#   Delta_B >= 0.05
#   AND case-level Delta >= 0 in >= 3/4 cases
#   AND case-level Delta > 0 in >= 2/4 cases
#
# BORDERLINE
#
#   Delta_B > 0 but full PASS rule not met
#
# NO-GO
#
#   Delta_B <= 0
#
# EVALUATION
#
#   native grid
#   prediction base threshold = 0.5
#   26-connectivity
#   no prediction-size removal
#   candidate score = mean probability inside fixed prediction component
#   eligible reference >= 0.1 mL
#   primary match IoU >= 0.10
#   one-to-one maximum-cardinality matching
#   max-total-IoU tie break
#
# FROC OPERATING POINT
#
#   Each model uses ONE common candidate-score threshold across the four
#   development patients.
#
#   R@1 is the operating point maximizing MACRO PATIENT lesion recall under:
#
#       average false-positive components per patient <= 1.0
#
#   Tie break:
#       1. higher macro patient recall
#       2. higher pooled lesion recall
#       3. fewer FP/patient
#       4. higher candidate-score threshold
#
# SUPPORTING METRICS CANNOT OVERRIDE GATE B.
#
# NO:
#   optimizer steps
#   checkpoint selection
#   final-CV access
#   CORA replay training
# ==========================================================================================

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import subprocess
import hashlib
import importlib
import json
import os
import sys
import gc
import textwrap
import shutil
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nibabel as nib

from nibabel.processing import (
    resample_from_to,
)

from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment

from tqdm.auto import tqdm
from kaggle_secrets import UserSecretsClient

import torch


# ==========================================================================================
# 0. CONSTANTS
# ==========================================================================================

REPO = Path(
    "/kaggle/working/CORA-LUNG"
)

CACHE_ROOT = Path(
    "/kaggle/working/cora_train_cache_v1_2"
)

CACHE_MANIFEST = (
    CACHE_ROOT
    / "manifest.csv"
)

FIT_ROOT = Path(
    "/kaggle/working/cora_gate_b_fits_v1_2"
)

EVAL_ROOT = Path(
    "/kaggle/working/cora_gate_b_eval_v1_2"
)

PREDICTION_ROOT = (
    EVAL_ROOT
    / "predictions"
)

EXPECTED_START_COMMIT = (
    "2dd4b8467d12"
)

PATCH_ZYX = (
    48,
    128,
    128,
)

SLIDING_OVERLAP = 0.50

GAUSSIAN_SIGMA_SCALE = 0.125

BASE_CHANNELS = 16

EMBEDDING_DIM = 64

BASE_PROBABILITY_THRESHOLD = 0.50

REFERENCE_MIN_VOLUME_ML = 0.10

PRIMARY_MATCH_IOU = 0.10

SECONDARY_MATCH_IOU = 0.25

FROC_BUDGETS = [
    0.5,
    1.0,
    2.0,
    4.0,
]

PRIMARY_FROC_BUDGET = 1.0

PRIMARY_MARGIN = 0.05

MIN_NONNEGATIVE_CASES = 3

MIN_STRICTLY_POSITIVE_CASES = 2

CONDITIONS = [
    "complete",
    "pixel_dropout_matched_50",
    "component_natural_50",
    "component_fixed_50",
]

PRIMARY_CONTROL = (
    "pixel_dropout_matched_50"
)

PRIMARY_OMISSION = (
    "component_natural_50"
)

COMPLEMENTARY_REFERENCE = (
    "complete"
)

COMPLEMENTARY_OMISSION = (
    "component_fixed_50"
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
        + "=" * 122
    )

    print(
        text
    )

    print(
        "=" * 122
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


def git_push():

    token = UserSecretsClient().get_secret(
        "pushCora"
    )

    if not token:

        raise RuntimeError(
            "Kaggle secret 'pushCora' unavailable."
        )

    token = token.strip()

    askpass = Path(
        "/tmp/cora_git_askpass_block08e.sh"
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

    try:

        askpass.unlink(
            missing_ok=True
        )

    except Exception:

        pass

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
# 2. PRECONDITIONS
# ==========================================================================================

heading(
    "CORA-LUNG BLOCK 08E — FIRST DENSE DEVELOPMENT EVALUATION"
)


if not (
    REPO
    / ".git"
).exists():

    raise RuntimeError(
        "CORA-LUNG repository missing."
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
        "Unexpected starting commit."
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
        "Repository must be clean before Block 08E."
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
    != "08D-LOCK"
):

    raise RuntimeError(
        "Gate-B decision rule is not the last completed block."
    )


if (
    state.get(
        "gate_b_decision_rule"
    )
    != "LOCKED"
):

    raise RuntimeError(
        "Gate-B decision rule is not locked."
    )


if (
    state.get(
        "dense_development_evaluation"
    )
    != "NOT_RUN"
):

    raise RuntimeError(
        "Dense development evaluation was already run."
    )


if (
    state.get(
        "gate_b_outcomes_opened"
    )
    is not False
):

    raise RuntimeError(
        "Gate-B outcomes have already been opened."
    )


lock_path = (
    REPO
    / "experiments/audits/"
    "block08d_gate_b_decision_lock.json"
)


lock_payload = json.loads(
    lock_path.read_text(
        encoding="utf-8"
    )
)


if (
    lock_payload.get(
        "status"
    )
    != "LOCKED_BEFORE_DENSE_OUTCOMES"
):

    raise RuntimeError(
        "Gate-B lock audit is invalid."
    )


locked_pass = (
    lock_payload[
        "primary_gate_rule"
    ][
        "pass"
    ]
)


if not np.isclose(
    float(
        locked_pass[
            "macro_delta_r_at_1_min"
        ]
    ),
    PRIMARY_MARGIN,
):

    raise RuntimeError(
        "Gate-B margin differs from frozen lock."
    )


if int(
    locked_pass[
        "case_level_nonnegative_count_min"
    ]
) != MIN_NONNEGATIVE_CASES:

    raise RuntimeError(
        "Nonnegative-case rule differs from frozen lock."
    )


if int(
    locked_pass[
        "case_level_strict_positive_count_min"
    ]
) != MIN_STRICTLY_POSITIVE_CASES:

    raise RuntimeError(
        "Positive-case rule differs from frozen lock."
    )


development_cases = list(
    lock_payload[
        "development_cases"
    ]
)


if len(
    development_cases
) != 4:

    raise RuntimeError(
        "Expected four locked development cases."
    )


print(
    "✓ Starting commit                    :",
    starting_commit[:12],
)

print(
    "✓ Gate-B rule                        : LOCKED"
)

print(
    "✓ Corrected final checkpoints        : SEALED"
)

print(
    "✓ Dense outcomes opened              : NO"
)

print(
    "✓ Development cases                 : 4"
)


# ==========================================================================================
# 3. VERIFY ALL FOUR MODEL HASHES BEFORE BUILDING EVALUATOR
# ==========================================================================================

heading(
    "STEP 1/10 — VERIFY FINAL MODEL BINARIES"
)


fit_audit = json.loads(
    (
        REPO
        / "experiments/audits/"
        "block08d_corrected_gate_b_weak_only_fits.json"
    ).read_text(
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
        "Block-08D fit audit is not PASS."
    )


locked_model_hashes = dict(
    fit_audit[
        "final_model_hashes"
    ]
)


model_paths = {}


for condition in CONDITIONS:

    path = (
        FIT_ROOT
        / (
            "gateb_v1_2_"
            + condition
            + "_seed17"
        )
        / "final_model.pt"
    )


    if not path.exists():

        raise RuntimeError(
            "Missing corrected model binary: "
            + str(
                path
            )
        )


    observed_sha = sha256_file(
        path
    )


    expected_sha = locked_model_hashes[
        condition
    ]


    if observed_sha != expected_sha:

        raise RuntimeError(
            "Final model SHA mismatch for "
            + condition
        )


    model_paths[
        condition
    ] = path


    print(
        "✓ {:29s} {}".format(
            condition,
            observed_sha[
                :12
            ],
        )
    )


# ==========================================================================================
# 4. WRITE EVALUATOR IMPLEMENTATION — STILL NO DENSE MASK ACCESS
# ==========================================================================================

heading(
    "STEP 2/10 — FREEZE EVALUATOR IMPLEMENTATION BEFORE OUTCOMES"
)


gate_b_module_source = r'''
"""Frozen Gate-B component and FROC evaluator for CORA-Lung."""

from __future__ import annotations

import numpy as np

from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment


CONNECTIVITY_26 = ndi.generate_binary_structure(
    3,
    3,
)


def dice_score(
    prediction,
    reference,
):
    prediction = np.asarray(
        prediction,
        dtype=bool,
    )

    reference = np.asarray(
        reference,
        dtype=bool,
    )

    intersection = int(
        np.logical_and(
            prediction,
            reference,
        ).sum()
    )

    denominator = (
        int(
            prediction.sum()
        )
        + int(
            reference.sum()
        )
    )

    if denominator == 0:
        return 1.0

    return (
        2.0
        * intersection
        / denominator
    )


def iou_score(
    prediction,
    reference,
):
    prediction = np.asarray(
        prediction,
        dtype=bool,
    )

    reference = np.asarray(
        reference,
        dtype=bool,
    )

    intersection = int(
        np.logical_and(
            prediction,
            reference,
        ).sum()
    )

    union = int(
        np.logical_or(
            prediction,
            reference,
        ).sum()
    )

    if union == 0:
        return 1.0

    return intersection / union


def label_binary_components(
    mask,
):
    return ndi.label(
        np.asarray(
            mask,
            dtype=bool,
        ),
        structure=CONNECTIVITY_26,
    )


def reference_component_partition(
    reference,
    *,
    voxel_volume_ml,
    minimum_volume_ml,
):
    labels, count = label_binary_components(
        reference
    )

    sizes = np.bincount(
        labels.reshape(
            -1
        ),
        minlength=count + 1,
    ).astype(
        np.int64
    )

    ids = np.arange(
        1,
        count + 1,
        dtype=np.int32,
    )

    volumes = (
        sizes[
            1:
        ].astype(
            np.float64
        )
        * float(
            voxel_volume_ml
        )
    )

    eligible = ids[
        volumes
        >= float(
            minimum_volume_ml
        )
    ]

    small = ids[
        volumes
        < float(
            minimum_volume_ml
        )
    ]

    return {
        "labels":
            labels.astype(
                np.int32,
                copy=False,
            ),

        "sizes":
            sizes,

        "eligible_ids":
            eligible.astype(
                np.int32
            ),

        "small_ids":
            small.astype(
                np.int32
            ),

        "eligible_count":
            int(
                len(
                    eligible
                )
            ),

        "small_count":
            int(
                len(
                    small
                )
            ),
    }


def prediction_component_partition(
    probability,
    *,
    base_threshold,
):
    probability = np.asarray(
        probability,
        dtype=np.float32,
    )

    binary = (
        probability
        >= float(
            base_threshold
        )
    )

    labels, count = label_binary_components(
        binary
    )

    labels = labels.astype(
        np.int32,
        copy=False,
    )

    sizes = np.bincount(
        labels.reshape(
            -1
        ),
        minlength=count + 1,
    ).astype(
        np.int64
    )

    if count == 0:

        scores = np.empty(
            0,
            dtype=np.float32,
        )

    else:

        scores = np.asarray(
            ndi.mean(
                probability,
                labels=labels,
                index=np.arange(
                    1,
                    count + 1,
                ),
            ),
            dtype=np.float32,
        )

    return {
        "binary":
            binary,

        "labels":
            labels,

        "sizes":
            sizes,

        "scores":
            scores,

        "count":
            int(
                count
            ),
    }


def component_iou_matrix(
    reference_labels,
    reference_sizes,
    reference_ids,
    prediction_labels,
    prediction_sizes,
):
    reference_ids = np.asarray(
        reference_ids,
        dtype=np.int32,
    )

    n_prediction = int(
        len(
            prediction_sizes
        )
        - 1
    )

    matrix = np.zeros(
        (
            len(
                reference_ids
            ),
            n_prediction,
        ),
        dtype=np.float32,
    )

    if (
        len(
            reference_ids
        )
        == 0
        or n_prediction
        == 0
    ):

        return matrix

    row_lookup = {
        int(
            component_id
        ):
            row_index
        for row_index, component_id
        in enumerate(
            reference_ids.tolist()
        )
    }

    overlap_mask = (
        (
            reference_labels
            > 0
        )
        & (
            prediction_labels
            > 0
        )
    )

    if not bool(
        overlap_mask.any()
    ):

        return matrix

    reference_overlap = (
        reference_labels[
            overlap_mask
        ].astype(
            np.int64,
            copy=False,
        )
    )

    prediction_overlap = (
        prediction_labels[
            overlap_mask
        ].astype(
            np.int64,
            copy=False,
        )
    )

    pair_code = (
        reference_overlap
        * (
            n_prediction
            + 1
        )
        + prediction_overlap
    )

    unique_codes, intersections = np.unique(
        pair_code,
        return_counts=True,
    )

    for code, intersection in zip(
        unique_codes.tolist(),
        intersections.tolist(),
    ):

        reference_id = int(
            code
            // (
                n_prediction
                + 1
            )
        )

        prediction_id = int(
            code
            % (
                n_prediction
                + 1
            )
        )

        row = row_lookup.get(
            reference_id
        )

        if row is None:
            continue

        if prediction_id <= 0:
            continue

        union = (
            int(
                reference_sizes[
                    reference_id
                ]
            )
            + int(
                prediction_sizes[
                    prediction_id
                ]
            )
            - int(
                intersection
            )
        )

        if union <= 0:
            continue

        matrix[
            row,
            prediction_id - 1,
        ] = (
            float(
                intersection
            )
            / float(
                union
            )
        )

    return matrix


def maximum_cardinality_iou_matching(
    iou_matrix,
    *,
    minimum_iou,
):
    iou_matrix = np.asarray(
        iou_matrix,
        dtype=np.float64,
    )

    n_reference, n_prediction = (
        iou_matrix.shape
    )

    if (
        n_reference == 0
        or n_prediction == 0
    ):

        return []


    if float(
        minimum_iou
    ) <= 0.0:

        admissible = (
            iou_matrix
            > 0.0
        )

    else:

        admissible = (
            iou_matrix
            >= float(
                minimum_iou
            )
        )


    if not bool(
        admissible.any()
    ):

        return []


    # A cardinality bonus larger than the maximum possible total IoU
    # ensures lexicographic optimization:
    #
    #   1. maximum number of admissible matches
    #   2. maximum total IoU among those matchings
    cardinality_bonus = (
        min(
            n_reference,
            n_prediction,
        )
        + 1.0
    )


    reward = np.where(
        admissible,
        cardinality_bonus
        + iou_matrix,
        0.0,
    )


    rows, columns = linear_sum_assignment(
        -reward
    )


    output = []


    for row, column in zip(
        rows.tolist(),
        columns.tolist(),
    ):

        if not admissible[
            row,
            column
        ]:
            continue

        output.append(
            (
                int(
                    row
                ),
                int(
                    column
                ),
                float(
                    iou_matrix[
                        row,
                        column
                    ]
                ),
            )
        )


    return output


def evaluate_selected_candidates(
    *,
    candidate_scores,
    eligible_iou,
    small_iou,
    score_threshold,
    minimum_iou,
):
    candidate_scores = np.asarray(
        candidate_scores,
        dtype=np.float32,
    )

    selected = np.flatnonzero(
        candidate_scores
        >= float(
            score_threshold
        )
    )


    selected_eligible_iou = (
        eligible_iou[
            :,
            selected,
        ]
        if len(
            selected
        )
        else eligible_iou[
            :,
            :0,
        ]
    )


    matches = maximum_cardinality_iou_matching(
        selected_eligible_iou,
        minimum_iou=minimum_iou,
    )


    matched_local_predictions = {
        int(
            item[
                1
            ]
        )
        for item
        in matches
    }


    matched_global_predictions = {
        int(
            selected[
                local_index
            ]
        )
        for local_index
        in matched_local_predictions
    }


    unmatched_global_predictions = [
        int(
            prediction_index
        )
        for prediction_index
        in selected.tolist()
        if int(
            prediction_index
        )
        not in matched_global_predictions
    ]


    neutral_small = 0


    for prediction_index in unmatched_global_predictions:

        if small_iou.shape[
            0
        ] == 0:

            continue


        max_small_iou = float(
            np.max(
                small_iou[
                    :,
                    prediction_index
                ]
            )
        )


        if max_small_iou >= float(
            minimum_iou
        ):

            neutral_small += 1


    false_positives = (
        len(
            unmatched_global_predictions
        )
        - neutral_small
    )


    eligible_count = int(
        eligible_iou.shape[
            0
        ]
    )


    true_positives = int(
        len(
            matches
        )
    )


    recall = (
        true_positives
        / eligible_count
        if eligible_count
        else 0.0
    )


    return {
        "selected_candidates":
            int(
                len(
                    selected
                )
            ),

        "true_positives":
            true_positives,

        "false_positives":
            int(
                false_positives
            ),

        "neutral_small_reference_predictions":
            int(
                neutral_small
            ),

        "eligible_references":
            eligible_count,

        "recall":
            float(
                recall
            ),
    }


def cohort_froc_curve(
    case_data,
    *,
    minimum_iou,
    maximum_fp_per_patient,
):
    case_ids = sorted(
        case_data.keys()
    )


    all_scores = []


    for case_id in case_ids:

        scores = np.asarray(
            case_data[
                case_id
            ][
                "candidate_scores"
            ],
            dtype=np.float32,
        )

        if len(
            scores
        ):

            all_scores.append(
                scores
            )


    if all_scores:

        unique_scores = np.unique(
            np.concatenate(
                all_scores
            )
        )

        unique_scores = np.sort(
            unique_scores
        )[
            ::-1
        ]

        thresholds = np.concatenate(
            [
                np.asarray(
                    [
                        np.inf
                    ],
                    dtype=np.float64,
                ),
                unique_scores.astype(
                    np.float64
                ),
            ]
        )

    else:

        thresholds = np.asarray(
            [
                np.inf
            ],
            dtype=np.float64,
        )


    rows = []

    previous_fp_total = -1


    for threshold in thresholds.tolist():

        case_results = {}

        fp_total = 0

        tp_total = 0

        eligible_total = 0

        recalls = []


        for case_id in case_ids:

            data = case_data[
                case_id
            ]


            result = evaluate_selected_candidates(
                candidate_scores=data[
                    "candidate_scores"
                ],
                eligible_iou=data[
                    "eligible_iou"
                ],
                small_iou=data[
                    "small_iou"
                ],
                score_threshold=threshold,
                minimum_iou=minimum_iou,
            )


            case_results[
                case_id
            ] = result


            fp_total += int(
                result[
                    "false_positives"
                ]
            )

            tp_total += int(
                result[
                    "true_positives"
                ]
            )

            eligible_total += int(
                result[
                    "eligible_references"
                ]
            )

            recalls.append(
                float(
                    result[
                        "recall"
                    ]
                )
            )


        fp_per_patient = (
            fp_total
            / len(
                case_ids
            )
        )


        macro_recall = float(
            np.mean(
                recalls
            )
        )


        pooled_recall = (
            tp_total
            / eligible_total
            if eligible_total
            else 0.0
        )


        row = {
            "score_threshold":
                float(
                    threshold
                ),

            "fp_total":
                int(
                    fp_total
                ),

            "fp_per_patient":
                float(
                    fp_per_patient
                ),

            "true_positives":
                int(
                    tp_total
                ),

            "eligible_references":
                int(
                    eligible_total
                ),

            "macro_recall":
                macro_recall,

            "pooled_recall":
                float(
                    pooled_recall
                ),

            "case_results":
                case_results,
        }


        rows.append(
            row
        )


        # FP must not decrease as the candidate-score threshold falls.
        # If monotonicity holds and we have moved beyond the largest
        # prespecified FP budget, no lower threshold can become feasible again.
        if (
            previous_fp_total >= 0
            and fp_total
            < previous_fp_total
        ):

            raise RuntimeError(
                "FROC false-positive count became non-monotonic."
            )


        previous_fp_total = (
            fp_total
        )


        if fp_per_patient > float(
            maximum_fp_per_patient
        ):

            break


    return rows


def select_froc_operating_point(
    curve,
    *,
    fp_budget,
):
    feasible = [
        row
        for row in curve
        if float(
            row[
                "fp_per_patient"
            ]
        )
        <= float(
            fp_budget
        )
        + 1e-12
    ]


    if not feasible:

        raise RuntimeError(
            "No feasible FROC operating point."
        )


    def key(
        row,
    ):

        threshold = float(
            row[
                "score_threshold"
            ]
        )

        threshold_tiebreak = (
            threshold
            if np.isfinite(
                threshold
            )
            else float(
                "inf"
            )
        )

        return (
            float(
                row[
                    "macro_recall"
                ]
            ),
            float(
                row[
                    "pooled_recall"
                ]
            ),
            -float(
                row[
                    "fp_per_patient"
                ]
            ),
            threshold_tiebreak,
        )


    return max(
        feasible,
        key=key,
    )


def fixed_threshold_metrics(
    *,
    probability,
    reference,
    voxel_volume_ml,
    minimum_reference_volume_ml,
    base_probability_threshold,
    primary_match_iou,
    secondary_match_iou,
):
    reference = np.asarray(
        reference,
        dtype=bool,
    )

    probability = np.asarray(
        probability,
        dtype=np.float32,
    )


    reference_partition = reference_component_partition(
        reference,
        voxel_volume_ml=voxel_volume_ml,
        minimum_volume_ml=minimum_reference_volume_ml,
    )


    prediction_partition = prediction_component_partition(
        probability,
        base_threshold=base_probability_threshold,
    )


    eligible_iou = component_iou_matrix(
        reference_partition[
            "labels"
        ],
        reference_partition[
            "sizes"
        ],
        reference_partition[
            "eligible_ids"
        ],
        prediction_partition[
            "labels"
        ],
        prediction_partition[
            "sizes"
        ],
    )


    small_iou = component_iou_matrix(
        reference_partition[
            "labels"
        ],
        reference_partition[
            "sizes"
        ],
        reference_partition[
            "small_ids"
        ],
        prediction_partition[
            "labels"
        ],
        prediction_partition[
            "sizes"
        ],
    )


    all_selected_primary = evaluate_selected_candidates(
        candidate_scores=prediction_partition[
            "scores"
        ],
        eligible_iou=eligible_iou,
        small_iou=small_iou,
        score_threshold=-np.inf,
        minimum_iou=primary_match_iou,
    )


    all_selected_secondary = evaluate_selected_candidates(
        candidate_scores=prediction_partition[
            "scores"
        ],
        eligible_iou=eligible_iou,
        small_iou=small_iou,
        score_threshold=-np.inf,
        minimum_iou=secondary_match_iou,
    )


    # "Any overlap" means one-to-one matching with strictly positive IoU.
    any_overlap_matches = maximum_cardinality_iou_matching(
        eligible_iou,
        minimum_iou=0.0,
    )


    eligible_count = int(
        reference_partition[
            "eligible_count"
        ]
    )


    any_overlap_recall = (
        len(
            any_overlap_matches
        )
        / eligible_count
        if eligible_count
        else 0.0
    )


    binary_prediction = prediction_partition[
        "binary"
    ]


    false_positive_voxels = int(
        np.logical_and(
            binary_prediction,
            np.logical_not(
                reference
            ),
        ).sum()
    )


    return {
        "dice":
            float(
                dice_score(
                    binary_prediction,
                    reference,
                )
            ),

        "iou":
            float(
                iou_score(
                    binary_prediction,
                    reference,
                )
            ),

        "predicted_components":
            int(
                prediction_partition[
                    "count"
                ]
            ),

        "eligible_reference_components":
            eligible_count,

        "small_reference_components":
            int(
                reference_partition[
                    "small_count"
                ]
            ),

        "fp_components_at_0p5":
            int(
                all_selected_primary[
                    "false_positives"
                ]
            ),

        "fp_volume_ml_at_0p5":
            float(
                false_positive_voxels
                * float(
                    voxel_volume_ml
                )
            ),

        "primary_iou_component_recall_at_0p5":
            float(
                all_selected_primary[
                    "recall"
                ]
            ),

        "iou_0p25_component_recall":
            float(
                all_selected_secondary[
                    "recall"
                ]
            ),

        "any_overlap_component_recall":
            float(
                any_overlap_recall
            ),

        "candidate_scores":
            prediction_partition[
                "scores"
            ],

        "prediction_sizes":
            prediction_partition[
                "sizes"
            ][
                1:
            ],

        "eligible_iou":
            eligible_iou,

        "small_iou":
            small_iou,
    }
'''


inference_module_source = r'''
"""Frozen sliding-window inference used for CORA-Lung Gate B."""

from __future__ import annotations

import numpy as np
import torch


def sliding_window_starts(
    length,
    patch,
    overlap,
):
    length = int(
        length
    )

    patch = int(
        patch
    )

    if length <= patch:

        return [
            0
        ]


    stride = max(
        1,
        int(
            round(
                patch
                * (
                    1.0
                    - float(
                        overlap
                    )
                )
            )
        ),
    )


    starts = list(
        range(
            0,
            length
            - patch
            + 1,
            stride,
        )
    )


    last = (
        length
        - patch
    )


    if starts[
        -1
    ] != last:

        starts.append(
            last
        )


    return starts


def gaussian_importance_map(
    patch_shape,
    *,
    sigma_scale,
):
    vectors = []


    for size in patch_shape:

        size = int(
            size
        )

        coordinate = np.arange(
            size,
            dtype=np.float32,
        )


        center = (
            size
            - 1
        ) / 2.0


        sigma = max(
            1.0,
            float(
                sigma_scale
            )
            * size,
        )


        vector = np.exp(
            -0.5
            * (
                (
                    coordinate
                    - center
                )
                / sigma
            )
            ** 2
        )


        vectors.append(
            vector.astype(
                np.float32
            )
        )


    weight = (
        vectors[
            0
        ][
            :,
            None,
            None,
        ]
        * vectors[
            1
        ][
            None,
            :,
            None,
        ]
        * vectors[
            2
        ][
            None,
            None,
            :,
        ]
    )


    weight = np.maximum(
        weight,
        1e-3,
    )


    return weight.astype(
        np.float32
    )


@torch.no_grad()
def sliding_window_probability(
    model,
    image_zyx,
    *,
    patch_shape,
    overlap,
    sigma_scale,
    device,
    amp=True,
):
    image_zyx = np.asarray(
        image_zyx,
        dtype=np.float32,
    )

    patch_shape = tuple(
        int(
            x
        )
        for x in patch_shape
    )


    original_shape = tuple(
        int(
            x
        )
        for x in image_zyx.shape
    )


    padded_shape = tuple(
        max(
            original_shape[
                axis
            ],
            patch_shape[
                axis
            ],
        )
        for axis in range(
            3
        )
    )


    pad_width = [
        (
            0,
            padded_shape[
                axis
            ]
            - original_shape[
                axis
            ],
        )
        for axis in range(
            3
        )
    ]


    padded = np.pad(
        image_zyx,
        pad_width,
        mode="constant",
        constant_values=-1.0,
    )


    starts = [
        sliding_window_starts(
            padded_shape[
                axis
            ],
            patch_shape[
                axis
            ],
            overlap,
        )
        for axis in range(
            3
        )
    ]


    importance = gaussian_importance_map(
        patch_shape,
        sigma_scale=sigma_scale,
    )


    logit_sum = np.zeros(
        padded_shape,
        dtype=np.float32,
    )


    weight_sum = np.zeros(
        padded_shape,
        dtype=np.float32,
    )


    patch_count = (
        len(
            starts[
                0
            ]
        )
        * len(
            starts[
                1
            ]
        )
        * len(
            starts[
                2
            ]
        )
    )


    model.eval()


    for z in starts[
        0
    ]:

        for y in starts[
            1
        ]:

            for x in starts[
                2
            ]:

                patch = padded[
                    z:
                    z
                    + patch_shape[
                        0
                    ],

                    y:
                    y
                    + patch_shape[
                        1
                    ],

                    x:
                    x
                    + patch_shape[
                        2
                    ],
                ]


                tensor = torch.from_numpy(
                    patch[
                        None,
                        None,
                        ...
                    ]
                ).to(
                    device=device,
                    dtype=torch.float32,
                )


                with torch.autocast(
                    device_type=device.type,
                    dtype=(
                        torch.float16
                        if device.type
                        == "cuda"
                        else torch.bfloat16
                    ),
                    enabled=bool(
                        amp
                        and device.type
                        == "cuda"
                    ),
                ):

                    output = model(
                        tensor
                    )


                    logits = output[
                        "logits"
                    ]


                patch_logits = (
                    logits[
                        0,
                        0
                    ]
                    .float()
                    .cpu()
                    .numpy()
                )


                logit_sum[
                    z:
                    z
                    + patch_shape[
                        0
                    ],

                    y:
                    y
                    + patch_shape[
                        1
                    ],

                    x:
                    x
                    + patch_shape[
                        2
                    ],
                ] += (
                    patch_logits
                    * importance
                )


                weight_sum[
                    z:
                    z
                    + patch_shape[
                        0
                    ],

                    y:
                    y
                    + patch_shape[
                        1
                    ],

                    x:
                    x
                    + patch_shape[
                        2
                    ],
                ] += importance


                del tensor
                del output
                del logits


    if np.any(
        weight_sum
        <= 0
    ):

        raise RuntimeError(
            "Sliding-window coverage contains zero-weight voxels."
        )


    merged_logits = (
        logit_sum
        / weight_sum
    )


    merged_logits = merged_logits[
        :
        original_shape[
            0
        ],

        :
        original_shape[
            1
        ],

        :
        original_shape[
            2
        ],
    ]


    # Stable sigmoid.
    probability = np.empty_like(
        merged_logits,
        dtype=np.float32,
    )


    positive = (
        merged_logits
        >= 0
    )


    probability[
        positive
    ] = (
        1.0
        / (
            1.0
            + np.exp(
                -merged_logits[
                    positive
                ]
            )
        )
    )


    exp_x = np.exp(
        merged_logits[
            ~positive
        ]
    )


    probability[
        ~positive
    ] = (
        exp_x
        / (
            1.0
            + exp_x
        )
    )


    return (
        probability.astype(
            np.float32
        ),
        int(
            patch_count
        ),
    )
'''


write_text(
    REPO
    / "src/cora_lung/eval/gate_b.py",
    gate_b_module_source,
)


write_text(
    REPO
    / "src/cora_lung/eval/inference.py",
    inference_module_source,
)


# ==========================================================================================
# 5. SYNTHETIC TESTS — BEFORE DENSE OUTCOMES
# ==========================================================================================

test_source = r'''
import numpy as np

from cora_lung.eval.gate_b import (
    maximum_cardinality_iou_matching,
    evaluate_selected_candidates,
    select_froc_operating_point,
)

from cora_lung.eval.inference import (
    sliding_window_starts,
)


def test_matching_prioritizes_cardinality_before_iou():

    # Greedy IoU would take 0.90 and leave only one match.
    # Correct lexicographic matching takes:
    #   ref0 -> pred1 (0.11)
    #   ref1 -> pred0 (0.10)
    # giving cardinality 2.
    matrix = np.asarray(
        [
            [0.90, 0.11],
            [0.10, 0.00],
        ],
        dtype=np.float32,
    )

    matches = maximum_cardinality_iou_matching(
        matrix,
        minimum_iou=0.10,
    )

    assert len(matches) == 2


def test_small_reference_prediction_is_not_false_positive():

    result = evaluate_selected_candidates(
        candidate_scores=np.asarray(
            [0.8],
            dtype=np.float32,
        ),
        eligible_iou=np.asarray(
            [
                [0.0],
            ],
            dtype=np.float32,
        ),
        small_iou=np.asarray(
            [
                [0.30],
            ],
            dtype=np.float32,
        ),
        score_threshold=0.5,
        minimum_iou=0.10,
    )

    assert result[
        "true_positives"
    ] == 0

    assert result[
        "false_positives"
    ] == 0

    assert result[
        "neutral_small_reference_predictions"
    ] == 1


def test_operating_point_respects_fp_budget_and_tie_break():

    curve = [
        {
            "score_threshold": 0.9,
            "fp_per_patient": 0.5,
            "macro_recall": 0.4,
            "pooled_recall": 0.4,
        },
        {
            "score_threshold": 0.8,
            "fp_per_patient": 1.0,
            "macro_recall": 0.6,
            "pooled_recall": 0.6,
        },
        {
            "score_threshold": 0.7,
            "fp_per_patient": 2.0,
            "macro_recall": 0.8,
            "pooled_recall": 0.8,
        },
    ]

    result = select_froc_operating_point(
        curve,
        fp_budget=1.0,
    )

    assert result[
        "score_threshold"
    ] == 0.8


def test_sliding_window_starts_cover_endpoint():

    starts = sliding_window_starts(
        101,
        48,
        0.5,
    )

    assert starts[
        0
    ] == 0

    assert starts[
        -1
    ] == (
        101
        - 48
    )
'''


write_text(
    REPO
    / "tests/test_gate_b_dense_evaluator.py",
    test_source,
)


# ==========================================================================================
# 6. FREEZE INFERENCE / FROC EXECUTION CONTRACT
# ==========================================================================================

evaluation_config = """
gate_b_dense_evaluation:
  version: "1.0"
  population: permanent_development_only
  outcomes_opened_before_this_lock: false

checkpoint:
  policy: final_epoch_only
  conditions:
    - complete
    - pixel_dropout_matched_50
    - component_natural_50
    - component_fixed_50

inference:
  image_only: true
  patch_zyx: [48, 128, 128]
  overlap: 0.50
  blending: gaussian_logit_blending
  gaussian_sigma_scale: 0.125
  amp: true
  test_time_augmentation: false

native_grid_restoration:
  interpolation: linear_probability
  outside_image_crop_probability: 0.0

prediction_components:
  base_probability_threshold: 0.50
  connectivity: 26
  size_removal: false
  candidate_score: mean_probability_within_fixed_component

references:
  connectivity: 26
  minimum_volume_ml: 0.10

matching:
  primary_iou: 0.10
  algorithm: maximum_cardinality_then_maximum_total_iou
  small_reference_predictions_after_eligible_matching: neutral_not_fp

froc:
  operating_threshold_scope: single_common_threshold_across_four_patients_per_model
  primary_sensitivity: macro_patient_component_recall
  supporting_sensitivity: pooled_component_recall
  budgets_fp_per_patient: [0.5, 1.0, 2.0, 4.0]
  operating_point_selection:
    - maximize_macro_patient_recall
    - maximize_pooled_component_recall
    - minimize_fp_per_patient
    - maximize_candidate_score_threshold

primary_gate:
  control: pixel_dropout_matched_50
  omission: component_natural_50
  statistic: control_macro_R_at_1_minus_omission_macro_R_at_1
  pass_margin: 0.05
  minimum_nonnegative_case_differences: 3
  minimum_strictly_positive_case_differences: 2

supporting_metrics_override_primary_gate: false
"""


write_text(
    REPO
    / "configs/gate_b_dense_evaluation.yaml",
    evaluation_config,
)


# Capture exact executable block BEFORE dense outcomes.
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
        "CORA-LUNG — CODE BLOCK 08E"
        in cell
    ):

        source_path = (
            REPO
            / "scripts/code_blocks/"
            "block08e_gate_b_dense_evaluation.py"
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


# ==========================================================================================
# 7. RUN EVALUATOR TESTS BEFORE MASK ACCESS
# ==========================================================================================

SRC = (
    REPO
    / "src"
)


pytest_env = os.environ.copy()


pytest_env[
    "PYTHONPATH"
] = (
    str(
        SRC
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
        "tests/test_gate_b_dense_evaluator.py",
        "tests/test_resunet3d.py",
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
        "Pre-outcome dense evaluator tests FAILED."
    )


print(
    "✓ Synthetic evaluator tests           : PASS"
)


# ==========================================================================================
# 8. PRE-OUTCOME EVALUATOR COMMIT
# ==========================================================================================

state.update(
    {
        "last_attempted_block":
            "08E",

        "current_stage":
            "gate_b_evaluator_locked_predictions_pending",

        "gate_b_dense_evaluator":
            "LOCKED",

        "gate_b_inference_protocol":
            "LOCKED",

        "dense_development_evaluation":
            "NOT_RUN",

        "gate_b_outcomes_opened":
            False,

        "updated_at_utc":
            NOW_ISO,
    }
)


write_json(
    state_path,
    state,
)


sh(
    [
        "git",
        "add",
        "PROJECT_STATE.json",
        "configs/gate_b_dense_evaluation.yaml",
        "src/cora_lung/eval/gate_b.py",
        "src/cora_lung/eval/inference.py",
        "tests/test_gate_b_dense_evaluator.py",
    ]
    + (
        [
            "scripts/code_blocks/"
            "block08e_gate_b_dense_evaluation.py"
        ]
        if (
            REPO
            / "scripts/code_blocks/"
            "block08e_gate_b_dense_evaluation.py"
        ).exists()
        else []
    ),
    cwd=REPO,
)


sh(
    [
        "git",
        "commit",
        "-m",
        "eval: freeze Gate-B evaluator before dense outcomes",
    ],
    cwd=REPO,
)


git_push()


evaluator_lock_commit = sh(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    cwd=REPO,
).stdout.strip()


print(
    "✓ Evaluator lock commit               :",
    evaluator_lock_commit[:12],
)


# ==========================================================================================
# 9. FRESH IMPORTS
# ==========================================================================================

if str(
    SRC
) not in sys.path:

    sys.path.insert(
        0,
        str(
            SRC
        ),
    )


for module_name in [
    "cora_lung.models.resunet3d",
    "cora_lung.eval.gate_b",
    "cora_lung.eval.inference",
]:

    sys.modules.pop(
        module_name,
        None,
    )


importlib.invalidate_caches()


from cora_lung.models.resunet3d import (
    CORALungResidualUNet,
)

from cora_lung.eval.inference import (
    sliding_window_probability,
)

from cora_lung.eval.gate_b import (
    fixed_threshold_metrics,
    cohort_froc_curve,
    select_froc_operating_point,
)


# ==========================================================================================
# 10. LOCATE DATA — NO DENSE MASK ARRAYS YET
# ==========================================================================================

heading(
    "STEP 3/10 — LOCATE DEVELOPMENT CT GEOMETRY AND CACHE IMAGES"
)


split_df = pd.read_csv(
    REPO
    / "data/splits/"
    "primary_volume_split_registry.csv"
)


development_df = (
    split_df[
        split_df[
            "case_id"
        ].astype(
            str
        ).isin(
            development_cases
        )
    ]
    .copy()
)


development_df[
    "case_id"
] = development_df[
    "case_id"
].astype(
    str
)


development_df = (
    development_df
    .set_index(
        "case_id"
    )
    .loc[
        development_cases
    ]
    .reset_index()
)


if len(
    development_df
) != 4:

    raise RuntimeError(
        "Development split lookup failed."
    )


candidate_roots = [
    Path(
        "/kaggle/input/datasets/andrewmvd/covid19-ct-scans"
    ),
    Path(
        "/kaggle/input/covid19-ct-scans"
    ),
]


primary_root = None


probe_ct = str(
    development_df.iloc[
        0
    ][
        "ct_scan"
    ]
)


for root in candidate_roots:

    if (
        root.exists()
        and (
            root
            / probe_ct
        ).exists()
    ):

        primary_root = root

        break


if primary_root is None:

    raise RuntimeError(
        "Primary CT dataset not found."
    )


cache_manifest_df = pd.read_csv(
    CACHE_MANIFEST
)


complete_rows = cache_manifest_df[
    cache_manifest_df[
        "condition"
    ].astype(
        str
    )
    == "complete"
].copy()


image_lookup = {
    str(
        row[
            "case_id"
        ]
    ):
        str(
            row[
                "image_file"
        ]
    )
    for _, row
    in complete_rows.iterrows()
}


for case_id in development_cases:

    if case_id not in image_lookup:

        raise RuntimeError(
            "Missing v1.2 cache image for "
            + case_id
        )


print(
    "✓ Native CT geometry cases            : 4"
)

print(
    "✓ Corrected v1.2 cache images         : 4"
)

print(
    "✓ Dense lesion masks opened so far    : 0"
)


# ==========================================================================================
# 11. IMAGE-ONLY PREDICTION — STILL NO DENSE MASK ACCESS
# ==========================================================================================

heading(
    "STEP 4/10 — RUN AND FREEZE 16 IMAGE-ONLY NATIVE-GRID PREDICTIONS"
)


if EVAL_ROOT.exists():

    # Only allow a fresh evaluation root because this is the first
    # dense-outcome invocation.
    raise RuntimeError(
        "Gate-B evaluation root already exists. "
        "Do not overwrite a previous prediction freeze."
    )


PREDICTION_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


if device.type != "cuda":

    raise RuntimeError(
        "Gate-B prediction generation requires CUDA."
    )


prediction_manifest_rows = []


for condition in CONDITIONS:

    print(
        "\n"
        + "-" * 122
    )

    print(
        "PREDICTING:",
        condition,
    )

    print(
        "-" * 122
    )


    model = CORALungResidualUNet(
        in_channels=1,
        base_channels=BASE_CHANNELS,
        embedding_dim=EMBEDDING_DIM,
    ).to(
        device
    )


    payload = torch.load(
        model_paths[
            condition
        ],
        map_location=device,
        weights_only=False,
    )


    if (
        payload.get(
            "condition"
        )
        != condition
    ):

        raise RuntimeError(
            "Checkpoint condition mismatch."
        )


    if (
        payload.get(
            "cache_version"
        )
        != "1.2"
    ):

        raise RuntimeError(
            "Checkpoint cache version mismatch."
        )


    model.load_state_dict(
        payload[
            "model_state_dict"
        ]
    )


    model.eval()


    for _, split_row in tqdm(
        development_df.iterrows(),
        total=len(
            development_df
        ),
        desc=condition,
    ):

        case_id = str(
            split_row[
                "case_id"
            ]
        )


        image_path = (
            CACHE_ROOT
            / image_lookup[
                case_id
            ]
        )


        with np.load(
            image_path,
            allow_pickle=False,
        ) as cached:

            crop_image_zyx = np.asarray(
                cached[
                    "ct_zyx"
                ],
                dtype=np.float32,
            )


            crop_origin_zyx = np.asarray(
                cached[
                    "crop_origin_zyx"
                ],
                dtype=np.int32,
            )


            full_shape_zyx = np.asarray(
                cached[
                    "full_shape_zyx"
                ],
                dtype=np.int32,
            )


            resampled_affine_xyz = np.asarray(
                cached[
                    "resampled_affine_xyz"
                ],
                dtype=np.float64,
            )


        start = time.perf_counter()


        crop_probability_zyx, patch_count = (
            sliding_window_probability(
                model,
                crop_image_zyx,
                patch_shape=PATCH_ZYX,
                overlap=SLIDING_OVERLAP,
                sigma_scale=GAUSSIAN_SIGMA_SCALE,
                device=device,
                amp=True,
            )
        )


        full_probability_zyx = np.zeros(
            tuple(
                int(
                    x
                )
                for x in full_shape_zyx
            ),
            dtype=np.float32,
        )


        crop_hi_zyx = (
            crop_origin_zyx
            + np.asarray(
                crop_probability_zyx.shape,
                dtype=np.int32,
            )
        )


        full_probability_zyx[
            int(
                crop_origin_zyx[
                    0
                ]
            ):
            int(
                crop_hi_zyx[
                    0
                ]
            ),

            int(
                crop_origin_zyx[
                    1
                ]
            ):
            int(
                crop_hi_zyx[
                    1
                ]
            ),

            int(
                crop_origin_zyx[
                    2
                ]
            ):
            int(
                crop_hi_zyx[
                    2
                ]
            ),
        ] = crop_probability_zyx


        full_probability_xyz = np.transpose(
            full_probability_zyx,
            (
                2,
                1,
                0,
            ),
        )


        resampled_probability_img = nib.Nifti1Image(
            full_probability_xyz,
            resampled_affine_xyz,
        )


        # Native CT HEADER/AFFINE only.
        # Dense lesion mask still has not been opened.
        native_ct_path = (
            primary_root
            / str(
                split_row[
                    "ct_scan"
                ]
            )
        )


        native_ct_img = nib.load(
            str(
                native_ct_path
            ),
            mmap=True,
        )


        native_probability_img = resample_from_to(
            resampled_probability_img,
            (
                native_ct_img.shape,
                native_ct_img.affine,
            ),
            order=1,
            mode="constant",
            cval=0.0,
        )


        native_probability_xyz = np.asarray(
            native_probability_img.dataobj,
            dtype=np.float32,
        )


        native_probability_xyz = np.clip(
            native_probability_xyz,
            0.0,
            1.0,
        )


        condition_dir = (
            PREDICTION_ROOT
            / condition
        )


        condition_dir.mkdir(
            parents=True,
            exist_ok=True,
        )


        prediction_path = (
            condition_dir
            / (
                case_id
                + ".npz"
            )
        )


        np.savez_compressed(
            prediction_path,
            probability_xyz=native_probability_xyz,
        )


        prediction_sha = sha256_file(
            prediction_path
        )


        inference_seconds = float(
            time.perf_counter()
            - start
        )


        prediction_manifest_rows.append(
            {
                "condition":
                    condition,

                "case_id":
                    case_id,

                "model_sha256":
                    locked_model_hashes[
                        condition
                    ],

                "prediction_file":
                    str(
                        prediction_path
                    ),

                "prediction_sha256":
                    prediction_sha,

                "native_shape_x":
                    int(
                        native_probability_xyz.shape[
                            0
                        ]
                    ),

                "native_shape_y":
                    int(
                        native_probability_xyz.shape[
                            1
                        ]
                    ),

                "native_shape_z":
                    int(
                        native_probability_xyz.shape[
                            2
                        ]
                    ),

                "sliding_window_patches":
                    int(
                        patch_count
                    ),

                "inference_seconds":
                    inference_seconds,

                "probability_min":
                    float(
                        native_probability_xyz.min()
                    ),

                "probability_max":
                    float(
                        native_probability_xyz.max()
                    ),

                "dense_mask_used":
                    False,
            }
        )


        del crop_image_zyx
        del crop_probability_zyx
        del full_probability_zyx
        del full_probability_xyz
        del native_probability_xyz

        gc.collect()


    del model
    del payload

    gc.collect()

    torch.cuda.empty_cache()


prediction_manifest_df = pd.DataFrame(
    prediction_manifest_rows
)


if len(
    prediction_manifest_df
) != 16:

    raise RuntimeError(
        "Expected exactly 16 frozen predictions."
    )


if prediction_manifest_df[
    "prediction_sha256"
].nunique() != 16:

    raise RuntimeError(
        "Prediction hash collision or duplicate artifact detected."
    )


print()
print(
    "✓ Frozen native-grid predictions      : 16/16"
)

print(
    "✓ Dense masks used for prediction     : 0"
)


# ==========================================================================================
# 12. COMMIT PREDICTION HASHES BEFORE DENSE OUTCOME ACCESS
# ==========================================================================================

heading(
    "STEP 5/10 — FREEZE PREDICTION HASHES BEFORE MASK ACCESS"
)


manifest_dir = (
    REPO
    / "data/manifests"
)


manifest_dir.mkdir(
    parents=True,
    exist_ok=True,
)


prediction_manifest_repo_path = (
    manifest_dir
    / "gate_b_v1_2_prediction_freeze.csv"
)


prediction_manifest_df[
    [
        "condition",
        "case_id",
        "model_sha256",
        "prediction_sha256",
        "native_shape_x",
        "native_shape_y",
        "native_shape_z",
        "sliding_window_patches",
        "inference_seconds",
        "probability_min",
        "probability_max",
        "dense_mask_used",
    ]
].to_csv(
    prediction_manifest_repo_path,
    index=False,
)


prediction_freeze_audit = {
    "project":
        "CORA-Lung",

    "block":
        "08E-PREDICTION-FREEZE",

    "status":
        "FROZEN_BEFORE_DENSE_OUTCOMES",

    "generated_at_utc":
        datetime.now(
            timezone.utc
        ).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),

    "evaluator_lock_commit":
        evaluator_lock_commit,

    "prediction_count":
        16,

    "conditions":
        CONDITIONS,

    "development_cases":
        development_cases,

    "dense_mask_arrays_accessed":
        0,

    "prediction_hashes": {
        (
            str(
                row[
                    "condition"
                ]
            )
            + "/"
            + str(
                row[
                    "case_id"
                ]
            )
        ):
            str(
                row[
                    "prediction_sha256"
                ]
            )
        for _, row
        in prediction_manifest_df.iterrows()
    },
}


write_json(
    REPO
    / "experiments/audits/"
    "block08e_prediction_freeze.json",
    prediction_freeze_audit,
)


state = json.loads(
    state_path.read_text(
        encoding="utf-8"
    )
)


state.update(
    {
        "current_stage":
            "gate_b_predictions_frozen_dense_outcomes_ready",

        "gate_b_dense_evaluator":
            "LOCKED",

        "gate_b_predictions":
            "FROZEN",

        "gate_b_prediction_count":
            16,

        "dense_development_evaluation":
            "NOT_RUN",

        "gate_b_outcomes_opened":
            False,

        "updated_at_utc":
            datetime.now(
                timezone.utc
            ).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
    }
)


write_json(
    state_path,
    state,
)


sh(
    [
        "git",
        "add",
        "PROJECT_STATE.json",
        "data/manifests/"
        "gate_b_v1_2_prediction_freeze.csv",
        "experiments/audits/"
        "block08e_prediction_freeze.json",
    ],
    cwd=REPO,
)


sh(
    [
        "git",
        "commit",
        "-m",
        "eval: freeze Gate-B predictions before dense outcomes",
    ],
    cwd=REPO,
)


git_push()


prediction_freeze_commit = sh(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    cwd=REPO,
).stdout.strip()


print(
    "✓ Prediction-freeze commit            :",
    prediction_freeze_commit[:12],
)

print(
    "✓ Dense lesion arrays accessed        : 0"
)

print()
print(
    ">>> PRE-OUTCOME FREEZE COMPLETE. DENSE DEVELOPMENT MASK ACCESS IS NOW AUTHORIZED."
)


# ==========================================================================================
# 13. FIRST DENSE OUTCOME ACCESS
# ==========================================================================================

heading(
    "STEP 6/10 — OPEN FOUR PERMANENT-DEVELOPMENT INFECTION MASKS"
)


source_hash_df = pd.read_csv(
    REPO
    / "data/manifests/"
    "primary_file_sha256.csv"
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


dense_reference_data = {}


for _, split_row in development_df.iterrows():

    case_id = str(
        split_row[
            "case_id"
        ]
    )


    relative_mask_path = str(
        split_row[
            "infection_mask"
        ]
    )


    mask_path = (
        primary_root
        / relative_mask_path
    )


    if relative_mask_path not in source_hash_lookup:

        raise RuntimeError(
            "Frozen source SHA missing for "
            + relative_mask_path
        )


    expected_mask_sha = (
        source_hash_lookup[
            relative_mask_path
        ]
    )


    observed_mask_sha = sha256_file(
        mask_path
    )


    if observed_mask_sha.lower() != expected_mask_sha:

        raise RuntimeError(
            "Dense development mask SHA mismatch for "
            + case_id
        )


    mask_img = nib.load(
        str(
            mask_path
        ),
        mmap=True,
    )


    mask_array = np.asarray(
        mask_img.dataobj
    )


    unique_values = np.unique(
        mask_array
    )


    if len(
        unique_values
    ) > 2:

        raise RuntimeError(
            "Development infection mask is not binary: "
            + case_id
        )


    reference = (
        mask_array
        > 0.5
    )


    voxel_volume_ml = float(
        abs(
            np.linalg.det(
                np.asarray(
                    mask_img.affine,
                    dtype=np.float64,
                )[
                    :3,
                    :3
                ]
            )
        )
        / 1000.0
    )


    dense_reference_data[
        case_id
    ] = {
        "reference":
            reference,

        "voxel_volume_ml":
            voxel_volume_ml,

        "shape":
            tuple(
                int(
                    x
                )
                for x in reference.shape
            ),

        "mask_sha256":
            observed_mask_sha.lower(),

        "mask_relative_path":
            relative_mask_path,
    }


    print(
        "✓ {:28s} mask_sha={} shape={}".format(
            case_id,
            observed_mask_sha[
                :12
            ],
            tuple(
                reference.shape
            ),
        )
    )


print()
print(
    "Dense development masks opened       : 4"
)

print(
    "Final outer-CV dense masks opened     : 0"
)


# ==========================================================================================
# 14. EVALUATE FIXED-THRESHOLD METRICS AND BUILD COMPACT FROC DATA
# ==========================================================================================

heading(
    "STEP 7/10 — COMPUTE NATIVE-GRID COMPONENT METRICS"
)


case_metric_rows = []

candidate_rows = []

compact_condition_data = {
    condition:
        {}
    for condition in CONDITIONS
}


expected_component_counts = {
    str(
        key
    ):
        int(
            value
        )
    for key, value
    in lock_payload[
        "eligible_components_per_case"
    ].items()
}


for condition in CONDITIONS:

    print(
        "\nEvaluating:",
        condition,
    )


    for case_id in tqdm(
        development_cases,
        desc=condition,
        leave=False,
    ):

        prediction_path = (
            PREDICTION_ROOT
            / condition
            / (
                case_id
                + ".npz"
            )
        )


        manifest_row = prediction_manifest_df[
            (
                prediction_manifest_df[
                    "condition"
                ]
                == condition
            )
            & (
                prediction_manifest_df[
                    "case_id"
                ]
                == case_id
            )
        ]


        if len(
            manifest_row
        ) != 1:

            raise RuntimeError(
                "Prediction manifest lookup failed."
            )


        expected_prediction_sha = str(
            manifest_row.iloc[
                0
            ][
                "prediction_sha256"
            ]
        )


        if sha256_file(
            prediction_path
        ) != expected_prediction_sha:

            raise RuntimeError(
                "Frozen prediction checksum mismatch."
            )


        with np.load(
            prediction_path,
            allow_pickle=False,
        ) as prediction_npz:

            probability = np.asarray(
                prediction_npz[
                    "probability_xyz"
                ],
                dtype=np.float32,
            )


        reference_info = dense_reference_data[
            case_id
        ]


        reference = reference_info[
            "reference"
        ]


        if probability.shape != reference.shape:

            raise RuntimeError(
                "Native prediction/reference shape mismatch for "
                + case_id
            )


        result = fixed_threshold_metrics(
            probability=probability,
            reference=reference,
            voxel_volume_ml=reference_info[
                "voxel_volume_ml"
            ],
            minimum_reference_volume_ml=REFERENCE_MIN_VOLUME_ML,
            base_probability_threshold=BASE_PROBABILITY_THRESHOLD,
            primary_match_iou=PRIMARY_MATCH_IOU,
            secondary_match_iou=SECONDARY_MATCH_IOU,
        )


        if int(
            result[
                "eligible_reference_components"
            ]
        ) != expected_component_counts[
            case_id
        ]:

            raise RuntimeError(
                "Eligible reference-component count differs "
                "from frozen pre-outcome audit for "
                + case_id
                + ": observed="
                + str(
                    result[
                        "eligible_reference_components"
                    ]
                )
                + ", expected="
                + str(
                    expected_component_counts[
                        case_id
                    ]
                )
            )


        compact_condition_data[
            condition
        ][
            case_id
        ] = {
            "candidate_scores":
                np.asarray(
                    result[
                        "candidate_scores"
                    ],
                    dtype=np.float32,
                ),

            "eligible_iou":
                np.asarray(
                    result[
                        "eligible_iou"
                    ],
                    dtype=np.float32,
                ),

            "small_iou":
                np.asarray(
                    result[
                        "small_iou"
                    ],
                    dtype=np.float32,
                ),
        }


        prediction_sizes = np.asarray(
            result[
                "prediction_sizes"
            ],
            dtype=np.int64,
        )


        candidate_scores = np.asarray(
            result[
                "candidate_scores"
            ],
            dtype=np.float32,
        )


        for prediction_index in range(
            len(
                candidate_scores
            )
        ):

            candidate_rows.append(
                {
                    "condition":
                        condition,

                    "case_id":
                        case_id,

                    "candidate_id":
                        prediction_index
                        + 1,

                    "candidate_score":
                        float(
                            candidate_scores[
                                prediction_index
                            ]
                        ),

                    "candidate_voxels":
                        int(
                            prediction_sizes[
                                prediction_index
                            ]
                        ),

                    "candidate_volume_ml":
                        float(
                            prediction_sizes[
                                prediction_index
                            ]
                            * reference_info[
                                "voxel_volume_ml"
                            ]
                        ),
                }
            )


        case_metric_rows.append(
            {
                "condition":
                    condition,

                "case_id":
                    case_id,

                "dice_at_0p5":
                    float(
                        result[
                            "dice"
                        ]
                    ),

                "iou_at_0p5":
                    float(
                        result[
                            "iou"
                        ]
                    ),

                "predicted_components_at_0p5":
                    int(
                        result[
                            "predicted_components"
                        ]
                    ),

                "eligible_reference_components":
                    int(
                        result[
                            "eligible_reference_components"
                        ]
                    ),

                "small_reference_components":
                    int(
                        result[
                            "small_reference_components"
                        ]
                    ),

                "fp_components_at_0p5":
                    int(
                        result[
                            "fp_components_at_0p5"
                        ]
                    ),

                "fp_volume_ml_at_0p5":
                    float(
                        result[
                            "fp_volume_ml_at_0p5"
                        ]
                    ),

                "primary_iou_component_recall_at_0p5":
                    float(
                        result[
                            "primary_iou_component_recall_at_0p5"
                        ]
                    ),

                "iou_0p25_component_recall":
                    float(
                        result[
                            "iou_0p25_component_recall"
                        ]
                    ),

                "any_overlap_component_recall":
                    float(
                        result[
                            "any_overlap_component_recall"
                        ]
                    ),
            }
        )


        del probability

        gc.collect()


case_metrics_df = pd.DataFrame(
    case_metric_rows
)


candidate_df = pd.DataFrame(
    candidate_rows
)


if len(
    case_metrics_df
) != 16:

    raise RuntimeError(
        "Expected 16 condition/case metric rows."
    )


print(
    "✓ Native-grid case evaluations        : 16/16"
)

print(
    "✓ Eligible reference counts           : EXACTLY MATCH PRE-OUTCOME AUDIT"
)

print(
    "✓ Prediction size filtering           : NONE"
)


# ==========================================================================================
# 15. FROC CURVES / OPERATING POINTS
# ==========================================================================================

heading(
    "STEP 8/10 — COMPUTE COHORT FROC AND APPLY LOCKED R@1 DEFINITION"
)


froc_rows = []

operating_rows = []

operating_points = {}


for condition in CONDITIONS:

    curve = cohort_froc_curve(
        compact_condition_data[
            condition
        ],
        minimum_iou=PRIMARY_MATCH_IOU,
        maximum_fp_per_patient=max(
            FROC_BUDGETS
        ),
    )


    operating_points[
        condition
    ] = {}


    for curve_index, row in enumerate(
        curve
    ):

        froc_rows.append(
            {
                "condition":
                    condition,

                "curve_index":
                    curve_index,

                "score_threshold":
                    float(
                        row[
                            "score_threshold"
                        ]
                    ),

                "fp_total":
                    int(
                        row[
                            "fp_total"
                        ]
                    ),

                "fp_per_patient":
                    float(
                        row[
                            "fp_per_patient"
                        ]
                    ),

                "true_positives":
                    int(
                        row[
                            "true_positives"
                        ]
                    ),

                "eligible_references":
                    int(
                        row[
                            "eligible_references"
                        ]
                    ),

                "macro_recall":
                    float(
                        row[
                            "macro_recall"
                        ]
                    ),

                "pooled_recall":
                    float(
                        row[
                            "pooled_recall"
                        ]
                    ),
            }
        )


    for budget in FROC_BUDGETS:

        point = select_froc_operating_point(
            curve,
            fp_budget=budget,
        )


        operating_points[
            condition
        ][
            float(
                budget
            )
        ] = point


        operating_rows.append(
            {
                "condition":
                    condition,

                "fp_budget":
                    float(
                        budget
                    ),

                "selected_score_threshold":
                    float(
                        point[
                            "score_threshold"
                        ]
                    ),

                "actual_fp_per_patient":
                    float(
                        point[
                            "fp_per_patient"
                        ]
                    ),

                "macro_recall":
                    float(
                        point[
                            "macro_recall"
                        ]
                    ),

                "pooled_recall":
                    float(
                        point[
                            "pooled_recall"
                        ]
                    ),

                "true_positives":
                    int(
                        point[
                            "true_positives"
                        ]
                    ),

                "eligible_references":
                    int(
                        point[
                            "eligible_references"
                        ]
                    ),
            }
        )


froc_df = pd.DataFrame(
    froc_rows
)


operating_df = pd.DataFrame(
    operating_rows
)


primary_control_point = (
    operating_points[
        PRIMARY_CONTROL
    ][
        PRIMARY_FROC_BUDGET
    ]
)


primary_omission_point = (
    operating_points[
        PRIMARY_OMISSION
    ][
        PRIMARY_FROC_BUDGET
    ]
)


control_r1 = float(
    primary_control_point[
        "macro_recall"
    ]
)


omission_r1 = float(
    primary_omission_point[
        "macro_recall"
    ]
)


macro_delta_r1 = (
    control_r1
    - omission_r1
)


case_delta_rows = []


for case_id in development_cases:

    control_case_recall = float(
        primary_control_point[
            "case_results"
        ][
            case_id
        ][
            "recall"
        ]
    )


    omission_case_recall = float(
        primary_omission_point[
            "case_results"
        ][
            case_id
        ][
            "recall"
        ]
    )


    delta = (
        control_case_recall
        - omission_case_recall
    )


    case_delta_rows.append(
        {
            "case_id":
                case_id,

            "eligible_components":
                expected_component_counts[
                    case_id
                ],

            "pixel_matched_R_at_1":
                control_case_recall,

            "component_omission_R_at_1":
                omission_case_recall,

            "delta_R_at_1":
                delta,

            "nonnegative":
                bool(
                    delta
                    >= -1e-12
                ),

            "strictly_positive":
                bool(
                    delta
                    > 1e-12
                ),
        }
    )


case_delta_df = pd.DataFrame(
    case_delta_rows
)


nonnegative_count = int(
    case_delta_df[
        "nonnegative"
    ].sum()
)


positive_count = int(
    case_delta_df[
        "strictly_positive"
    ].sum()
)


# ==========================================================================================
# 16. AUTOMATIC LOCKED GATE-B DECISION
# ==========================================================================================

if (
    macro_delta_r1
    >= PRIMARY_MARGIN
    - 1e-12
    and nonnegative_count
    >= MIN_NONNEGATIVE_CASES
    and positive_count
    >= MIN_STRICTLY_POSITIVE_CASES
):

    gate_b_decision = "PASS"


elif macro_delta_r1 > 1e-12:

    gate_b_decision = "BORDERLINE"


else:

    gate_b_decision = "NO_GO"


print()
print(
    "Primary cohort R@1:"
)

print(
    "  pixel_dropout_matched_50 :",
    "{:.6f}".format(
        control_r1
    ),
)

print(
    "  component_natural_50     :",
    "{:.6f}".format(
        omission_r1
    ),
)

print(
    "  Delta                    :",
    "{:+.6f}".format(
        macro_delta_r1
    ),
)

print()
print(
    case_delta_df.to_string(
        index=False
    )
)

print()
print(
    "Nonnegative cases                   :",
    str(
        nonnegative_count
    )
    + "/4",
)

print(
    "Strictly positive cases             :",
    str(
        positive_count
    )
    + "/4",
)

print()
print(
    "LOCKED GATE-B DECISION              :",
    gate_b_decision,
)


# ==========================================================================================
# 17. SUPPORTING AGGREGATES
# ==========================================================================================

supporting_summary = (
    case_metrics_df.groupby(
        "condition",
        as_index=False,
    )
    .agg(
        mean_dice_at_0p5=(
            "dice_at_0p5",
            "mean",
        ),

        mean_iou_at_0p5=(
            "iou_at_0p5",
            "mean",
        ),

        mean_fp_components_at_0p5=(
            "fp_components_at_0p5",
            "mean",
        ),

        mean_fp_volume_ml_at_0p5=(
            "fp_volume_ml_at_0p5",
            "mean",
        ),

        mean_any_overlap_component_recall=(
            "any_overlap_component_recall",
            "mean",
        ),

        mean_iou_0p25_component_recall=(
            "iou_0p25_component_recall",
            "mean",
        ),

        mean_iou_0p10_component_recall_at_0p5=(
            "primary_iou_component_recall_at_0p5",
            "mean",
        ),
    )
)


r1_summary = operating_df[
    operating_df[
        "fp_budget"
    ]
    == PRIMARY_FROC_BUDGET
][
    [
        "condition",
        "selected_score_threshold",
        "actual_fp_per_patient",
        "macro_recall",
        "pooled_recall",
    ]
].rename(
    columns={
        "macro_recall":
            "macro_R_at_1",

        "pooled_recall":
            "pooled_R_at_1",
    }
)


supporting_summary = supporting_summary.merge(
    r1_summary,
    on="condition",
    how="left",
)


# ==========================================================================================
# 18. SAVE TABLES
# ==========================================================================================

heading(
    "STEP 9/10 — SAVE GATE-B TABLES AND PUBLICATION-QUALITY FIGURES"
)


case_metrics_df.to_csv(
    REPO
    / "data/manifests/"
    "gate_b_case_metrics.csv",
    index=False,
)


candidate_df.to_csv(
    REPO
    / "data/manifests/"
    "gate_b_prediction_candidates.csv",
    index=False,
)


froc_df.to_csv(
    REPO
    / "data/manifests/"
    "gate_b_froc_curve.csv",
    index=False,
)


operating_df.to_csv(
    REPO
    / "data/manifests/"
    "gate_b_froc_operating_points.csv",
    index=False,
)


case_delta_df.to_csv(
    REPO
    / "data/manifests/"
    "gate_b_primary_case_differences.csv",
    index=False,
)


supporting_summary.to_csv(
    REPO
    / "data/manifests/"
    "gate_b_condition_summary.csv",
    index=False,
)


# ==========================================================================================
# 19. FIGURE 14 — DEVELOPMENT FROC
# ==========================================================================================

figure_dir = (
    REPO
    / "figures/results"
)


figure_dir.mkdir(
    parents=True,
    exist_ok=True,
)


fig, ax = plt.subplots(
    figsize=(
        9.5,
        6.5,
    )
)


for condition in CONDITIONS:

    subset = froc_df[
        froc_df[
            "condition"
        ]
        == condition
    ].copy()


    subset = subset[
        subset[
            "fp_per_patient"
        ]
        <= max(
            FROC_BUDGETS
        )
        + 0.5
    ]


    ax.step(
        subset[
            "fp_per_patient"
        ],
        subset[
            "macro_recall"
        ],
        where="post",
        linewidth=2,
        label=condition,
    )


ax.axvline(
    1.0,
    linestyle="--",
    linewidth=1.5,
)


ax.set_xlim(
    left=0,
    right=4.25,
)


ax.set_ylim(
    0,
    1.02,
)


ax.set_xlabel(
    "False-Positive Components per Patient",
    fontweight="bold",
)


ax.set_ylabel(
    "Macro Patient Lesion Recall",
    fontweight="bold",
)


ax.set_title(
    "Gate-B Development FROC After Corrected v1.2 Training\n"
    "Whole-Component Omission vs. Matched Random-Pixel Sparsity",
    fontweight="bold",
)


ax.grid(
    alpha=0.25,
)


ax.legend(
    fontsize=8,
)


fig.tight_layout()


fig.savefig(
    figure_dir
    / "fig14_gate_b_development_froc.png",
    dpi=600,
    bbox_inches="tight",
)


fig.savefig(
    figure_dir
    / "fig14_gate_b_development_froc.pdf",
    bbox_inches="tight",
)


plt.close(
    fig
)


# ==========================================================================================
# 20. FIGURE 15 — CASE-LEVEL PRIMARY DIFFERENCE
# ==========================================================================================

fig, ax = plt.subplots(
    figsize=(
        10,
        6.2,
    )
)


x = np.arange(
    len(
        case_delta_df
    )
)


ax.plot(
    x,
    case_delta_df[
        "pixel_matched_R_at_1"
    ],
    marker="o",
    linewidth=2,
    label="Matched random-pixel sparsity",
)


ax.plot(
    x,
    case_delta_df[
        "component_omission_R_at_1"
    ],
    marker="o",
    linewidth=2,
    label="Whole-component omission",
)


for index, row in case_delta_df.iterrows():

    ax.plot(
        [
            index,
            index,
        ],
        [
            row[
                "component_omission_R_at_1"
            ],
            row[
                "pixel_matched_R_at_1"
            ],
        ],
        linestyle=":",
        linewidth=1.2,
    )


ax.set_xticks(
    x
)


ax.set_xticklabels(
    case_delta_df[
        "case_id"
    ].tolist(),
    rotation=20,
    ha="right",
    fontweight="bold",
)


ax.set_ylim(
    0,
    1.02,
)


ax.set_ylabel(
    "Lesion Recall at ≤1 FP/Patient",
    fontweight="bold",
)


ax.set_title(
    "Gate-B Case-Level Lesion Recovery at the Locked R@1 Operating Point\n"
    "Equal Positive-Voxel Budget: Pixel Thinning vs. Whole-Component Omission",
    fontweight="bold",
)


ax.grid(
    axis="y",
    alpha=0.25,
)


ax.legend()


fig.tight_layout()


fig.savefig(
    figure_dir
    / "fig15_gate_b_case_level_r_at_1.png",
    dpi=600,
    bbox_inches="tight",
)


fig.savefig(
    figure_dir
    / "fig15_gate_b_case_level_r_at_1.pdf",
    bbox_inches="tight",
)


plt.close(
    fig
)


# ==========================================================================================
# 21. FIGURE 16 — PRIMARY GATE SUMMARY
# ==========================================================================================

primary_plot = pd.DataFrame(
    {
        "condition": [
            PRIMARY_CONTROL,
            PRIMARY_OMISSION,
        ],

        "macro_R_at_1": [
            control_r1,
            omission_r1,
        ],
    }
)


fig, ax = plt.subplots(
    figsize=(
        8,
        5.8,
    )
)


bars = ax.bar(
    np.arange(
        2
    ),
    primary_plot[
        "macro_R_at_1"
    ],
)


ax.set_xticks(
    np.arange(
        2
    )
)


ax.set_xticklabels(
    [
        "Matched\nRandom-Pixel",
        "Whole-Component\nOmission",
    ],
    fontweight="bold",
)


ax.set_ylim(
    0,
    1.02,
)


ax.set_ylabel(
    "Macro R@1",
    fontweight="bold",
)


ax.set_title(
    "Gate-B Primary Problem-Validation Contrast\n"
    + "Locked Decision: "
    + gate_b_decision
    + " | ΔR@1 = "
    + "{:+.3f}".format(
        macro_delta_r1
    ),
    fontweight="bold",
)


ax.grid(
    axis="y",
    alpha=0.25,
)


for bar, value in zip(
    bars,
    primary_plot[
        "macro_R_at_1"
    ].tolist(),
):

    ax.text(
        bar.get_x()
        + bar.get_width()
        / 2,
        value
        + 0.02,
        "{:.3f}".format(
            value
        ),
        ha="center",
        va="bottom",
        fontweight="bold",
    )


fig.tight_layout()


fig.savefig(
    figure_dir
    / "fig16_gate_b_primary_contrast.png",
    dpi=600,
    bbox_inches="tight",
)


fig.savefig(
    figure_dir
    / "fig16_gate_b_primary_contrast.pdf",
    bbox_inches="tight",
)


plt.close(
    fig
)


# ==========================================================================================
# 22. FINAL SCIENTIFIC AUDIT
# ==========================================================================================

if gate_b_decision == "PASS":

    next_action = (
        "Gate B passed. Proceed to Gate C: implement and test "
        "CORA component-omission replay on the permanent development cases "
        "without touching final outer-CV outcomes."
    )


elif gate_b_decision == "BORDERLINE":

    next_action = (
        "Gate B is borderline. Do not proceed automatically to CORA. "
        "Perform protocol-level scientific review without changing the "
        "locked Gate-B result."
    )


else:

    next_action = (
        "Gate B is NO-GO. Do not proceed to CORA efficacy experiments "
        "under the current benchmark claim."
    )


gate_b_audit = {
    "project":
        "CORA-Lung",

    "block":
        "08E",

    "status":
        "PASS",

    "completed_at_utc":
        datetime.now(
            timezone.utc
        ).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),

    "first_dense_development_evaluation":
        True,

    "evaluator_lock_commit":
        evaluator_lock_commit,

    "prediction_freeze_commit":
        prediction_freeze_commit,

    "development_cases":
        development_cases,

    "final_outer_cv_cases_accessed":
        0,

    "dense_development_masks_opened":
        4,

    "eligible_components_total":
        int(
            sum(
                expected_component_counts.values()
            )
        ),

    "primary_control":
        PRIMARY_CONTROL,

    "primary_omission":
        PRIMARY_OMISSION,

    "control_macro_R_at_1":
        control_r1,

    "omission_macro_R_at_1":
        omission_r1,

    "macro_delta_R_at_1":
        macro_delta_r1,

    "locked_margin":
        PRIMARY_MARGIN,

    "nonnegative_case_count":
        nonnegative_count,

    "strictly_positive_case_count":
        positive_count,

    "required_nonnegative_cases":
        MIN_NONNEGATIVE_CASES,

    "required_positive_cases":
        MIN_STRICTLY_POSITIVE_CASES,

    "gate_b_decision":
        gate_b_decision,

    "supporting_metrics_override_gate":
        False,

    "checkpoint_selection_after_dense_outcomes":
        False,

    "optimizer_steps_in_block08e":
        0,

    "cora_replay_evaluated":
        False,

    "next_action":
        next_action,
}


write_json(
    REPO
    / "experiments/audits/"
    "block08e_gate_b_dense_evaluation.json",
    gate_b_audit,
)


# ==========================================================================================
# 23. UPDATE PROJECT STATE
# ==========================================================================================

state = json.loads(
    state_path.read_text(
        encoding="utf-8"
    )
)


state.update(
    {
        "last_attempted_block":
            "08E",

        "last_completed_block":
            "08E",

        "last_completed_block_name":
            "gate_b_dense_development_evaluation",

        "current_stage":
            (
                "gate_b_pass_ready_for_gate_c"
                if gate_b_decision
                == "PASS"
                else "gate_b_not_passed_scientific_review"
            ),

        "current_gate":
            "B",

        "gate_b":
            gate_b_decision,

        "gate_b_decision_rule":
            "LOCKED_AND_APPLIED",

        "gate_b_dense_evaluator":
            "PASS",

        "gate_b_predictions":
            "FROZEN",

        "dense_development_evaluation":
            "PASS",

        "gate_b_outcomes_opened":
            True,

        "gate_b_primary_control_R_at_1":
            control_r1,

        "gate_b_primary_omission_R_at_1":
            omission_r1,

        "gate_b_primary_delta_R_at_1":
            macro_delta_r1,

        "gate_b_nonnegative_cases":
            nonnegative_count,

        "gate_b_strict_positive_cases":
            positive_count,

        "training_authorized":
            False,

        "pilot_training_authorized":
            False,

        "next_action":
            next_action,

        "updated_at_utc":
            datetime.now(
                timezone.utc
            ).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
    }
)


write_json(
    state_path,
    state,
)


# ==========================================================================================
# 24. REPOSITORY MANIFEST
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
            datetime.now(
                timezone.utc
            ).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),

        "completed_block":
            "08E",

        "gate_b_decision":
            gate_b_decision,

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
# 25. FINAL COMMIT + PUSH
# ==========================================================================================

heading(
    "STEP 10/10 — COMMIT LOCKED GATE-B OUTCOME"
)


git_paths = [
    "PROJECT_STATE.json",
    "REPOSITORY_MANIFEST.json",
    "data/manifests/gate_b_case_metrics.csv",
    "data/manifests/gate_b_prediction_candidates.csv",
    "data/manifests/gate_b_froc_curve.csv",
    "data/manifests/gate_b_froc_operating_points.csv",
    "data/manifests/gate_b_primary_case_differences.csv",
    "data/manifests/gate_b_condition_summary.csv",
    "experiments/audits/"
    "block08e_gate_b_dense_evaluation.json",
    "figures/results/"
    "fig14_gate_b_development_froc.png",
    "figures/results/"
    "fig14_gate_b_development_froc.pdf",
    "figures/results/"
    "fig15_gate_b_case_level_r_at_1.png",
    "figures/results/"
    "fig15_gate_b_case_level_r_at_1.pdf",
    "figures/results/"
    "fig16_gate_b_primary_contrast.png",
    "figures/results/"
    "fig16_gate_b_primary_contrast.pdf",
]


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
        "No Gate-B dense evaluation artifacts to commit."
    )


print(
    status
)


sh(
    [
        "git",
        "commit",
        "-m",
        (
            "eval: apply locked Gate-B dense development decision "
            + gate_b_decision.lower()
        ),
    ],
    cwd=REPO,
)


git_push()


final_commit = sh(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    cwd=REPO,
).stdout.strip()


# ==========================================================================================
# 26. FINAL REPORT
# ==========================================================================================

print(
    "\n"
    + "=" * 122
)

print(
    "CORA-LUNG CODE BLOCK 08E — FINAL GATE-B DENSE DEVELOPMENT REPORT"
)

print(
    "=" * 122
)


print(
    "Evaluator implementation locked       : PASS"
)

print(
    "Synthetic evaluator tests             : PASS"
)

print(
    "Predictions frozen before outcomes    : 16 / 16"
)

print(
    "Dense development masks opened        : 4 / 4"
)

print(
    "Final outer-CV masks opened           : 0"
)

print(
    "Eligible lesion components            :",
    int(
        sum(
            expected_component_counts.values()
        )
    ),
)

print(
    "Primary match IoU                     :",
    PRIMARY_MATCH_IOU,
)

print(
    "Primary FROC budget                   : 1 FP / patient"
)

print(
    "Candidate threshold scope             : ONE COHORT THRESHOLD / MODEL"
)

print()
print(
    "PRIMARY GATE-B RESULT"
)

print(
    "---------------------"
)

print(
    "Matched random-pixel macro R@1        :",
    "{:.6f}".format(
        control_r1
    ),
)

print(
    "Whole-component omission macro R@1    :",
    "{:.6f}".format(
        omission_r1
    ),
)

print(
    "Delta R@1                             :",
    "{:+.6f}".format(
        macro_delta_r1
    ),
)

print(
    "Locked meaningful margin              :",
    "{:.3f}".format(
        PRIMARY_MARGIN
    ),
)

print(
    "Nonnegative case differences          :",
    str(
        nonnegative_count
    )
    + "/4",
)

print(
    "Strictly positive case differences    :",
    str(
        positive_count
    )
    + "/4",
)

print()
print(
    "CASE-LEVEL PRIMARY RESULTS"
)

print(
    "--------------------------"
)

print(
    case_delta_df[
        [
            "case_id",
            "eligible_components",
            "pixel_matched_R_at_1",
            "component_omission_R_at_1",
            "delta_R_at_1",
        ]
    ].to_string(
        index=False
    )
)

print()
print(
    "CONDITION SUMMARY"
)

print(
    "-----------------"
)

print(
    supporting_summary.to_string(
        index=False
    )
)

print()
print(
    "=============================================================="
)

print(
    "LOCKED GATE-B DECISION                 :",
    gate_b_decision,
)

print(
    "=============================================================="
)

print()

print(
    "Supporting metrics override decision  : NO"
)

print(
    "Checkpoint reselection after outcomes : NO"
)

print(
    "Optimizer steps in Block 08E          : 0"
)

print(
    "CORA replay evaluated                 : NO"
)

print(
    "Evaluator lock commit                 :",
    evaluator_lock_commit[:12],
)

print(
    "Prediction freeze commit              :",
    prediction_freeze_commit[:12],
)

print(
    "Final Gate-B result commit            :",
    final_commit[:12],
)

print(
    "GitHub synchronization                : PASS"
)

print()

print(
    "NEXT ACTION:"
)

print(
    next_action
)

print()
print(
    "Send me this COMPLETE final report and the CONDITION SUMMARY."
)

print(
    "=" * 122
)