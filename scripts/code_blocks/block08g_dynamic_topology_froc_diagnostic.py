# ==========================================================================================
# CORA-LUNG — CODE BLOCK 08G
# Calibration/Topology-Robust Dynamic-FROC Diagnostic
#
# PURPOSE
#
# Gate B is permanently NO_GO under the locked evaluator.
#
# Block 08F showed:
#   - severe fragmentation in all conditions,
#   - strong threshold sensitivity,
#   - better lesion/background separation under natural component omission,
#   - no implementation defect proven.
#
# The locked evaluator creates prediction components at p=0.5 and then
# thresholds COMPONENT SCORES. If p=0.5 causes giant merged regions and
# thousands of fragments, component-score thresholding cannot change that
# topology.
#
# This diagnostic asks:
#
#   What happens if connected components are RECOMPUTED at each voxel
#   probability threshold?
#
# This is a calibration/topology robustness analysis ONLY.
#
# IT DOES NOT:
#   - alter Gate B
#   - replace the committed Gate-B result
#   - retrain models
#   - reselect checkpoints
#   - regenerate predictions
#   - access final outer-CV cases
#   - implement CORA replay
#   - perform optimizer steps
#
# PRIMARY DIAGNOSTIC CONTRAST
#
#   dynamic_delta =
#       dynamic R@1(pixel_dropout_matched_50)
#       -
#       dynamic R@1(component_natural_50)
#
# INTERPRETATION
#
#   dynamic_delta <= 0
#       -> original premise remains unsupported even after removing
#          fixed-p=0.5 topology as the main calibration confounder.
#
#   0 < dynamic_delta < 0.05
#       -> topology/calibration contributed, but evidence for the premise
#          remains weak.
#
#   dynamic_delta >= 0.05
#       -> strong evidence that the locked fixed-component evaluator materially
#          masked the proposed omission deficit.
#
# IMPORTANT:
#   NONE of these outcomes changes Gate B = NO_GO.
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
import textwrap
import gc

import numpy as np
import pandas as pd

import nibabel as nib

import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from kaggle_secrets import UserSecretsClient


# ==========================================================================================
# 0. CONSTANTS
# ==========================================================================================

REPO = Path(
    "/kaggle/working/CORA-LUNG"
)

EVAL_ROOT = Path(
    "/kaggle/working/cora_gate_b_eval_v1_2"
)

PREDICTION_ROOT = (
    EVAL_ROOT
    / "predictions"
)

EXPECTED_START_COMMIT = (
    "5c486cb0e871"
)

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

REFERENCE_MIN_VOLUME_ML = 0.10

PRIMARY_IOU = 0.10

FP_BUDGETS = [
    0.5,
    1.0,
    2.0,
    4.0,
]

PRIMARY_FP_BUDGET = 1.0

LOCKED_MARGIN = 0.05


# Fine high-probability resolution is intentional because Block 08F showed
# that the models remain highly fragmented even around p=0.9.
VOXEL_THRESHOLDS = np.asarray(
    [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.85,
        0.90,
        0.925,
        0.950,
        0.960,
        0.970,
        0.980,
        0.985,
        0.990,
        0.9925,
        0.995,
        0.9975,
        0.999,
        0.9995,
        1.000001,  # guaranteed zero-prediction fallback
    ],
    dtype=np.float64,
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
        + "=" * 122
    )

    print(text)

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

    path = Path(path)

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

    path = Path(path)

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
        "/tmp/cora_git_askpass_block08g.sh"
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
    "CORA-LUNG BLOCK 08G — DYNAMIC-TOPOLOGY FROC DIAGNOSTIC"
)


if not (
    REPO
    / ".git"
).exists():

    raise RuntimeError(
        "Repository missing."
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
        "Repository must be clean before Block 08G."
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
    != "08F"
):

    raise RuntimeError(
        "Expected Block 08F as last completed block."
    )


if (
    state.get(
        "gate_b"
    )
    != "NO_GO"
):

    raise RuntimeError(
        "Gate B must remain NO_GO."
    )


if (
    state.get(
        "gate_c"
    )
    != "NOT_RUN"
):

    raise RuntimeError(
        "Gate C must remain NOT_RUN."
    )


if (
    state.get(
        "gate_b_failure_audit"
    )
    != "PASS"
):

    raise RuntimeError(
        "Block-08F mechanism audit is not PASS."
    )


print(
    "✓ Starting commit                    :",
    starting_commit[:12],
)

print(
    "✓ Gate B                             : NO_GO"
)

print(
    "✓ Gate C                             : NOT RUN"
)

print(
    "✓ Training in Block 08G              : NO"
)

print(
    "✓ Optimizer steps                    : 0"
)


# ==========================================================================================
# 3. IMPORT FROZEN COMPONENT EVALUATOR
# ==========================================================================================

SRC = (
    REPO
    / "src"
)


if str(SRC) not in sys.path:

    sys.path.insert(
        0,
        str(SRC),
    )


sys.modules.pop(
    "cora_lung.eval.gate_b",
    None,
)


importlib.invalidate_caches()


from cora_lung.eval.gate_b import (
    label_binary_components,
    reference_component_partition,
    component_iou_matrix,
    evaluate_selected_candidates,
)


# ==========================================================================================
# 4. VERIFY DEVELOPMENT SPLIT AND FROZEN PREDICTIONS
# ==========================================================================================

heading(
    "STEP 1/7 — VERIFY DEVELOPMENT-ONLY ARTIFACTS"
)


split_df = pd.read_csv(
    REPO
    / "data/splits/"
    "primary_volume_split_registry.csv"
)


development_df = (
    split_df[
        split_df[
            "role"
        ]
        == "permanent_development"
    ]
    .copy()
)


development_df[
    "case_id"
] = development_df[
    "case_id"
].astype(str)


development_cases = sorted(
    development_df[
        "case_id"
    ].tolist()
)


final_cases = split_df[
    split_df[
        "role"
    ]
    == "final_outer_cv"
][
    "case_id"
].astype(
    str
).tolist()


if len(development_cases) != 4:

    raise RuntimeError(
        "Expected four development cases."
    )


if len(final_cases) != 16:

    raise RuntimeError(
        "Expected sixteen sealed final cases."
    )


prediction_freeze_df = pd.read_csv(
    REPO
    / "data/manifests/"
    "gate_b_v1_2_prediction_freeze.csv"
)


prediction_hash_lookup = {
    (
        str(
            row[
                "condition"
            ]
        ),
        str(
            row[
                "case_id"
            ]
        ),
    ):
        str(
            row[
                "prediction_sha256"
            ]
        )
    for _, row
    in prediction_freeze_df.iterrows()
}


for condition in CONDITIONS:

    for case_id in development_cases:

        prediction_path = (
            PREDICTION_ROOT
            / condition
            / (
                case_id
                + ".npz"
            )
        )


        if not prediction_path.exists():

            raise RuntimeError(
                "Frozen prediction missing:\n"
                + str(
                    prediction_path
                )
            )


        expected_sha = prediction_hash_lookup[
            (
                condition,
                case_id,
            )
        ]


        if sha256_file(
            prediction_path
        ) != expected_sha:

            raise RuntimeError(
                "Frozen prediction checksum mismatch."
            )


print(
    "✓ Frozen predictions                  : PASS 16/16"
)

print(
    "✓ Development cases                   : 4"
)

print(
    "✓ Final outer-CV access               : 0"
)


# ==========================================================================================
# 5. LOCATE FROZEN SOURCE MASKS
# ==========================================================================================

candidate_roots = [
    Path(
        "/kaggle/input/datasets/andrewmvd/covid19-ct-scans"
    ),
    Path(
        "/kaggle/input/covid19-ct-scans"
    ),
]


primary_root = None


probe_mask = str(
    development_df.iloc[
        0
    ][
        "infection_mask"
    ]
)


for root in candidate_roots:

    if (
        root.exists()
        and (
            root
            / probe_mask
        ).exists()
    ):

        primary_root = root

        break


if primary_root is None:

    raise RuntimeError(
        "Primary dataset root not found."
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


case_lookup = {
    str(
        row[
            "case_id"
        ]
    ):
        row
    for _, row
    in development_df.iterrows()
}


reference_data = {}


for case_id in development_cases:

    row = case_lookup[
        case_id
    ]


    relative_mask = str(
        row[
            "infection_mask"
        ]
    )


    mask_path = (
        primary_root
        / relative_mask
    )


    if relative_mask not in source_hash_lookup:

        raise RuntimeError(
            "Missing frozen source-mask hash."
        )


    if (
        sha256_file(
            mask_path
        ).lower()
        != source_hash_lookup[
            relative_mask
        ]
    ):

        raise RuntimeError(
            "Source-mask checksum mismatch."
        )


    image = nib.load(
        str(
            mask_path
        ),
        mmap=True,
    )


    reference = (
        np.asarray(
            image.dataobj
        )
        > 0.5
    )


    voxel_volume_ml = float(
        abs(
            np.linalg.det(
                np.asarray(
                    image.affine,
                    dtype=np.float64,
                )[
                    :3,
                    :3
                ]
            )
        )
        / 1000.0
    )


    partition = reference_component_partition(
        reference,
        voxel_volume_ml=voxel_volume_ml,
        minimum_volume_ml=REFERENCE_MIN_VOLUME_ML,
    )


    reference_data[
        case_id
    ] = {
        "reference":
            reference,

        "partition":
            partition,

        "voxel_volume_ml":
            voxel_volume_ml,

        "source_origin":
            str(
                row[
                    "source_origin"
                ]
            ),
    }


print(
    "✓ Development reference masks         : VERIFIED 4/4"
)


# ==========================================================================================
# 6. DYNAMIC COMPONENTIZATION
# ==========================================================================================

heading(
    "STEP 2/7 — RECOMPUTE 3-D COMPONENT TOPOLOGY AT EVERY VOXEL THRESHOLD"
)


case_rows = []


for condition in CONDITIONS:

    print(
        "\nCondition:",
        condition,
    )


    for case_id in tqdm(
        development_cases,
        desc=condition,
        leave=False,
    ):

        with np.load(
            PREDICTION_ROOT
            / condition
            / (
                case_id
                + ".npz"
            ),
            allow_pickle=False,
        ) as npz:

            probability = np.asarray(
                npz[
                    "probability_xyz"
                ],
                dtype=np.float32,
            )


        reference = reference_data[
            case_id
        ][
            "reference"
        ]


        partition = reference_data[
            case_id
        ][
            "partition"
        ]


        if probability.shape != reference.shape:

            raise RuntimeError(
                "Prediction/reference shape mismatch."
            )


        eligible_ids = np.asarray(
            partition[
                "eligible_ids"
            ],
            dtype=np.int32,
        )


        small_ids = np.asarray(
            partition[
                "small_ids"
            ],
            dtype=np.int32,
        )


        all_reference_ids = np.concatenate(
            [
                eligible_ids,
                small_ids,
            ]
        )


        for threshold in VOXEL_THRESHOLDS:

            binary = (
                probability
                >= float(
                    threshold
                )
            )


            prediction_labels, prediction_count = (
                label_binary_components(
                    binary
                )
            )


            prediction_labels = prediction_labels.astype(
                np.int32,
                copy=False,
            )


            prediction_sizes = np.bincount(
                prediction_labels.reshape(
                    -1
                ),
                minlength=prediction_count
                + 1,
            ).astype(
                np.int64
            )


            all_iou = component_iou_matrix(
                partition[
                    "labels"
                ],
                partition[
                    "sizes"
                ],
                all_reference_ids,
                prediction_labels,
                prediction_sizes,
            )


            eligible_iou = all_iou[
                :
                len(
                    eligible_ids
                ),
                :,
            ]


            small_iou = all_iou[
                len(
                    eligible_ids
                ):
                ,
                :,
            ]


            dummy_scores = np.ones(
                prediction_count,
                dtype=np.float32,
            )


            result = evaluate_selected_candidates(
                candidate_scores=dummy_scores,
                eligible_iou=eligible_iou,
                small_iou=small_iou,
                score_threshold=-np.inf,
                minimum_iou=PRIMARY_IOU,
            )


            case_rows.append(
                {
                    "condition":
                        condition,

                    "case_id":
                        case_id,

                    "source_origin":
                        reference_data[
                            case_id
                        ][
                            "source_origin"
                        ],

                    "voxel_threshold":
                        float(
                            threshold
                        ),

                    "prediction_components":
                        int(
                            prediction_count
                        ),

                    "eligible_references":
                        int(
                            result[
                                "eligible_references"
                            ]
                        ),

                    "true_positives":
                        int(
                            result[
                                "true_positives"
                            ]
                        ),

                    "false_positives":
                        int(
                            result[
                                "false_positives"
                            ]
                        ),

                    "neutral_small":
                        int(
                            result[
                                "neutral_small_reference_predictions"
                            ]
                        ),

                    "recall":
                        float(
                            result[
                                "recall"
                            ]
                        ),
                }
            )


        del probability

        gc.collect()


dynamic_case_df = pd.DataFrame(
    case_rows
)


expected_rows = (
    len(
        CONDITIONS
    )
    * len(
        development_cases
    )
    * len(
        VOXEL_THRESHOLDS
    )
)


if len(
    dynamic_case_df
) != expected_rows:

    raise RuntimeError(
        "Dynamic-topology row count mismatch."
    )


print(
    "✓ Dynamic case-threshold evaluations  :",
    len(
        dynamic_case_df
    ),
)


# ==========================================================================================
# 7. BUILD CALIBRATION-ROBUST FROC CURVES
# ==========================================================================================

heading(
    "STEP 3/7 — BUILD DYNAMIC-TOPOLOGY FROC OPERATING POINTS"
)


cohort_rows = []


for condition in CONDITIONS:

    for threshold in VOXEL_THRESHOLDS:

        subset = dynamic_case_df[
            (
                dynamic_case_df[
                    "condition"
                ]
                == condition
            )
            & np.isclose(
                dynamic_case_df[
                    "voxel_threshold"
                ],
                threshold,
            )
        ]


        if len(
            subset
        ) != 4:

            raise RuntimeError(
                "Dynamic cohort aggregation failed."
            )


        fp_total = int(
            subset[
                "false_positives"
            ].sum()
        )


        tp_total = int(
            subset[
                "true_positives"
            ].sum()
        )


        eligible_total = int(
            subset[
                "eligible_references"
            ].sum()
        )


        cohort_rows.append(
            {
                "condition":
                    condition,

                "voxel_threshold":
                    float(
                        threshold
                    ),

                "fp_total":
                    fp_total,

                "fp_per_patient":
                    float(
                        fp_total
                        / 4.0
                    ),

                "macro_recall":
                    float(
                        subset[
                            "recall"
                        ].mean()
                    ),

                "pooled_recall":
                    float(
                        tp_total
                        / eligible_total
                        if eligible_total
                        else 0.0
                    ),

                "true_positives":
                    tp_total,

                "eligible_references":
                    eligible_total,

                "mean_prediction_components":
                    float(
                        subset[
                            "prediction_components"
                        ].mean()
                    ),
            }
        )


dynamic_froc_df = pd.DataFrame(
    cohort_rows
)


operating_rows = []


for condition in CONDITIONS:

    condition_curve = dynamic_froc_df[
        dynamic_froc_df[
            "condition"
        ]
        == condition
    ]


    for budget in FP_BUDGETS:

        feasible = condition_curve[
            condition_curve[
                "fp_per_patient"
            ]
            <= float(
                budget
            )
            + 1e-12
        ].copy()


        if len(
            feasible
        ) == 0:

            raise RuntimeError(
                "No feasible dynamic-FROC operating point."
            )


        # Lexicographic diagnostic selection:
        #   1. highest macro patient lesion recall
        #   2. highest pooled lesion recall
        #   3. fewer FP/patient
        #   4. higher voxel threshold
        selected = (
            feasible.sort_values(
                [
                    "macro_recall",
                    "pooled_recall",
                    "fp_per_patient",
                    "voxel_threshold",
                ],
                ascending=[
                    False,
                    False,
                    True,
                    False,
                ],
            )
            .iloc[
                0
            ]
        )


        operating_rows.append(
            {
                "condition":
                    condition,

                "fp_budget":
                    float(
                        budget
                    ),

                "selected_voxel_threshold":
                    float(
                        selected[
                            "voxel_threshold"
                        ]
                    ),

                "actual_fp_per_patient":
                    float(
                        selected[
                            "fp_per_patient"
                        ]
                    ),

                "macro_recall":
                    float(
                        selected[
                            "macro_recall"
                        ]
                    ),

                "pooled_recall":
                    float(
                        selected[
                            "pooled_recall"
                        ]
                    ),

                "true_positives":
                    int(
                        selected[
                            "true_positives"
                        ]
                    ),

                "eligible_references":
                    int(
                        selected[
                            "eligible_references"
                        ]
                    ),

                "mean_prediction_components":
                    float(
                        selected[
                            "mean_prediction_components"
                        ]
                    ),
            }
        )


dynamic_operating_df = pd.DataFrame(
    operating_rows
)


# ==========================================================================================
# 8. PRIMARY DYNAMIC R@1 CONTRAST
# ==========================================================================================

heading(
    "STEP 4/7 — COMPUTE PRIMARY DYNAMIC-TOPOLOGY R@1 CONTRAST"
)


primary_operating = dynamic_operating_df[
    np.isclose(
        dynamic_operating_df[
            "fp_budget"
        ],
        PRIMARY_FP_BUDGET,
    )
].copy()


control_row = primary_operating[
    primary_operating[
        "condition"
    ]
    == PRIMARY_CONTROL
].iloc[
    0
]


omission_row = primary_operating[
    primary_operating[
        "condition"
    ]
    == PRIMARY_OMISSION
].iloc[
    0
]


control_dynamic_r1 = float(
    control_row[
        "macro_recall"
    ]
)


omission_dynamic_r1 = float(
    omission_row[
        "macro_recall"
    ]
)


dynamic_delta = (
    control_dynamic_r1
    - omission_dynamic_r1
)


control_threshold = float(
    control_row[
        "selected_voxel_threshold"
    ]
)


omission_threshold = float(
    omission_row[
        "selected_voxel_threshold"
    ]
)


# ------------------------------------------------------------------------------------------
# Case-level recalls at each model's selected cohort-wide operating threshold.
# ------------------------------------------------------------------------------------------

case_difference_rows = []


for case_id in development_cases:

    control_case = dynamic_case_df[
        (
            dynamic_case_df[
                "condition"
            ]
            == PRIMARY_CONTROL
        )
        & (
            dynamic_case_df[
                "case_id"
            ]
            == case_id
        )
        & np.isclose(
            dynamic_case_df[
                "voxel_threshold"
            ],
            control_threshold,
        )
    ].iloc[
        0
    ]


    omission_case = dynamic_case_df[
        (
            dynamic_case_df[
                "condition"
            ]
            == PRIMARY_OMISSION
        )
        & (
            dynamic_case_df[
                "case_id"
            ]
            == case_id
        )
        & np.isclose(
            dynamic_case_df[
                "voxel_threshold"
            ],
            omission_threshold,
        )
    ].iloc[
        0
    ]


    case_difference_rows.append(
        {
            "case_id":
                case_id,

            "source_origin":
                str(
                    control_case[
                        "source_origin"
                    ]
                ),

            "eligible_components":
                int(
                    control_case[
                        "eligible_references"
                    ]
                ),

            "pixel_dynamic_R_at_1":
                float(
                    control_case[
                        "recall"
                    ]
                ),

            "omission_dynamic_R_at_1":
                float(
                    omission_case[
                        "recall"
                    ]
                ),

            "dynamic_delta_R_at_1":
                float(
                    control_case[
                        "recall"
                    ]
                    - omission_case[
                        "recall"
                    ]
                ),
        }
    )


dynamic_case_difference_df = pd.DataFrame(
    case_difference_rows
)


source_dynamic_df = (
    dynamic_case_difference_df.groupby(
        "source_origin",
        as_index=False,
    )
    .agg(
        cases=(
            "case_id",
            "count",
        ),

        mean_pixel_dynamic_R_at_1=(
            "pixel_dynamic_R_at_1",
            "mean",
        ),

        mean_omission_dynamic_R_at_1=(
            "omission_dynamic_R_at_1",
            "mean",
        ),

        mean_dynamic_delta_R_at_1=(
            "dynamic_delta_R_at_1",
            "mean",
        ),
    )
)


if dynamic_delta <= 1e-12:

    topology_interpretation = (
        "PRIMARY_PREMISE_REMAINS_UNSUPPORTED"
    )


elif dynamic_delta < LOCKED_MARGIN:

    topology_interpretation = (
        "TOPOLOGY_CONTRIBUTES_BUT_PREMISE_REMAINS_WEAK"
    )


else:

    topology_interpretation = (
        "STRONG_FIXED_TOPOLOGY_CONFOUND_SIGNAL"
    )


print(
    "Matched-pixel dynamic macro R@1      :",
    "{:.6f}".format(
        control_dynamic_r1
    ),
)

print(
    "  selected voxel threshold           :",
    "{:.6f}".format(
        control_threshold
    ),
)

print(
    "Natural-omission dynamic macro R@1   :",
    "{:.6f}".format(
        omission_dynamic_r1
    ),
)

print(
    "  selected voxel threshold           :",
    "{:.6f}".format(
        omission_threshold
    ),
)

print(
    "Dynamic delta R@1                    :",
    "{:+.6f}".format(
        dynamic_delta
    ),
)

print(
    "Diagnostic interpretation            :",
    topology_interpretation,
)

print()

print(
    dynamic_case_difference_df.to_string(
        index=False
    )
)


# ==========================================================================================
# 9. LOSS-PRIOR DIAGNOSTIC
# ==========================================================================================

heading(
    "STEP 5/7 — QUANTIFY CURRENT SPARSE-LOSS CLASS-PRIOR EFFECT"
)


exposure_df = pd.read_csv(
    REPO
    / "data/manifests/"
    "gate_b_failure_supervision_exposure_summary.csv"
)


loss_prior_rows = []


def constant_sparse_loss(
    probability,
    positive_fraction,
):

    p = float(
        probability
    )

    pi = float(
        positive_fraction
    )


    bce = (
        -pi
        * np.log(
            p
        )
        - (
            1.0
            - pi
        )
        * np.log(
            1.0
            - p
        )
    )


    dice_loss = (
        1.0
        - (
            2.0
            * p
            * pi
        )
        / (
            p
            + pi
        )
    )


    return float(
        bce
        + dice_loss
    )


probability_grid = np.linspace(
    0.001,
    0.999,
    999,
)


for _, row in exposure_df.iterrows():

    condition = str(
        row[
            "condition"
        ]
    )


    fg = float(
        row[
            "foreground_voxels_seen"
        ]
    )


    bg = float(
        row[
            "background_voxels_seen"
        ]
    )


    labelled = (
        fg
        + bg
    )


    positive_fraction = (
        fg
        / labelled
    )


    losses = np.asarray(
        [
            constant_sparse_loss(
                p,
                positive_fraction,
            )
            for p
            in probability_grid
        ],
        dtype=np.float64,
    )


    minimum_index = int(
        np.argmin(
            losses
        )
    )


    loss_prior_rows.append(
        {
            "condition":
                condition,

            "foreground_exposures":
                int(
                    fg
                ),

            "background_exposures":
                int(
                    bg
                ),

            "positive_label_fraction_seen":
                float(
                    positive_fraction
                ),

            "constant_probability_minimizer_under_current_BCE_plus_Dice":
                float(
                    probability_grid[
                        minimum_index
                    ]
                ),

            "minimum_constant_sparse_loss":
                float(
                    losses[
                        minimum_index
                    ]
                ),
        }
    )


loss_prior_df = pd.DataFrame(
    loss_prior_rows
)


print(
    loss_prior_df.to_string(
        index=False
    )
)


pixel_prior = float(
    loss_prior_df[
        loss_prior_df[
            "condition"
        ]
        == PRIMARY_CONTROL
    ][
        "positive_label_fraction_seen"
    ].iloc[
        0
    ]
)


natural_prior = float(
    loss_prior_df[
        loss_prior_df[
            "condition"
        ]
        == PRIMARY_OMISSION
    ][
        "positive_label_fraction_seen"
    ].iloc[
        0
    ]
)


primary_prior_difference = abs(
    pixel_prior
    - natural_prior
)


print()

print(
    "Primary-pair labelled-class-prior difference :",
    "{:.6f}".format(
        primary_prior_difference
    ),
)


# ==========================================================================================
# 10. SAVE TABLES / FIGURES
# ==========================================================================================

heading(
    "STEP 6/7 — FREEZE POST-HOC DIAGNOSTIC ARTIFACTS"
)


manifest_dir = (
    REPO
    / "data/manifests"
)


audit_dir = (
    REPO
    / "experiments/audits"
)


figure_dir = (
    REPO
    / "figures/diagnostics/"
    "gate_b_failure"
)


manifest_dir.mkdir(
    parents=True,
    exist_ok=True,
)


audit_dir.mkdir(
    parents=True,
    exist_ok=True,
)


figure_dir.mkdir(
    parents=True,
    exist_ok=True,
)


dynamic_case_df.to_csv(
    manifest_dir
    / "gate_b_dynamic_topology_case_threshold.csv",
    index=False,
)


dynamic_froc_df.to_csv(
    manifest_dir
    / "gate_b_dynamic_topology_froc.csv",
    index=False,
)


dynamic_operating_df.to_csv(
    manifest_dir
    / "gate_b_dynamic_topology_operating_points.csv",
    index=False,
)


dynamic_case_difference_df.to_csv(
    manifest_dir
    / "gate_b_dynamic_topology_primary_case_difference.csv",
    index=False,
)


source_dynamic_df.to_csv(
    manifest_dir
    / "gate_b_dynamic_topology_source_summary.csv",
    index=False,
)


loss_prior_df.to_csv(
    manifest_dir
    / "gate_b_sparse_loss_prior_audit.csv",
    index=False,
)


# ------------------------------------------------------------------------------------------
# Figure 21 — Dynamic topology FROC.
# ------------------------------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(
        10,
        6.4,
    )
)


for condition in CONDITIONS:

    subset = dynamic_froc_df[
        dynamic_froc_df[
            "condition"
        ]
        == condition
    ].copy()


    subset = subset.sort_values(
        "fp_per_patient"
    )


    ax.plot(
        subset[
            "fp_per_patient"
        ],
        subset[
            "macro_recall"
        ],
        marker="o",
        linewidth=1.8,
        markersize=3.5,
        label=condition,
    )


ax.axvline(
    1.0,
    linestyle="--",
    linewidth=1.3,
)


ax.set_xlim(
    0,
    min(
        10,
        max(
            4.5,
            float(
                dynamic_froc_df[
                    "fp_per_patient"
                ].quantile(
                    0.75
                )
            ),
        ),
    ),
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
    "Post-hoc Dynamic-Topology FROC Diagnostic\n"
    "Connected Components Recomputed at Every Voxel Threshold",
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
    / "fig21_dynamic_topology_froc.png",
    dpi=600,
    bbox_inches="tight",
)


fig.savefig(
    figure_dir
    / "fig21_dynamic_topology_froc.pdf",
    bbox_inches="tight",
)


plt.close(
    fig
)


# ------------------------------------------------------------------------------------------
# Figure 22 — Primary dynamic R@1 contrast.
# ------------------------------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(
        8,
        5.8,
    )
)


values = [
    control_dynamic_r1,
    omission_dynamic_r1,
]


bars = ax.bar(
    np.arange(
        2
    ),
    values,
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
    max(
        0.20,
        max(
            values
        )
        * 1.25
        + 0.02,
    ),
)


ax.set_ylabel(
    "Dynamic-Topology Macro R@1",
    fontweight="bold",
)


ax.set_title(
    "Post-hoc Calibration/Topology Robustness Test\n"
    + "ΔR@1 = "
    + "{:+.3f}".format(
        dynamic_delta
    )
    + " | "
    + topology_interpretation,
    fontweight="bold",
)


ax.grid(
    axis="y",
    alpha=0.25,
)


for bar, value in zip(
    bars,
    values,
):

    ax.text(
        bar.get_x()
        + bar.get_width()
        / 2,
        value
        + 0.005,
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
    / "fig22_dynamic_topology_primary_contrast.png",
    dpi=600,
    bbox_inches="tight",
)


fig.savefig(
    figure_dir
    / "fig22_dynamic_topology_primary_contrast.pdf",
    bbox_inches="tight",
)


plt.close(
    fig
)


# ------------------------------------------------------------------------------------------
# Figure 23 — Sparse loss class-prior diagnostic.
# ------------------------------------------------------------------------------------------

plot_loss_prior = (
    loss_prior_df.set_index(
        "condition"
    ).loc[
        CONDITIONS
    ]
)


fig, ax = plt.subplots(
    figsize=(
        10,
        6.0,
    )
)


x = np.arange(
    len(
        CONDITIONS
    )
)


width = 0.35


ax.bar(
    x
    - width
    / 2,
    plot_loss_prior[
        "positive_label_fraction_seen"
    ],
    width,
    label="Observed positive fraction",
)


ax.bar(
    x
    + width
    / 2,
    plot_loss_prior[
        "constant_probability_minimizer_under_current_BCE_plus_Dice"
    ],
    width,
    label="Constant-p loss minimizer",
)


ax.set_xticks(
    x
)


ax.set_xticklabels(
    [
        "Complete",
        "Matched Pixel",
        "Natural Omission",
        "Fixed Omission",
    ],
    rotation=15,
    ha="right",
    fontweight="bold",
)


ax.set_ylim(
    0,
    1,
)


ax.set_ylabel(
    "Probability / Label Fraction",
    fontweight="bold",
)


ax.set_title(
    "Sparse BCE + Partial Dice: Label-Prior Diagnostic\n"
    "Unknown Voxels Do Not Contribute to the Training Loss",
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
    / "fig23_sparse_loss_prior_diagnostic.png",
    dpi=600,
    bbox_inches="tight",
)


fig.savefig(
    figure_dir
    / "fig23_sparse_loss_prior_diagnostic.pdf",
    bbox_inches="tight",
)


plt.close(
    fig
)


# ==========================================================================================
# 11. AUDIT / STATE
# ==========================================================================================

audit_payload = {
    "project":
        "CORA-Lung",

    "block":
        "08G",

    "status":
        "PASS",

    "generated_at_utc":
        NOW_ISO,

    "purpose":
        (
            "post-hoc calibration/topology robustness diagnostic "
            "after locked Gate-B NO-GO"
        ),

    "gate_b_before":
        "NO_GO",

    "gate_b_after":
        "NO_GO",

    "gate_b_modified":
        False,

    "gate_c":
        "NOT_RUN",

    "optimizer_steps":
        0,

    "models_retrained":
        False,

    "predictions_regenerated":
        False,

    "final_outer_cv_cases_accessed":
        0,

    "dynamic_topology_definition":
        (
            "recompute 26-connected prediction components "
            "at every voxel probability threshold"
        ),

    "dynamic_voxel_thresholds":
        VOXEL_THRESHOLDS.tolist(),

    "primary_control_dynamic_R_at_1":
        control_dynamic_r1,

    "primary_omission_dynamic_R_at_1":
        omission_dynamic_r1,

    "dynamic_delta_R_at_1":
        dynamic_delta,

    "control_selected_voxel_threshold":
        control_threshold,

    "omission_selected_voxel_threshold":
        omission_threshold,

    "interpretation":
        topology_interpretation,

    "locked_gate_b_margin_reference_only":
        LOCKED_MARGIN,

    "primary_label_prior_difference":
        primary_prior_difference,

    "scientific_constraint":
        (
            "This post-hoc diagnostic cannot replace or modify "
            "the locked Gate-B result."
        ),

    "next_action":
        (
            "Review whether the component-omission premise remains "
            "supportable after calibration/topology robustness analysis. "
            "Do not retrain automatically."
        ),
}


write_json(
    audit_dir
    / "block08g_dynamic_topology_froc_diagnostic.json",
    audit_payload,
)


note = """
# Block 08G — Dynamic-Topology FROC Diagnostic

The official Gate-B result remains NO-GO.

This analysis recomputes connected components independently at each voxel
probability threshold. It therefore asks whether the fixed p=0.5 candidate
topology used by the locked evaluator materially affected the scientific
ordering between matched random-pixel sparsity and natural component omission.

The analysis is post-hoc and diagnostic. It cannot replace Gate B.

The same block also quantifies the class-prior dependence of the current
partial BCE + partial Dice loss using the sparse supervision actually observed
during the frozen training schedule.

No model is retrained and no final outer-CV case is accessed.
"""


write_text(
    REPO
    / "docs/gate_b_dynamic_topology_diagnostic.md",
    note,
)


state.update(
    {
        "last_attempted_block":
            "08G",

        "last_completed_block":
            "08G",

        "last_completed_block_name":
            "gate_b_dynamic_topology_froc_diagnostic",

        "current_stage":
            "gate_b_posthoc_topology_review_complete",

        "current_gate":
            "B",

        "gate_b":
            "NO_GO",

        "gate_c":
            "NOT_RUN",

        "gate_b_dynamic_topology_audit":
            "PASS",

        "gate_b_dynamic_topology_delta_R_at_1":
            dynamic_delta,

        "gate_b_dynamic_topology_interpretation":
            topology_interpretation,

        "gate_b_primary_label_prior_difference":
            primary_prior_difference,

        "training_authorized":
            False,

        "pilot_training_authorized":
            False,

        "final_outer_cv_dynamic_audit_access":
            0,

        "next_action":
            (
                "Scientific review of Block 08G. "
                "Do not retrain until the project direction is explicitly chosen."
            ),

        "updated_at_utc":
            NOW_ISO,
    }
)


if (
    state[
        "gate_b"
    ]
    != "NO_GO"
):

    raise RuntimeError(
        "Block 08G illegally changed Gate B."
    )


if (
    state[
        "gate_c"
    ]
    != "NOT_RUN"
):

    raise RuntimeError(
        "Block 08G illegally changed Gate C."
    )


write_json(
    state_path,
    state,
)


# ==========================================================================================
# 12. REPOSITORY MANIFEST
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
            "08G",

        "gate_b":
            "NO_GO",

        "gate_c":
            "NOT_RUN",

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
# 13. CAPTURE SOURCE
# ==========================================================================================

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
        "CORA-LUNG — CODE BLOCK 08G"
        in cell
    ):

        source_path = (
            REPO
            / "scripts/code_blocks/"
            "block08g_dynamic_topology_froc_diagnostic.py"
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
# 14. COMMIT + PUSH
# ==========================================================================================

heading(
    "STEP 7/7 — COMMIT POST-HOC TOPOLOGY DIAGNOSTIC"
)


git_paths = [
    "PROJECT_STATE.json",
    "REPOSITORY_MANIFEST.json",
    "data/manifests/gate_b_dynamic_topology_case_threshold.csv",
    "data/manifests/gate_b_dynamic_topology_froc.csv",
    "data/manifests/gate_b_dynamic_topology_operating_points.csv",
    "data/manifests/gate_b_dynamic_topology_primary_case_difference.csv",
    "data/manifests/gate_b_dynamic_topology_source_summary.csv",
    "data/manifests/gate_b_sparse_loss_prior_audit.csv",
    "experiments/audits/block08g_dynamic_topology_froc_diagnostic.json",
    "docs/gate_b_dynamic_topology_diagnostic.md",
    "figures/diagnostics/gate_b_failure/"
    "fig21_dynamic_topology_froc.png",
    "figures/diagnostics/gate_b_failure/"
    "fig21_dynamic_topology_froc.pdf",
    "figures/diagnostics/gate_b_failure/"
    "fig22_dynamic_topology_primary_contrast.png",
    "figures/diagnostics/gate_b_failure/"
    "fig22_dynamic_topology_primary_contrast.pdf",
    "figures/diagnostics/gate_b_failure/"
    "fig23_sparse_loss_prior_diagnostic.png",
    "figures/diagnostics/gate_b_failure/"
    "fig23_sparse_loss_prior_diagnostic.pdf",
]


source_path = (
    REPO
    / "scripts/code_blocks/"
    "block08g_dynamic_topology_froc_diagnostic.py"
)


if source_path.exists():

    git_paths.append(
        "scripts/code_blocks/"
        "block08g_dynamic_topology_froc_diagnostic.py"
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
        "No Block-08G artifacts available to commit."
    )


print(status)


sh(
    [
        "git",
        "commit",
        "-m",
        "audit: test Gate-B calibration and topology robustness",
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
# 15. FINAL REPORT
# ==========================================================================================

print(
    "\n"
    + "=" * 122
)

print(
    "CORA-LUNG CODE BLOCK 08G — FINAL DYNAMIC-TOPOLOGY REPORT"
)

print(
    "=" * 122
)


print(
    "Official locked Gate-B result         : NO_GO"
)

print(
    "Gate-B result modified                : NO"
)

print(
    "Gate C                                : NOT RUN"
)

print(
    "Models retrained                      : NO"
)

print(
    "Predictions regenerated               : NO"
)

print(
    "Optimizer steps                       : 0"
)

print(
    "Final outer-CV cases accessed         : 0"
)

print(
    "Voxel thresholds tested               :",
    len(
        VOXEL_THRESHOLDS
    ),
)

print()
print(
    "DYNAMIC-TOPOLOGY PRIMARY RESULT"
)

print(
    "--------------------------------"
)

print(
    "Matched-pixel dynamic macro R@1       :",
    "{:.6f}".format(
        control_dynamic_r1
    ),
)

print(
    "Matched-pixel selected voxel threshold:",
    "{:.6f}".format(
        control_threshold
    ),
)

print(
    "Natural-omission dynamic macro R@1    :",
    "{:.6f}".format(
        omission_dynamic_r1
    ),
)

print(
    "Natural-omission selected threshold   :",
    "{:.6f}".format(
        omission_threshold
    ),
)

print(
    "Dynamic delta R@1                     :",
    "{:+.6f}".format(
        dynamic_delta
    ),
)

print(
    "Diagnostic interpretation             :",
    topology_interpretation,
)

print()
print(
    "CASE-LEVEL DYNAMIC RESULT"
)

print(
    "-------------------------"
)

print(
    dynamic_case_difference_df.to_string(
        index=False
    )
)

print()
print(
    "SOURCE-SPECIFIC DYNAMIC RESULT"
)

print(
    "------------------------------"
)

print(
    source_dynamic_df.to_string(
        index=False
    )
)

print()
print(
    "CURRENT SPARSE-LOSS PRIOR AUDIT"
)

print(
    "-------------------------------"
)

print(
    loss_prior_df[
        [
            "condition",
            "positive_label_fraction_seen",
            "constant_probability_minimizer_under_current_BCE_plus_Dice",
        ]
    ].to_string(
        index=False
    )
)

print()
print(
    "Primary matched-pair label-prior diff :",
    "{:.6f}".format(
        primary_prior_difference
    ),
)

print()
print(
    "Exact Block-08G source captured       :",
    source_capture,
)

print(
    "Starting commit                       :",
    starting_commit[:12],
)

print(
    "Final audit commit                    :",
    final_commit[:12],
)

print(
    "GitHub synchronization                : PASS"
)

print()
print(
    "NEXT:"
)

print(
    "Send me this COMPLETE report."
)

print(
    "Do NOT retrain yet."
)

print(
    "=" * 122
)