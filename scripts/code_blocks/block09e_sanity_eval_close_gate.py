# ==========================================================================================
# COVA-3D — BLOCK 09E-SANITY-EVAL-CLOSE-GATE
#
# PURPOSE
# -------
# 1. FIRST freeze + push R2 prediction provenance and the exact dense sanity
#    evaluation rule BEFORE opening any dense annotation.
# 2. THEN open ONLY the four permanent-development infection masks.
# 3. Apply the already-locked sanity gate.
# 4. Authorize factorial FITTING only if the complete gate passes.
#
# NO:
#   - Dice / IoU / FROC
#   - threshold search
#   - hyperparameter selection
#   - retraining
#   - optimizer.step()
#   - final outer-CV dense masks
#
# EXPECTED START HEAD:
#   fa70f4db7d23
# ==========================================================================================

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import hashlib
import json
import os
import subprocess
import sys
import textwrap

import nibabel as nib
import numpy as np
import pandas as pd
import yaml

from nibabel.processing import resample_to_output
from kaggle_secrets import UserSecretsClient


# ==========================================================================================
# CONSTANTS
# ==========================================================================================

REPO = Path("/kaggle/working/CORA-LUNG")

EXPECTED_START_HEAD = "fa70f4db7d23"

DATASET_ROOT = Path(
    "/kaggle/input/datasets/andrewmvd/covid19-ct-scans"
)

PRED_ROOT = Path(
    "/kaggle/working/cova3d_sanity_eval_v1_0/predictions"
)

RUNTIME_PRED_MANIFEST = Path(
    "/kaggle/working/cova3d_sanity_eval_v1_0/prediction_manifest.csv"
)

MODEL_SHA = (
    "fce56f781cbd4ffd3e733cccb04c81f3b5bb2a664aceed4b8c0688e06ac4891c"
)

EFFECTIVE_PROTOCOL = (
    "COVA3D_1.0+A1+A1.1+A1.2+A1.3+N1+N2"
)

TARGET_SPACING_XYZ = (
    1.5,
    1.5,
    3.0,
)

TARGET_SPACING_ZYX = (
    3.0,
    1.5,
    1.5,
)

LOSS_RATIO_MAX = 0.95
ZERO_LABEL_MAX = 0.30
STD_MIN = 0.001
POSITIVE_SEPARATION_MIN_CASES = 3

SPLIT_PATH = (
    REPO
    / "data/splits/primary_volume_split_registry.csv"
)

GRID_GEOMETRY_PATH = (
    REPO
    / "data/manifests/cova3d_training_grid_geometry_v1_2.csv"
)

R2_EPOCH_LOG_PATH = (
    REPO
    / "data/manifests/cova3d_sanity_fit_R2_epoch_log_v1_0.csv"
)

BASELINE_CONFIG_PATH = (
    REPO
    / "configs/cova3d_baseline_training_v1_0.yaml"
)

STATE_PATH = REPO / "COVA3D_STATE.json"
PROJECT_STATE_PATH = REPO / "PROJECT_STATE.json"

PRED_FREEZE_REPO_PATH = (
    REPO
    / "data/manifests/cova3d_sanity_R2_prediction_freeze_v1_0.csv"
)

INFERENCE_LOCK_PATH = (
    REPO
    / "configs/cova3d_sanity_inference_implementation_lock_v1_0.yaml"
)

OUTCOME_LOCK_PATH = (
    REPO
    / "configs/cova3d_sanity_outcome_evaluation_lock_v1_0.yaml"
)

PREFREEZE_AUDIT_PATH = (
    REPO
    / "experiments/audits/block09e_sanity_prediction_freeze.json"
)

CASE_METRICS_PATH = (
    REPO
    / "data/manifests/cova3d_sanity_gate_case_metrics_v1_0.csv"
)

GATE_AUDIT_PATH = (
    REPO
    / "experiments/audits/block09e_sanity_gate_close.json"
)

TEST_PATH = (
    REPO
    / "tests/test_cova3d_sanity_gate_close.py"
)

NOW_ISO = datetime.now(
    timezone.utc
).strftime("%Y-%m-%dT%H:%M:%SZ")


# ==========================================================================================
# HELPERS
# ==========================================================================================

def heading(text):
    print("\n" + "=" * 128)
    print(text)
    print("=" * 128)


def sh(cmd, *, env=None, check=True):

    r = subprocess.run(
        cmd,
        cwd=str(REPO),
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if check and r.returncode != 0:

        raise RuntimeError(
            "COMMAND FAILED\n"
            + " ".join(cmd)
            + "\n\nSTDOUT:\n"
            + (r.stdout or "")
            + "\nSTDERR:\n"
            + (r.stderr or "")
        )

    return r


def git(*args):
    return sh(
        ["git", *args]
    ).stdout.strip()


def sha256_file(path):

    h = hashlib.sha256()

    with open(path, "rb") as f:

        for chunk in iter(
            lambda: f.read(8 * 1024 * 1024),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def write_json(path, obj):

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


def write_text(path, text):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        textwrap.dedent(text).strip()
        + "\n",
        encoding="utf-8",
    )


def git_auth():

    token = (
        UserSecretsClient()
        .get_secret("pushCora")
        .strip()
    )

    askpass = Path(
        "/tmp/cova3d_git_askpass_sanity_gate.sh"
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

    askpass.chmod(0o700)

    env = os.environ.copy()
    env["GITHUB_TOKEN"] = token
    env["GIT_ASKPASS"] = str(askpass)
    env["GIT_TERMINAL_PROMPT"] = "0"

    return token, askpass, env


def push(env, token):

    r = sh(
        ["git", "push", "origin", "main"],
        env=env,
        check=False,
    )

    if r.returncode != 0:

        raise RuntimeError(
            "Git push failed:\n"
            + (r.stderr or "").replace(
                token,
                "***REDACTED***",
            )
        )


# ==========================================================================================
# 1. PREFLIGHT
# ==========================================================================================

heading(
    "COVA-3D 09E-SANITY-EVAL — PREFLIGHT"
)

head = git(
    "rev-parse",
    "HEAD",
)

if not head.startswith(
    EXPECTED_START_HEAD
):
    raise RuntimeError(
        f"Unexpected HEAD: {head}"
    )

dirty = git(
    "status",
    "--porcelain",
)

if dirty:

    # Only known abandoned inference config from prior failed block is allowed.
    allowed = (
        "?? configs/"
        "cova3d_sanity_inference_implementation_lock_v1_0.yaml"
    )

    lines = [
        x
        for x in dirty.splitlines()
        if x.strip()
    ]

    if lines == [allowed]:

        orphan = INFERENCE_LOCK_PATH

        if orphan.exists():
            orphan.unlink()

        dirty = git(
            "status",
            "--porcelain",
        )

if dirty:

    raise RuntimeError(
        "Unexpected dirty repository:\n"
        + dirty
    )


state = json.loads(
    STATE_PATH.read_text(
        encoding="utf-8"
    )
)

project_state = json.loads(
    PROJECT_STATE_PATH.read_text(
        encoding="utf-8"
    )
)


if state.get(
    "A1_3_status"
) != "FROZEN_PASS":

    raise RuntimeError(
        "A1.3 is not frozen PASS."
    )


if state.get(
    "dense_outcomes_opened_in_cova3d"
) is not False:

    raise RuntimeError(
        "Dense outcomes were already opened."
    )


if int(
    state.get(
        "final_outer_cv_access_in_cova3d",
        -1,
    )
) != 0:

    raise RuntimeError(
        "Final outer-CV access is not zero."
    )


if state.get(
    "factorial_training_authorized"
) is not False:

    raise RuntimeError(
        "Factorial training should still be locked."
    )


print(
    "✓ Git HEAD                         :",
    head[:12],
)

print(
    "✓ Repository                       : CLEAN"
)

print(
    "✓ A1.3                             : FROZEN PASS"
)

print(
    "✓ Dense outcomes                   : SEALED"
)

print(
    "✓ Final outer-CV access            : 0"
)


# ==========================================================================================
# 2. VERIFY THE FOUR FROZEN PREDICTIONS
# ==========================================================================================

heading(
    "STEP 1/6 — VERIFY FROZEN R2 PREDICTIONS"
)


if not RUNTIME_PRED_MANIFEST.exists():

    raise RuntimeError(
        "Runtime prediction manifest missing."
    )


pred_df = pd.read_csv(
    RUNTIME_PRED_MANIFEST
)


if len(pred_df) != 4:

    raise RuntimeError(
        "Expected exactly four prediction rows."
    )


split_df = pd.read_csv(
    SPLIT_PATH
)


dev_df = split_df[
    split_df["role"]
    == "permanent_development"
].copy()


dev_cases = sorted(
    dev_df[
        "case_id"
    ].astype(str).tolist()
)


if len(dev_cases) != 4:

    raise RuntimeError(
        "Expected four permanent-development cases."
    )


if sorted(
    pred_df[
        "case_id"
    ].astype(str).tolist()
) != dev_cases:

    raise RuntimeError(
        "Prediction cases do not match development cases."
    )


for _, row in pred_df.iterrows():

    path = Path(
        str(
            row["prediction_file"]
        )
    )

    if not path.exists():

        raise RuntimeError(
            f"Frozen prediction missing: {path}"
        )

    if sha256_file(path) != str(
        row["prediction_sha256"]
    ):

        raise RuntimeError(
            f"Prediction SHA mismatch: {row['case_id']}"
        )

    if not bool(
        row["std_gate_pass"]
    ):

        raise RuntimeError(
            "Prediction-std gate is not 4/4 PASS."
        )


print(
    "✓ Prediction files                 : 4 /4"
)

print(
    "✓ Prediction SHA checks            : 4 /4"
)

print(
    "✓ Prediction std >= 0.001          : 4 /4 PASS"
)

print(
    "✓ Dense masks opened               : 0"
)


# ==========================================================================================
# 3. RECOMPUTE LOCKED TRAINING SUBCRITERIA
# ==========================================================================================

heading(
    "STEP 2/6 — RECOMPUTE TRAINING SANITY SUBCRITERIA"
)


epoch_df = (
    pd.read_csv(
        R2_EPOCH_LOG_PATH
    )
    .sort_values("epoch")
    .reset_index(drop=True)
)


if len(epoch_df) != 20:

    raise RuntimeError(
        "Expected 20 R2 epochs."
    )


finite_training = bool(
    np.isfinite(
        epoch_df[
            [
                "mean_loss",
                "mean_partial_bce",
                "mean_partial_dice_fp32",
            ]
        ].to_numpy(
            dtype=np.float64
        )
    ).all()
)


first3 = float(
    epoch_df[
        "mean_loss"
    ].iloc[:3].mean()
)


last3 = float(
    epoch_df[
        "mean_loss"
    ].iloc[-3:].mean()
)


loss_ratio = (
    last3
    / first3
)


loss_pass = bool(
    finite_training
    and loss_ratio
    <= LOSS_RATIO_MAX
)


zero_label_fraction = float(
    epoch_df[
        "zero_label_microbatches"
    ].sum()
    / epoch_df[
        "logical_microbatches"
    ].sum()
)


zero_label_pass = bool(
    zero_label_fraction
    <= ZERO_LABEL_MAX
)


print(
    "First-3 mean loss                 :",
    f"{first3:.12f}",
)

print(
    "Last-3 mean loss                  :",
    f"{last3:.12f}",
)

print(
    "Last3 / First3                    :",
    f"{loss_ratio:.12f}",
)

print(
    "Loss-ratio criterion <= 0.95      :",
    "PASS"
    if loss_pass
    else "FAIL",
)

print(
    "Zero-label fraction               :",
    f"{zero_label_fraction:.8f}",
)

print(
    "Zero-label criterion <= 0.30      :",
    "PASS"
    if zero_label_pass
    else "FAIL",
)


# ==========================================================================================
# 4. FREEZE EVALUATION RULE + PREDICTION PROVENANCE BEFORE MASK ACCESS
# ==========================================================================================

heading(
    "STEP 3/6 — PRE-OUTCOME FREEZE + GITHUB PUSH"
)


baseline_cfg = yaml.safe_load(
    BASELINE_CONFIG_PATH.read_text(
        encoding="utf-8"
    )
)


locked_gate = baseline_cfg[
    "sanity_gate"
]


if float(
    locked_gate[
        "last_3_epoch_mean_loss_div_first_3_epoch_mean_loss_max"
    ]
) != LOSS_RATIO_MAX:

    raise RuntimeError(
        "Baseline sanity rule changed."
    )


if float(
    locked_gate[
        "zero_label_microbatch_fraction_max"
    ]
) != ZERO_LABEL_MAX:

    raise RuntimeError(
        "Baseline zero-label rule changed."
    )


if int(
    locked_gate[
        "development_cases_with_positive_separation_min"
    ]
) != POSITIVE_SEPARATION_MIN_CASES:

    raise RuntimeError(
        "Baseline separation rule changed."
    )


if float(
    locked_gate[
        "minimum_prediction_probability_std_per_case"
    ]
) != STD_MIN:

    raise RuntimeError(
        "Baseline prediction-std rule changed."
    )


# ------------------------------------------------------------------
# Store frozen prediction manifest in repository.
# ------------------------------------------------------------------

repo_pred_df = pred_df.copy()

repo_pred_df[
    "R2_model_sha256"
] = MODEL_SHA

repo_pred_df[
    "condition"
] = "C100_COH"

repo_pred_df[
    "seed"
] = 17

repo_pred_df[
    "dense_mask_accessed_before_prediction"
] = False


PRED_FREEZE_REPO_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


repo_pred_df.to_csv(
    PRED_FREEZE_REPO_PATH,
    index=False,
)


# ------------------------------------------------------------------
# Freeze inference implementation.
# ------------------------------------------------------------------

inference_lock = {
    "project":
        "COVA-3D",

    "status":
        "FROZEN_BEFORE_DENSE_SANITY_OUTCOME",

    "model_sha256":
        MODEL_SHA,

    "condition":
        "C100_COH",

    "seed":
        17,

    "inference":
        {
            "patch_zyx":
                [48, 128, 128],

            "overlap":
                0.50,

            "weighting":
                "gaussian",

            "gaussian_sigma_scale":
                0.125,

            "blend_domain":
                "logits",

            "probability":
                "sigmoid_after_merged_logits",

            "TTA":
                False,
        },

    "prediction_count":
        4,

    "prediction_std_threshold":
        STD_MIN,

    "prediction_std_cases_passing":
        4,

    "dense_masks_opened":
        False,

    "frozen_at_utc":
        NOW_ISO,
}


INFERENCE_LOCK_PATH.write_text(
    yaml.safe_dump(
        inference_lock,
        sort_keys=False,
    ),
    encoding="utf-8",
)


# ------------------------------------------------------------------
# Freeze the exact dense sanity evaluation procedure.
# ------------------------------------------------------------------

outcome_lock = {
    "project":
        "COVA-3D",

    "status":
        "FROZEN_BEFORE_FIRST_DENSE_SANITY_MASK_ACCESS",

    "development_cases":
        dev_cases,

    "dense_reference":
        "infection_mask only",

    "mask_preprocessing":
        {
            "native_geometry":
                (
                    "Use original dataset infection mask corresponding "
                    "to the frozen development case."
                ),

            "canonicalization":
                "nibabel.as_closest_canonical",

            "resampling":
                "nibabel.processing.resample_to_output",

            "target_spacing_xyz_mm":
                list(
                    TARGET_SPACING_XYZ
                ),

            "interpolation_order":
                0,

            "constant_value":
                0,

            "foreground_definition":
                "resampled mask > 0",

            "transpose":
                "XYZ to ZYX",

            "crop":
                (
                    "Use frozen crop_origin and crop_shape from "
                    "cova3d_training_grid_geometry_v1_2.csv"
                ),
        },

    "per_case_statistic":
        {
            "inside":
                "mean frozen predicted probability where dense lesion mask == 1",

            "outside":
                "mean frozen predicted probability where dense lesion mask == 0 within same crop",

            "positive_separation":
                "inside_mean > outside_mean",
        },

    "gate":
        {
            "finite_training_and_probabilities":
                True,

            "loss_ratio_max":
                LOSS_RATIO_MAX,

            "zero_label_fraction_max":
                ZERO_LABEL_MAX,

            "prediction_std_min_each_case":
                STD_MIN,

            "positive_separation_cases_min":
                POSITIVE_SEPARATION_MIN_CASES,
        },

    "explicitly_forbidden":
        [
            "Dice",
            "IoU",
            "FROC",
            "threshold search",
            "performance model selection",
            "final outer-CV dense masks",
        ],

    "frozen_at_utc":
        NOW_ISO,
}


OUTCOME_LOCK_PATH.write_text(
    yaml.safe_dump(
        outcome_lock,
        sort_keys=False,
    ),
    encoding="utf-8",
)


prefreeze_audit = {
    "status":
        "PREDICTIONS_AND_OUTCOME_RULE_FROZEN",

    "git_parent":
        head,

    "prediction_count":
        4,

    "prediction_SHA_verified":
        True,

    "prediction_std_cases_passing":
        4,

    "dense_masks_opened":
        False,

    "final_outer_cv_access":
        0,

    "optimizer_steps":
        0,

    "training":
        False,

    "generated_at_utc":
        NOW_ISO,
}


write_json(
    PREFREEZE_AUDIT_PATH,
    prefreeze_audit,
)


state[
    "last_completed_block"
] = "09E-SANITY-EVAL-PRED-FREEZE"

state[
    "sanity_prediction_freeze_status"
] = "PASS"

state[
    "sanity_prediction_std_subcriterion"
] = "PASS"

state[
    "sanity_prediction_std_cases_passing"
] = 4

state[
    "dense_outcomes_opened_in_cova3d"
] = False

state[
    "factorial_training_authorized"
] = False

state[
    "final_outer_cv_access_in_cova3d"
] = 0

state[
    "next_block"
] = "09E-SANITY-EVAL-OUTCOME"

state[
    "updated_at_utc"
] = NOW_ISO


write_json(
    STATE_PATH,
    state,
)


project_state[
    "last_completed_block"
] = "09E-SANITY-EVAL-PRED-FREEZE"

project_state[
    "cova3d_dense_outcomes_opened"
] = False

project_state[
    "cova3d_factorial_training_authorized"
] = False

project_state[
    "cova3d_final_outer_cv_access"
] = 0

project_state[
    "updated_at_utc"
] = NOW_ISO


write_json(
    PROJECT_STATE_PATH,
    project_state,
)


git(
    "add",
    "COVA3D_STATE.json",
    "PROJECT_STATE.json",
    "configs/cova3d_sanity_inference_implementation_lock_v1_0.yaml",
    "configs/cova3d_sanity_outcome_evaluation_lock_v1_0.yaml",
    "data/manifests/cova3d_sanity_R2_prediction_freeze_v1_0.csv",
    "experiments/audits/block09e_sanity_prediction_freeze.json",
)


check = sh(
    [
        "git",
        "diff",
        "--cached",
        "--check",
    ],
    check=False,
)


if check.returncode != 0:

    raise RuntimeError(
        check.stdout
        + check.stderr
    )


git(
    "commit",
    "-m",
    "eval: freeze COVA-3D R2 sanity predictions before dense outcomes",
)


prefreeze_commit = git(
    "rev-parse",
    "HEAD",
)


token, askpass, git_env = git_auth()


push(
    git_env,
    token,
)


if git(
    "status",
    "--porcelain",
):

    raise RuntimeError(
        "Repository not clean after pre-outcome freeze."
    )


print(
    "✓ Prediction/rule freeze commit    :",
    prefreeze_commit[:12],
)

print(
    "✓ GitHub push                      : PASS"
)

print(
    "✓ Dense masks opened so far        : 0"
)


# ==========================================================================================
# 5. OPEN EXACTLY FOUR DEVELOPMENT INFECTION MASKS
# ==========================================================================================

heading(
    "STEP 4/6 — LOCKED FOUR-CASE DENSE SANITY EVALUATION"
)


geometry_df = pd.read_csv(
    GRID_GEOMETRY_PATH
)


geometry_lookup = {
    str(row["case_id"]):
        row
    for _, row in geometry_df.iterrows()
}


split_lookup = {
    str(row["case_id"]):
        row
    for _, row in split_df.iterrows()
}


pred_lookup = {
    str(row["case_id"]):
        row
    for _, row in pred_df.iterrows()
}


case_rows = []


for case_id in dev_cases:

    split_row = split_lookup[
        case_id
    ]

    geometry_row = geometry_lookup[
        case_id
    ]

    pred_row = pred_lookup[
        case_id
    ]


    # ------------------------------------------------------------------
    # Re-verify frozen prediction.
    # ------------------------------------------------------------------

    prediction_path = Path(
        str(
            pred_row[
                "prediction_file"
            ]
        )
    )


    if sha256_file(
        prediction_path
    ) != str(
        pred_row[
            "prediction_sha256"
        ]
    ):

        raise RuntimeError(
            "Frozen prediction changed for "
            + case_id
        )


    with np.load(
        prediction_path,
        allow_pickle=False,
    ) as prediction_npz:

        probability = np.asarray(
            prediction_npz[
                "probability_zyx"
            ],
            dtype=np.float32,
        )


    if not np.isfinite(
        probability
    ).all():

        raise RuntimeError(
            "Non-finite prediction for "
            + case_id
        )


    # ------------------------------------------------------------------
    # DENSE OUTCOME ACCESS OCCURS HERE.
    # ONLY the permanent-development infection mask is opened.
    # ------------------------------------------------------------------

    mask_path = (
        DATASET_ROOT
        / str(
            split_row[
                "infection_mask"
            ]
        )
    )


    if not mask_path.exists():

        raise RuntimeError(
            "Development infection mask missing:\n"
            + str(mask_path)
        )


    native_mask_img = nib.load(
        str(mask_path),
        mmap=True,
    )


    canonical_mask_img = nib.as_closest_canonical(
        native_mask_img
    )


    resampled_mask_img = resample_to_output(
        canonical_mask_img,
        voxel_sizes=TARGET_SPACING_XYZ,
        order=0,
        mode="constant",
        cval=0,
    )


    resampled_mask_xyz = np.asarray(
        resampled_mask_img.dataobj
    )


    dense_full_zyx = np.transpose(
        resampled_mask_xyz,
        (
            2,
            1,
            0,
        ),
    ) > 0


    expected_full_shape = (
        int(
            geometry_row[
                "full_shape_z"
            ]
        ),
        int(
            geometry_row[
                "full_shape_y"
            ]
        ),
        int(
            geometry_row[
                "full_shape_x"
            ]
        ),
    )


    if tuple(
        dense_full_zyx.shape
    ) != expected_full_shape:

        raise RuntimeError(
            "Dense-mask resampled full-grid shape mismatch for "
            + case_id
            + "\nExpected: "
            + str(expected_full_shape)
            + "\nObserved: "
            + str(
                dense_full_zyx.shape
            )
        )


    origin = np.asarray(
        [
            int(
                geometry_row[
                    "crop_origin_z"
                ]
            ),
            int(
                geometry_row[
                    "crop_origin_y"
                ]
            ),
            int(
                geometry_row[
                    "crop_origin_x"
                ]
            ),
        ],
        dtype=np.int32,
    )


    crop_shape = np.asarray(
        [
            int(
                geometry_row[
                    "crop_shape_z"
                ]
            ),
            int(
                geometry_row[
                    "crop_shape_y"
                ]
            ),
            int(
                geometry_row[
                    "crop_shape_x"
                ]
            ),
        ],
        dtype=np.int32,
    )


    hi = (
        origin
        + crop_shape
    )


    dense_crop = dense_full_zyx[
        int(origin[0]):
        int(hi[0]),

        int(origin[1]):
        int(hi[1]),

        int(origin[2]):
        int(hi[2]),
    ]


    if tuple(
        dense_crop.shape
    ) != tuple(
        crop_shape.tolist()
    ):

        raise RuntimeError(
            "Dense crop shape mismatch."
        )


    if dense_crop.shape != probability.shape:

        raise RuntimeError(
            "Prediction/dense-mask crop shape mismatch for "
            + case_id
        )


    lesion_voxels = int(
        dense_crop.sum()
    )


    outside_voxels = int(
        (
            ~dense_crop
        ).sum()
    )


    if lesion_voxels == 0:

        raise RuntimeError(
            "Dense development lesion became empty for "
            + case_id
        )


    if outside_voxels == 0:

        raise RuntimeError(
            "No outside-lesion voxels for "
            + case_id
        )


    inside_mean = float(
        probability[
            dense_crop
        ].astype(
            np.float64
        ).mean()
    )


    outside_mean = float(
        probability[
            ~dense_crop
        ].astype(
            np.float64
        ).mean()
    )


    delta = (
        inside_mean
        - outside_mean
    )


    positive = bool(
        delta > 0.0
    )


    probability_std = float(
        probability.astype(
            np.float64
        ).std()
    )


    case_rows.append(
        {
            "case_id":
                case_id,

            "role":
                "permanent_development",

            "prediction_sha256":
                str(
                    pred_row[
                        "prediction_sha256"
                    ]
                ),

            "infection_mask_file":
                str(
                    split_row[
                        "infection_mask"
                    ]
                ),

            "infection_mask_sha256":
                sha256_file(
                    mask_path
                ),

            "lesion_voxels_in_crop":
                lesion_voxels,

            "outside_voxels_in_crop":
                outside_voxels,

            "mean_probability_inside_lesion":
                inside_mean,

            "mean_probability_outside_lesion":
                outside_mean,

            "probability_separation_delta":
                delta,

            "positive_separation":
                positive,

            "probability_std":
                probability_std,

            "prediction_std_pass":
                probability_std
                >= STD_MIN,
        }
    )


    print()
    print(case_id)

    print(
        "  lesion voxels     :",
        lesion_voxels,
    )

    print(
        "  P inside          :",
        f"{inside_mean:.8f}",
    )

    print(
        "  P outside         :",
        f"{outside_mean:.8f}",
    )

    print(
        "  delta             :",
        f"{delta:.8f}",
    )

    print(
        "  separation        :",
        "PASS"
        if positive
        else "FAIL",
    )


case_metrics = pd.DataFrame(
    case_rows
)


positive_cases = int(
    case_metrics[
        "positive_separation"
    ].sum()
)


separation_pass = bool(
    positive_cases
    >= POSITIVE_SEPARATION_MIN_CASES
)


std_pass = bool(
    case_metrics[
        "prediction_std_pass"
    ].all()
)


finite_probability_pass = bool(
    np.isfinite(
        case_metrics[
            [
                "mean_probability_inside_lesion",
                "mean_probability_outside_lesion",
                "probability_separation_delta",
                "probability_std",
            ]
        ].to_numpy(
            dtype=np.float64
        )
    ).all()
)


finite_gate_pass = bool(
    finite_training
    and finite_probability_pass
)


overall_gate_pass = bool(
    finite_gate_pass
    and loss_pass
    and zero_label_pass
    and std_pass
    and separation_pass
)


# ==========================================================================================
# 6. CLOSE SANITY GATE
# ==========================================================================================

heading(
    "STEP 5/6 — CLOSE LOCKED SANITY GATE"
)


print(
    "Finite loss/probabilities          :",
    "PASS"
    if finite_gate_pass
    else "FAIL",
)

print(
    "Loss ratio <= 0.95                :",
    "PASS"
    if loss_pass
    else "FAIL",
)

print(
    "Zero-label fraction <= 0.30       :",
    "PASS"
    if zero_label_pass
    else "FAIL",
)

print(
    "Prediction std >= 0.001 (4/4)     :",
    "PASS"
    if std_pass
    else "FAIL",
)

print(
    "Positive lesion separation        :",
    f"{positive_cases}/4",
)

print(
    "Required positive separation      : >=3/4"
)

print(
    "Separation criterion              :",
    "PASS"
    if separation_pass
    else "FAIL",
)

print()
print(
    "OVERALL SANITY GATE               :",
    "PASS"
    if overall_gate_pass
    else "FAIL",
)


CASE_METRICS_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


case_metrics.to_csv(
    CASE_METRICS_PATH,
    index=False,
)


gate_audit = {
    "project":
        "COVA-3D",

    "block":
        "09E-SANITY-EVAL",

    "status":
        (
            "SANITY_GATE_PASS"
            if overall_gate_pass
            else "SANITY_GATE_FAIL"
        ),

    "effective_protocol":
        EFFECTIVE_PROTOCOL,

    "model_sha256":
        MODEL_SHA,

    "condition":
        "C100_COH",

    "seed":
        17,

    "training_subcriteria":
        {
            "finite_training":
                finite_training,

            "first3_mean_loss":
                first3,

            "last3_mean_loss":
                last3,

            "loss_ratio":
                loss_ratio,

            "loss_ratio_max":
                LOSS_RATIO_MAX,

            "loss_ratio_pass":
                loss_pass,

            "zero_label_fraction":
                zero_label_fraction,

            "zero_label_max":
                ZERO_LABEL_MAX,

            "zero_label_pass":
                zero_label_pass,
        },

    "prediction_subcriteria":
        {
            "finite_probabilities":
                finite_probability_pass,

            "prediction_std_min":
                STD_MIN,

            "prediction_std_cases_passing":
                int(
                    case_metrics[
                        "prediction_std_pass"
                    ].sum()
                ),

            "prediction_std_pass":
                std_pass,
        },

    "dense_development_separation":
        {
            "cases_evaluated":
                4,

            "positive_cases":
                positive_cases,

            "minimum_required":
                POSITIVE_SEPARATION_MIN_CASES,

            "pass":
                separation_pass,
        },

    "overall_gate_pass":
        overall_gate_pass,

    "firewall":
        {
            "development_infection_masks_accessed":
                4,

            "development_lung_masks_accessed":
                0,

            "final_outer_cv_dense_masks_accessed":
                0,

            "optimizer_steps":
                0,

            "training_performed":
                False,

            "threshold_search":
                False,

            "Dice_computed":
                False,

            "IoU_computed":
                False,

            "FROC_computed":
                False,
        },

    "authorization":
        {
            "factorial_fitting":
                overall_gate_pass,

            "method_development":
                False,

            "final_outer_cv_dense_outcomes":
                False,
        },

    "next_block":
        (
            "10A-FACTORIAL-PREFLIGHT"
            if overall_gate_pass
            else "09E-SANITY-GATE-FAIL-AUDIT"
        ),

    "generated_at_utc":
        NOW_ISO,
}


write_json(
    GATE_AUDIT_PATH,
    gate_audit,
)


# ------------------------------------------------------------------
# State transition.
# ------------------------------------------------------------------

state[
    "last_completed_block"
] = "09E-SANITY-EVAL"

state[
    "current_stage"
] = (
    "SANITY_GATE_PASS_FACTORIAL_PREFLIGHT_PENDING"
    if overall_gate_pass
    else "SANITY_GATE_FAIL_AUDIT_PENDING"
)

state[
    "sanity_gate_status"
] = (
    "PASS"
    if overall_gate_pass
    else "FAIL"
)

state[
    "sanity_gate_positive_separation_cases"
] = positive_cases

state[
    "sanity_gate_loss_ratio"
] = loss_ratio

state[
    "sanity_gate_zero_label_fraction"
] = zero_label_fraction

state[
    "sanity_prediction_std_cases_passing"
] = int(
    case_metrics[
        "prediction_std_pass"
    ].sum()
)

state[
    "dense_outcomes_opened_in_cova3d"
] = True

state[
    "development_dense_masks_accessed_in_cova3d"
] = 4

state[
    "factorial_training_authorized"
] = overall_gate_pass

state[
    "method_development_authorized"
] = False

state[
    "final_outer_cv_outcomes_authorized"
] = False

state[
    "final_outer_cv_access_in_cova3d"
] = 0

state[
    "next_block"
] = (
    "10A-FACTORIAL-PREFLIGHT"
    if overall_gate_pass
    else "09E-SANITY-GATE-FAIL-AUDIT"
)

state[
    "next_action"
] = (
    "Run frozen factorial preflight; do not alter method or annotations."
    if overall_gate_pass
    else
    "Stop. Audit implementation only under the prospectively locked failure action."
)

state[
    "updated_at_utc"
] = NOW_ISO


write_json(
    STATE_PATH,
    state,
)


project_state[
    "last_completed_block"
] = "09E-SANITY-EVAL"

project_state[
    "current_stage"
] = (
        "cova3d_factorial_preflight_pending"
        if overall_gate_pass
        else "cova3d_sanity_gate_fail_audit_pending"
)

project_state[
    "cova3d_sanity_gate"
] = (
        "PASS"
        if overall_gate_pass
        else "FAIL"
)

project_state[
    "cova3d_dense_outcomes_opened"
] = True

project_state[
    "cova3d_factorial_training_authorized"
] = overall_gate_pass

project_state[
    "cova3d_final_outer_cv_access"
] = 0

project_state[
    "updated_at_utc"
] = NOW_ISO


write_json(
    PROJECT_STATE_PATH,
    project_state,
)


# ==========================================================================================
# 7. REGRESSION TEST + FINAL COMMIT
# ==========================================================================================

heading(
    "STEP 6/6 — REGRESSION + FINAL COMMIT"
)


write_text(
    TEST_PATH,
    r'''
from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_sanity_gate_case_table():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_sanity_gate_case_metrics_v1_0.csv"
    )

    assert len(frame) == 4

    assert (
        frame["role"]
        == "permanent_development"
    ).all()

    assert (
        frame["lesion_voxels_in_crop"]
        > 0
    ).all()

    assert (
        frame["outside_voxels_in_crop"]
        > 0
    ).all()


def test_sanity_gate_audit():

    audit = json.loads(
        (
            ROOT
            / "experiments/audits/"
            "block09e_sanity_gate_close.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert audit[
        "dense_development_separation"
    ][
        "cases_evaluated"
    ] == 4

    assert audit[
        "firewall"
    ][
        "development_infection_masks_accessed"
    ] == 4

    assert audit[
        "firewall"
    ][
        "development_lung_masks_accessed"
    ] == 0

    assert audit[
        "firewall"
    ][
        "final_outer_cv_dense_masks_accessed"
    ] == 0

    assert audit[
        "firewall"
    ][
        "optimizer_steps"
    ] == 0

    assert audit[
        "firewall"
    ][
        "threshold_search"
    ] is False

    assert audit[
        "firewall"
    ][
        "Dice_computed"
    ] is False


def test_state_matches_gate():

    audit = json.loads(
        (
            ROOT
            / "experiments/audits/"
            "block09e_sanity_gate_close.json"
        ).read_text(
            encoding="utf-8"
        )
    )

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
    ] == "09E-SANITY-EVAL"

    assert state[
        "sanity_gate_status"
    ] == (
        "PASS"
        if audit[
            "overall_gate_pass"
        ]
        else "FAIL"
    )

    assert state[
        "factorial_training_authorized"
    ] is audit[
        "overall_gate_pass"
    ]

    assert state[
        "final_outer_cv_access_in_cova3d"
    ] == 0
'''
)


tests = sh(
    [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_cova3d_sanity_gate_close.py",
        "-q",
        "-p",
        "no:cacheprovider",
    ],
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
        "Sanity-gate regression tests failed."
    )


# ------------------------------------------------------------------
# Capture this executed source if available.
# ------------------------------------------------------------------

source_capture = "NOT_AVAILABLE"


try:

    cell = (
        get_ipython()
        .history_manager
        .input_hist_raw[-1]
    )

    if (
        "# COVA-3D — BLOCK 09E-SANITY-EVAL-CLOSE-GATE"
        in cell
    ):

        source_path = (
            REPO
            / "scripts/code_blocks/"
            "block09e_sanity_eval_close_gate.py"
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

        source_capture = "PASS"

except Exception:
    pass


gate_audit[
    "source_capture"
] = source_capture


write_json(
    GATE_AUDIT_PATH,
    gate_audit,
)


git_paths = [
    "COVA3D_STATE.json",
    "PROJECT_STATE.json",

    "data/manifests/"
    "cova3d_sanity_gate_case_metrics_v1_0.csv",

    "experiments/audits/"
    "block09e_sanity_gate_close.json",

    "tests/"
    "test_cova3d_sanity_gate_close.py",
]


source_path = (
    REPO
    / "scripts/code_blocks/"
    "block09e_sanity_eval_close_gate.py"
)


if source_path.exists():

    git_paths.append(
        "scripts/code_blocks/"
        "block09e_sanity_eval_close_gate.py"
    )


git(
    "add",
    *git_paths,
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
        diff_check.stdout
        + diff_check.stderr
    )


staged = git(
    "diff",
    "--cached",
    "--name-only",
).splitlines()


if any(
    x.endswith(".pt")
    or x.endswith(".npz")
    for x in staged
):

    raise RuntimeError(
        "Runtime model/prediction accidentally staged."
    )


git(
    "commit",
    "-m",
    (
        "eval: close COVA-3D sanity gate "
        + (
            "PASS"
            if overall_gate_pass
            else "FAIL"
        )
    ),
)


gate_commit = git(
    "rev-parse",
    "HEAD",
)


push(
    git_env,
    token,
)


try:
    askpass.unlink(
        missing_ok=True
    )
except Exception:
    pass


final_status = git(
    "status",
    "--porcelain",
)


if final_status:

    raise RuntimeError(
        "Repository not clean after sanity-gate commit:\n"
        + final_status
    )


# ==========================================================================================
# FINAL REPORT
# ==========================================================================================

heading(
    "COVA-3D 09E-SANITY-EVAL — FINAL REPORT"
)


print(
    "Starting commit                    :",
    head[:12],
)

print(
    "Prediction-freeze commit           :",
    prefreeze_commit[:12],
)

print(
    "Sanity-gate commit                 :",
    gate_commit[:12],
)

print(
    "GitHub synchronization             : PASS"
)

print(
    "Repository                         : CLEAN"
)

print()
print(
    "LOCKED SANITY CRITERIA"
)
print(
    "----------------------"
)

print(
    "Finite loss/probabilities          :",
    "PASS"
    if finite_gate_pass
    else "FAIL",
)

print(
    "Loss ratio                        :",
    f"{loss_ratio:.8f}",
    "PASS"
    if loss_pass
    else "FAIL",
)

print(
    "Zero-label fraction               :",
    f"{zero_label_fraction:.8f}",
    "PASS"
    if zero_label_pass
    else "FAIL",
)

print(
    "Prediction std                    :",
    f"{int(case_metrics['prediction_std_pass'].sum())}/4",
    "PASS"
    if std_pass
    else "FAIL",
)

print(
    "Positive lesion separation        :",
    f"{positive_cases}/4",
    "PASS"
    if separation_pass
    else "FAIL",
)

print()


for _, row in case_metrics.iterrows():

    print(
        "{:28s} inside={:.8f}  outside={:.8f}  "
        "delta={:+.8f}  {}".format(
            str(row["case_id"]),
            float(
                row[
                    "mean_probability_inside_lesion"
                ]
            ),
            float(
                row[
                    "mean_probability_outside_lesion"
                ]
            ),
            float(
                row[
                    "probability_separation_delta"
                ]
            ),
            (
                "PASS"
                if bool(
                    row[
                        "positive_separation"
                    ]
                )
                else "FAIL"
            ),
        )
    )


print()
print(
    "OVERALL SANITY GATE               :",
    "PASS"
    if overall_gate_pass
    else "FAIL",
)

print()
print(
    "FIREWALL"
)
print(
    "--------"
)

print(
    "Development infection masks opened: 4"
)

print(
    "Development lung masks opened     : 0"
)

print(
    "Final outer-CV dense masks opened : 0"
)

print(
    "Dice / IoU / FROC computed        : NO"
)

print(
    "Threshold search                  : NO"
)

print(
    "optimizer.step() calls            : 0"
)

print()
print(
    "AUTHORIZATION"
)
print(
    "-------------"
)

print(
    "Factorial fitting authorized      :",
    "YES"
    if overall_gate_pass
    else "NO",
)

print(
    "Method development authorized     : NO"
)

print(
    "Final outer-CV dense outcomes     : NO"
)

print(
    "Next block                        :",
    state[
        "next_block"
    ],
)

print()
print(
    "Exact source captured             :",
    source_capture,
)

print()
print(
    "Send me this COMPLETE report."
)

print(
    "Do NOT begin the 72 factorial fits until I audit this gate result."
)

print(
    "=" * 128
)
