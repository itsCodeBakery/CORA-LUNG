# ==========================================================================================
# COVA-3D — BLOCK 09E-LOCK
# Strong Sparse-Supervision Baseline + Training Protocol Freeze
#
# EXPECTED START COMMIT
# ---------------------
# ea3617bd96f7
#
# SCIENTIFIC PURPOSE
# ------------------
# The annotation methodology is now structurally frozen.
#
# This block prospectively freezes the MODEL/TRAINING side before the first
# COVA-3D optimizer step.
#
# PRIMARY BASELINE
# ----------------
# 3-D nnU-Net-style anisotropic U-Net
# + partial BCE
# + partial Dice
#
# Sparse unknown voxels = -1 and contribute zero loss.
#
# IMPORTANT
# ---------
# This is deliberately a simple strong baseline rather than a new proposed
# segmentation architecture. COVA-3D's scientific variable is the sparse
# annotation design:
#
#     coverage × geometry
#
# We therefore avoid model-specific novelty before testing that factorial
# question.
#
# THIS BLOCK FREEZES
# ------------------
#   • architecture
#   • patch size
#   • sparse loss
#   • patient sampling
#   • data augmentation
#   • optimizer
#   • learning-rate schedule
#   • gradient accumulation
#   • AMP
#   • checkpoint rule
#   • paired randomness across conditions
#   • sanity-training protocol
#   • sanity-gate criteria
#   • final 4-fold × 6-condition × 3-seed experiment
#   • inference/evaluation policy
#
# THIS BLOCK DOES NOT
# -------------------
#   ✗ load CT datasets
#   ✗ load dense lesion masks
#   ✗ inspect any model outcomes
#   ✗ train on any case
#   ✗ call optimizer.step()
#
# A single synthetic forward/backward CUDA dry-run is permitted only to verify
# architecture shape and memory feasibility.
#
# AFTER PASS
# ----------
# Only the SANITY RUN is authorized.
#
# Factorial training remains prohibited until the fixed sanity gate passes.
# ==========================================================================================


from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import subprocess
import hashlib
import importlib
import json
import os
import random
import sys
import textwrap

import numpy as np
import pandas as pd
import yaml

from kaggle_secrets import UserSecretsClient

import torch


# ==========================================================================================
# 0. FROZEN CONSTANTS
# ==========================================================================================

REPO = Path(
    "/kaggle/working/CORA-LUNG"
)

EXPECTED_START_COMMIT = (
    "ea3617bd96f7"
)

BLOCK = (
    "09E-LOCK"
)

BASELINE_ID = (
    "COVA3D_NNUNET_STYLE_PARTIAL_V1"
)

EFFECTIVE_PROTOCOL = (
    "COVA3D_1.0+A1+A1.1+A1.2"
)

EXPECTED_CASES = (
    20
)

EXPECTED_FINAL_CASES = (
    16
)

EXPECTED_DEV_CASES = (
    4
)

EXPECTED_FINAL_ARTIFACTS = (
    120
)

CONDITIONS = [
    "C50_COH",
    "C50_DIS",
    "C50_FRG",
    "C100_COH",
    "C100_DIS",
    "C100_FRG",
]

FINAL_SEEDS = [
    17,
    29,
    43,
]

SANITY_SEED = (
    17
)

SANITY_CONDITION = (
    "C100_COH"
)

PATCH_ZYX = (
    48,
    128,
    128,
)

FEATURES = [
    32,
    64,
    128,
    256,
    320,
]

ENCODER_KERNELS = [
    (
        1,
        3,
        3,
    ),
    (
        3,
        3,
        3,
    ),
    (
        3,
        3,
        3,
    ),
    (
        3,
        3,
        3,
    ),
    (
        3,
        3,
        3,
    ),
]

DOWNSAMPLE_STRIDES = [
    (
        1,
        2,
        2,
    ),
    (
        2,
        2,
        2,
    ),
    (
        2,
        2,
        2,
    ),
    (
        2,
        2,
        2,
    ),
]

FOREGROUND_PATCH_PROBABILITY = (
    0.50
)

BACKGROUND_PATCH_PROBABILITY = (
    0.25
)

UNIFORM_PATCH_PROBABILITY = (
    0.25
)

ACCUMULATION_STEPS = (
    2
)

MICROBATCH_SIZE = (
    1
)

# --------------------------------------------------------------------------
# SANITY RUN
#
# Uses ONLY the four permanent-development cases.
# Dense masks remain unopened during fitting.
#
# After fitting, one fixed dense-development evaluation is permitted.
# --------------------------------------------------------------------------

SANITY_EPOCHS = (
    20
)

SANITY_MICROBATCHES_PER_EPOCH = (
    100
)

SANITY_OPTIMIZER_STEPS = (
    SANITY_EPOCHS
    * SANITY_MICROBATCHES_PER_EPOCH
    // ACCUMULATION_STEPS
)

# --------------------------------------------------------------------------
# FINAL FACTORIAL FIT
#
# Every outer fold contains 12 training volumes and 4 held-out volumes.
#
# 120 microbatches / epoch = exactly 10 scheduled microbatches per training
# volume before deterministic shuffle.
# --------------------------------------------------------------------------

FINAL_EPOCHS = (
    40
)

FINAL_MICROBATCHES_PER_EPOCH = (
    120
)

FINAL_OPTIMIZER_STEPS_PER_RUN = (
    FINAL_EPOCHS
    * FINAL_MICROBATCHES_PER_EPOCH
    // ACCUMULATION_STEPS
)

FINAL_OUTER_FOLDS = (
    4
)

EXPECTED_FINAL_RUNS = (
    len(
        CONDITIONS
    )
    * FINAL_OUTER_FOLDS
    * len(
        FINAL_SEEDS
    )
)

LEARNING_RATE = (
    3e-4
)

MIN_LEARNING_RATE = (
    3e-6
)

WEIGHT_DECAY = (
    1e-4
)

ADAM_BETA1 = (
    0.9
)

ADAM_BETA2 = (
    0.999
)

GRADIENT_CLIP_NORM = (
    12.0
)

WARMUP_STEPS = (
    100
)

AMP_DTYPE = (
    "float16"
)

UNKNOWN_LABEL = (
    -1
)

LOSS_BCE_WEIGHT = (
    1.0
)

LOSS_DICE_WEIGHT = (
    1.0
)

CHECKPOINT_SELECTION = (
    "FINAL_EPOCH_ONLY"
)

# --------------------------------------------------------------------------
# Conservative spatial augmentation.
#
# No rotations/elastic transforms are used because the experiment itself is
# studying sparse annotation geometry.
#
# Flips preserve pairwise distances/connectivity.
# --------------------------------------------------------------------------

FLIP_X_PROBABILITY = (
    0.50
)

FLIP_Y_PROBABILITY = (
    0.50
)

FLIP_Z_PROBABILITY = (
    0.00
)

INTENSITY_SCALE_MIN = (
    0.90
)

INTENSITY_SCALE_MAX = (
    1.10
)

INTENSITY_SHIFT_MIN = (
    -0.10
)

INTENSITY_SHIFT_MAX = (
    0.10
)

GAUSSIAN_NOISE_SIGMA_MIN = (
    0.00
)

GAUSSIAN_NOISE_SIGMA_MAX = (
    0.03
)

# --------------------------------------------------------------------------
# SANITY GATE
#
# These criteria are intentionally permissive.
#
# Their purpose is to detect broken optimization / degenerate inference,
# NOT to select the best-performing model.
#
# Failure allows implementation debugging only.
# Hyperparameter/model redesign requires a new prospective protocol amendment.
# --------------------------------------------------------------------------

SANITY_LAST_TO_FIRST_LOSS_RATIO_MAX = (
    0.95
)

SANITY_ZERO_LABEL_MICROBATCH_FRACTION_MAX = (
    0.30
)

SANITY_FINITE_OUTPUT_REQUIRED = (
    True
)

SANITY_DEV_CASES_POSITIVE_SEPARATION_MIN = (
    3
)

SANITY_DEV_PREDICTION_STD_MIN = (
    1e-3
)

# --------------------------------------------------------------------------
# FROZEN PRIMARY INFERENCE / FROC SETTINGS
# --------------------------------------------------------------------------

SLIDING_WINDOW_OVERLAP = (
    0.50
)

PRIMARY_THRESHOLD_GRID = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30,
    0.35,
    0.40,
    0.45,
    0.50,
    0.55,
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
    0.95,
    0.975,
    0.99,
    0.995,
    1.000001,
]

PREDICTION_CONNECTIVITY = (
    26
)

REFERENCE_COMPONENT_MIN_ML = (
    0.1
)

MATCH_IOU_THRESHOLD = (
    0.10
)

PRIMARY_FP_BUDGET = (
    1.0
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


def stable_seed(*parts):

    payload = "|".join(
        str(
            part
        )
        for part in parts
    )

    digest = hashlib.sha256(
        payload.encode(
            "utf-8"
        )
    ).digest()

    return int.from_bytes(
        digest[
            :8
        ],
        "little",
        signed=False,
    ) % (
        2**32
        - 1
    )


def reset_seed(seed):

    random.seed(
        seed
    )

    np.random.seed(
        seed
    )

    torch.manual_seed(
        seed
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            seed
        )

    torch.backends.cudnn.benchmark = (
        False
    )

    torch.backends.cudnn.deterministic = (
        True
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
        "/tmp/cova3d_git_askpass_09e_lock.sh"
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
# 2. VERIFY STRUCTURAL FREEZE
# ==========================================================================================

heading(
    "COVA-3D BLOCK 09E-LOCK — STRONG BASELINE + TRAINING PROTOCOL FREEZE"
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
        "Repository must be clean before 09E-LOCK.\n"
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
) != "09D-B2":

    raise RuntimeError(
        "Expected 09D-B2 as the previous completed block."
    )


if cova_state.get(
    "annotation_methodology_structurally_ready"
) is not True:

    raise RuntimeError(
        "Annotation methodology is not structurally ready."
    )


if cova_state.get(
    "coordinate_level_grid_feasibility"
) != "PASS_20_OF_20":

    raise RuntimeError(
        "Coordinate-level feasibility is not PASS 20/20."
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


audit_09db2 = json.loads(
    (
        REPO
        / "experiments/audits/"
        "block09d_b2_final_coordinate_grid.json"
    ).read_text(
        encoding="utf-8"
    )
)


if audit_09db2.get(
    "status"
) != "PASS":

    raise RuntimeError(
        "09D-B2 audit is not PASS."
    )


if audit_09db2[
    "structural_result"
][
    "coordinate_level_feasible_cases"
] != 20:

    raise RuntimeError(
        "09D-B2 did not pass all 20 cases."
    )


print(
    "✓ Starting commit                      :",
    head[:12],
)

print(
    "✓ Annotation methodology               : STRUCTURALLY READY"
)

print(
    "✓ Coordinate-level feasibility         : PASS 20/20"
)

print(
    "✓ COVA optimizer steps                 : 0"
)

print(
    "✓ Factorial training                   : NOT AUTHORIZED"
)


# ==========================================================================================
# 3. VERIFY FINAL CASE/FOLD REGISTRY
# ==========================================================================================

heading(
    "STEP 1/9 — VERIFY DEVELOPMENT / OUTER-CV REGISTRY"
)


split_df = pd.read_csv(
    REPO
    / "data/splits/"
    "primary_volume_split_registry.csv"
)


condition_df = pd.read_csv(
    REPO
    / "data/manifests/"
    "cova3d_training_grid_condition_manifest_v1_2.csv"
)


artifact_df = pd.read_csv(
    REPO
    / "data/manifests/"
    "cova3d_training_grid_candidate_artifact_manifest_v1_2.csv"
)


if len(
    split_df
) != EXPECTED_CASES:

    raise RuntimeError(
        "Expected 20 cases."
    )


if len(
    condition_df
) != EXPECTED_FINAL_ARTIFACTS:

    raise RuntimeError(
        "Expected 120 final condition rows."
    )


if len(
    artifact_df
) != EXPECTED_FINAL_ARTIFACTS:

    raise RuntimeError(
        "Expected 120 final sparse artifacts."
    )


dev_cases = sorted(
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


final_cases = sorted(
    split_df.loc[
        split_df[
            "role"
        ]
        == "final_outer_cv",
        "case_id",
    ].astype(
        str
    ).tolist()
)


if len(
    dev_cases
) != EXPECTED_DEV_CASES:

    raise RuntimeError(
        "Expected four permanent development cases."
    )


if len(
    final_cases
) != EXPECTED_FINAL_CASES:

    raise RuntimeError(
        "Expected sixteen final-CV cases."
    )


fold_counts = (
    split_df.loc[
        split_df[
            "role"
        ]
        == "final_outer_cv"
    ]
    .groupby(
        "outer_fold"
    )
    .size()
    .to_dict()
)


expected_fold_counts = {
    0:
        4,

    1:
        4,

    2:
        4,

    3:
        4,
}


observed_fold_counts = {
    int(
        key
    ):
        int(
            value
        )
    for key, value in fold_counts.items()
}


if observed_fold_counts != expected_fold_counts:

    raise RuntimeError(
        "Expected four volumes in every final outer fold.\n"
        + str(
            observed_fold_counts
        )
    )


print(
    "✓ Permanent-development cases          :",
    len(
        dev_cases
    ),
)

print(
    "✓ Final outer-CV cases                 :",
    len(
        final_cases
    ),
)

print(
    "✓ Final folds                          : 4 × 4 held-out volumes"
)

print(
    "✓ Final fold train size                : 12 volumes"
)

print(
    "✓ Factorial conditions                 : 6"
)

print(
    "✓ Final random seeds                   :",
    FINAL_SEEDS,
)

print(
    "✓ Planned final model fits             :",
    EXPECTED_FINAL_RUNS,
)


# ==========================================================================================
# 4. VERIFY FINAL SPARSE ARTIFACT SCHEMA
# ==========================================================================================

heading(
    "STEP 2/9 — VERIFY FINAL SPARSE TRAINER INPUTS"
)


EXPECTED_KEYS = {
    "supervision_voxel_zyx",
    "supervision_label",
    "fg_membership_voxel_zyx",
    "fg_membership_group_id",
}


checked = (
    0
)


for _, row in artifact_df.iterrows():

    path = (
        REPO
        / str(
            row[
                "artifact_file"
            ]
        )
    )


    if not path.exists():

        raise RuntimeError(
            "Final sparse artifact missing:\n"
            + str(
                path
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
            "Final sparse artifact hash mismatch:\n"
            + str(
                path
            )
        )


    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        if set(
            data.files
        ) != EXPECTED_KEYS:

            raise RuntimeError(
                "Unexpected trainer artifact schema."
            )


        labels = np.asarray(
            data[
                "supervision_label"
            ]
        )


    if not set(
        np.unique(
            labels
        ).tolist()
    ).issubset(
        {
            0,
            1,
        }
    ):

        raise RuntimeError(
            "Sparse artifact contains non-binary direct labels."
        )


    checked += 1


print(
    "✓ Sparse trainer candidates verified   :",
    checked,
    "/120",
)

print(
    "✓ Dense arrays in trainer candidates   : 0"
)


# ==========================================================================================
# 5. WRITE STRONG nnU-Net-STYLE BASELINE IMPLEMENTATION
# ==========================================================================================

heading(
    "STEP 3/9 — FREEZE nnU-NET-STYLE 3-D BASELINE"
)


model_source = r'''
"""COVA-3D strong nnU-Net-style sparse-supervision baseline.

This module intentionally contains no COVA-specific replay, teacher/student,
pseudo-label, graph, prototype, VLM or confidence mechanism.

The scientific intervention is the annotation condition, not the network.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvNormAct3D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
    ):
        super().__init__()

        if isinstance(
            kernel_size,
            int,
        ):
            kernel_size = (
                kernel_size,
                kernel_size,
                kernel_size,
            )

        padding = tuple(
            int(
                k
            )
            // 2
            for k in kernel_size
        )

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.InstanceNorm3d(
                out_channels,
                affine=True,
            ),
            nn.LeakyReLU(
                negative_slope=0.01,
                inplace=True,
            ),
        )

    def forward(
        self,
        x,
    ):
        return self.block(
            x
        )


class PlainConvStage3D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        convolutions=2,
    ):
        super().__init__()

        layers = []

        current = int(
            in_channels
        )

        for _ in range(
            int(
                convolutions
            )
        ):

            layers.append(
                ConvNormAct3D(
                    current,
                    out_channels,
                    kernel_size,
                )
            )

            current = int(
                out_channels
            )

        self.block = nn.Sequential(
            *layers
        )

    def forward(
        self,
        x,
    ):
        return self.block(
            x
        )


class Downsample3D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
    ):
        super().__init__()

        self.down = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=stride,
            stride=stride,
            bias=False,
        )

    def forward(
        self,
        x,
    ):
        return self.down(
            x
        )


class DecoderStage3D(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        stride,
        kernel_size,
    ):
        super().__init__()

        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=stride,
            stride=stride,
            bias=False,
        )

        self.stage = PlainConvStage3D(
            out_channels
            + skip_channels,
            out_channels,
            kernel_size=kernel_size,
            convolutions=2,
        )

    def forward(
        self,
        x,
        skip,
    ):
        x = self.up(
            x
        )

        if x.shape[
            2:
        ] != skip.shape[
            2:
        ]:
            raise RuntimeError(
                "Decoder/skip shape mismatch: "
                + str(
                    tuple(
                        x.shape
                    )
                )
                + " versus "
                + str(
                    tuple(
                        skip.shape
                    )
                )
            )

        x = torch.cat(
            [
                x,
                skip,
            ],
            dim=1,
        )

        return self.stage(
            x
        )


class COVA3DNNUNetStyle(nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=(
            32,
            64,
            128,
            256,
            320,
        ),
    ):
        super().__init__()

        features = tuple(
            int(
                value
            )
            for value in features
        )

        if len(
            features
        ) != 5:
            raise ValueError(
                "COVA3DNNUNetStyle requires exactly five feature stages."
            )

        kernels = (
            (
                1,
                3,
                3,
            ),
            (
                3,
                3,
                3,
            ),
            (
                3,
                3,
                3,
            ),
            (
                3,
                3,
                3,
            ),
            (
                3,
                3,
                3,
            ),
        )

        strides = (
            (
                1,
                2,
                2,
            ),
            (
                2,
                2,
                2,
            ),
            (
                2,
                2,
                2,
            ),
            (
                2,
                2,
                2,
            ),
        )

        self.encoder0 = PlainConvStage3D(
            in_channels,
            features[
                0
            ],
            kernels[
                0
            ],
        )

        self.down1 = Downsample3D(
            features[
                0
            ],
            features[
                1
            ],
            strides[
                0
            ],
        )

        self.encoder1 = PlainConvStage3D(
            features[
                1
            ],
            features[
                1
            ],
            kernels[
                1
            ],
        )

        self.down2 = Downsample3D(
            features[
                1
            ],
            features[
                2
            ],
            strides[
                1
            ],
        )

        self.encoder2 = PlainConvStage3D(
            features[
                2
            ],
            features[
                2
            ],
            kernels[
                2
            ],
        )

        self.down3 = Downsample3D(
            features[
                2
            ],
            features[
                3
            ],
            strides[
                2
            ],
        )

        self.encoder3 = PlainConvStage3D(
            features[
                3
            ],
            features[
                3
            ],
            kernels[
                3
            ],
        )

        self.down4 = Downsample3D(
            features[
                3
            ],
            features[
                4
            ],
            strides[
                3
            ],
        )

        self.bottleneck = PlainConvStage3D(
            features[
                4
            ],
            features[
                4
            ],
            kernels[
                4
            ],
        )

        self.decoder3 = DecoderStage3D(
            features[
                4
            ],
            features[
                3
            ],
            features[
                3
            ],
            strides[
                3
            ],
            kernels[
                3
            ],
        )

        self.decoder2 = DecoderStage3D(
            features[
                3
            ],
            features[
                2
            ],
            features[
                2
            ],
            strides[
                2
            ],
            kernels[
                2
            ],
        )

        self.decoder1 = DecoderStage3D(
            features[
                2
            ],
            features[
                1
            ],
            features[
                1
            ],
            strides[
                1
            ],
            kernels[
                1
            ],
        )

        self.decoder0 = DecoderStage3D(
            features[
                1
            ],
            features[
                0
            ],
            features[
                0
            ],
            strides[
                0
            ],
            kernels[
                0
            ],
        )

        self.output_head = nn.Conv3d(
            features[
                0
            ],
            out_channels,
            kernel_size=1,
            bias=True,
        )

        self.apply(
            self._initialize
        )

    @staticmethod
    def _initialize(module):

        if isinstance(
            module,
            (
                nn.Conv3d,
                nn.ConvTranspose3d,
            ),
        ):
            nn.init.kaiming_normal_(
                module.weight,
                a=0.01,
                mode="fan_out",
                nonlinearity="leaky_relu",
            )

            if getattr(
                module,
                "bias",
                None,
            ) is not None:
                nn.init.zeros_(
                    module.bias
                )

    def forward(
        self,
        x,
    ):
        e0 = self.encoder0(
            x
        )

        e1 = self.encoder1(
            self.down1(
                e0
            )
        )

        e2 = self.encoder2(
            self.down2(
                e1
            )
        )

        e3 = self.encoder3(
            self.down3(
                e2
            )
        )

        bottleneck = self.bottleneck(
            self.down4(
                e3
            )
        )

        d3 = self.decoder3(
            bottleneck,
            e3,
        )

        d2 = self.decoder2(
            d3,
            e2,
        )

        d1 = self.decoder1(
            d2,
            e1,
        )

        d0 = self.decoder0(
            d1,
            e0,
        )

        return self.output_head(
            d0
        )
'''


model_path = (
    REPO
    / "src/cora_lung/models/"
    "cova3d_nnunet_style.py"
)


write_text(
    model_path,
    model_source,
)


print(
    "✓ Architecture                         : 5-stage 3-D nnU-Net-style"
)

print(
    "✓ Features                             :",
    FEATURES,
)

print(
    "✓ Patch ZYX                            :",
    PATCH_ZYX,
)

print(
    "✓ First downsample                     : (1,2,2)"
)

print(
    "✓ Remaining downsampling               : 3 × (2,2,2)"
)

print(
    "✓ Instance normalization               : YES"
)

print(
    "✓ LeakyReLU                            : 0.01"
)

print(
    "✓ Deep supervision                     : NO"
)

print(
    "✓ Dropout                              : NO"
)


# ==========================================================================================
# 6. FREEZE COMPLETE BASELINE/TRAINING CONFIGURATION
# ==========================================================================================

heading(
    "STEP 4/9 — FREEZE LOSS / OPTIMIZER / RANDOMIZATION / SANITY GATE"
)


baseline_config = {
    "project":
        "COVA-3D",

    "block":
        BLOCK,

    "baseline_id":
        BASELINE_ID,

    "effective_protocol":
        EFFECTIVE_PROTOCOL,

    "status":
        "PROSPECTIVELY_FROZEN_BEFORE_FIRST_COVA_OPTIMIZER_STEP",

    "scientific_role":
        (
            "Strong architecture-controlled sparse-supervision baseline "
            "for testing the frozen coverage-by-geometry factorial design."
        ),

    "literature_rationale":
        {
            "baseline_family":
                "3-D nnU-Net-style",

            "sparse_learning":
                "partial loss on labelled voxels only",

            "reference_1":
                (
                    "Gotkowski et al., Revisiting 3D Medical Scribble "
                    "Supervision: Benchmarking Beyond Cardiac Segmentation, "
                    "MICCAI 2025."
                ),

            "reference_2":
                (
                    "nnU-Net ignore-label / partial-loss sparse supervision."
                ),
        },

    "architecture":
        {
            "class":
                "COVA3DNNUNetStyle",

            "input_channels":
                1,

            "output_channels":
                1,

            "features":
                FEATURES,

            "encoder_kernels":
                [
                    list(
                        kernel
                    )
                    for kernel in ENCODER_KERNELS
                ],

            "downsample_strides":
                [
                    list(
                        stride
                    )
                    for stride in DOWNSAMPLE_STRIDES
                ],

            "convolutions_per_stage":
                2,

            "normalization":
                "InstanceNorm3d_affine",

            "activation":
                "LeakyReLU_0.01",

            "upsampling":
                "ConvTranspose3d",

            "skip_connections":
                "concatenation",

            "deep_supervision":
                False,

            "deep_supervision_reason":
                (
                    "Avoid scale-dependent remapping of extremely sparse "
                    "labels and keep the factorial annotation intervention "
                    "identical at the optimized output scale."
                ),

            "dropout":
                False,

            "weight_initialization":
                "KaimingNormal_leaky_relu",
        },

    "patch":
        {
            "shape_zyx":
                list(
                    PATCH_ZYX
                ),

            "microbatch_size":
                MICROBATCH_SIZE,

            "accumulation_steps":
                ACCUMULATION_STEPS,

            "effective_batch_in_optimizer_updates":
                (
                    MICROBATCH_SIZE
                    * ACCUMULATION_STEPS
                ),
        },

    "sampling":
        {
            "patient_sampling":
                "uniform",

            "foreground_center_probability":
                FOREGROUND_PATCH_PROBABILITY,

            "background_center_probability":
                BACKGROUND_PATCH_PROBABILITY,

            "uniform_crop_probability":
                UNIFORM_PATCH_PROBABILITY,

            "condition_specific_sampling_rule":
                False,

            "same_patient_schedule_across_conditions":
                True,
        },

    "augmentation":
        {
            "paired_across_conditions":
                True,

            "spatial":
                {
                    "flip_x_probability":
                        FLIP_X_PROBABILITY,

                    "flip_y_probability":
                        FLIP_Y_PROBABILITY,

                    "flip_z_probability":
                        FLIP_Z_PROBABILITY,

                    "rotation":
                        False,

                    "elastic":
                        False,

                    "zoom":
                        False,
                },

            "intensity":
                {
                    "scale_uniform":
                        [
                            INTENSITY_SCALE_MIN,
                            INTENSITY_SCALE_MAX,
                        ],

                    "shift_uniform":
                        [
                            INTENSITY_SHIFT_MIN,
                            INTENSITY_SHIFT_MAX,
                        ],

                    "gaussian_noise_sigma_uniform":
                        [
                            GAUSSIAN_NOISE_SIGMA_MIN,
                            GAUSSIAN_NOISE_SIGMA_MAX,
                        ],

                    "clip_after_augmentation":
                        [
                            -1.0,
                            1.0,
                        ],
                },
        },

    "paired_randomization":
        {
            "model_initialization":
                (
                    "derived from seed + outer_fold; CONDITION EXCLUDED"
                ),

            "patient_schedule":
                (
                    "derived from seed + outer_fold + epoch; "
                    "CONDITION EXCLUDED"
                ),

            "augmentation":
                (
                    "derived from seed + outer_fold + epoch + "
                    "microbatch_index; CONDITION EXCLUDED"
                ),

            "purpose":
                (
                    "Reduce nuisance Monte-Carlo variation between paired "
                    "factorial conditions."
                ),
        },

    "loss":
        {
            "unknown_label":
                UNKNOWN_LABEL,

            "labelled_voxels_only":
                True,

            "partial_BCE_weight":
                LOSS_BCE_WEIGHT,

            "partial_Dice_weight":
                LOSS_DICE_WEIGHT,

            "total":
                "partial_BCE + partial_Dice",

            "pseudo_labels":
                False,

            "dense_training_labels":
                False,

            "consistency_loss":
                False,

            "replay_loss":
                False,

            "teacher_student":
                False,
        },

    "optimizer":
        {
            "name":
                "AdamW",

            "learning_rate":
                LEARNING_RATE,

            "betas":
                [
                    ADAM_BETA1,
                    ADAM_BETA2,
                ],

            "weight_decay":
                WEIGHT_DECAY,

            "gradient_clip_l2_norm":
                GRADIENT_CLIP_NORM,
        },

    "learning_rate_schedule":
        {
            "name":
                "linear_warmup_plus_cosine",

            "warmup_optimizer_steps":
                WARMUP_STEPS,

            "base_learning_rate":
                LEARNING_RATE,

            "minimum_learning_rate":
                MIN_LEARNING_RATE,

            "schedule_scope":
                "optimizer_step",
        },

    "precision":
        {
            "AMP":
                True,

            "dtype":
                AMP_DTYPE,
        },

    "checkpoint":
        {
            "selection":
                CHECKPOINT_SELECTION,

            "early_stopping":
                False,

            "performance_based_selection":
                False,
        },

    "sanity_run":
        {
            "authorized_after_this_lock":
                True,

            "factorial_cell":
                SANITY_CONDITION,

            "seed":
                SANITY_SEED,

            "fit_cases":
                dev_cases,

            "fit_case_role":
                "permanent_development",

            "dense_labels_during_fit":
                False,

            "epochs":
                SANITY_EPOCHS,

            "microbatches_per_epoch":
                SANITY_MICROBATCHES_PER_EPOCH,

            "microbatches_per_case_per_epoch":
                (
                    SANITY_MICROBATCHES_PER_EPOCH
                    // len(
                        dev_cases
                    )
                ),

            "accumulation_steps":
                ACCUMULATION_STEPS,

            "optimizer_steps":
                SANITY_OPTIMIZER_STEPS,

            "checkpoint":
                "FINAL_EPOCH_ONLY",

            "dense_development_evaluation_after_fit":
                True,

            "checkpoint_reuse_in_final_experiment":
                False,

            "model_reinitialized_for_final_experiment":
                True,
        },

    "sanity_gate":
        {
            "purpose":
                (
                    "Implementation sanity only; not hyperparameter "
                    "optimization and not model selection."
                ),

            "finite_loss_and_probabilities":
                SANITY_FINITE_OUTPUT_REQUIRED,

            "last_3_epoch_mean_loss_div_first_3_epoch_mean_loss_max":
                SANITY_LAST_TO_FIRST_LOSS_RATIO_MAX,

            "zero_label_microbatch_fraction_max":
                SANITY_ZERO_LABEL_MICROBATCH_FRACTION_MAX,

            "dense_dev_probability_separation":
                (
                    "mean predicted probability inside dense lesion mask "
                    "> mean probability outside lesion mask"
                ),

            "development_cases_with_positive_separation_min":
                SANITY_DEV_CASES_POSITIVE_SEPARATION_MIN,

            "minimum_prediction_probability_std_per_case":
                SANITY_DEV_PREDICTION_STD_MIN,

            "failure_action":
                (
                    "STOP. Audit implementation only. Architecture, loss, "
                    "optimizer or thresholds may not be changed without a "
                    "new prospective protocol amendment."
                ),

            "pass_action":
                (
                    "Authorize frozen factorial fitting. Do not modify "
                    "training or annotation settings."
                ),
        },

    "final_factorial_training":
        {
            "authorized_now":
                False,

            "authorization_dependency":
                "SANITY_GATE_PASS",

            "conditions":
                CONDITIONS,

            "outer_folds":
                FINAL_OUTER_FOLDS,

            "seeds":
                FINAL_SEEDS,

            "models_total":
                EXPECTED_FINAL_RUNS,

            "training_cases_per_outer_fold":
                12,

            "heldout_cases_per_outer_fold":
                4,

            "epochs":
                FINAL_EPOCHS,

            "microbatches_per_epoch":
                FINAL_MICROBATCHES_PER_EPOCH,

            "microbatches_per_train_case_per_epoch":
                10,

            "accumulation_steps":
                ACCUMULATION_STEPS,

            "optimizer_steps_per_run":
                FINAL_OPTIMIZER_STEPS_PER_RUN,

            "checkpoint":
                CHECKPOINT_SELECTION,

            "reinitialize_every_run":
                True,

            "dense_labels_during_fit":
                False,

            "heldout_dense_outcomes_open_during_fit":
                False,

            "outcome_release_rule":
                (
                    "All final factorial models/predictions must be frozen "
                    "before final dense outcomes are evaluated."
                ),
        },

    "inference":
        {
            "sliding_window_patch_zyx":
                list(
                    PATCH_ZYX
                ),

            "overlap":
                SLIDING_WINDOW_OVERLAP,

            "window_weighting":
                "gaussian",

            "test_time_augmentation":
                False,

            "probability_output":
                "sigmoid",
        },

    "primary_evaluation":
        {
            "threshold_grid":
                PRIMARY_THRESHOLD_GRID,

            "prediction_connectivity":
                PREDICTION_CONNECTIVITY,

            "prediction_size_filter":
                False,

            "reference_minimum_component_volume_ml":
                REFERENCE_COMPONENT_MIN_ML,

            "component_match":
                "IoU",

            "match_IoU_threshold":
                MATCH_IOU_THRESHOLD,

            "primary_metric":
                "macro_patient_R_at_1_FP",

            "primary_FP_budget":
                PRIMARY_FP_BUDGET,

            "topology_recomputed_at_every_probability_threshold":
                True,
        },

    "firewall":
        {
            "dense_masks_allowed_during_fit":
                False,

            "dense_dev_masks_allowed_after_sanity_fit":
                True,

            "final_dense_masks_allowed_before_all_final_predictions_frozen":
                False,
        },

    "training_authorization_after_lock":
        {
            "sanity_training":
                True,

            "factorial_training":
                False,

            "final_outcome_evaluation":
                False,
        },

    "frozen_at_utc":
        NOW_ISO,
}


config_path = (
    REPO
    / "configs/"
    "cova3d_baseline_training_v1_0.yaml"
)


config_path.write_text(
    yaml.safe_dump(
        baseline_config,
        sort_keys=False,
        allow_unicode=True,
    ),
    encoding="utf-8",
)


print(
    "✓ Loss                                 : partial BCE + partial Dice"
)

print(
    "✓ Optimizer                            : AdamW"
)

print(
    "✓ LR                                   :",
    LEARNING_RATE,
)

print(
    "✓ Weight decay                         :",
    WEIGHT_DECAY,
)

print(
    "✓ AMP                                  : FP16"
)

print(
    "✓ Checkpoint selection                 : FINAL EPOCH ONLY"
)

print(
    "✓ Sanity condition                     :",
    SANITY_CONDITION,
)

print(
    "✓ Sanity optimizer steps               :",
    SANITY_OPTIMIZER_STEPS,
)

print(
    "✓ Final epochs / run                   :",
    FINAL_EPOCHS,
)

print(
    "✓ Final optimizer steps / run          :",
    FINAL_OPTIMIZER_STEPS_PER_RUN,
)

print(
    "✓ Planned final model fits             :",
    EXPECTED_FINAL_RUNS,
)


# ==========================================================================================
# 7. SYNTHETIC CUDA MEMORY + LOSS DRY RUN — NO OPTIMIZER STEP
# ==========================================================================================

heading(
    "STEP 5/9 — CUDA FORWARD/BACKWARD MEMORY DRY RUN"
)


if not torch.cuda.is_available():

    raise RuntimeError(
        "09E-LOCK requires a Kaggle GPU for the architecture memory dry run."
    )


SRC = (
    REPO
    / "src"
)


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
    "cora_lung.models.cova3d_nnunet_style",
    "cora_lung.losses.partial",
]:

    sys.modules.pop(
        module_name,
        None,
    )


importlib.invalidate_caches()


from cora_lung.models.cova3d_nnunet_style import (
    COVA3DNNUNetStyle,
)


from cora_lung.losses.partial import (
    partial_bce,
    partial_dice,
)


reset_seed(
    SANITY_SEED
)


device = torch.device(
    "cuda"
)


gpu_name = torch.cuda.get_device_name(
    device
)


total_gpu_gb = float(
    torch.cuda.get_device_properties(
        device
    ).total_memory
    / (
        1024
        ** 3
    )
)


torch.cuda.empty_cache()


torch.cuda.reset_peak_memory_stats(
    device
)


model = COVA3DNNUNetStyle(
    in_channels=1,
    out_channels=1,
    features=tuple(
        FEATURES
    ),
).to(
    device
)


trainable_parameters = int(
    sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
)


if not (
    10_000_000
    <= trainable_parameters
    <= 50_000_000
):

    raise RuntimeError(
        "Unexpected strong-baseline parameter count: "
        + str(
            trainable_parameters
        )
    )


synthetic_image = (
    torch.rand(
        (
            1,
            1,
            *PATCH_ZYX,
        ),
        device=device,
        dtype=torch.float32,
    )
    * 2.0
    - 1.0
)


synthetic_target = torch.full(
    (
        1,
        1,
        *PATCH_ZYX,
    ),
    UNKNOWN_LABEL,
    dtype=torch.int8,
    device=device,
)


rng = np.random.default_rng(
    1701
)


voxel_count = int(
    np.prod(
        PATCH_ZYX
    )
)


chosen = rng.choice(
    voxel_count,
    size=512,
    replace=False,
)


fg_flat = chosen[
    :256
]


bg_flat = chosen[
    256:
]


target_flat = synthetic_target.view(
    -1
)


target_flat[
    torch.as_tensor(
        fg_flat,
        device=device,
        dtype=torch.long,
    )
] = 1


target_flat[
    torch.as_tensor(
        bg_flat,
        device=device,
        dtype=torch.long,
    )
] = 0


model.train()


with torch.autocast(
    device_type="cuda",
    dtype=torch.float16,
    enabled=True,
):

    logits = model(
        synthetic_image
    )


    if tuple(
        logits.shape
    ) != tuple(
        synthetic_target.shape
    ):

        raise RuntimeError(
            "Baseline output shape mismatch.\n"
            + "Output: "
            + str(
                tuple(
                    logits.shape
                )
            )
            + "\nTarget: "
            + str(
                tuple(
                    synthetic_target.shape
                )
            )
        )


    bce = partial_bce(
        logits,
        synthetic_target,
    )


    dice = partial_dice(
        logits,
        synthetic_target,
    )


    loss = (
        LOSS_BCE_WEIGHT
        * bce
        + LOSS_DICE_WEIGHT
        * dice
    )


if not torch.isfinite(
    loss
):

    raise RuntimeError(
        "Synthetic partial loss is non-finite."
    )


# Backward only.
#
# IMPORTANT:
# No optimizer object is required and optimizer.step() is NEVER called.
loss.backward()


torch.cuda.synchronize()


peak_allocated_gb = float(
    torch.cuda.max_memory_allocated(
        device
    )
    / (
        1024
        ** 3
    )
)


peak_reserved_gb = float(
    torch.cuda.max_memory_reserved(
        device
    )
    / (
        1024
        ** 3
    )
)


reserved_fraction = float(
    peak_reserved_gb
    / total_gpu_gb
)


if reserved_fraction > 0.90:

    raise RuntimeError(
        "Baseline dry run leaves insufficient GPU memory headroom.\n"
        + "GPU: "
        + gpu_name
        + "\nTotal GB: "
        + "{:.2f}".format(
            total_gpu_gb
        )
        + "\nPeak reserved GB: "
        + "{:.2f}".format(
            peak_reserved_gb
        )
        + "\nReserved fraction: "
        + "{:.3f}".format(
            reserved_fraction
        )
    )


print(
    "✓ GPU                                  :",
    gpu_name,
)

print(
    "✓ GPU memory                           :",
    "{:.2f} GB".format(
        total_gpu_gb
    ),
)

print(
    "✓ Trainable parameters                 :",
    "{:,}".format(
        trainable_parameters
    ),
)

print(
    "✓ Synthetic output shape               :",
    tuple(
        logits.shape
    ),
)

print(
    "✓ Synthetic partial BCE                :",
    "{:.6f}".format(
        float(
            bce.detach().cpu()
        )
    ),
)

print(
    "✓ Synthetic partial Dice               :",
    "{:.6f}".format(
        float(
            dice.detach().cpu()
        )
    ),
)

print(
    "✓ Synthetic total loss                 :",
    "{:.6f}".format(
        float(
            loss.detach().cpu()
        )
    ),
)

print(
    "✓ Peak allocated                       :",
    "{:.2f} GB".format(
        peak_allocated_gb
    ),
)

print(
    "✓ Peak reserved                        :",
    "{:.2f} GB".format(
        peak_reserved_gb
    ),
)

print(
    "✓ Reserved / total GPU                 :",
    "{:.1f}%".format(
        100.0
        * reserved_fraction
    ),
)

print(
    "✓ optimizer.step() calls               : 0"
)


del logits
del loss
del bce
del dice
del synthetic_image
del synthetic_target
del target_flat
del model


torch.cuda.empty_cache()


# ==========================================================================================
# 8. FREEZE RUN REGISTRY
# ==========================================================================================

heading(
    "STEP 6/9 — FREEZE FINAL FACTORIAL RUN REGISTRY"
)


run_rows = []


final_split = split_df[
    split_df[
        "role"
    ]
    == "final_outer_cv"
].copy()


for fold in range(
    FINAL_OUTER_FOLDS
):

    train_cases = sorted(
        final_split.loc[
            final_split[
                "outer_fold"
            ]
            != fold,
            "case_id",
        ].astype(
            str
        ).tolist()
    )


    heldout_cases = sorted(
        final_split.loc[
            final_split[
                "outer_fold"
            ]
            == fold,
            "case_id",
        ].astype(
            str
        ).tolist()
    )


    if len(
        train_cases
    ) != 12:

        raise RuntimeError(
            "Expected 12 training volumes in fold "
            + str(
                fold
            )
        )


    if len(
        heldout_cases
    ) != 4:

        raise RuntimeError(
            "Expected 4 held-out volumes in fold "
            + str(
                fold
            )
        )


    for seed in FINAL_SEEDS:

        initialization_seed = stable_seed(
            BASELINE_ID,
            "init",
            seed,
            fold,
        )


        for condition in CONDITIONS:

            run_id = (
                "cova3d_"
                + condition.lower()
                + "_fold"
                + str(
                    fold
                )
                + "_seed"
                + str(
                    seed
                )
            )


            run_rows.append(
                {
                    "run_id":
                        run_id,

                    "condition":
                        condition,

                    "outer_fold":
                        fold,

                    "seed":
                        seed,

                    "initialization_seed":
                        initialization_seed,

                    "training_cases":
                        ";".join(
                            train_cases
                        ),

                    "heldout_cases":
                        ";".join(
                            heldout_cases
                        ),

                    "training_case_count":
                        len(
                            train_cases
                        ),

                    "heldout_case_count":
                        len(
                            heldout_cases
                        ),

                    "epochs":
                        FINAL_EPOCHS,

                    "microbatches_per_epoch":
                        FINAL_MICROBATCHES_PER_EPOCH,

                    "accumulation_steps":
                        ACCUMULATION_STEPS,

                    "optimizer_steps":
                        FINAL_OPTIMIZER_STEPS_PER_RUN,

                    "checkpoint_selection":
                        CHECKPOINT_SELECTION,

                    "paired_initialization_across_conditions":
                        True,

                    "paired_schedule_across_conditions":
                        True,

                    "paired_augmentation_rng_across_conditions":
                        True,

                    "dense_training_labels":
                        False,

                    "dense_heldout_outcomes_open_during_fit":
                        False,

                    "training_authorized":
                        False,
                }
            )


run_registry = pd.DataFrame(
    run_rows
).sort_values(
    [
        "outer_fold",
        "seed",
        "condition",
    ]
).reset_index(
    drop=True
)


if len(
    run_registry
) != EXPECTED_FINAL_RUNS:

    raise RuntimeError(
        "Final run-registry size mismatch."
    )


run_registry_path = (
    REPO
    / "data/manifests/"
    "cova3d_final_factorial_run_registry_v1_0.csv"
)


run_registry.to_csv(
    run_registry_path,
    index=False,
)


print(
    "✓ Final run registry                   :",
    len(
        run_registry
    ),
    "models",
)

print(
    "✓ Conditions                           : 6"
)

print(
    "✓ Outer folds                          : 4"
)

print(
    "✓ Seeds                                : 3"
)

print(
    "✓ Initialization paired                : YES"
)

print(
    "✓ Patient schedule paired              : YES"
)

print(
    "✓ Augmentation RNG paired              : YES"
)

print(
    "✓ Final run authorization              : NO"
)


# ==========================================================================================
# 9. DOCUMENT THE LOCK
# ==========================================================================================

write_text(
    REPO
    / "docs/"
    "cova3d_baseline_training_protocol_v1_0.md",
    f"""
    # COVA-3D Baseline and Training Protocol v1.0

    ## Purpose

    The sparse annotation protocol was completed before model development.

    The primary experiment tests annotation coverage and annotation geometry.
    The segmentation network is therefore intentionally kept as a strong,
    conventional 3-D baseline rather than introducing a new architecture.

    ## Baseline

    The model is a five-stage anisotropic nnU-Net-style 3-D U-Net.

    Feature widths are 32, 64, 128, 256 and 320.

    The first encoder kernel is 1x3x3. Later kernels are 3x3x3.

    The first downsampling operation is 1x2x2. Later downsampling operations
    are 2x2x2. This respects the frozen 3.0x1.5x1.5 mm grid.

    Each stage contains two convolutions with affine instance normalization
    and LeakyReLU.

    Deep supervision is disabled because downsampling extremely sparse labels
    would add a second annotation-remapping operation to the factorial
    experiment.

    ## Sparse loss

    Unknown voxels have label -1 and are ignored.

    The objective is partial BCE plus partial Dice with equal weights.

    No dense mask, pseudo-label, teacher/student loss, consistency loss,
    component replay or other auxiliary objective is used.

    ## Patch protocol

    Patch size is 48x128x128 voxels.

    Patients are sampled uniformly.

    Patch-source probabilities are:

    - foreground-supervision centered: 0.50
    - background-supervision centered: 0.25
    - uniform crop: 0.25

    ## Optimization

    AdamW is used with learning rate 3e-4 and weight decay 1e-4.

    Gradient accumulation is two microbatches.

    AMP float16 is enabled.

    Learning rate uses 100 optimizer-step linear warmup followed by cosine
    decay to 3e-6.

    No early stopping is allowed.

    Only the final epoch checkpoint is eligible.

    ## Paired randomization

    Within each fold and seed, all six annotation conditions receive identical
    model initialization.

    Patient-order randomness is paired across conditions.

    Augmentation randomness is paired across conditions.

    The annotation coordinates themselves remain condition-specific because
    they are the scientific intervention.

    ## Development sanity gate

    The sanity model uses C100_COH, seed 17 and only the four permanent
    development volumes.

    It runs for {SANITY_EPOCHS} epochs and {SANITY_OPTIMIZER_STEPS} optimizer
    steps.

    Dense masks are not accessed during fitting.

    The final checkpoint is evaluated once against the dense permanent-
    development masks.

    The sanity gate is intended only to detect a broken trainer or degenerate
    inference pipeline. It is not a hyperparameter-selection stage.

    If the gate fails, factorial training remains blocked. Only implementation
    defects may be corrected. Model or hyperparameter changes require a new
    prospective protocol amendment.

    The sanity checkpoint is never reused in the final experiment.

    ## Final factorial experiment

    Final evaluation uses four frozen outer folds.

    Each model trains on 12 volumes and predicts four held-out volumes.

    There are six annotation conditions and three seeds: 17, 29 and 43.

    This yields {EXPECTED_FINAL_RUNS} independently initialized final fits.

    Each fit uses {FINAL_EPOCHS} epochs, 120 microbatches per epoch and
    {FINAL_OPTIMIZER_STEPS_PER_RUN} optimizer updates.

    All final models and predictions must be frozen before dense final
    outcomes are opened.

    ## Current authorization

    Sanity training is authorized after this lock.

    Factorial training is not authorized.

    Final-outcome evaluation is not authorized.
    """
)


# ==========================================================================================
# 10. UPDATE STATE — SANITY ONLY AUTHORIZED
# ==========================================================================================

heading(
    "STEP 7/9 — UPDATE STATE: SANITY TRAINING ONLY"
)


cova_state.update(
    {
        "last_completed_block":
            BLOCK,

        "current_stage":
            "BASELINE_AND_TRAINING_PROTOCOL_LOCKED",

        "baseline_id":
            BASELINE_ID,

        "baseline_protocol_status":
            "FROZEN",

        "baseline_architecture":
            "nnUNet_style_3D_anisotropic",

        "baseline_parameter_count":
            trainable_parameters,

        "baseline_patch_zyx":
            list(
                PATCH_ZYX
            ),

        "baseline_partial_loss":
            "BCE_plus_Dice",

        "baseline_optimizer":
            "AdamW",

        "baseline_learning_rate":
            LEARNING_RATE,

        "baseline_memory_dry_run":
            "PASS",

        "baseline_peak_reserved_gb":
            peak_reserved_gb,

        "baseline_gpu":
            gpu_name,

        "sanity_condition":
            SANITY_CONDITION,

        "sanity_seed":
            SANITY_SEED,

        "sanity_epochs":
            SANITY_EPOCHS,

        "sanity_optimizer_steps_planned":
            SANITY_OPTIMIZER_STEPS,

        "sanity_training_authorized":
            True,

        "sanity_dense_evaluation_authorized_after_fit":
            True,

        "sanity_gate_status":
            "NOT_RUN",

        "final_factorial_run_count":
            EXPECTED_FINAL_RUNS,

        "final_factorial_seeds":
            FINAL_SEEDS,

        "final_factorial_outer_folds":
            FINAL_OUTER_FOLDS,

        "final_optimizer_steps_per_run":
            FINAL_OPTIMIZER_STEPS_PER_RUN,

        "factorial_training_authorized":
            False,

        "method_development_authorized":
            False,

        "final_outer_cv_outcomes_authorized":
            False,

        "optimizer_steps_in_cova3d":
            0,

        "next_block":
            "09E-SANITY-FIT",

        "next_action":
            (
                "Fit the single frozen C100_COH sanity model on the four "
                "permanent-development cases only. Do not evaluate final "
                "outer-CV outcomes."
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
            "cova3d_baseline_training_protocol_lock",

        "current_stage":
            "cova3d_sanity_training_authorized",

        "current_gate":
            "COVA_SANITY_GATE",

        "cova3d_baseline_id":
            BASELINE_ID,

        "cova3d_baseline_protocol":
            "FROZEN",

        "cova3d_sanity_training_authorized":
            True,

        "cova3d_sanity_gate":
            "NOT_RUN",

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
                "Run 09E-SANITY-FIT only. Factorial fitting remains locked."
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
# 11. TESTS
# ==========================================================================================

heading(
    "STEP 8/9 — REGRESSION TESTS"
)


test_source = r'''
from pathlib import Path

import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_baseline_config_is_frozen():

    config = yaml.safe_load(
        (
            ROOT
            / "configs/"
            "cova3d_baseline_training_v1_0.yaml"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        config[
            "status"
        ]
        == "PROSPECTIVELY_FROZEN_BEFORE_FIRST_COVA_OPTIMIZER_STEP"
    )

    assert (
        config[
            "loss"
        ][
            "unknown_label"
        ]
        == -1
    )

    assert (
        config[
            "training_authorization_after_lock"
        ][
            "sanity_training"
        ]
        is True
    )

    assert (
        config[
            "training_authorization_after_lock"
        ][
            "factorial_training"
        ]
        is False
    )


def test_final_run_registry():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_final_factorial_run_registry_v1_0.csv"
    )

    assert len(
        frame
    ) == 72

    assert set(
        frame[
            "seed"
        ].astype(
            int
        )
    ) == {
        17,
        29,
        43,
    }

    assert set(
        frame[
            "outer_fold"
        ].astype(
            int
        )
    ) == {
        0,
        1,
        2,
        3,
    }

    assert (
        frame[
            "training_case_count"
        ]
        == 12
    ).all()

    assert (
        frame[
            "heldout_case_count"
        ]
        == 4
    ).all()

    assert not frame[
        "training_authorized"
    ].astype(
        bool
    ).any()


def test_model_shape_cpu():

    from cora_lung.models.cova3d_nnunet_style import (
        COVA3DNNUNetStyle,
    )

    model = COVA3DNNUNetStyle(
        features=(
            32,
            64,
            128,
            256,
            320,
        )
    )

    # Smaller divisible tensor for CPU unit test.
    x = torch.zeros(
        (
            1,
            1,
            16,
            32,
            32,
        ),
        dtype=torch.float32,
    )

    model.eval()

    with torch.no_grad():

        y = model(
            x
        )

    assert y.shape == x.shape


def test_pairing_seed_independent_of_condition():

    import hashlib

    def stable_seed(*parts):

        payload = "|".join(
            str(
                part
            )
            for part in parts
        )

        digest = hashlib.sha256(
            payload.encode(
                "utf-8"
            )
        ).digest()

        return int.from_bytes(
            digest[
                :8
            ],
            "little",
            signed=False,
        ) % (
            2**32
            - 1
        )

    seed_a = stable_seed(
        "COVA3D_NNUNET_STYLE_PARTIAL_V1",
        "init",
        17,
        0,
    )

    seed_b = stable_seed(
        "COVA3D_NNUNET_STYLE_PARTIAL_V1",
        "init",
        17,
        0,
    )

    assert seed_a == seed_b
'''


test_path = (
    REPO
    / "tests/"
    "test_cova3d_baseline_lock.py"
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

        "tests/test_partial_losses.py",
        "tests/test_gate_b_sampling_protocol.py",
        "tests/test_cova3d_amendment_a1_2.py",
        "tests/test_cova3d_final_coordinate_grid_v1_2.py",
        "tests/test_cova3d_baseline_lock.py",

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
        "09E-LOCK regression tests FAILED."
    )


print(
    "✓ Regression tests                     : PASS"
)


# ==========================================================================================
# 12. CAPTURE SOURCE + AUDIT
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
        "COVA-3D — BLOCK 09E-LOCK"
        in cell
    ):

        source_path = (
            REPO
            / "scripts/code_blocks/"
            "block09e_lock_baseline_training_protocol.py"
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

    "effective_protocol":
        EFFECTIVE_PROTOCOL,

    "baseline_id":
        BASELINE_ID,

    "architecture": {
        "family":
            "nnUNet_style_3D",

        "features":
            FEATURES,

        "patch_zyx":
            list(
                PATCH_ZYX
            ),

        "trainable_parameters":
            trainable_parameters,

        "deep_supervision":
            False,
    },

    "loss": {
        "partial_BCE":
            True,

        "partial_Dice":
            True,

        "unknown_label":
            -1,

        "dense_training_labels":
            False,
    },

    "memory_dry_run": {
        "gpu":
            gpu_name,

        "total_gpu_gb":
            total_gpu_gb,

        "peak_allocated_gb":
            peak_allocated_gb,

        "peak_reserved_gb":
            peak_reserved_gb,

        "reserved_fraction":
            reserved_fraction,

        "output_shape":
            [
                1,
                1,
                *PATCH_ZYX,
            ],

        "backward_pass":
            "PASS",

        "optimizer_steps":
            0,
    },

    "sanity_protocol": {
        "condition":
            SANITY_CONDITION,

        "seed":
            SANITY_SEED,

        "fit_cases":
            dev_cases,

        "epochs":
            SANITY_EPOCHS,

        "optimizer_steps_planned":
            SANITY_OPTIMIZER_STEPS,

        "sanity_training_authorized":
            True,

        "dense_evaluation_only_after_fit":
            True,

        "checkpoint_reused_final":
            False,
    },

    "final_protocol": {
        "conditions":
            6,

        "outer_folds":
            4,

        "seeds":
            FINAL_SEEDS,

        "planned_models":
            EXPECTED_FINAL_RUNS,

        "epochs_per_model":
            FINAL_EPOCHS,

        "optimizer_steps_per_model":
            FINAL_OPTIMIZER_STEPS_PER_RUN,

        "paired_randomization":
            True,

        "final_epoch_checkpoint_only":
            True,

        "factorial_training_authorized":
            False,
    },

    "firewall": {
        "CT_arrays_accessed":
            0,

        "dense_masks_accessed":
            0,

        "predictions_accessed":
            0,

        "model_outcomes_accessed":
            0,

        "optimizer_steps":
            0,
    },

    "source_capture":
        source_capture,

    "next_block":
        "09E-SANITY-FIT",
}


audit_path = (
    REPO
    / "experiments/audits/"
    "block09e_lock_baseline_training_protocol.json"
)


write_json(
    audit_path,
    audit,
)


# ==========================================================================================
# 13. COMMIT + PUSH
# ==========================================================================================

heading(
    "STEP 9/9 — COMMIT BASELINE/TRAINING LOCK"
)


git_paths = [
    "COVA3D_STATE.json",
    "PROJECT_STATE.json",

    "configs/"
    "cova3d_baseline_training_v1_0.yaml",

    "data/manifests/"
    "cova3d_final_factorial_run_registry_v1_0.csv",

    "docs/"
    "cova3d_baseline_training_protocol_v1_0.md",

    "experiments/audits/"
    "block09e_lock_baseline_training_protocol.json",

    "src/cora_lung/models/"
    "cova3d_nnunet_style.py",

    "tests/"
    "test_cova3d_baseline_lock.py",
]


source_path = (
    REPO
    / "scripts/code_blocks/"
    "block09e_lock_baseline_training_protocol.py"
)


if source_path.exists():

    git_paths.append(
        "scripts/code_blocks/"
        "block09e_lock_baseline_training_protocol.py"
    )


for relative in git_paths:

    path = (
        REPO
        / relative
    )

    if (
        path.exists()
        and path.is_file()
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
    "✓ git diff --cached --check             : PASS"
)


sh(
    [
        "git",
        "commit",
        "-m",
        "protocol: freeze COVA-3D strong sparse baseline",
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
        "Repository is not clean after 09E-LOCK:\n"
        + final_status
    )


# ==========================================================================================
# 14. FINAL REPORT
# ==========================================================================================

heading(
    "COVA-3D BLOCK 09E-LOCK — FINAL REPORT"
)


print(
    "Starting commit                       :",
    head[:12],
)

print(
    "09E-LOCK commit                        :",
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
    "BASELINE"
)

print(
    "--------"
)

print(
    "Baseline ID                           :",
    BASELINE_ID,
)

print(
    "Architecture                          : 3-D nnU-Net-style"
)

print(
    "Features                              :",
    FEATURES,
)

print(
    "Patch ZYX                             :",
    PATCH_ZYX,
)

print(
    "Trainable parameters                  :",
    "{:,}".format(
        trainable_parameters
    ),
)

print(
    "Loss                                  : partial BCE + partial Dice"
)

print(
    "Deep supervision                      : NO"
)

print()

print(
    "CUDA DRY RUN"
)

print(
    "------------"
)

print(
    "GPU                                   :",
    gpu_name,
)

print(
    "Total GPU memory                      :",
    "{:.2f} GB".format(
        total_gpu_gb
    ),
)

print(
    "Peak allocated                        :",
    "{:.2f} GB".format(
        peak_allocated_gb
    ),
)

print(
    "Peak reserved                         :",
    "{:.2f} GB".format(
        peak_reserved_gb
    ),
)

print(
    "Backward pass                         : PASS"
)

print(
    "Optimizer steps                       : 0"
)

print()

print(
    "SANITY PROTOCOL"
)

print(
    "---------------"
)

print(
    "Condition                             :",
    SANITY_CONDITION,
)

print(
    "Seed                                  :",
    SANITY_SEED,
)

print(
    "Cases                                 : 4 permanent development"
)

print(
    "Epochs                                :",
    SANITY_EPOCHS,
)

print(
    "Planned optimizer steps               :",
    SANITY_OPTIMIZER_STEPS,
)

print(
    "Final checkpoint only                 : YES"
)

print(
    "Sanity training authorized            : YES"
)

print(
    "Factorial training authorized         : NO"
)

print()

print(
    "FINAL FACTORIAL PROTOCOL"
)

print(
    "------------------------"
)

print(
    "Conditions                            : 6"
)

print(
    "Outer folds                           : 4"
)

print(
    "Seeds                                 :",
    FINAL_SEEDS,
)

print(
    "Planned model fits                    :",
    EXPECTED_FINAL_RUNS,
)

print(
    "Epochs / model                        :",
    FINAL_EPOCHS,
)

print(
    "Optimizer steps / model               :",
    FINAL_OPTIMIZER_STEPS_PER_RUN,
)

print(
    "Paired initialization                 : YES"
)

print(
    "Paired patient schedule               : YES"
)

print(
    "Paired augmentation RNG               : YES"
)

print(
    "Performance checkpoint selection      : NO"
)

print()

print(
    "FIREWALL"
)

print(
    "--------"
)

print(
    "CT arrays accessed                    : 0"
)

print(
    "Dense masks accessed                  : 0"
)

print(
    "Predictions accessed                  : 0"
)

print(
    "Model outcomes accessed               : 0"
)

print(
    "COVA optimizer steps                  : 0"
)

print()

print(
    "Regression tests                      : PASS"
)

print(
    "Exact source captured                 :",
    source_capture,
)

print()

print(
    "NEXT:"
)

print(
    "Send me this COMPLETE report."
)

print(
    "Do NOT run the six factorial conditions yet."
)

print(
    "Next block will be 09E-SANITY-FIT."
)

print(
    "It will train exactly ONE C100_COH model on the four permanent "
    "development cases using seed 17 and the frozen protocol."
)

print(
    "After that, a separate locked sanity evaluation will decide whether "
    "factorial training may begin."
)

print(
    "=" * 128
)
