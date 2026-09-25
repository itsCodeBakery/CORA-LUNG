# ==========================================================================================
# COVA-3D — BLOCK 10B-FACTORIAL-BATCH-09
#
# REAL PRIMARY FACTORIAL TRAINING
#
# BATCH
# -----
# Outer fold : 0
# Seed       : 43
#
# CONDITIONS
# ----------
# C50_COH
# C50_DIS
# C50_FRG
# C100_COH
# C100_DIS
# C100_FRG
#
# EACH RUN
# --------
# 12 training patients
# 40 epochs
# 120 microbatches / epoch
# accumulation = 2
# 2400 successful optimizer steps
# final checkpoint only
#
# PAIRING
# -------
# Same model initialization across six conditions
# Same patient schedule
# Same sampling-source random numbers
# Same augmentation random numbers
# Condition is excluded from nuisance RNG
#
# NUMERICS
# --------
# N1:
#   AMP initial scale = 4096
#   overflow retry = exact same accumulation pair
#
# N2:
#   sparse Dice arithmetic = FP32
#
# FIREWALL
# --------
# NO dense lesion masks
# NO lung masks
# NO held-out outcome evaluation
# NO Dice/FROC evaluation
# NO threshold search
#
# Runtime final checkpoints are NOT committed to normal Git.
#
# ==========================================================================================

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import copy
import gc
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import textwrap

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from tqdm.auto import tqdm
from kaggle_secrets import UserSecretsClient


# ==========================================================================================
# 0. FROZEN CONSTANTS
# ==========================================================================================

REPO = Path(
    "/kaggle/working/CORA-LUNG"
)

EXPECTED_HEAD = (
    "520d672d1105"
)

BLOCK = (
    "10B-FACTORIAL-BATCH-09"
)

NEXT_BLOCK = (
    "10C-FACTORIAL-BATCH-09-PRED-FREEZE"
)

EFFECTIVE_PROTOCOL = (
    "COVA3D_1.0+A1+A1.1+A1.2+A1.3+N1+N2"
)

BATCH_ID = (
    "fold2_seed43"
)

OUTER_FOLD = (
    2
)

SEED = (
    43
)

EXPECTED_INITIALIZATION_SEED = (
    2566699473
)

CONDITIONS = [
    "C50_COH",
    "C50_DIS",
    "C50_FRG",
    "C100_COH",
    "C100_DIS",
    "C100_FRG",
]

PATCH_ZYX = (
    48,
    128,
    128,
)

FEATURES = (
    32,
    64,
    128,
    256,
    320,
)

EPOCHS = (
    40
)

MICROBATCHES_PER_EPOCH = (
    120
)

MICROBATCHES_PER_CASE_PER_EPOCH = (
    10
)

ACCUMULATION_STEPS = (
    2
)

OPTIMIZER_STEPS_PER_EPOCH = (
    60
)

TOTAL_OPTIMIZER_STEPS = (
    2400
)

BASE_LR = (
    3e-4
)

MIN_LR = (
    3e-6
)

WARMUP_STEPS = (
    100
)

WEIGHT_DECAY = (
    1e-4
)

BETAS = (
    0.9,
    0.999,
)

GRAD_CLIP = (
    12.0
)

AMP_INITIAL_SCALE = (
    4096.0
)

AMP_GROWTH_FACTOR = (
    2.0
)

AMP_BACKOFF_FACTOR = (
    0.5
)

AMP_GROWTH_INTERVAL = (
    1_000_000
)

AMP_MIN_SCALE = (
    1.0
)

MAX_OVERFLOW_RETRIES = (
    16
)

UNKNOWN_LABEL = (
    -1
)

FG_PROB = (
    0.50
)

BG_PROB = (
    0.25
)

UNIFORM_PROB = (
    0.25
)

EXPECTED_PARAMETER_COUNT = (
    19_166_529
)


# ==========================================================================================
# 1. PATHS
# ==========================================================================================

STATE_PATH = (
    REPO
    / "COVA3D_STATE.json"
)

PROJECT_STATE_PATH = (
    REPO
    / "PROJECT_STATE.json"
)

BASELINE_CONFIG_PATH = (
    REPO
    / "configs/"
    "cova3d_baseline_training_v1_0.yaml"
)

EXECUTION_REGISTRY_PATH = (
    REPO
    / "data/manifests/"
    "cova3d_final_factorial_execution_registry_v1_0.csv"
)

BATCH_REGISTRY_PATH = (
    REPO
    / "data/manifests/"
    "cova3d_final_factorial_batch_registry_v1_0.csv"
)

SPARSE_MANIFEST_PATH = (
    REPO
    / "data/manifests/"
    "cova3d_training_grid_candidate_artifact_manifest_v1_3.csv"
)

CACHE_MANIFEST_PATH = (
    REPO
    / "data/manifests/"
    "cova3d_final16_CT_cache_manifest_v1_0.csv"
)

CT_CACHE_ROOT = Path(
    "/kaggle/working/cova3d_final16_ct_cache_v1_0"
)

BATCH_ROOT = Path(
    "/kaggle/working/cova3d_factorial_v1_0/"
    "batch09_fold2_seed43"
)

BATCH_RUNTIME_STATE = (
    BATCH_ROOT
    / "batch_state.json"
)

BATCH_EPOCH_LOG_REPO = (
    REPO
    / "data/manifests/"
    "cova3d_factorial_batch09_epoch_log_v1_0.csv"
)

BATCH_RUN_SUMMARY_REPO = (
    REPO
    / "data/manifests/"
    "cova3d_factorial_batch09_fit_summary_v1_0.csv"
)

BATCH_AUDIT_PATH = (
    REPO
    / "experiments/audits/"
    "block10b_factorial_batch09_fit.json"
)

FIG_PNG = (
    REPO
    / "figures/audit/"
    "fig_cova3d_factorial_batch09_training_curves.png"
)

FIG_PDF = (
    REPO
    / "figures/audit/"
    "fig_cova3d_factorial_batch09_training_curves.pdf"
)

TEST_PATH = (
    REPO
    / "tests/"
    "test_cova3d_factorial_batch09_fit.py"
)

SOURCE_PATH = (
    REPO
    / "scripts/code_blocks/"
    "block10b_factorial_batch09_fit.py"
)

NOW_ISO = datetime.now(
    timezone.utc
).strftime(
    "%Y-%m-%dT%H:%M:%SZ"
)


# ==========================================================================================
# 2. GENERAL HELPERS
# ==========================================================================================

def heading(text):

    print(
        "\n"
        + "=" * 132
    )

    print(text)

    print(
        "=" * 132
    )


def sh(
    cmd,
    *,
    env=None,
    check=True,
):

    result = subprocess.run(
        cmd,
        cwd=str(REPO),
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
                map(str, cmd)
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


def git(*args):

    return sh(
        [
            "git",
            *args,
        ]
    ).stdout.strip()


def sha256_file(path):

    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as handle:

        for chunk in iter(
            lambda:
                handle.read(
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
        str(x)
        for x in parts
    ).encode(
        "utf-8"
    )

    digest = hashlib.sha256(
        payload
    ).digest()

    return int.from_bytes(
        digest[:8],
        byteorder="little",
        signed=False,
    ) % (
        2**32
    )


def state_dict_sha256(state_dict):

    h = hashlib.sha256()

    for key in sorted(
        state_dict.keys()
    ):

        tensor = (
            state_dict[
                key
            ]
            .detach()
            .cpu()
            .contiguous()
        )

        h.update(
            key.encode(
                "utf-8"
            )
        )

        h.update(
            str(
                tensor.dtype
            ).encode(
                "utf-8"
            )
        )

        h.update(
            np.asarray(
                tensor.shape,
                dtype=np.int64,
            ).tobytes()
        )

        h.update(
            tensor.numpy().tobytes()
        )

    return h.hexdigest()


def make_git_auth():

    token = (
        UserSecretsClient()
        .get_secret(
            "pushCora"
        )
    )

    if not token:

        raise RuntimeError(
            "Kaggle secret pushCora unavailable."
        )

    token = token.strip()

    askpass = Path(
        "/tmp/cova3d_git_askpass_batch09.sh"
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
            "HEAD:cova-batch09",
        ],
        env=env,
        check=False,
    )

    if result.returncode != 0:

        raise RuntimeError(
            "GitHub push failed:\n"
            + (
                result.stderr
                or ""
            ).replace(
                token,
                "***TOKEN_REDACTED***",
            )
        )


# ==========================================================================================
# 3. TRAINING HELPERS
# ==========================================================================================

def learning_rate_for_step(
    step,
):

    step = int(step)

    if step <= WARMUP_STEPS:

        return (
            BASE_LR
            * step
            / WARMUP_STEPS
        )

    progress = (
        step
        - WARMUP_STEPS
    ) / (
        TOTAL_OPTIMIZER_STEPS
        - WARMUP_STEPS
    )

    progress = min(
        max(
            progress,
            0.0,
        ),
        1.0,
    )

    return (
        MIN_LR
        + 0.5
        * (
            BASE_LR
            - MIN_LR
        )
        * (
            1.0
            + math.cos(
                math.pi
                * progress
            )
        )
    )


def make_scaler():

    try:

        return torch.amp.GradScaler(
            "cuda",
            init_scale=AMP_INITIAL_SCALE,
            growth_factor=AMP_GROWTH_FACTOR,
            backoff_factor=AMP_BACKOFF_FACTOR,
            growth_interval=AMP_GROWTH_INTERVAL,
            enabled=True,
        )

    except Exception:

        return torch.cuda.amp.GradScaler(
            init_scale=AMP_INITIAL_SCALE,
            growth_factor=AMP_GROWTH_FACTOR,
            backoff_factor=AMP_BACKOFF_FACTOR,
            growth_interval=AMP_GROWTH_INTERVAL,
            enabled=True,
        )


def create_epoch_schedule(
    training_cases,
    epoch,
):

    schedule = np.repeat(
        np.asarray(
            training_cases,
            dtype=object,
        ),
        MICROBATCHES_PER_CASE_PER_EPOCH,
    )

    rng = np.random.default_rng(
        stable_seed(
            "patient_schedule",
            SEED,
            OUTER_FOLD,
            epoch,
        )
    )

    rng.shuffle(
        schedule
    )

    return schedule.tolist()


def create_micro_plan(
    case_id,
    epoch,
    microbatch_index,
):

    rng = np.random.default_rng(
        stable_seed(
            "paired_microbatch",
            SEED,
            OUTER_FOLD,
            epoch,
            microbatch_index,
        )
    )

    source_u = float(
        rng.random()
    )

    center_u = float(
        rng.random()
    )

    uniform_origin_u = (
        float(
            rng.random()
        ),
        float(
            rng.random()
        ),
        float(
            rng.random()
        ),
    )

    flip_x_u = float(
        rng.random()
    )

    flip_y_u = float(
        rng.random()
    )

    scale_u = float(
        rng.random()
    )

    shift_u = float(
        rng.random()
    )

    noise_sigma_u = float(
        rng.random()
    )

    noise_seed = int(
        rng.integers(
            0,
            2**32,
            dtype=np.uint64,
        )
    )


    if source_u < FG_PROB:

        source = (
            "foreground"
        )

    elif source_u < (
        FG_PROB
        + BG_PROB
    ):

        source = (
            "background"
        )

    else:

        source = (
            "uniform"
        )


    return {
        "case_id":
            str(
                case_id
            ),

        "source":
            source,

        "center_u":
            center_u,

        "uniform_origin_u":
            uniform_origin_u,

        "flip_x":
            bool(
                flip_x_u
                < 0.5
            ),

        "flip_y":
            bool(
                flip_y_u
                < 0.5
            ),

        "scale":
            0.9
            + 0.2
            * scale_u,

        "shift":
            -0.1
            + 0.2
            * shift_u,

        "noise_sigma":
            0.03
            * noise_sigma_u,

        "noise_seed":
            noise_seed,
    }


def build_all_paired_plans(
    training_cases,
):

    plans_by_epoch = []

    h = hashlib.sha256()


    for epoch in range(
        EPOCHS
    ):

        schedule = create_epoch_schedule(
            training_cases,
            epoch,
        )


        if len(
            schedule
        ) != MICROBATCHES_PER_EPOCH:

            raise RuntimeError(
                "Patient schedule length mismatch."
            )


        counts = pd.Series(
            schedule
        ).value_counts()


        for case_id in training_cases:

            if int(
                counts.get(
                    case_id,
                    0,
                )
            ) != MICROBATCHES_PER_CASE_PER_EPOCH:

                raise RuntimeError(
                    "Patient schedule is not exactly balanced."
                )


        epoch_plans = []


        for microbatch_index, case_id in enumerate(
            schedule
        ):

            plan = create_micro_plan(
                case_id,
                epoch,
                microbatch_index,
            )

            epoch_plans.append(
                plan
            )


            h.update(
                json.dumps(
                    plan,
                    sort_keys=True,
                    separators=(
                        ",",
                        ":",
                    ),
                ).encode(
                    "utf-8"
                )
            )


        plans_by_epoch.append(
            epoch_plans
        )


    return (
        plans_by_epoch,
        h.hexdigest(),
    )


def choose_origin(
    image_shape,
    sparse_coords,
    sparse_labels,
    plan,
):

    image_shape = np.asarray(
        image_shape,
        dtype=np.int32,
    )

    patch_shape = np.asarray(
        PATCH_ZYX,
        dtype=np.int32,
    )


    if np.any(
        image_shape
        < patch_shape
    ):

        raise RuntimeError(
            "Final16 crop is smaller than frozen patch."
        )


    source = plan[
        "source"
    ]


    if source == "foreground":

        candidates = sparse_coords[
            sparse_labels
            == 1
        ]


        if len(
            candidates
        ) == 0:

            raise RuntimeError(
                "Foreground-centered sample requested but no FG labels exist."
            )


        index = min(
            int(
                plan[
                    "center_u"
                ]
                * len(
                    candidates
                )
            ),
            len(
                candidates
            )
            - 1,
        )


        center = candidates[
            index
        ].astype(
            np.int32
        )


        origin = (
            center
            - patch_shape
            // 2
        )


        origin = np.maximum(
            origin,
            0,
        )


        origin = np.minimum(
            origin,
            image_shape
            - patch_shape,
        )


    elif source == "background":

        candidates = sparse_coords[
            sparse_labels
            == 0
        ]


        if len(
            candidates
        ) == 0:

            raise RuntimeError(
                "Background-centered sample requested but no BG labels exist."
            )


        index = min(
            int(
                plan[
                    "center_u"
                ]
                * len(
                    candidates
                )
            ),
            len(
                candidates
            )
            - 1,
        )


        center = candidates[
            index
        ].astype(
            np.int32
        )


        origin = (
            center
            - patch_shape
            // 2
        )


        origin = np.maximum(
            origin,
            0,
        )


        origin = np.minimum(
            origin,
            image_shape
            - patch_shape,
        )


    elif source == "uniform":

        available = (
            image_shape
            - patch_shape
        )


        origin = np.asarray(
            [
                int(
                    math.floor(
                        plan[
                            "uniform_origin_u"
                        ][
                            axis
                        ]
                        * (
                            int(
                                available[
                                    axis
                                ]
                            )
                            + 1
                        )
                    )
                )
                for axis in range(
                    3
                )
            ],
            dtype=np.int32,
        )


        origin = np.minimum(
            origin,
            available,
        )


    else:

        raise RuntimeError(
            "Unknown patch source."
        )


    return origin.astype(
        np.int32
    )


def make_augmented_patch(
    image,
    sparse_coords,
    sparse_labels,
    plan,
):

    origin = choose_origin(
        image.shape,
        sparse_coords,
        sparse_labels,
        plan,
    )


    hi = (
        origin
        + np.asarray(
            PATCH_ZYX,
            dtype=np.int32,
        )
    )


    patch = np.asarray(
        image[
            int(
                origin[
                    0
                ]
            ):
            int(
                hi[
                    0
                ]
            ),

            int(
                origin[
                    1
                ]
            ):
            int(
                hi[
                    1
                ]
            ),

            int(
                origin[
                    2
                ]
            ):
            int(
                hi[
                    2
                ]
            ),
        ],
        dtype=np.float32,
    ).copy()


    target = np.full(
        PATCH_ZYX,
        UNKNOWN_LABEL,
        dtype=np.int8,
    )


    relative = (
        sparse_coords.astype(
            np.int32
        )
        - origin[
            None,
            :
        ]
    )


    inside = np.all(
        (
            relative
            >= 0
        )
        & (
            relative
            < np.asarray(
                PATCH_ZYX,
                dtype=np.int32,
            )[
                None,
                :
            ]
        ),
        axis=1,
    )


    relative_inside = relative[
        inside
    ]


    labels_inside = sparse_labels[
        inside
    ]


    if len(
        relative_inside
    ):

        target[
            relative_inside[
                :,
                0
            ],
            relative_inside[
                :,
                1
            ],
            relative_inside[
                :,
                2
            ],
        ] = labels_inside


    # ------------------------------------------------------------------
    # Paired spatial augmentation.
    # ------------------------------------------------------------------

    if plan[
        "flip_x"
    ]:

        patch = np.flip(
            patch,
            axis=2,
        )

        target = np.flip(
            target,
            axis=2,
        )


    if plan[
        "flip_y"
    ]:

        patch = np.flip(
            patch,
            axis=1,
        )

        target = np.flip(
            target,
            axis=1,
        )


    # ------------------------------------------------------------------
    # Paired intensity augmentation.
    # ------------------------------------------------------------------

    patch = (
        patch
        * float(
            plan[
                "scale"
            ]
        )
        + float(
            plan[
                "shift"
            ]
        )
    )


    sigma = float(
        plan[
            "noise_sigma"
        ]
    )


    if sigma > 0.0:

        noise_rng = np.random.default_rng(
            int(
                plan[
                    "noise_seed"
                ]
            )
        )

        noise = noise_rng.normal(
            loc=0.0,
            scale=sigma,
            size=PATCH_ZYX,
        ).astype(
            np.float32
        )

        patch = (
            patch
            + noise
        )


    patch = np.clip(
        patch,
        -1.0,
        1.0,
    ).astype(
        np.float32
    )


    patch = np.ascontiguousarray(
        patch
    )

    target = np.ascontiguousarray(
        target
    )


    labelled = (
        target
        != UNKNOWN_LABEL
    )


    labelled_count = int(
        labelled.sum()
    )

    foreground_count = int(
        (
            target
            == 1
        ).sum()
    )


    return {
        "image":
            patch,

        "target":
            target,

        "source":
            plan[
                "source"
            ],

        "case_id":
            plan[
                "case_id"
            ],

        "labelled_count":
            labelled_count,

        "foreground_count":
            foreground_count,
    }


# ==========================================================================================
# 4. PREFLIGHT
# ==========================================================================================

heading(
    "COVA-3D 10B-FACTORIAL-BATCH-09 — PREFLIGHT"
)


head = git(
    "rev-parse",
    "HEAD",
)


if not head.startswith(
    EXPECTED_HEAD
):

    raise RuntimeError(
        "Unexpected Git HEAD.\n"
        + "Expected: "
        + EXPECTED_HEAD
        + "\nObserved: "
        + head
    )


dirty = git(
    "status",
    "--porcelain",
)


if dirty:

    raise RuntimeError(
        "Repository must be clean before real factorial training:\n"
        + dirty
    )


state = json.loads(
    STATE_PATH.read_text(
        encoding="utf-8"
    )
)


if state.get(
    "last_completed_block"
) != "10C-FACTORIAL-BATCH-08-PRED-FREEZE":

    raise RuntimeError(
        "Batch-03 prediction freeze is not the frozen parallel-execution base."
    )


if state.get(
    "factorial_preflight_status"
) != "PASS":

    raise RuntimeError(
        "Factorial preflight is not PASS."
    )


if state.get(
    "factorial_training_authorized"
) is not True:

    raise RuntimeError(
        "Factorial fitting is not authorized."
    )


if state.get(
    "next_factorial_batch"
) != "fold2_seed43":

    raise RuntimeError(
        "Unexpected frozen main-line state for parallel Batch-09 execution."
    )


if BATCH_ID != "fold2_seed43":

    raise RuntimeError(
        "Parallel Batch-09 authorization mismatch."
    )


if state.get(
    "final_outer_cv_outcomes_authorized"
) is not False:

    raise RuntimeError(
        "Final dense outcomes unexpectedly authorized."
    )


if int(
    state.get(
        "final_outer_cv_access_in_cova3d",
        -1,
    )
) != 0:

    raise RuntimeError(
        "Final dense outcome access must remain zero."
    )


if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA GPU is required."
    )


print(
    "✓ Git HEAD                             :",
    head[:12],
)

print(
    "✓ Factorial preflight                  : PASS"
)

print(
    "✓ Factorial fitting                    : AUTHORIZED"
)

print(
    "✓ Final dense outcomes                 : SEALED"
)

print(
    "✓ CUDA                                 :",
    torch.cuda.get_device_name(
        0
    ),
)


# ==========================================================================================
# 5. VERIFY FROZEN TRAINING CONFIG
# ==========================================================================================

heading(
    "STEP 1/8 — VERIFY FROZEN BATCH DEFINITION"
)


baseline_cfg = yaml.safe_load(
    BASELINE_CONFIG_PATH.read_text(
        encoding="utf-8"
    )
)


final_cfg = baseline_cfg[
    "final_factorial_training"
]


if int(
    final_cfg[
        "epochs"
    ]
) != EPOCHS:

    raise RuntimeError(
        "Frozen epoch count changed."
    )


if int(
    final_cfg[
        "microbatches_per_epoch"
    ]
) != MICROBATCHES_PER_EPOCH:

    raise RuntimeError(
        "Frozen microbatch count changed."
    )


if int(
    final_cfg[
        "accumulation_steps"
    ]
) != ACCUMULATION_STEPS:

    raise RuntimeError(
        "Frozen accumulation changed."
    )


if int(
    final_cfg[
        "optimizer_steps_per_run"
    ]
) != TOTAL_OPTIMIZER_STEPS:

    raise RuntimeError(
        "Frozen optimizer-step count changed."
    )


execution_df = pd.read_csv(
    EXECUTION_REGISTRY_PATH
)


batch_runs = (
    execution_df[
        execution_df[
            "paired_batch_id"
        ]
        == BATCH_ID
    ]
    .sort_values(
        "condition_execution_order_within_batch"
    )
    .reset_index(
        drop=True
    )
)


if len(
    batch_runs
) != 6:

    raise RuntimeError(
        "Batch-09 does not contain six runs."
    )


if batch_runs[
    "initialization_seed"
].nunique() != 1:

    raise RuntimeError(
        "Paired initialization seed mismatch."
    )


initialization_seed = int(
    batch_runs.iloc[
        0
    ][
        "initialization_seed"
    ]
)


if initialization_seed != EXPECTED_INITIALIZATION_SEED:

    raise RuntimeError(
        "Unexpected initialization seed."
    )


if batch_runs[
    "training_cases"
].nunique() != 1:

    raise RuntimeError(
        "Training-case set differs across conditions."
    )


training_cases = [
    x
    for x in str(
        batch_runs.iloc[
            0
        ][
            "training_cases"
        ]
    ).split(
        ";"
    )
    if x
]


heldout_cases = [
    x
    for x in str(
        batch_runs.iloc[
            0
        ][
            "heldout_cases"
        ]
    ).split(
        ";"
    )
    if x
]


if len(
    training_cases
) != 12:

    raise RuntimeError(
        "Expected twelve training cases."
    )


if len(
    heldout_cases
) != 4:

    raise RuntimeError(
        "Expected four held-out cases."
    )


if set(
    batch_runs[
        "condition"
    ]
) != set(
    CONDITIONS
):

    raise RuntimeError(
        "Unexpected condition set."
    )


print(
    "Batch                                 :",
    BATCH_ID,
)

print(
    "Outer fold                            : 2"
)

print(
    "Seed                                  : 43"
)

print(
    "Initialization seed                   :",
    initialization_seed,
)

print(
    "Training patients                     : 12"
)

print(
    "Held-out patients                     : 4"
)

print(
    "Conditions                            : 6"
)

print(
    "Optimizer steps / run                 :",
    TOTAL_OPTIMIZER_STEPS,
)


# ==========================================================================================
# 6. IMPORT FROZEN MODEL + N2 LOSS
# ==========================================================================================

heading(
    "STEP 2/8 — LOAD FROZEN MODEL / N1 / N2 IMPLEMENTATION"
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


from cora_lung.models.cova3d_nnunet_style import (
    COVA3DNNUNetStyle,
)

from cora_lung.losses.partial import (
    partial_bce,
    partial_dice_fp32,
)


random.seed(
    initialization_seed
)

np.random.seed(
    initialization_seed
    % (
        2**32
    )
)

torch.manual_seed(
    initialization_seed
)

torch.cuda.manual_seed_all(
    initialization_seed
)


base_model = COVA3DNNUNetStyle(
    in_channels=1,
    out_channels=1,
    features=FEATURES,
)


parameter_count = sum(
    p.numel()
    for p in base_model.parameters()
)


if parameter_count != EXPECTED_PARAMETER_COUNT:

    raise RuntimeError(
        "Frozen model parameter count changed."
    )


base_initial_state = {
    key:
        value.detach().cpu().clone()
    for key, value in base_model.state_dict().items()
}


initialization_sha = state_dict_sha256(
    base_initial_state
)


del base_model


print(
    "✓ Model parameters                     :",
    parameter_count,
)

print(
    "✓ Shared initialization SHA            :",
    initialization_sha,
)

print(
    "✓ N1 AMP initial scale                 :",
    AMP_INITIAL_SCALE,
)

print(
    "✓ N2 sparse Dice                       : FP32"
)


# ==========================================================================================
# 7. LOAD TRAINING CTs
# ==========================================================================================

heading(
    "STEP 3/8 — LOAD 12 TRAINING CT CROPS"
)


cache_manifest = pd.read_csv(
    CACHE_MANIFEST_PATH
)


cache_lookup = {
    str(
        row[
            "case_id"
        ]
    ):
        row
    for _, row in cache_manifest.iterrows()
}


ct_images = {}


for case_id in tqdm(
    training_cases,
    desc="Training CTs",
):

    if case_id not in cache_lookup:

        raise RuntimeError(
            "Missing CT cache manifest row for "
            + case_id
        )


    path = (
        CT_CACHE_ROOT
        / (
            case_id
            + ".npz"
        )
    )


    if not path.exists():

        raise RuntimeError(
            "Runtime CT cache missing:\n"
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
        ) != {
            "ct_zyx"
        }:

            raise RuntimeError(
                "Unexpected CT cache schema."
            )


        image = np.asarray(
            data[
                "ct_zyx"
            ]
        )


    if image.dtype != np.float16:

        raise RuntimeError(
            "CT cache must use frozen float16 representation."
        )


    if not np.isfinite(
        image
    ).all():

        raise RuntimeError(
            "Non-finite CT cache."
        )


    ct_images[
        case_id
    ] = image


print(
    "✓ Training CTs loaded                  : 12"
)

print(
    "✓ Held-out CTs used during fit         : 0"
)

print(
    "✓ Dense masks opened                   : 0"
)


# ==========================================================================================
# 8. VERIFY / INDEX V1.3 SPARSE ARTIFACTS
# ==========================================================================================

heading(
    "STEP 4/8 — INDEX FROZEN V1.3 SPARSE SUPERVISION"
)


sparse_manifest = pd.read_csv(
    SPARSE_MANIFEST_PATH
)


sparse_manifest_lookup = {
    (
        str(
            row[
                "case_id"
            ]
        ),
        str(
            row[
                "condition_id"
            ]
        ),
    ):
        row
    for _, row in sparse_manifest.iterrows()
}


sparse_by_condition = {}


for condition in CONDITIONS:

    case_map = {}


    for case_id in training_cases:

        manifest_row = sparse_manifest_lookup.get(
            (
                case_id,
                condition,
            )
        )


        if manifest_row is None:

            raise RuntimeError(
                "Missing sparse artifact "
                + case_id
                + " / "
                + condition
            )


        artifact_path = (
            REPO
            / str(
                manifest_row[
                    "artifact_file"
                ]
            )
        )


        if sha256_file(
            artifact_path
        ) != str(
            manifest_row[
                "artifact_file_sha256"
            ]
        ):

            raise RuntimeError(
                "Sparse artifact SHA mismatch."
            )


        with np.load(
            artifact_path,
            allow_pickle=False,
        ) as data:

            coords = np.asarray(
                data[
                    "supervision_voxel_zyx"
                ],
                dtype=np.int32,
            )

            labels = np.asarray(
                data[
                    "supervision_label"
                ],
                dtype=np.int8,
            )


        if coords.ndim != 2 or coords.shape[
            1
        ] != 3:

            raise RuntimeError(
                "Invalid sparse coordinate shape."
            )


        if len(
            coords
        ) != len(
            labels
        ):

            raise RuntimeError(
                "Sparse coordinate/label length mismatch."
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
                "Invalid sparse labels."
            )


        if not np.any(
            labels
            == 1
        ):

            raise RuntimeError(
                "Training artifact lacks foreground supervision."
            )


        if not np.any(
            labels
            == 0
        ):

            raise RuntimeError(
                "Training artifact lacks background supervision."
            )


        shape = np.asarray(
            ct_images[
                case_id
            ].shape,
            dtype=np.int32,
        )


        if np.any(
            coords
            < 0
        ) or np.any(
            coords
            >= shape[
                None,
                :
            ]
        ):

            raise RuntimeError(
                "Sparse coordinate outside CT crop."
            )


        case_map[
            case_id
        ] = {
            "coords":
                coords,

            "labels":
                labels,
        }


    sparse_by_condition[
        condition
    ] = case_map


print(
    "✓ Sparse training artifacts indexed    : 72 /72"
)

print(
    "✓ Dense training labels                : NONE"
)


# ==========================================================================================
# 9. FREEZE PAIRED RANDOMIZATION PLAN
# ==========================================================================================

heading(
    "STEP 5/8 — BUILD CONDITION-INDEPENDENT PAIRED RANDOMIZATION"
)


plans_by_epoch, paired_plan_sha = build_all_paired_plans(
    training_cases
)


source_counts = {
    "foreground":
        0,

    "background":
        0,

    "uniform":
        0,
}


for epoch_plans in plans_by_epoch:

    for plan in epoch_plans:

        source_counts[
            plan[
                "source"
            ]
        ] += 1


print(
    "✓ Paired microbatch plans              :",
    EPOCHS
    * MICROBATCHES_PER_EPOCH,
)

print(
    "✓ Paired-plan SHA                      :",
    paired_plan_sha,
)

print(
    "✓ Patch source FG/BG/U                 :",
    source_counts[
        "foreground"
    ],
    "/",
    source_counts[
        "background"
    ],
    "/",
    source_counts[
        "uniform"
    ],
)

print(
    "✓ Condition included in RNG            : NO"
)


# ==========================================================================================
# 10. RUNTIME BATCH ROOT
# ==========================================================================================

BATCH_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


write_json(
    BATCH_RUNTIME_STATE,
    {
        "status":
            "RUNNING",

        "block":
            BLOCK,

        "batch_id":
            BATCH_ID,

        "outer_fold":
            OUTER_FOLD,

        "seed":
            SEED,

        "initialization_seed":
            initialization_seed,

        "shared_initialization_sha256":
            initialization_sha,

        "paired_plan_sha256":
            paired_plan_sha,

        "conditions":
            CONDITIONS,

        "completed_conditions":
            [],

        "dense_masks_opened":
            0,

        "final_outer_cv_outcome_access":
            0,

        "started_at_utc":
            NOW_ISO,
    },
)


# ==========================================================================================
# 11. TRAIN SIX CONDITIONS
# ==========================================================================================

heading(
    "STEP 6/8 — TRAIN SIX REAL FACTORIAL MODELS"
)


device = torch.device(
    "cuda:0"
)


all_epoch_rows = []

run_summary_rows = []

completed_conditions = []


for condition_index, condition in enumerate(
    CONDITIONS,
    start=1,
):

    run_row = batch_runs[
        batch_runs[
            "condition"
        ]
        == condition
    ]


    if len(
        run_row
    ) != 1:

        raise RuntimeError(
            "Could not uniquely resolve run for "
            + condition
        )


    run_row = run_row.iloc[
        0
    ]


    run_id = str(
        run_row[
            "run_id"
        ]
    )


    run_root = (
        BATCH_ROOT
        / condition
    )


    final_model_path = (
        run_root
        / "final_model.pt"
    )


    epoch_log_path = (
        run_root
        / "epoch_log.csv"
    )


    run_state_path = (
        run_root
        / "run_state.json"
    )


    # --------------------------------------------------------------------------------------
    # SAFE RUNTIME RESUME
    # --------------------------------------------------------------------------------------

    reusable = False


    if (
        final_model_path.exists()
        and epoch_log_path.exists()
        and run_state_path.exists()
    ):

        existing_state = json.loads(
            run_state_path.read_text(
                encoding="utf-8"
            )
        )


        if (
            existing_state.get(
                "status"
            )
            == "FIT_COMPLETE"
            and existing_state.get(
                "run_id"
            )
            == run_id
            and int(
                existing_state.get(
                    "successful_optimizer_steps",
                    -1,
                )
            )
            == TOTAL_OPTIMIZER_STEPS
            and existing_state.get(
                "shared_initialization_sha256"
            )
            == initialization_sha
            and existing_state.get(
                "paired_plan_sha256"
            )
            == paired_plan_sha
            and sha256_file(
                final_model_path
            )
            == existing_state.get(
                "final_model_sha256"
            )
        ):

            existing_epoch_df = pd.read_csv(
                epoch_log_path
            )


            if len(
                existing_epoch_df
            ) == EPOCHS:

                reusable = True


    if reusable:

        print()
        print(
            f"[{condition_index}/6] {condition}: "
            "VALID COMPLETED RUNTIME FIT FOUND — REUSING"
        )


        existing_epoch_df = pd.read_csv(
            epoch_log_path
        )


        all_epoch_rows.extend(
            existing_epoch_df.to_dict(
                orient="records"
            )
        )


        run_summary_rows.append(
            existing_state[
                "summary"
            ]
        )


        completed_conditions.append(
            condition
        )


        continue


    if run_root.exists():

        shutil.rmtree(
            run_root
        )


    run_root.mkdir(
        parents=True,
        exist_ok=True,
    )


    print()
    print(
        "-" * 132
    )

    print(
        f"[{condition_index}/6] TRAINING {condition}"
    )

    print(
        "Run ID                                :",
        run_id,
    )

    print(
        "Shared initialization SHA             :",
        initialization_sha,
    )

    print(
        "Paired-plan SHA                       :",
        paired_plan_sha,
    )

    print(
        "-" * 132
    )


    # --------------------------------------------------------------------------------------
    # EXACT SHARED INITIALIZATION
    # --------------------------------------------------------------------------------------

    model = COVA3DNNUNetStyle(
        in_channels=1,
        out_channels=1,
        features=FEATURES,
    )


    model.load_state_dict(
        base_initial_state,
        strict=True,
    )


    condition_init_sha = state_dict_sha256(
        model.state_dict()
    )


    if condition_init_sha != initialization_sha:

        raise RuntimeError(
            "Paired initialization failed."
        )


    model = model.to(
        device
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=BASE_LR,
        betas=BETAS,
        weight_decay=WEIGHT_DECAY,
    )


    scaler = make_scaler()


    successful_steps = (
        0
    )

    total_overflows = (
        0
    )

    total_retry_attempts = (
        0
    )


    condition_epoch_rows = []


    torch.cuda.reset_peak_memory_stats(
        device
    )


    run_start = time.perf_counter()


    epoch_bar = tqdm(
        range(
            EPOCHS
        ),
        desc=condition,
        position=0,
        leave=True,
    )


    for epoch in epoch_bar:

        model.train()


        epoch_loss_sum = (
            0.0
        )

        epoch_bce_sum = (
            0.0
        )

        epoch_dice_sum = (
            0.0
        )

        labelled_voxels_seen = (
            0
        )

        foreground_voxels_seen = (
            0
        )

        zero_label_microbatches = (
            0
        )

        patch_counts = {
            "foreground":
                0,

            "background":
                0,

            "uniform":
                0,
        }


        case_counts = {
            case_id:
                0
            for case_id in training_cases
        }


        epoch_overflows = (
            0
        )

        epoch_retries = (
            0
        )

        epoch_grad_norms = []

        epoch_scale_min = float(
            scaler.get_scale()
        )

        epoch_scale_max = float(
            scaler.get_scale()
        )


        epoch_start = time.perf_counter()


        plans = plans_by_epoch[
            epoch
        ]


        # 120 microbatches = 60 exact accumulation pairs.
        for pair_start in range(
            0,
            MICROBATCHES_PER_EPOCH,
            ACCUMULATION_STEPS,
        ):

            pair_items = []


            for offset in range(
                ACCUMULATION_STEPS
            ):

                micro_index = (
                    pair_start
                    + offset
                )


                plan = plans[
                    micro_index
                ]


                case_id = plan[
                    "case_id"
                ]


                sparse = sparse_by_condition[
                    condition
                ][
                    case_id
                ]


                item = make_augmented_patch(
                    ct_images[
                        case_id
                    ],
                    sparse[
                        "coords"
                    ],
                    sparse[
                        "labels"
                    ],
                    plan,
                )


                pair_items.append(
                    item
                )


            retries = (
                0
            )


            while True:

                if retries > MAX_OVERFLOW_RETRIES:

                    raise RuntimeError(
                        condition
                        + ": exceeded maximum AMP overflow retries."
                    )


                prospective_step = (
                    successful_steps
                    + 1
                )


                lr = learning_rate_for_step(
                    prospective_step
                )


                for group in optimizer.param_groups:

                    group[
                        "lr"
                    ] = lr


                optimizer.zero_grad(
                    set_to_none=True
                )


                pair_loss_values = []

                pair_bce_values = []

                pair_dice_values = []


                for item in pair_items:

                    image_tensor = torch.from_numpy(
                        item[
                            "image"
                        ][
                            None,
                            None,
                            ...
                        ]
                    ).to(
                        device=device,
                        dtype=torch.float32,
                        non_blocking=True,
                    )


                    target_tensor = torch.from_numpy(
                        item[
                            "target"
                        ][
                            None,
                            None,
                            ...
                        ]
                    ).to(
                        device=device,
                        dtype=torch.int8,
                        non_blocking=True,
                    )


                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                        enabled=True,
                    ):

                        logits = model(
                            image_tensor
                        )


                        bce = partial_bce(
                            logits,
                            target_tensor,
                        )


                        dice = partial_dice_fp32(
                            logits,
                            target_tensor,
                        )


                        loss = (
                            bce
                            + dice
                        )


                    if not torch.isfinite(
                        loss.detach()
                    ):

                        raise RuntimeError(
                            condition
                            + ": non-finite sparse loss."
                        )


                    scaler.scale(
                        loss
                        / ACCUMULATION_STEPS
                    ).backward()


                    pair_loss_values.append(
                        float(
                            loss.detach()
                            .float()
                            .cpu()
                        )
                    )


                    pair_bce_values.append(
                        float(
                            bce.detach()
                            .float()
                            .cpu()
                        )
                    )


                    pair_dice_values.append(
                        float(
                            dice.detach()
                            .float()
                            .cpu()
                        )
                    )


                    del (
                        image_tensor,
                        target_tensor,
                        logits,
                        bce,
                        dice,
                        loss,
                    )


                old_scale = float(
                    scaler.get_scale()
                )


                scaler.unscale_(
                    optimizer
                )


                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=GRAD_CLIP,
                    error_if_nonfinite=False,
                )


                grad_norm_value = float(
                    grad_norm.detach()
                    .float()
                    .cpu()
                )


                scaler.step(
                    optimizer
                )


                scaler.update()


                new_scale = float(
                    scaler.get_scale()
                )


                epoch_scale_min = min(
                    epoch_scale_min,
                    new_scale,
                )


                epoch_scale_max = max(
                    epoch_scale_max,
                    old_scale,
                    new_scale,
                )


                overflow = bool(
                    new_scale
                    < old_scale
                )


                if overflow:

                    retries += (
                        1
                    )

                    total_retry_attempts += (
                        1
                    )

                    epoch_retries += (
                        1
                    )

                    total_overflows += (
                        1
                    )

                    epoch_overflows += (
                        1
                    )


                    if new_scale < AMP_MIN_SCALE:

                        raise RuntimeError(
                            condition
                            + ": AMP scale fell below frozen minimum."
                        )


                    continue


                if not math.isfinite(
                    grad_norm_value
                ):

                    raise RuntimeError(
                        condition
                        + ": non-finite gradient norm without AMP overflow."
                    )


                successful_steps += (
                    1
                )


                epoch_grad_norms.append(
                    grad_norm_value
                )


                # Logical microbatches are counted ONCE only after
                # their accumulation pair successfully updates.
                for index_in_pair, item in enumerate(
                    pair_items
                ):

                    epoch_loss_sum += pair_loss_values[
                        index_in_pair
                    ]

                    epoch_bce_sum += pair_bce_values[
                        index_in_pair
                    ]

                    epoch_dice_sum += pair_dice_values[
                        index_in_pair
                    ]


                    labelled_voxels_seen += int(
                        item[
                            "labelled_count"
                        ]
                    )


                    foreground_voxels_seen += int(
                        item[
                            "foreground_count"
                        ]
                    )


                    if int(
                        item[
                            "labelled_count"
                        ]
                    ) == 0:

                        zero_label_microbatches += (
                            1
                        )


                    patch_counts[
                        item[
                            "source"
                        ]
                    ] += 1


                    case_counts[
                        item[
                            "case_id"
                        ]
                    ] += 1


                break


        expected_steps = (
            (
                epoch
                + 1
            )
            * OPTIMIZER_STEPS_PER_EPOCH
        )


        if successful_steps != expected_steps:

            raise RuntimeError(
                condition
                + ": successful optimizer-step count drifted."
            )


        for case_id in training_cases:

            if case_counts[
                case_id
            ] != MICROBATCHES_PER_CASE_PER_EPOCH:

                raise RuntimeError(
                    "Patient balance changed during training."
                )


        epoch_seconds = (
            time.perf_counter()
            - epoch_start
        )


        mean_loss = (
            epoch_loss_sum
            / MICROBATCHES_PER_EPOCH
        )


        mean_bce = (
            epoch_bce_sum
            / MICROBATCHES_PER_EPOCH
        )


        mean_dice = (
            epoch_dice_sum
            / MICROBATCHES_PER_EPOCH
        )


        current_lr = float(
            optimizer.param_groups[
                0
            ][
                "lr"
            ]
        )


        epoch_row = {
            "batch_id":
                BATCH_ID,

            "run_id":
                run_id,

            "condition":
                condition,

            "outer_fold":
                OUTER_FOLD,

            "seed":
                SEED,

            "initialization_seed":
                initialization_seed,

            "shared_initialization_sha256":
                initialization_sha,

            "paired_plan_sha256":
                paired_plan_sha,

            "epoch":
                epoch,

            "epoch_1based":
                epoch
                + 1,

            "logical_microbatches":
                MICROBATCHES_PER_EPOCH,

            "successful_optimizer_steps_total":
                successful_steps,

            "mean_loss":
                mean_loss,

            "mean_partial_bce":
                mean_bce,

            "mean_partial_dice_fp32":
                mean_dice,

            "labelled_voxels_seen":
                labelled_voxels_seen,

            "foreground_voxels_seen":
                foreground_voxels_seen,

            "zero_label_microbatches":
                zero_label_microbatches,

            "foreground_centered":
                patch_counts[
                    "foreground"
                ],

            "background_centered":
                patch_counts[
                    "background"
                ],

            "uniform_crop":
                patch_counts[
                    "uniform"
                ],

            "amp_overflow_events":
                epoch_overflows,

            "amp_retry_attempts":
                epoch_retries,

            "amp_scale_min":
                epoch_scale_min,

            "amp_scale_max":
                epoch_scale_max,

            "amp_scale_end":
                float(
                    scaler.get_scale()
                ),

            "mean_preclip_grad_norm":
                float(
                    np.mean(
                        epoch_grad_norms
                    )
                ),

            "max_preclip_grad_norm":
                float(
                    np.max(
                        epoch_grad_norms
                    )
                ),

            "learning_rate_end":
                current_lr,

            "epoch_seconds":
                epoch_seconds,

            "peak_allocated_gb":
                float(
                    torch.cuda.max_memory_allocated(
                        device
                    )
                    / (
                        1024
                        ** 3
                    )
                ),

            "peak_reserved_gb":
                float(
                    torch.cuda.max_memory_reserved(
                        device
                    )
                    / (
                        1024
                        ** 3
                    )
                ),
        }


        condition_epoch_rows.append(
            epoch_row
        )


        epoch_bar.set_postfix(
            loss=f"{mean_loss:.4f}",
            lr=f"{current_lr:.2e}",
            steps=successful_steps,
            amp=f"{float(scaler.get_scale()):.0f}",
        )


    epoch_bar.close()


    if successful_steps != TOTAL_OPTIMIZER_STEPS:

        raise RuntimeError(
            condition
            + ": final optimizer-step count is not 2400."
        )


    run_seconds = (
        time.perf_counter()
        - run_start
    )


    # --------------------------------------------------------------------------------------
    # FINAL-EPOCH CHECKPOINT ONLY
    # --------------------------------------------------------------------------------------

    final_state_cpu = {
        key:
            value.detach().cpu()
        for key, value in model.state_dict().items()
    }


    torch.save(
        {
            "project":
                "COVA-3D",

            "effective_protocol":
                EFFECTIVE_PROTOCOL,

            "block":
                BLOCK,

            "batch_id":
                BATCH_ID,

            "run_id":
                run_id,

            "condition":
                condition,

            "outer_fold":
                OUTER_FOLD,

            "seed":
                SEED,

            "initialization_seed":
                initialization_seed,

            "shared_initialization_sha256":
                initialization_sha,

            "paired_plan_sha256":
                paired_plan_sha,

            "epochs":
                EPOCHS,

            "successful_optimizer_steps":
                successful_steps,

            "checkpoint_selection":
                "FINAL_EPOCH_ONLY",

            "N1_active":
                True,

            "N2_active":
                True,

            "model_state_dict":
                final_state_cpu,
        },
        final_model_path,
    )


    final_model_sha = sha256_file(
        final_model_path
    )


    condition_epoch_df = pd.DataFrame(
        condition_epoch_rows
    )


    condition_epoch_df.to_csv(
        epoch_log_path,
        index=False,
    )


    first3_mean = float(
        condition_epoch_df[
            "mean_loss"
        ].head(
            3
        ).mean()
    )


    last3_mean = float(
        condition_epoch_df[
            "mean_loss"
        ].tail(
            3
        ).mean()
    )


    summary = {
        "batch_id":
            BATCH_ID,

        "run_id":
            run_id,

        "condition":
            condition,

        "outer_fold":
            OUTER_FOLD,

        "seed":
            SEED,

        "initialization_seed":
            initialization_seed,

        "shared_initialization_sha256":
            initialization_sha,

        "paired_plan_sha256":
            paired_plan_sha,

        "epochs":
            EPOCHS,

        "successful_optimizer_steps":
            successful_steps,

        "AMP_overflow_events":
            total_overflows,

        "AMP_retry_attempts":
            total_retry_attempts,

        "final_AMP_scale":
            float(
                scaler.get_scale()
            ),

        "first3_mean_loss":
            first3_mean,

        "last3_mean_loss":
            last3_mean,

        "last3_first3_ratio":
            float(
                last3_mean
                / first3_mean
            ),

        "final_epoch_loss":
            float(
                condition_epoch_df.iloc[
                    -1
                ][
                    "mean_loss"
                ]
            ),

        "runtime_minutes":
            run_seconds
            / 60.0,

        "final_model_path":
            str(
                final_model_path
            ),

        "final_model_sha256":
            final_model_sha,

        "checkpoint_selection":
            "FINAL_EPOCH_ONLY",

        "dense_training_masks_accessed":
            0,

        "heldout_dense_outcomes_accessed":
            0,

        "status":
            "FIT_COMPLETE",
    }


    write_json(
        run_state_path,
        {
            "status":
                "FIT_COMPLETE",

            "run_id":
                run_id,

            "condition":
                condition,

            "successful_optimizer_steps":
                successful_steps,

            "shared_initialization_sha256":
                initialization_sha,

            "paired_plan_sha256":
                paired_plan_sha,

            "final_model_sha256":
                final_model_sha,

            "summary":
                summary,
        },
    )


    all_epoch_rows.extend(
        condition_epoch_rows
    )


    run_summary_rows.append(
        summary
    )


    completed_conditions.append(
        condition
    )


    write_json(
        BATCH_RUNTIME_STATE,
        {
            "status":
                (
                    "FIT_COMPLETE"
                    if len(
                        completed_conditions
                    )
                    == 6
                    else "RUNNING"
                ),

            "block":
                BLOCK,

            "batch_id":
                BATCH_ID,

            "outer_fold":
                OUTER_FOLD,

            "seed":
                SEED,

            "initialization_seed":
                initialization_seed,

            "shared_initialization_sha256":
                initialization_sha,

            "paired_plan_sha256":
                paired_plan_sha,

            "conditions":
                CONDITIONS,

            "completed_conditions":
                completed_conditions,

            "dense_masks_opened":
                0,

            "final_outer_cv_outcome_access":
                0,

            "updated_at_utc":
                datetime.now(
                    timezone.utc
                ).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"
                ),
        },
    )


    print()
    print(
        "✓ Completed                            :",
        condition,
    )

    print(
        "  successful optimizer steps           :",
        successful_steps,
    )

    print(
        "  final loss                           :",
        f"{summary['final_epoch_loss']:.6f}",
    )

    print(
        "  AMP overflows                        :",
        total_overflows,
    )

    print(
        "  final model SHA                      :",
        final_model_sha,
    )

    print(
        "  runtime                              :",
        f"{summary['runtime_minutes']:.2f} min",
    )


    del (
        model,
        optimizer,
        scaler,
        final_state_cpu,
    )

    gc.collect()

    torch.cuda.empty_cache()


# ==========================================================================================
# 12. VERIFY COMPLETE BATCH
# ==========================================================================================

heading(
    "STEP 7/8 — VERIFY SIX COMPLETED FITS"
)


if set(
    completed_conditions
) != set(
    CONDITIONS
):

    raise RuntimeError(
        "Not all six conditions completed."
    )


epoch_df = pd.DataFrame(
    all_epoch_rows
)


summary_df = pd.DataFrame(
    run_summary_rows
)


if len(
    epoch_df
) != (
    6
    * EPOCHS
):

    raise RuntimeError(
        "Expected 240 epoch rows."
    )


if len(
    summary_df
) != 6:

    raise RuntimeError(
        "Expected six run summaries."
    )


if not (
    summary_df[
        "successful_optimizer_steps"
    ]
    == TOTAL_OPTIMIZER_STEPS
).all():

    raise RuntimeError(
        "At least one fit does not contain 2400 successful optimizer steps."
    )


if summary_df[
    "shared_initialization_sha256"
].nunique() != 1:

    raise RuntimeError(
        "Paired initialization was not identical."
    )


if summary_df[
    "paired_plan_sha256"
].nunique() != 1:

    raise RuntimeError(
        "Paired randomization plan was not identical."
    )


for _, row in summary_df.iterrows():

    model_path = Path(
        row[
            "final_model_path"
        ]
    )


    if sha256_file(
        model_path
    ) != row[
        "final_model_sha256"
    ]:

        raise RuntimeError(
            "Final checkpoint SHA verification failed."
        )


BATCH_EPOCH_LOG_REPO.parent.mkdir(
    parents=True,
    exist_ok=True,
)


epoch_df.to_csv(
    BATCH_EPOCH_LOG_REPO,
    index=False,
)


summary_df.to_csv(
    BATCH_RUN_SUMMARY_REPO,
    index=False,
)


print(
    "✓ Completed models                     : 6 /6"
)

print(
    "✓ Successful optimizer steps           :",
    int(
        summary_df[
            "successful_optimizer_steps"
        ].sum()
    ),
)

print(
    "✓ Expected batch optimizer steps       : 14400"
)

print(
    "✓ Shared initialization                : PASS"
)

print(
    "✓ Shared randomization plan            : PASS"
)

print(
    "✓ Final checkpoint SHA checks          : 6 /6"
)

print(
    "✓ Held-out dense outcomes opened       : 0"
)


# ==========================================================================================
# 13. TRAINING-CURVE FIGURE
# ==========================================================================================

FIG_PNG.parent.mkdir(
    parents=True,
    exist_ok=True,
)


fig, ax = plt.subplots(
    figsize=(
        9.0,
        5.5,
    )
)


for condition in CONDITIONS:

    sub = epoch_df[
        epoch_df[
            "condition"
        ]
        == condition
    ].sort_values(
        "epoch"
    )


    ax.plot(
        sub[
            "epoch_1based"
        ],
        sub[
            "mean_loss"
        ],
        linewidth=1.8,
        label=condition,
    )


ax.set_xlabel(
    "Epoch"
)

ax.set_ylabel(
    "Sparse training loss"
)

ax.set_title(
    "COVA-3D Factorial Batch 02 — Fold 0, Seed 29"
)

ax.grid(
    True,
    alpha=0.20,
)

ax.legend(
    ncol=2,
    frameon=False,
)

fig.tight_layout()


fig.savefig(
    FIG_PNG,
    dpi=300,
    bbox_inches="tight",
)


fig.savefig(
    FIG_PDF,
    bbox_inches="tight",
)


plt.close(
    fig
)


# ==========================================================================================
# 14. AUDIT + STATE
# ==========================================================================================

audit = {
    "project":
        "COVA-3D",

    "block":
        BLOCK,

    "status":
        "FIT_COMPLETE",

    "batch_id":
        BATCH_ID,

    "outer_fold":
        OUTER_FOLD,

    "seed":
        SEED,

    "effective_protocol":
        EFFECTIVE_PROTOCOL,

    "conditions":
        CONDITIONS,

    "fits_completed":
        6,

    "optimizer_steps_per_fit":
        TOTAL_OPTIMIZER_STEPS,

    "optimizer_steps_batch":
        int(
            summary_df[
                "successful_optimizer_steps"
            ].sum()
        ),

    "shared_initialization_sha256":
        initialization_sha,

    "paired_plan_sha256":
        paired_plan_sha,

    "pairing":
        {
            "model_initialization":
                "PASS",

            "patient_schedule":
                "PASS",

            "sampling_randomization":
                "PASS",

            "augmentation_randomization":
                "PASS",
        },

    "checkpoint_policy":
        "FINAL_EPOCH_ONLY",

    "final_model_SHA_verified":
        6,

    "runtime_minutes_total":
        float(
            summary_df[
                "runtime_minutes"
            ].sum()
        ),

    "AMP_overflow_events_total":
        int(
            summary_df[
                "AMP_overflow_events"
            ].sum()
        ),

    "firewall":
        {
            "dense_training_masks_accessed":
                0,

            "heldout_dense_outcomes_accessed":
                0,

            "final_outer_cv_outcome_access":
                0,

            "threshold_search":
                False,

            "performance_based_checkpoint_selection":
                False,
        },

    "next_block":
        NEXT_BLOCK,

    "generated_at_utc":
        datetime.now(
            timezone.utc
        ).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),
}


write_json(
    BATCH_AUDIT_PATH,
    audit,
)


state[
    "last_completed_block"
] = BLOCK

state[
    "current_stage"
] = (
    "FACTORIAL_BATCH09_FIT_COMPLETE_PREDICTION_FREEZE_PENDING"
)

state[
    "factorial_runs_completed"
] = (
    30
)

state[
    "factorial_batches_completed"
] = (
    5
)

state[
    "factorial_batch09_fit_status"
] = (
    "COMPLETE"
)

state[
    "factorial_batch09_outer_fold"
] = (
    OUTER_FOLD
)

state[
    "factorial_batch09_seed"
] = (
    SEED
)

state[
    "factorial_batch09_models"
] = (
    6
)

state[
    "factorial_batch09_shared_initialization_sha256"
] = (
    initialization_sha
)

state[
    "factorial_batch09_paired_plan_sha256"
] = (
    paired_plan_sha
)

state[
    "factorial_training_authorized"
] = (
    True
)

state[
    "method_development_authorized"
] = (
    False
)

state[
    "final_outer_cv_outcomes_authorized"
] = (
    False
)

state[
    "final_outer_cv_access_in_cova3d"
] = (
    0
)

state[
    "next_block"
] = (
    NEXT_BLOCK
)

state[
    "next_factorial_batch"
] = (
    BATCH_ID
)

state[
    "next_action"
] = (
    "Freeze image-only held-out predictions for the six completed "
    "Batch-09 models before moving to fold3_seed17."
)

state[
    "updated_at_utc"
] = datetime.now(
    timezone.utc
).strftime(
    "%Y-%m-%dT%H:%M:%SZ"
)


write_json(
    STATE_PATH,
    state,
)


project_state = json.loads(
    PROJECT_STATE_PATH.read_text(
        encoding="utf-8"
    )
)


project_state[
    "last_completed_block"
] = BLOCK

project_state[
    "current_stage"
] = (
    "cova3d_factorial_batch09_prediction_freeze_pending"
)

project_state[
    "cova3d_factorial_runs_completed"
] = (
    30
)

project_state[
    "cova3d_factorial_batches_completed"
] = (
    5
)

project_state[
    "cova3d_final_outer_cv_access"
] = (
    0
)

project_state[
    "next_action"
] = (
    "Run 10C-FACTORIAL-BATCH-09-PRED-FREEZE."
)

project_state[
    "updated_at_utc"
] = datetime.now(
    timezone.utc
).strftime(
    "%Y-%m-%dT%H:%M:%SZ"
)


write_json(
    PROJECT_STATE_PATH,
    project_state,
)


# ==========================================================================================
# 15. REGRESSION TEST
# ==========================================================================================

heading(
    "STEP 8/8 — REGRESSION / COMMIT / PUSH"
)


write_text(
    TEST_PATH,
    r'''
from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_batch09_fit_summary():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_factorial_batch09_fit_summary_v1_0.csv"
    )

    assert len(frame) == 6

    assert set(frame["condition"]) == {
        "C50_COH",
        "C50_DIS",
        "C50_FRG",
        "C100_COH",
        "C100_DIS",
        "C100_FRG",
    }

    assert (frame["outer_fold"] == 2).all()
    assert (frame["seed"] == 43).all()
    assert (frame["successful_optimizer_steps"] == 2400).all()

    assert frame["shared_initialization_sha256"].nunique() == 1
    assert frame["paired_plan_sha256"].nunique() == 1

    assert (frame["dense_training_masks_accessed"] == 0).all()
    assert (frame["heldout_dense_outcomes_accessed"] == 0).all()


def test_batch09_epoch_log():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_factorial_batch09_epoch_log_v1_0.csv"
    )

    assert len(frame) == 240

    assert (
        frame.groupby("condition")["epoch"].count()
        == 40
    ).all()

    assert (
        frame.groupby("condition")[
            "successful_optimizer_steps_total"
        ].max()
        == 2400
    ).all()


def test_batch09_audit():

    audit = json.loads(
        (
            ROOT
            / "experiments/audits/"
            "block10b_factorial_batch09_fit.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert audit["status"] == "FIT_COMPLETE"
    assert audit["batch_id"] == "fold2_seed43"
    assert audit["fits_completed"] == 6
    assert audit["optimizer_steps_batch"] == 14400

    assert audit["pairing"]["model_initialization"] == "PASS"
    assert audit["pairing"]["patient_schedule"] == "PASS"
    assert audit["pairing"]["augmentation_randomization"] == "PASS"

    assert audit["firewall"]["dense_training_masks_accessed"] == 0
    assert audit["firewall"]["heldout_dense_outcomes_accessed"] == 0
    assert audit["firewall"]["final_outer_cv_outcome_access"] == 0


def test_batch09_state():

    state = json.loads(
        (
            ROOT
            / "COVA3D_STATE.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert state["last_completed_block"] == "10B-FACTORIAL-BATCH-09"
    assert state["factorial_runs_completed"] == 30
    assert state["factorial_batches_completed"] == 5
    assert state["factorial_batch09_fit_status"] == "COMPLETE"
    assert state["final_outer_cv_outcomes_authorized"] is False
    assert state["final_outer_cv_access_in_cova3d"] == 0

    assert (
        state["next_block"]
        == "10C-FACTORIAL-BATCH-09-PRED-FREEZE"
    )
'''
)


pytest_env = os.environ.copy()


pytest_env[
    "PYTHONPATH"
] = (
    str(
        REPO
        / "src"
    )
)


tests = sh(
    [
        sys.executable,
        "-m",
        "pytest",

        "tests/test_cova3d_factorial_batch09_fit.py",

        "-q",
        "-p",
        "no:cacheprovider",
    ],
    env=pytest_env,
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
        "Batch-09 regression tests failed."
    )


# ==========================================================================================
# 16. SOURCE CAPTURE
# ==========================================================================================

source_capture = "PASS"

SOURCE_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)

SOURCE_PATH.write_text(
    Path(__file__).read_text(
        encoding="utf-8"
    ),
    encoding="utf-8",
)


audit[
    "source_capture"
] = (
    source_capture
)


write_json(
    BATCH_AUDIT_PATH,
    audit,
)


# ==========================================================================================
# 17. NORMALIZE TEXT BEFORE REPOSITORY MANIFEST
# ==========================================================================================

text_paths = [
    BATCH_EPOCH_LOG_REPO,
    BATCH_RUN_SUMMARY_REPO,
    BATCH_AUDIT_PATH,
    TEST_PATH,
    STATE_PATH,
    PROJECT_STATE_PATH,
]


if SOURCE_PATH.exists():

    text_paths.append(
        SOURCE_PATH
    )


for path in text_paths:

    content = path.read_text(
        encoding="utf-8"
    )

    path.write_text(
        content.rstrip()
        + "\n",
        encoding="utf-8",
    )


# ==========================================================================================
# 18. REPOSITORY MANIFEST
# ==========================================================================================

repo_manifest_path = (
    REPO
    / "REPOSITORY_MANIFEST.json"
)


EXCLUDED_DIRS = {
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
            and path
            != repo_manifest_path
            and not any(
                part in EXCLUDED_DIRS
                for part in path.parts
            )
        )
    ],
    key=lambda p:
        str(
            p.relative_to(
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
            BLOCK,

        "active_track":
            "COVA3D",

        "effective_protocol":
            EFFECTIVE_PROTOCOL,

        "factorial_runs_completed":
            30,

        "factorial_batches_completed":
            5,

        "final_outer_cv_outcome_access":
            0,

        "next_block":
            NEXT_BLOCK,

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
# 19. GIT COMMIT / PUSH
# ==========================================================================================

git_paths = [
    "COVA3D_STATE.json",
    "PROJECT_STATE.json",
    "REPOSITORY_MANIFEST.json",

    "data/manifests/"
    "cova3d_factorial_batch09_epoch_log_v1_0.csv",

    "data/manifests/"
    "cova3d_factorial_batch09_fit_summary_v1_0.csv",

    "experiments/audits/"
    "block10b_factorial_batch09_fit.json",

    "figures/audit/"
    "fig_cova3d_factorial_batch09_training_curves.png",

    "figures/audit/"
    "fig_cova3d_factorial_batch09_training_curves.pdf",

    "tests/"
    "test_cova3d_factorial_batch09_fit.py",
]


if SOURCE_PATH.exists():

    git_paths.append(
        "scripts/code_blocks/"
        "block10b_factorial_batch09_fit.py"
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
    path.endswith(
        ".pt"
    )
    or path.endswith(
        ".npz"
    )
    for path in staged
):

    raise RuntimeError(
        "Runtime checkpoint or data artifact was accidentally staged."
    )


unstaged = git(
    "diff",
    "--name-only",
)


if unstaged:

    raise RuntimeError(
        "Tracked unstaged files remain:\n"
        + unstaged
    )


git(
    "commit",
    "-m",
    "experiment: complete COVA-3D factorial batch 08 fits [parallel]",
)


final_commit = git(
    "rev-parse",
    "HEAD",
)


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


final_status = git(
    "status",
    "--porcelain",
)


if final_status:

    raise RuntimeError(
        "Repository not clean after Batch-09 commit:\n"
        + final_status
    )


# ==========================================================================================
# 20. FINAL REPORT
# ==========================================================================================

heading(
    "COVA-3D 10B-FACTORIAL-BATCH-09 — FINAL REPORT"
)


print(
    "Starting commit                        :",
    head[:12],
)

print(
    "Batch-09 fit commit                    :",
    final_commit[:12],
)

print(
    "GitHub synchronization                 : PASS"
)

print(
    "Repository state                       : CLEAN"
)

print()

print(
    "REAL FACTORIAL FITS"
)

print(
    "-------------------"
)

print(
    "Batch                                  : fold2_seed43"
)

print(
    "Models completed                       : 6 /6"
)

print(
    "Conditions                             :",
    ", ".join(
        CONDITIONS
    ),
)

print(
    "Successful optimizer steps / model     : 2400"
)

print(
    "Successful optimizer steps / batch     :",
    int(
        summary_df[
            "successful_optimizer_steps"
        ].sum()
    ),
)

print(
    "Shared initialization                  : PASS"
)

print(
    "Shared patient / augmentation RNG      : PASS"
)

print()

print(
    "TRAINING SUMMARY"
)

print(
    "----------------"
)


for _, row in summary_df.iterrows():

    print(
        "{:10s} final_loss={:.6f}  "
        "ratio={:.4f}  overflows={}  runtime={:.1f}m".format(
            str(
                row[
                    "condition"
                ]
            ),
            float(
                row[
                    "final_epoch_loss"
                ]
            ),
            float(
                row[
                    "last3_first3_ratio"
                ]
            ),
            int(
                row[
                    "AMP_overflow_events"
                ]
            ),
            float(
                row[
                    "runtime_minutes"
                ]
            ),
        )
    )


print()

print(
    "RUNTIME CHECKPOINTS"
)

print(
    "-------------------"
)

print(
    "Final checkpoints                     : 6"
)

print(
    "Checkpoint SHA verification            : 6 /6 PASS"
)

print(
    "Normal Git checkpoint upload           : NO"
)

print()

print(
    "FIREWALL"
)

print(
    "--------"
)

print(
    "Dense training masks opened            : 0"
)

print(
    "Held-out dense outcomes opened         : 0"
)

print(
    "Final outer-CV outcome access          : 0"
)

print(
    "Threshold search                       : NO"
)

print(
    "Performance checkpoint selection       : NO"
)

print()

print(
    "PROGRESS"
)

print(
    "--------"
)

print(
    "Factorial fits represented on branch   : 30 /72"
)

print(
    "Fit batches represented on branch      : 5 /12"
)

print(
    "Next block                             :",
    NEXT_BLOCK,
)

print()

print(
    "Exact source captured                  :",
    source_capture,
)

print()

print(
    "IMPORTANT:"
)

print(
    "Do NOT shut down the Kaggle session after this report."
)

print(
    "The six model checkpoints are runtime files."
)

print(
    "Send me this COMPLETE report next."
)

print(
    "I will immediately freeze the four held-out image-only predictions "
    "for each model before this parallel Batch-09 run is backed up."
)

print(
    "=" * 132
)
