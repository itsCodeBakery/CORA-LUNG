# ==========================================================================================
# CORA-LUNG — CODE BLOCK 08D
# Corrected Gate-B Weak-Only Fits on Training Cache v1.2
#
# FIRST VALID GATE-B FITS AFTER PRE-OUTCOME PREPROCESSING CORRECTION
#
# SCIENTIFIC PURPOSE
#
# Fit the identical partial-loss backbone under the corrected v1.2 intensity
# preprocessing and the corrected 0.50 / 0.25 / 0.25 patch-source protocol.
#
# CONDITIONS
#
#   1. complete
#   2. pixel_dropout_matched_50
#   3. component_natural_50
#   4. component_fixed_50
#
# PRIMARY GATE-B CONTRAST
#
#   pixel_dropout_matched_50
#       versus
#   component_natural_50
#
# They have the same unique positive-voxel count on each paired development
# case. This tests annotation-removal structure rather than label quantity.
#
# COMPLEMENTARY CONTRAST
#
#   complete
#       versus
#   component_fixed_50
#
# This is reported separately. It is NOT substituted for the primary
# equal-count pixel-vs-component comparison.
#
# FIT PROTOCOL
#
#   permanent development cases : 4
#   final outer-CV cases         : SEALED
#   seed                         : 17
#   epochs                       : 30
#   microbatches / epoch         : 100
#   accumulation                 : 2
#   optimizer                    : AdamW
#   LR                           : 3e-4
#   weight decay                 : 1e-4
#   AMP                          : yes
#   patch                        : 48 x 128 x 128
#   patient sampling             : uniform
#   FG-stroke patch              : 0.50
#   BG-stroke patch              : 0.25
#   uniform crop                 : 0.25
#   checkpoint selection         : FINAL EPOCH ONLY
#
# LOSS
#
#   partial BCE + partial Dice
#
# FIREWALL
#
#   Every optimizer-bound microbatch passes guard_optimizer_batch().
#
# THIS BLOCK DOES NOT:
#
#   - open infection masks
#   - open lung masks
#   - evaluate Dice against dense masks
#   - evaluate lesion recall
#   - evaluate FROC
#   - implement CORA replay
#   - use final outer-fold cases
#
# Block 08A checkpoints remain INVALIDATED and untouched.
#
# The new run root is intentionally different:
#
#   /kaggle/working/cora_gate_b_fits_v1_2
#
# so old and corrected runs can never be confused.
# ==========================================================================================


from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import subprocess
import hashlib
import importlib
import inspect
import json
import os
import sys
import random
import shutil
import time
import gc
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

from tqdm.auto import tqdm
from kaggle_secrets import UserSecretsClient

import torch


# ==========================================================================================
# 0. FROZEN CONSTANTS
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

RUN_ROOT = Path(
    "/kaggle/working/cora_gate_b_fits_v1_2"
)

EXPECTED_START_COMMIT = (
    "34f3c3f2de1c"
)

SEED = 17

PATCH_ZYX = (
    48,
    128,
    128,
)

EPOCHS = 30

MICROBATCHES_PER_EPOCH = 100

ACCUMULATION_STEPS = 2

BASE_CHANNELS = 16

EMBEDDING_DIM = 64

LEARNING_RATE = 3e-4

WEIGHT_DECAY = 1e-4

EXPECTED_FG_PROBABILITY = 0.50

EXPECTED_BG_PROBABILITY = 0.25

EXPECTED_RANDOM_PROBABILITY = 0.25

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
        + "=" * 120
    )

    print(
        text
    )

    print(
        "=" * 120
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


def atomic_json(
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

    tmp = path.with_suffix(
        path.suffix
        + ".tmp"
    )

    tmp.write_text(
        json.dumps(
            obj,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
    )


def stable_seed(
    *parts,
):

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


def reset_seed(
    seed,
):

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

    torch.backends.cudnn.benchmark = False

    torch.backends.cudnn.deterministic = True


def git_push():

    token = UserSecretsClient().get_secret(
        "pushCora"
    )

    if not token:

        raise RuntimeError(
            "Kaggle secret 'pushCora' is unavailable."
        )

    token = token.strip()

    askpass = Path(
        "/tmp/cora_git_askpass_block08d.sh"
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


def make_case_schedule(
    cases,
    epoch,
):

    if (
        MICROBATCHES_PER_EPOCH
        % len(
            cases
        )
        != 0
    ):

        raise RuntimeError(
            "Microbatches per epoch must be divisible "
            "by development-case count."
        )

    per_case = (
        MICROBATCHES_PER_EPOCH
        // len(
            cases
        )
    )

    schedule = []

    for case_id in cases:

        schedule.extend(
            [
                case_id
            ]
            * per_case
        )

    rng = np.random.default_rng(
        stable_seed(
            SEED,
            "gate_b_v1_2_patient_schedule",
            epoch,
        )
    )

    rng.shuffle(
        schedule
    )

    return schedule


def make_scaler():

    try:

        return torch.amp.GradScaler(
            "cuda",
            enabled=True,
        )

    except Exception:

        return torch.cuda.amp.GradScaler(
            enabled=True,
        )


reset_seed(
    SEED
)


# ==========================================================================================
# 2. PRECONDITIONS
# ==========================================================================================

heading(
    "CORA-LUNG BLOCK 08D — CORRECTED GATE-B WEAK-ONLY FITS"
)


if not (
    REPO
    / ".git"
).exists():

    raise RuntimeError(
        "CORA-LUNG repository missing."
    )


current_head = sh(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    cwd=REPO,
).stdout.strip()


repository_dirty = sh(
    [
        "git",
        "status",
        "--porcelain",
    ],
    cwd=REPO,
).stdout.strip()


if repository_dirty:

    raise RuntimeError(
        "Repository must be clean before Block 08D."
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


fresh_start = bool(
    state.get(
        "last_completed_block"
    )
    == "08C"
)


resume_start = bool(
    state.get(
        "current_stage"
    )
    in {
        "gate_b_corrected_fits_authorized",
        "gate_b_corrected_fits_in_progress",
    }
    and state.get(
        "gate_b_outcomes_opened"
    )
    is False
)


if not (
    fresh_start
    or resume_start
):

    raise RuntimeError(
        "Project state is not valid for Block 08D."
    )


if fresh_start:

    if not current_head.startswith(
        EXPECTED_START_COMMIT
    ):

        raise RuntimeError(
            "Unexpected Block-08C starting commit."
        )


if (
    state.get(
        "preprocess_version"
    )
    != "1.2"
):

    raise RuntimeError(
        "Active preprocessing is not v1.2."
    )


if (
    Path(
        state.get(
            "training_cache_root",
            "",
        )
    )
    != CACHE_ROOT
):

    raise RuntimeError(
        "PROJECT_STATE training cache does not point to v1.2."
    )


if (
    state.get(
        "block08a_fit_validity"
    )
    != "INVALIDATED_BEFORE_DENSE_EVALUATION"
):

    raise RuntimeError(
        "Block-08A invalidation lineage is missing."
    )


if (
    state.get(
        "dense_development_evaluation"
    )
    != "NOT_RUN"
):

    raise RuntimeError(
        "Dense development evaluation has already run."
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


if not CACHE_MANIFEST.exists():

    raise RuntimeError(
        "Corrected v1.2 cache manifest is missing."
    )


audit_08c = json.loads(
    (
        REPO
        / "experiments/audits/"
        "block08c_preprocessing_rebuild_v1_2.json"
    ).read_text(
        encoding="utf-8"
    )
)


if (
    audit_08c.get(
        "status"
    )
    != "PASS"
):

    raise RuntimeError(
        "Block-08C audit is not PASS."
    )


if (
    audit_08c.get(
        "dense_development_outcomes_opened"
    )
    is not False
):

    raise RuntimeError(
        "Block-08C audit indicates opened dense outcomes."
    )


print(
    "✓ Current commit                     :",
    current_head[
        :12
    ],
)

print(
    "✓ Corrected preprocessing            : v1.2"
)

print(
    "✓ Corrected training cache           : v1.2"
)

print(
    "✓ Dense Gate-B outcomes              : SEALED"
)

print(
    "✓ Block-08A models                   : INVALIDATED"
)

print(
    "✓ Final outer-CV cases               : SEALED"
)


# ==========================================================================================
# 3. IMPORT FROZEN TRAINING MODULES
# ==========================================================================================

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
    "cora_lung.data.training_cache",
    "cora_lung.data.pilot_dataset",
    "cora_lung.losses.partial",
    "cora_lung.models.resunet3d",
    "cora_lung.engine.checkpointing",
    "cora_lung.engine.trainer_firewall",
]:

    sys.modules.pop(
        module_name,
        None,
    )


importlib.invalidate_caches()


from cora_lung.data.training_cache import (
    validate_training_cache,
)

from cora_lung.data.pilot_dataset import (
    SparseTrainingCache,
    deterministic_patch_origin,
    extract_patch,
)

from cora_lung.losses.partial import (
    partial_bce,
    partial_dice,
)

from cora_lung.models.resunet3d import (
    CORALungResidualUNet,
)

from cora_lung.engine.checkpointing import (
    save_training_checkpoint,
    load_training_checkpoint,
)

from cora_lung.engine.trainer_firewall import (
    validate_cache_sample,
    guard_optimizer_batch,
)


# ==========================================================================================
# 4. VERIFY CACHE AND SAMPLER CONTRACT
# ==========================================================================================

heading(
    "STEP 1/8 — VERIFY v1.2 CACHE AND PATCH-SAMPLING CONTRACT"
)


firewall_summary = validate_training_cache(
    CACHE_ROOT
)


if (
    firewall_summary[
        "manifest_rows"
    ]
    != 260
):

    raise RuntimeError(
        "v1.2 training-cache firewall failed."
    )


sampler_signature = inspect.signature(
    deterministic_patch_origin
)


fg_default = float(
    sampler_signature.parameters[
        "foreground_probability"
    ].default
)


bg_default = float(
    sampler_signature.parameters[
        "background_probability"
    ].default
)


random_default = (
    1.0
    - fg_default
    - bg_default
)


if not np.isclose(
    fg_default,
    EXPECTED_FG_PROBABILITY,
):

    raise RuntimeError(
        "Foreground patch probability is not 0.50."
    )


if not np.isclose(
    bg_default,
    EXPECTED_BG_PROBABILITY,
):

    raise RuntimeError(
        "Background patch probability is not 0.25."
    )


if not np.isclose(
    random_default,
    EXPECTED_RANDOM_PROBABILITY,
):

    raise RuntimeError(
        "Uniform-crop probability is not 0.25."
    )


print(
    "✓ Cache firewall                     : PASS"
)

print(
    "✓ Foreground-stroke probability      :",
    fg_default,
)

print(
    "✓ Background-stroke probability      :",
    bg_default,
)

print(
    "✓ Uniform-crop probability           :",
    random_default,
)


# ==========================================================================================
# 5. DEVELOPMENT SPLIT + DENSE CHECKSUM FIREWALL
# ==========================================================================================

heading(
    "STEP 2/8 — VERIFY DEVELOPMENT BOUNDARY AND PRIMARY PAIR"
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
    development_cases
) != 4:

    raise RuntimeError(
        "Expected four permanent development cases."
    )


if len(
    final_cases
) != 16:

    raise RuntimeError(
        "Expected sixteen sealed final-CV volumes."
    )


dense_hash_df = pd.read_csv(
    REPO
    / "data/manifests/"
    "trainer_dense_sha256_denylist.csv"
)


dense_sha256_denylist = set(
    dense_hash_df[
        "sha256"
    ].astype(
        str
    ).str.lower()
)


if len(
    dense_sha256_denylist
) != 60:

    raise RuntimeError(
        "Expected 60 dense-file hashes."
    )


cache_manifest_df = pd.read_csv(
    CACHE_MANIFEST
)


available_conditions = set(
    cache_manifest_df[
        "condition"
    ].astype(
        str
    )
)


missing_conditions = (
    set(
        CONDITIONS
    )
    - available_conditions
)


if missing_conditions:

    raise RuntimeError(
        "Missing corrected Gate-B conditions: "
        + str(
            sorted(
                missing_conditions
            )
        )
    )


cache = SparseTrainingCache(
    CACHE_ROOT,
    development_cases,
)


# ------------------------------------------------------------------------------------------
# Verify real cache samples and the primary equal-count pair.
# ------------------------------------------------------------------------------------------

prefit_rows = []


def split_fg_bg(
    sample,
):

    coords = np.asarray(
        sample[
            "supervision_voxel_zyx"
        ],
        dtype=np.int64,
    )


    labels = np.asarray(
        sample[
            "supervision_label"
        ],
        dtype=np.int8,
    )


    fg = {
        tuple(
            int(
                x
            )
            for x in coord
        )
        for coord
        in coords[
            labels == 1
        ]
    }


    bg = {
        tuple(
            int(
                x
            )
            for x in coord
        )
        for coord
        in coords[
            labels == 0
        ]
    }


    return (
        fg,
        bg,
    )


for case_id in development_cases:

    samples = {}


    for condition in CONDITIONS:

        sample = cache.get(
            case_id,
            condition,
        )


        validate_cache_sample(
            sample,
            dense_sha256_denylist,
        )


        samples[
            condition
        ] = sample


    pixel_fg, pixel_bg = split_fg_bg(
        samples[
            PRIMARY_CONTROL
        ]
    )


    natural_fg, natural_bg = split_fg_bg(
        samples[
            PRIMARY_OMISSION
        ]
    )


    complete_fg, complete_bg = split_fg_bg(
        samples[
            COMPLEMENTARY_REFERENCE
        ]
    )


    fixed_fg, fixed_bg = split_fg_bg(
        samples[
            COMPLEMENTARY_OMISSION
        ]
    )


    if len(
        pixel_fg
    ) != len(
        natural_fg
    ):

        raise RuntimeError(
            f"Primary equal-count FG mismatch: {case_id}"
        )


    if pixel_bg != natural_bg:

        raise RuntimeError(
            f"Primary paired BG mismatch: {case_id}"
        )


    prefit_rows.append(
        {
            "case_id":
                case_id,

            "pixel_dropout_matched_fg":
                len(
                    pixel_fg
                ),

            "component_natural_fg":
                len(
                    natural_fg
                ),

            "primary_fg_count_equal":
                True,

            "primary_bg_coordinates_equal":
                True,

            "complete_fg":
                len(
                    complete_fg
                ),

            "component_fixed_fg":
                len(
                    fixed_fg
                ),

            "complete_bg":
                len(
                    complete_bg
                ),

            "component_fixed_bg":
                len(
                    fixed_bg
                ),
        }
    )


prefit_df = pd.DataFrame(
    prefit_rows
)


print(
    "Permanent development cases:"
)


for case_id in development_cases:

    print(
        "  •",
        case_id,
    )


print()
print(
    "✓ Primary equal-count FG pair        : 4/4"
)

print(
    "✓ Primary paired BG realization      : 4/4"
)

print(
    "✓ Final outer-CV volumes accessed    : 0"
)

print(
    "✓ Dense arrays accessed              : 0"
)


# ==========================================================================================
# 6. VERIFY FROZEN GATE-B CONFIG
# ==========================================================================================

heading(
    "STEP 3/8 — VERIFY CORRECTED GATE-B CONFIGURATION"
)


gate_b_config_path = (
    REPO
    / "configs/"
    "gate_b_problem_validation.yaml"
)


gate_b_cfg = yaml.safe_load(
    gate_b_config_path.read_text(
        encoding="utf-8"
    )
)


configured_conditions = list(
    gate_b_cfg[
        "conditions"
    ]
)


if configured_conditions != CONDITIONS:

    raise RuntimeError(
        "Gate-B condition registry differs from Block-08D contract.\n"
        f"Config={configured_conditions}\n"
        f"Expected={CONDITIONS}"
    )


if (
    str(
        gate_b_cfg[
            "data"
        ][
            "training_cache_version"
        ]
    )
    != "1.2"
):

    raise RuntimeError(
        "Gate-B config does not specify training cache v1.2."
    )


if not np.isclose(
    float(
        gate_b_cfg[
            "patch_sampling"
        ][
            "foreground_stroke_probability"
        ]
    ),
    0.50,
):

    raise RuntimeError(
        "Gate-B config FG patch probability mismatch."
    )


if not np.isclose(
    float(
        gate_b_cfg[
            "patch_sampling"
        ][
            "background_stroke_probability"
        ]
    ),
    0.25,
):

    raise RuntimeError(
        "Gate-B config BG patch probability mismatch."
    )


if not np.isclose(
    float(
        gate_b_cfg[
            "patch_sampling"
        ][
            "uniform_crop_probability"
        ]
    ),
    0.25,
):

    raise RuntimeError(
        "Gate-B config uniform-crop probability mismatch."
    )


print(
    "✓ Gate-B conditions                  : 4"
)

print(
    "✓ Training cache                     : v1.2"
)

print(
    "✓ Seed                               : 17"
)

print(
    "✓ Epochs                             : 30"
)

print(
    "✓ Microbatches / epoch               : 100"
)

print(
    "✓ Accumulation                       : 2"
)

print(
    "✓ Final checkpoint policy            : FINAL EPOCH"
)


# ==========================================================================================
# 7. AUTHORIZE CORRECTED FITS BEFORE OPTIMIZER STEP 1
# ==========================================================================================

heading(
    "STEP 4/8 — FREEZE CORRECTED TRAINING AUTHORIZATION"
)


authorization_commit = None


if fresh_start:

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
            "CORA-LUNG — CODE BLOCK 08D"
            in cell
        ):

            source_path = (
                REPO
                / "scripts/code_blocks/"
                "block08d_corrected_gate_b_fits_v1_2.py"
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

        source_capture = "NOT_AVAILABLE"


    prefit_path = (
        REPO
        / "data/manifests/"
        "gate_b_v1_2_prefit_audit.csv"
    )


    prefit_df.to_csv(
        prefit_path,
        index=False,
    )


    authorization = {
        "project":
            "CORA-Lung",

        "block":
            "08D",

        "status":
            "AUTHORIZED",

        "authorized_at_utc":
            NOW_ISO,

        "parent_commit":
            current_head,

        "training_cache":
            str(
                CACHE_ROOT
            ),

        "training_cache_version":
            "1.2",

        "development_cases":
            development_cases,

        "sealed_final_cases":
            16,

        "conditions":
            CONDITIONS,

        "primary_equal_count_pair":
            [
                PRIMARY_CONTROL,
                PRIMARY_OMISSION,
            ],

        "complementary_pair":
            [
                COMPLEMENTARY_REFERENCE,
                COMPLEMENTARY_OMISSION,
            ],

        "seed":
            SEED,

        "epochs":
            EPOCHS,

        "microbatches_per_epoch":
            MICROBATCHES_PER_EPOCH,

        "accumulation_steps":
            ACCUMULATION_STEPS,

        "patch_sampling": {
            "foreground":
                EXPECTED_FG_PROBABILITY,

            "background":
                EXPECTED_BG_PROBABILITY,

            "uniform_crop":
                EXPECTED_RANDOM_PROBABILITY,
        },

        "dense_training_access":
            False,

        "dense_development_evaluation":
            False,

        "checkpoint_selection":
            "FINAL_EPOCH_ONLY",

        "block08a_models":
            "INVALIDATED_AND_EXCLUDED",

        "source_capture":
            source_capture,
    }


    write_json(
        REPO
        / "experiments/audits/"
        "block08d_corrected_training_authorization.json",
        authorization,
    )


    state.update(
        {
            "last_attempted_block":
                "08D",

            "current_stage":
                "gate_b_corrected_fits_authorized",

            "current_gate":
                "B",

            "gate_b":
                "IN_PROGRESS_CORRECTED_FITS",

            "corrected_gate_b_training_authorized":
                True,

            "valid_gate_b_optimizer_steps":
                0,

            "dense_development_evaluation":
                "NOT_RUN",

            "gate_b_outcomes_opened":
                False,

            "next_action":
                (
                    "Run four corrected v1.2 weak-only Gate-B fits."
                ),

            "updated_at_utc":
                NOW_ISO,
        }
    )


    write_json(
        state_path,
        state,
    )


    git_paths = [
        "PROJECT_STATE.json",
        "data/manifests/gate_b_v1_2_prefit_audit.csv",
        "experiments/audits/"
        "block08d_corrected_training_authorization.json",
    ]


    captured_source_path = (
        REPO
        / "scripts/code_blocks/"
        "block08d_corrected_gate_b_fits_v1_2.py"
    )


    if captured_source_path.exists():

        git_paths.append(
            "scripts/code_blocks/"
            "block08d_corrected_gate_b_fits_v1_2.py"
        )


    sh(
        [
            "git",
            "add",
            *git_paths,
        ],
        cwd=REPO,
    )


    sh(
        [
            "git",
            "commit",
            "-m",
            "train: authorize corrected Gate-B fits on cache v1.2",
        ],
        cwd=REPO,
    )


    git_push()


    authorization_commit = sh(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=REPO,
    ).stdout.strip()


    print(
        "✓ Authorization commit              :",
        authorization_commit[
            :12
        ],
    )


else:

    authorization_commit = current_head


    print(
        "✓ Resume authorization commit       :",
        authorization_commit[
            :12
        ],
    )


# Repository must be clean before actual optimizer activity.
if sh(
    [
        "git",
        "status",
        "--porcelain",
    ],
    cwd=REPO,
).stdout.strip():

    raise RuntimeError(
        "Repository became dirty before corrected fitting."
    )


config_hash = sha256_file(
    gate_b_config_path
)


cache_manifest_hash = sha256_file(
    CACHE_MANIFEST
)


# ==========================================================================================
# 8. INITIALIZE TRAINING PROGRAM
# ==========================================================================================

heading(
    "STEP 5/8 — INITIALIZE CORRECTED SCIENTIFIC FITS"
)


device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


if device.type != "cuda":

    raise RuntimeError(
        "Corrected Gate-B fitting requires Kaggle CUDA."
    )


RUN_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


print(
    "GPU                                :",
    torch.cuda.get_device_name(
        0
    ),
)

print(
    "Corrected run root                 :",
    RUN_ROOT,
)

print(
    "Models                             :",
    len(
        CONDITIONS
    ),
)

print(
    "Optimizer steps / model            :",
    EPOCHS
    * MICROBATCHES_PER_EPOCH
    // ACCUMULATION_STEPS,
)

print(
    "Total valid planned steps          :",
    len(
        CONDITIONS
    )
    * EPOCHS
    * MICROBATCHES_PER_EPOCH
    // ACCUMULATION_STEPS,
)


def load_condition_samples(
    condition,
):

    output = {}


    for case_id in development_cases:

        sample = cache.get(
            case_id,
            condition,
        )


        validate_cache_sample(
            sample,
            dense_sha256_denylist,
        )


        output[
            case_id
        ] = sample


    return output


# ==========================================================================================
# 9. FIT FOUR CORRECTED MODELS
# ==========================================================================================

heading(
    "STEP 6/8 — RUN FOUR CORRECTED GATE-B FITS"
)


fit_summaries = []

all_epoch_rows = []


for condition_number, condition in enumerate(
    CONDITIONS,
    start=1,
):

    print(
        "\n"
        + "#" * 120
    )

    print(
        "CORRECTED FIT "
        + str(
            condition_number
        )
        + "/"
        + str(
            len(
                CONDITIONS
            )
        )
        + " — "
        + condition
    )

    print(
        "#" * 120
    )


    run_id = (
        "gateb_v1_2_"
        + condition
        + "_seed17"
    )


    run_dir = (
        RUN_ROOT
        / run_id
    )


    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )


    run_spec = {
        "run_id":
            run_id,

        "condition":
            condition,

        "cache_version":
            "1.2",

        "seed":
            SEED,

        "development_cases":
            development_cases,

        "epochs":
            EPOCHS,

        "microbatches_per_epoch":
            MICROBATCHES_PER_EPOCH,

        "accumulation_steps":
            ACCUMULATION_STEPS,

        "patch_zyx":
            list(
                PATCH_ZYX
            ),

        "patch_probabilities": {
            "foreground":
                EXPECTED_FG_PROBABILITY,

            "background":
                EXPECTED_BG_PROBABILITY,

            "uniform":
                EXPECTED_RANDOM_PROBABILITY,
        },

        "optimizer":
            "AdamW",

        "learning_rate":
            LEARNING_RATE,

        "weight_decay":
            WEIGHT_DECAY,

        "base_channels":
            BASE_CHANNELS,

        "embedding_dim":
            EMBEDDING_DIM,

        "config_sha256":
            config_hash,

        "cache_manifest_sha256":
            cache_manifest_hash,

        "authorization_commit":
            authorization_commit,

        "dense_training_access":
            False,

        "dense_evaluation_during_fit":
            False,

        "checkpoint_selection":
            "FINAL_EPOCH_ONLY",
    }


    run_spec_path = (
        run_dir
        / "run_spec.json"
    )


    if run_spec_path.exists():

        existing_spec = json.loads(
            run_spec_path.read_text(
                encoding="utf-8"
            )
        )


        if existing_spec != run_spec:

            raise RuntimeError(
                "Run specification mismatch for "
                + run_id
            )


    else:

        atomic_json(
            run_spec_path,
            run_spec,
        )


    final_model_path = (
        run_dir
        / "final_model.pt"
    )


    final_manifest_path = (
        run_dir
        / "final_manifest.json"
    )


    epoch_log_path = (
        run_dir
        / "epoch_log.csv"
    )


    latest_checkpoint_path = (
        run_dir
        / "latest_checkpoint.pt"
    )


    # --------------------------------------------------------------------------------------
    # Completed corrected run: validate and reuse.
    # --------------------------------------------------------------------------------------

    if (
        final_model_path.exists()
        and final_manifest_path.exists()
    ):

        final_manifest = json.loads(
            final_manifest_path.read_text(
                encoding="utf-8"
            )
        )


        if (
            final_manifest.get(
                "status"
            )
            != "COMPLETE"
        ):

            raise RuntimeError(
                "Invalid completed run state for "
                + run_id
            )


        if (
            final_manifest.get(
                "cache_manifest_sha256"
            )
            != cache_manifest_hash
        ):

            raise RuntimeError(
                "Completed run cache lineage mismatch."
            )


        if (
            final_manifest.get(
                "config_sha256"
            )
            != config_hash
        ):

            raise RuntimeError(
                "Completed run config lineage mismatch."
            )


        if (
            sha256_file(
                final_model_path
            )
            != final_manifest[
                "final_model_sha256"
            ]
        ):

            raise RuntimeError(
                "Completed final model checksum mismatch."
            )


        print(
            "✓ Existing corrected final model verified."
        )


        fit_summaries.append(
            final_manifest
        )


        existing_epochs = pd.read_csv(
            epoch_log_path
        )


        all_epoch_rows.extend(
            existing_epochs.to_dict(
                "records"
            )
        )


        continue


    # --------------------------------------------------------------------------------------
    # Identical initial parameter seed across every condition.
    # --------------------------------------------------------------------------------------

    reset_seed(
        SEED
    )


    model = CORALungResidualUNet(
        in_channels=1,
        base_channels=BASE_CHANNELS,
        embedding_dim=EMBEDDING_DIM,
    ).to(
        device
    )


    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    scaler = make_scaler()


    samples = load_condition_samples(
        condition
    )


    start_epoch = 0

    optimizer_steps = 0


    if latest_checkpoint_path.exists():

        payload = load_training_checkpoint(
            latest_checkpoint_path,
            expected_config_hash=config_hash,
            expected_cache_manifest_hash=cache_manifest_hash,
        )


        if (
            payload[
                "run_id"
            ]
            != run_id
        ):

            raise RuntimeError(
                "Checkpoint run ID mismatch."
            )


        model.load_state_dict(
            payload[
                "model"
            ]
        )


        optimizer.load_state_dict(
            payload[
                "optimizer"
            ]
        )


        if (
            payload.get(
                "scaler"
            )
            is not None
        ):

            scaler.load_state_dict(
                payload[
                    "scaler"
                ]
            )


        start_epoch = (
            int(
                payload[
                    "epoch"
                ]
            )
            + 1
        )


        optimizer_steps = int(
            payload[
                "global_step"
            ]
        )


        print(
            "✓ Resuming at epoch "
            + str(
                start_epoch
                + 1
            )
            + "/"
            + str(
                EPOCHS
            )
            + "; optimizer steps="
            + str(
                optimizer_steps
            )
        )


    if epoch_log_path.exists():

        epoch_rows = pd.read_csv(
            epoch_log_path
        ).to_dict(
            "records"
        )


    else:

        epoch_rows = []


    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(
        device
    )


    fit_start = time.perf_counter()


    for epoch in range(
        start_epoch,
        EPOCHS,
    ):

        model.train()


        schedule = make_case_schedule(
            development_cases,
            epoch,
        )


        # Exact patient-uniform schedule.
        case_counts = pd.Series(
            schedule
        ).value_counts()


        if not (
            case_counts
            == (
                MICROBATCHES_PER_EPOCH
                // len(
                    development_cases
                )
            )
        ).all():

            raise RuntimeError(
                "Patient-uniform epoch schedule failed."
            )


        optimizer.zero_grad(
            set_to_none=True
        )


        loss_sum = 0.0

        bce_sum = 0.0

        dice_sum = 0.0

        labelled_voxel_sum = 0

        foreground_voxel_sum = 0

        zero_label_microbatches = 0


        source_counts = {
            "foreground":
                0,

            "background":
                0,

            "random":
                0,
        }


        epoch_start = time.perf_counter()


        progress = tqdm(
            range(
                MICROBATCHES_PER_EPOCH
            ),
            desc=(
                condition
                + " | epoch "
                + str(
                    epoch
                    + 1
                ).zfill(
                    2
                )
                + "/"
                + str(
                    EPOCHS
                )
            ),
            leave=False,
        )


        for microbatch_index in progress:

            case_id = schedule[
                microbatch_index
            ]


            global_microbatch_index = (
                epoch
                * MICROBATCHES_PER_EPOCH
                + microbatch_index
            )


            patch = extract_patch(
                samples[
                    case_id
                ],
                PATCH_ZYX,
                seed=SEED,
                sample_index=global_microbatch_index,
            )


            # ==================================================================================
            # HARD OPTIMIZER-BOUNDARY FIREWALL
            # ==================================================================================

            guard_optimizer_batch(
                patch,
                dense_sha256_denylist,
            )


            source = str(
                patch[
                    "sampling_source"
                ]
            )


            if source not in source_counts:

                raise RuntimeError(
                    "Unexpected patch source: "
                    + source
                )


            source_counts[
                source
            ] += 1


            image = (
                patch[
                    "image"
                ]
                .unsqueeze(
                    0
                )
                .to(
                    device=device,
                    dtype=torch.float32,
                    non_blocking=True,
                )
            )


            target = (
                patch[
                    "target"
                ]
                .unsqueeze(
                    0
                )
                .to(
                    device=device,
                    dtype=torch.int8,
                    non_blocking=True,
                )
            )


            labelled_count = int(
                (
                    target
                    != -1
                ).sum().item()
            )


            foreground_count = int(
                (
                    target
                    == 1
                ).sum().item()
            )


            labelled_voxel_sum += (
                labelled_count
            )


            foreground_voxel_sum += (
                foreground_count
            )


            if labelled_count == 0:

                zero_label_microbatches += 1


            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=True,
            ):

                output = model(
                    image
                )


                bce = partial_bce(
                    output[
                        "logits"
                    ],
                    target,
                )


                dice = partial_dice(
                    output[
                        "logits"
                    ],
                    target,
                )


                raw_loss = (
                    bce
                    + dice
                )


                scaled_loss = (
                    raw_loss
                    / ACCUMULATION_STEPS
                )


            scaler.scale(
                scaled_loss
            ).backward()


            loss_sum += float(
                raw_loss.detach().cpu()
            )


            bce_sum += float(
                bce.detach().cpu()
            )


            dice_sum += float(
                dice.detach().cpu()
            )


            step_due = bool(
                (
                    microbatch_index
                    + 1
                )
                % ACCUMULATION_STEPS
                == 0
            )


            if step_due:

                scaler.step(
                    optimizer
                )


                scaler.update()


                optimizer.zero_grad(
                    set_to_none=True
                )


                optimizer_steps += 1


            progress.set_postfix(
                loss=(
                    loss_sum
                    / (
                        microbatch_index
                        + 1
                    )
                ),
                labels=labelled_count,
                source=source,
            )


            del image
            del target
            del output
            del bce
            del dice
            del raw_loss
            del scaled_loss


        if (
            MICROBATCHES_PER_EPOCH
            % ACCUMULATION_STEPS
            != 0
        ):

            raise RuntimeError(
                "Partial accumulation flush would be required."
            )


        torch.cuda.synchronize()


        epoch_seconds = float(
            time.perf_counter()
            - epoch_start
        )


        # The exact realized count is logged. We do not force every
        # 100-sample epoch to equal 50/25/25 exactly; the sampler is stochastic
        # with frozen deterministic RNG and the distribution is validated
        # separately by the protocol regression test.
        epoch_row = {
            "run_id":
                run_id,

            "condition":
                condition,

            "seed":
                SEED,

            "cache_version":
                "1.2",

            "epoch":
                epoch,

            "epoch_1based":
                epoch
                + 1,

            "mean_loss":
                loss_sum
                / MICROBATCHES_PER_EPOCH,

            "mean_partial_bce":
                bce_sum
                / MICROBATCHES_PER_EPOCH,

            "mean_partial_dice":
                dice_sum
                / MICROBATCHES_PER_EPOCH,

            "labelled_voxels_seen":
                labelled_voxel_sum,

            "foreground_voxels_seen":
                foreground_voxel_sum,

            "zero_label_microbatches":
                zero_label_microbatches,

            "foreground_centered":
                source_counts[
                    "foreground"
                ],

            "background_centered":
                source_counts[
                    "background"
                ],

            "uniform_crop":
                source_counts[
                    "random"
                ],

            "optimizer_steps_total":
                optimizer_steps,

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


        # Idempotent resume logging.
        epoch_rows = [
            row
            for row in epoch_rows
            if int(
                row[
                    "epoch"
                ]
            )
            != epoch
        ]


        epoch_rows.append(
            epoch_row
        )


        pd.DataFrame(
            epoch_rows
        ).sort_values(
            "epoch"
        ).to_csv(
            epoch_log_path,
            index=False,
        )


        checkpoint_sha = save_training_checkpoint(
            latest_checkpoint_path,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            global_step=optimizer_steps,
            config_hash=config_hash,
            cache_manifest_hash=cache_manifest_hash,
            run_id=run_id,
        )


        atomic_json(
            run_dir
            / "run_state.json",
            {
                "status":
                    (
                        "COMPLETE"
                        if epoch
                        == EPOCHS
                        - 1
                        else "IN_PROGRESS"
                    ),

                "run_id":
                    run_id,

                "condition":
                    condition,

                "cache_version":
                    "1.2",

                "completed_epoch":
                    epoch,

                "optimizer_steps":
                    optimizer_steps,

                "latest_checkpoint":
                    latest_checkpoint_path.name,

                "latest_checkpoint_sha256":
                    checkpoint_sha,

                "updated_at_utc":
                    datetime.now(
                        timezone.utc
                    ).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
            },
        )


        print(
            "  Epoch "
            + str(
                epoch
                + 1
            ).zfill(
                2
            )
            + "/"
            + str(
                EPOCHS
            )
            + " | loss="
            + "{:.5f}".format(
                epoch_row[
                    "mean_loss"
                ]
            )
            + " | BCE="
            + "{:.5f}".format(
                epoch_row[
                    "mean_partial_bce"
                ]
            )
            + " | DiceLoss="
            + "{:.5f}".format(
                epoch_row[
                    "mean_partial_dice"
                ]
            )
            + " | FG/BG/U="
            + str(
                source_counts[
                    "foreground"
                ]
            )
            + "/"
            + str(
                source_counts[
                    "background"
                ]
            )
            + "/"
            + str(
                source_counts[
                    "random"
                ]
            )
            + " | steps="
            + str(
                optimizer_steps
            )
            + " | "
            + "{:.1f}".format(
                epoch_seconds
            )
            + "s"
        )


    # ======================================================================================
    # FINAL CHECKPOINT — NO PERFORMANCE-BASED MODEL SELECTION
    # ======================================================================================

    expected_optimizer_steps = (
        EPOCHS
        * MICROBATCHES_PER_EPOCH
        // ACCUMULATION_STEPS
    )


    if optimizer_steps != expected_optimizer_steps:

        raise RuntimeError(
            condition
            + " optimizer-step count is "
            + str(
                optimizer_steps
            )
            + "; expected "
            + str(
                expected_optimizer_steps
            )
        )


    final_epoch_df = pd.DataFrame(
        epoch_rows
    ).sort_values(
        "epoch"
    )


    if len(
        final_epoch_df
    ) != EPOCHS:

        raise RuntimeError(
            "Expected 30 logged epochs for "
            + condition
        )


    final_row = final_epoch_df.iloc[
        -1
    ]


    final_payload = {
        "run_id":
            run_id,

        "condition":
            condition,

        "cache_version":
            "1.2",

        "seed":
            SEED,

        "final_epoch":
            EPOCHS
            - 1,

        "optimizer_steps":
            optimizer_steps,

        "config_sha256":
            config_hash,

        "cache_manifest_sha256":
            cache_manifest_hash,

        "authorization_commit":
            authorization_commit,

        "model_state_dict":
            model.state_dict(),
    }


    tmp_final_path = (
        run_dir
        / "final_model.pt.tmp"
    )


    torch.save(
        final_payload,
        tmp_final_path,
    )


    os.replace(
        tmp_final_path,
        final_model_path,
    )


    final_model_sha = sha256_file(
        final_model_path
    )


    total_fit_seconds = float(
        time.perf_counter()
        - fit_start
    )


    final_manifest = {
        "status":
            "COMPLETE",

        "run_id":
            run_id,

        "condition":
            condition,

        "cache_version":
            "1.2",

        "seed":
            SEED,

        "epochs":
            EPOCHS,

        "microbatches_per_epoch":
            MICROBATCHES_PER_EPOCH,

        "accumulation_steps":
            ACCUMULATION_STEPS,

        "optimizer_steps":
            optimizer_steps,

        "final_epoch":
            EPOCHS
            - 1,

        "checkpoint_selection":
            "FINAL_EPOCH_ONLY",

        "dense_training_access":
            False,

        "dense_development_evaluation":
            False,

        "development_cases":
            development_cases,

        "final_outer_cv_cases_used":
            0,

        "patch_sampling": {
            "foreground":
                EXPECTED_FG_PROBABILITY,

            "background":
                EXPECTED_BG_PROBABILITY,

            "uniform":
                EXPECTED_RANDOM_PROBABILITY,
        },

        "config_sha256":
            config_hash,

        "cache_manifest_sha256":
            cache_manifest_hash,

        "authorization_commit":
            authorization_commit,

        "final_model_file":
            str(
                final_model_path
            ),

        "final_model_sha256":
            final_model_sha,

        "final_mean_loss":
            float(
                final_row[
                    "mean_loss"
                ]
            ),

        "final_mean_partial_bce":
            float(
                final_row[
                    "mean_partial_bce"
                ]
            ),

        "final_mean_partial_dice":
            float(
                final_row[
                    "mean_partial_dice"
                ]
            ),

        "fit_seconds_this_session":
            total_fit_seconds,

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

        "completed_at_utc":
            datetime.now(
                timezone.utc
            ).strftime(
                "%Y-%m-%dT%H:%M:%SZ"
            ),
    }


    atomic_json(
        final_manifest_path,
        final_manifest,
    )


    fit_summaries.append(
        final_manifest
    )


    all_epoch_rows.extend(
        final_epoch_df.to_dict(
            "records"
        )
    )


    print(
        "✓ Corrected final model frozen:",
        final_model_sha[
            :12
        ],
    )


    del model
    del optimizer
    del scaler
    del samples

    gc.collect()

    torch.cuda.empty_cache()


# ==========================================================================================
# 10. VERIFY ALL FOUR CORRECTED FITS
# ==========================================================================================

heading(
    "STEP 7/8 — VERIFY SEALED CORRECTED FITS"
)


summary_df = pd.DataFrame(
    fit_summaries
)


if len(
    summary_df
) != 4:

    raise RuntimeError(
        "Expected exactly four corrected Gate-B final models."
    )


if set(
    summary_df[
        "condition"
    ]
) != set(
    CONDITIONS
):

    raise RuntimeError(
        "Corrected Gate-B final condition set is incomplete."
    )


if not (
    summary_df[
        "status"
    ]
    == "COMPLETE"
).all():

    raise RuntimeError(
        "At least one corrected fit is incomplete."
    )


expected_steps_per_condition = (
    EPOCHS
    * MICROBATCHES_PER_EPOCH
    // ACCUMULATION_STEPS
)


if not (
    summary_df[
        "optimizer_steps"
    ]
    == expected_steps_per_condition
).all():

    raise RuntimeError(
        "Corrected optimizer-step count is inconsistent."
    )


if not (
    summary_df[
        "checkpoint_selection"
    ]
    == "FINAL_EPOCH_ONLY"
).all():

    raise RuntimeError(
        "Non-final checkpoint selection detected."
    )


if not (
    summary_df[
        "dense_training_access"
    ]
    == False
).all():

    raise RuntimeError(
        "Dense training access detected."
    )


if not (
    summary_df[
        "dense_development_evaluation"
    ]
    == False
).all():

    raise RuntimeError(
        "Dense development evaluation occurred during fitting."
    )


valid_optimizer_steps = int(
    summary_df[
        "optimizer_steps"
    ].sum()
)


print(
    "✓ Corrected frozen models             : 4/4"
)

print(
    "✓ Optimizer steps / model             :",
    expected_steps_per_condition,
)

print(
    "✓ Total valid Gate-B optimizer steps  :",
    valid_optimizer_steps,
)

print(
    "✓ Checkpoint selection                : FINAL EPOCH ONLY"
)

print(
    "✓ Dense labels during fit             : NONE"
)

print(
    "✓ Final outer-CV cases used           : 0"
)


# ==========================================================================================
# 11. TRAINING-ONLY AUDIT / CURVES
# ==========================================================================================

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
    / "figures/audit"
)


summary_repo_path = (
    manifest_dir
    / "gate_b_v1_2_weak_only_fit_summary.csv"
)


summary_df[
    [
        "run_id",
        "condition",
        "cache_version",
        "seed",
        "epochs",
        "microbatches_per_epoch",
        "accumulation_steps",
        "optimizer_steps",
        "checkpoint_selection",
        "dense_training_access",
        "dense_development_evaluation",
        "final_outer_cv_cases_used",
        "config_sha256",
        "cache_manifest_sha256",
        "authorization_commit",
        "final_model_sha256",
        "final_mean_loss",
        "final_mean_partial_bce",
        "final_mean_partial_dice",
        "fit_seconds_this_session",
        "peak_allocated_gb",
        "peak_reserved_gb",
    ]
].to_csv(
    summary_repo_path,
    index=False,
)


epoch_df = pd.DataFrame(
    all_epoch_rows
)


epoch_df = (
    epoch_df.sort_values(
        [
            "condition",
            "epoch",
        ]
    )
    .drop_duplicates(
        [
            "condition",
            "epoch",
        ],
        keep="last",
    )
)


if len(
    epoch_df
) != (
    len(
        CONDITIONS
    )
    * EPOCHS
):

    raise RuntimeError(
        "Corrected epoch-log row count mismatch."
    )


epoch_repo_path = (
    manifest_dir
    / "gate_b_v1_2_epoch_log.csv"
)


epoch_df.to_csv(
    epoch_repo_path,
    index=False,
)


# ------------------------------------------------------------------------------------------
# Training-loss figure is diagnostic only.
# ------------------------------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(
        11,
        6.2,
    )
)


for condition in CONDITIONS:

    subset = epoch_df[
        epoch_df[
            "condition"
        ]
        == condition
    ].sort_values(
        "epoch"
    )


    ax.plot(
        subset[
            "epoch_1based"
        ],
        subset[
            "mean_loss"
        ],
        marker="o",
        markersize=2.5,
        linewidth=1.4,
        label=condition,
    )


ax.set_xlabel(
    "Epoch",
    fontweight="bold",
)


ax.set_ylabel(
    "Mean Sparse Training Loss",
    fontweight="bold",
)


ax.set_title(
    "Corrected Gate-B Weak-Only Fits on CT Cache v1.2\n"
    "Training Diagnostics Only — Dense Development Outcomes Remain Sealed",
    fontweight="bold",
)


ax.grid(
    alpha=0.25,
)


ax.legend(
    fontsize=8,
)


fig.tight_layout()


figure_dir.mkdir(
    parents=True,
    exist_ok=True,
)


fig.savefig(
    figure_dir
    / "fig13_gate_b_v1_2_training_curves.png",
    dpi=600,
    bbox_inches="tight",
)


fig.savefig(
    figure_dir
    / "fig13_gate_b_v1_2_training_curves.pdf",
    bbox_inches="tight",
)


plt.close(
    fig
)


# ------------------------------------------------------------------------------------------
# Aggregate realized patch-source distribution.
# ------------------------------------------------------------------------------------------

sampling_summary = (
    epoch_df.groupby(
        "condition",
        as_index=False,
    )[
        [
            "foreground_centered",
            "background_centered",
            "uniform_crop",
        ]
    ]
    .sum()
)


for _, row in sampling_summary.iterrows():

    total = (
        int(
            row[
                "foreground_centered"
            ]
        )
        + int(
            row[
                "background_centered"
            ]
        )
        + int(
            row[
                "uniform_crop"
            ]
        )
    )


    if total != (
        EPOCHS
        * MICROBATCHES_PER_EPOCH
    ):

        raise RuntimeError(
            "Patch-source accounting mismatch."
        )


sampling_summary.to_csv(
    manifest_dir
    / "gate_b_v1_2_patch_sampling_realization.csv",
    index=False,
)


# ==========================================================================================
# 12. FREEZE BLOCK-08D SCIENTIFIC AUDIT
# ==========================================================================================

fit_audit = {
    "project":
        "CORA-Lung",

    "block":
        "08D",

    "status":
        "PASS",

    "completed_at_utc":
        datetime.now(
            timezone.utc
        ).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),

    "stage":
        "CORRECTED_WEAK_ONLY_FITS_COMPLETE_DENSE_EVAL_PENDING",

    "cache_version":
        "1.2",

    "preprocess_version":
        "1.2",

    "development_cases":
        development_cases,

    "final_outer_cv_cases_used":
        0,

    "dense_mask_arrays_accessed":
        0,

    "conditions":
        CONDITIONS,

    "primary_equal_count_pair":
        [
            PRIMARY_CONTROL,
            PRIMARY_OMISSION,
        ],

    "complementary_pair":
        [
            COMPLEMENTARY_REFERENCE,
            COMPLEMENTARY_OMISSION,
        ],

    "seed":
        SEED,

    "epochs":
        EPOCHS,

    "microbatches_per_epoch":
        MICROBATCHES_PER_EPOCH,

    "accumulation_steps":
        ACCUMULATION_STEPS,

    "patch_sampling": {
        "foreground":
            EXPECTED_FG_PROBABILITY,

        "background":
            EXPECTED_BG_PROBABILITY,

        "uniform":
            EXPECTED_RANDOM_PROBABILITY,
    },

    "optimizer_steps_per_condition":
        expected_steps_per_condition,

    "valid_optimizer_steps":
        valid_optimizer_steps,

    "invalidated_block08a_optimizer_steps":
        7500,

    "checkpoint_policy":
        "FINAL_EPOCH_ONLY",

    "trainer_boundary_firewall":
        "ENFORCED_EVERY_MICROBATCH",

    "final_model_hashes": {
        str(
            row[
                "condition"
            ]
        ):
            str(
                row[
                    "final_model_sha256"
                ]
            )
        for _, row
        in summary_df.iterrows()
    },

    "gate_b_decision":
        "NOT_YET_SCORED",

    "dense_development_evaluation":
        "BLOCK_08E_ONLY",

    "gate_b_outcomes_opened":
        False,

    "replay_model":
        "NOT_IMPLEMENTED",

    "independent_generalization_claim":
        False,
}


write_json(
    audit_dir
    / "block08d_corrected_gate_b_weak_only_fits.json",
    fit_audit,
)


# ==========================================================================================
# 13. UPDATE PROJECT STATE
# ==========================================================================================

state = json.loads(
    state_path.read_text(
        encoding="utf-8"
    )
)


state.update(
    {
        "last_attempted_block":
            "08D",

        "last_completed_block":
            "08D",

        "last_completed_block_name":
            "corrected_gate_b_weak_only_fits_v1_2",

        "current_stage":
            "gate_b_corrected_fits_complete_dense_eval_pending",

        "current_gate":
            "B",

        "gate_b":
            "IN_PROGRESS",

        "corrected_gate_b_training_authorized":
            False,

        "gate_b_corrected_weak_only_fits":
            "PASS",

        "valid_gate_b_optimizer_steps":
            valid_optimizer_steps,

        "invalidated_gate_b_optimizer_steps":
            7500,

        # Historical optimizer count is retained separately.
        "optimizer_steps_performed":
            (
                7500
                + valid_optimizer_steps
            ),

        "dense_development_evaluation":
            "NOT_RUN",

        "gate_b_outcomes_opened":
            False,

        "pilot_training_authorized":
            False,

        "training_authorized":
            False,

        "next_action":
            (
                "Audit Block 08D. Then perform frozen-checkpoint "
                "dense development evaluation for Gate B."
            ),

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
# 14. CAPTURE / REFRESH REPOSITORY MANIFEST
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
            "08D",

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
# 15. COMMIT TRAINING-ONLY AUDIT
# ==========================================================================================

heading(
    "STEP 8/8 — SEAL CORRECTED FITS AND SYNCHRONIZE"
)


git_paths = [
    "PROJECT_STATE.json",
    "REPOSITORY_MANIFEST.json",
    "data/manifests/gate_b_v1_2_weak_only_fit_summary.csv",
    "data/manifests/gate_b_v1_2_epoch_log.csv",
    "data/manifests/gate_b_v1_2_patch_sampling_realization.csv",
    "experiments/audits/"
    "block08d_corrected_gate_b_weak_only_fits.json",
    "figures/audit/fig13_gate_b_v1_2_training_curves.png",
    "figures/audit/fig13_gate_b_v1_2_training_curves.pdf",
]


sh(
    [
        "git",
        "add",
        *git_paths,
    ],
    cwd=REPO,
)


git_status = sh(
    [
        "git",
        "status",
        "--short",
    ],
    cwd=REPO,
).stdout.strip()


if not git_status:

    raise RuntimeError(
        "No corrected fit audit changes to commit."
    )


print(
    git_status
)


sh(
    [
        "git",
        "commit",
        "-m",
        "train: seal corrected Gate-B weak-only fits on cache v1.2",
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
# 16. FINAL REPORT
# ==========================================================================================

print(
    "\n"
    + "=" * 120
)

print(
    "CORA-LUNG CODE BLOCK 08D — FINAL CORRECTED GATE-B FIT REPORT"
)

print(
    "=" * 120
)


print(
    "Corrected cache version             : v1.2"
)

print(
    "Corrected preprocessing             : PASS"
)

print(
    "Patch sampling                      : 0.50 / 0.25 / 0.25"
)

print(
    "Development volumes                 :",
    len(
        development_cases
    ),
)

print(
    "Final outer-CV volumes used         : 0"
)

print(
    "Dense lesion-mask arrays accessed   : 0"
)

print(
    "Conditions fitted                   :",
    len(
        CONDITIONS
    ),
)


for condition in CONDITIONS:

    row = summary_df[
        summary_df[
            "condition"
        ]
        == condition
    ].iloc[
        0
    ]


    print(
        "  {:29s} steps={:4d} final_loss={:.6f} model_sha={}".format(
            condition,
            int(
                row[
                    "optimizer_steps"
                ]
            ),
            float(
                row[
                    "final_mean_loss"
                ]
            ),
            str(
                row[
                    "final_model_sha256"
                ]
            )[
                :12
            ],
        )
    )


print(
    "Primary Gate-B pair                 : "
    "pixel_dropout_matched_50 vs component_natural_50"
)

print(
    "Primary pair FG budget equality     : PASS 4/4"
)

print(
    "Primary pair BG realization         : PASS 4/4"
)

print(
    "Complementary contrast              : "
    "complete vs component_fixed_50"
)

print(
    "Seed                                :",
    SEED,
)

print(
    "Epochs / condition                  :",
    EPOCHS,
)

print(
    "Microbatches / epoch                :",
    MICROBATCHES_PER_EPOCH,
)

print(
    "Optimizer steps / condition         :",
    expected_steps_per_condition,
)

print(
    "Valid corrected optimizer steps     :",
    valid_optimizer_steps,
)

print(
    "Invalidated historical steps        : 7500"
)

print(
    "Trainer firewall every microbatch   : PASS"
)

print(
    "Checkpoint selection                : FINAL EPOCH ONLY"
)

print(
    "All corrected final models sealed   : PASS"
)

print(
    "Block-08A models eligible           : NO"
)

print(
    "Gate B                              : IN PROGRESS"
)

print(
    "Dense development evaluation        : NOT RUN"
)

print(
    "Gate-B outcomes opened              : NO"
)

print(
    "CORA replay                         : NOT RUN"
)

print(
    "Authorization commit                :",
    authorization_commit[
        :12
    ],
)

print(
    "Final audit commit                  :",
    final_commit[
        :12
    ],
)

print(
    "GitHub synchronization              : PASS"
)

print()
print(
    "NEXT: Send me this COMPLETE report. "
    "Do not open dense masks manually."
)

print(
    "=" * 120
)