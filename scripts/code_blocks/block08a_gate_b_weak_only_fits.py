# ==========================================================================================
# CORA-LUNG — CODE BLOCK 08A
# Gate-B Weak-Only Problem-Validation Fits
#
# FIRST SCIENTIFIC OPTIMIZER UPDATES IN THE PROJECT
#
# PURPOSE
#   Fit the identical partial-loss backbone under five frozen development-only
#   supervision conditions, WITHOUT accessing dense lesion masks.
#
# CONDITIONS
#   1. complete
#   2. pixel_dropout_matched_50
#   3. component_natural_50
#   4. complete_fixed_50
#   5. component_fixed_50
#
# DECISIVE GATE-B PAIRS
#   PRIMARY:
#       pixel_dropout_matched_50  vs  component_natural_50
#
#   SECONDARY FIXED-BUDGET:
#       complete_fixed_50         vs  component_fixed_50
#
# FROZEN FIT
#   Cases                 : 4 permanent-development volumes only
#   Seed                  : 17
#   Epochs                : 30
#   Microbatches / epoch  : 100
#   Batch                 : 1
#   Gradient accumulation : 2
#   Optimizer             : AdamW
#   LR                    : 3e-4
#   Weight decay          : 1e-4
#   AMP                   : yes on CUDA
#   Patch                 : 48 x 128 x 128 (z,y,x)
#   Checkpoint            : final epoch is PRIMARY
#
# TRAINING LOSS
#   partial BCE + partial Dice
#
# CRITICAL FIREWALL
#   guard_optimizer_batch(...) is called on EVERY microbatch immediately before
#   the optimizer-bound training path consumes it.
#
# THIS BLOCK DOES NOT:
#   - read infection-mask arrays
#   - read lung-mask arrays
#   - evaluate Dice against dense ground truth
#   - evaluate FROC
#   - select checkpoints from dense labels
#   - implement CORA replay
#
# Dense development evaluation happens ONLY in Block 08B after checkpoints freeze.
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
import random
import shutil
import time
import gc
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
    "/kaggle/working/cora_train_cache_v1_1"
)

CACHE_MANIFEST = (
    CACHE_ROOT
    / "manifest.csv"
)

RUN_ROOT = Path(
    "/kaggle/working/cora_gate_b_fits_v1"
)

BASE_ACCEPTED_COMMIT = "a5f33d922d0f"

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

LR = 3e-4

WEIGHT_DECAY = 1e-4

CONDITIONS = [
    "complete",
    "pixel_dropout_matched_50",
    "component_natural_50",
    "complete_fixed_50",
    "component_fixed_50",
]

PRIMARY_PAIR = (
    "pixel_dropout_matched_50",
    "component_natural_50",
)

FIXED_PAIR = (
    "complete_fixed_50",
    "component_fixed_50",
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
        + "=" * 118
    )

    print(text)

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


def stable_seed(
    *parts,
):

    digest = hashlib.sha256(
        "|".join(
            str(x)
            for x in parts
        ).encode(
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

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


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
        )
        + "\n",
        encoding="utf-8",
    )

    os.replace(
        tmp,
        path,
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
            "Microbatches/epoch must divide evenly across development cases."
        )

    repeats = (
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
            * repeats
        )

    rng = np.random.default_rng(
        stable_seed(
            SEED,
            "gate_b_case_schedule",
            epoch,
        )
    )

    rng.shuffle(
        schedule
    )

    return schedule


def git_push_with_secret():

    token = UserSecretsClient().get_secret(
        "pushCora"
    )

    if not token:

        raise RuntimeError(
            "Kaggle secret 'pushCora' unavailable."
        )

    token = token.strip()

    askpass = Path(
        "/tmp/cora_git_askpass_block08a.sh"
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

        error = (
            result.stderr
            or ""
        ).replace(
            token,
            "***TOKEN_REDACTED***",
        )

        raise RuntimeError(
            "GitHub push failed:\n"
            + error
        )


reset_seed(
    SEED
)


# ==========================================================================================
# 2. PRECONDITIONS
# ==========================================================================================

heading(
    "CORA-LUNG BLOCK 08A — GATE-B WEAK-ONLY DEVELOPMENT FITS"
)


if not (
    REPO
    / ".git"
).exists():

    raise RuntimeError(
        "CORA-LUNG repository missing."
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


current_head = sh(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    cwd=REPO,
).stdout.strip()


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
        "Repository must be clean before Block 08A."
    )


first_authorization = (
    state.get(
        "last_completed_block"
    )
    == "07D"
)


resume_authorization = (
    state.get(
        "current_stage"
    )
    == "gate_b_problem_validation_authorized"
    and state.get(
        "gate_b"
    )
    == "IN_PROGRESS"
)


if not (
    first_authorization
    or resume_authorization
):

    raise RuntimeError(
        "Project state is not valid for Gate-B fitting."
    )


if first_authorization:

    if not current_head.startswith(
        BASE_ACCEPTED_COMMIT
    ):

        raise RuntimeError(
            "Unexpected Block-07D starting commit."
        )


    audit_07d = json.loads(
        (
            REPO
            / "experiments/audits/"
            "block07d_pretraining_safety_closure.json"
        ).read_text(
            encoding="utf-8"
        )
    )


    if audit_07d.get(
        "status"
    ) != "PASS":

        raise RuntimeError(
            "Block 07D audit is not PASS."
        )


    if state.get(
        "trainer_boundary_firewall"
    ) != "PASS":

        raise RuntimeError(
            "Trainer-boundary firewall is not PASS."
        )


    if state.get(
        "native_voxel_volume_header_audit"
    ) != "PASS":

        raise RuntimeError(
            "Native geometry closure is not PASS."
        )


elif resume_authorization:

    print(
        "✓ Existing Gate-B authorization detected; "
        "checkpoint resume is permitted."
    )


if not CACHE_MANIFEST.exists():

    raise RuntimeError(
        "Accepted v1.1 training cache missing."
    )


print(
    f"✓ Current commit                  : "
    f"{current_head[:12]}"
)

print(
    "✓ Block 07D safety closure        : PASS"
)

print(
    "✓ Dense-label access during fits  : PROHIBITED"
)

print(
    "✓ Final outer-CV cases            : SEALED"
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


from cora_lung.data.pilot_dataset import (
    SparseTrainingCache,
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
# 4. DEVELOPMENT BOUNDARY + BUDGET RECHECK
# ==========================================================================================

heading(
    "STEP 1/7 — VERIFY DEVELOPMENT BOUNDARY AND CAUSAL BUDGETS"
)


split_registry = pd.read_csv(
    REPO
    / "data/splits/primary_volume_split_registry.csv"
)


development_cases = sorted(
    split_registry.loc[
        split_registry[
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


final_cases = split_registry.loc[
    split_registry[
        "role"
    ]
    == "final_outer_cv",
    "case_id",
].astype(
    str
).tolist()


if len(
    final_cases
) != 16:

    raise RuntimeError(
        "Expected exactly 16 sealed final-CV volumes."
    )


cache_manifest = pd.read_csv(
    CACHE_MANIFEST
)


observed_conditions = set(
    cache_manifest[
        "condition"
    ].astype(
        str
    )
)


missing_conditions = (
    set(
        CONDITIONS
    )
    - observed_conditions
)


if missing_conditions:

    raise RuntimeError(
        "Missing Gate-B cache conditions: "
        + str(
            sorted(
                missing_conditions
            )
        )
    )


dense_hash_df = pd.read_csv(
    REPO
    / "data/manifests/trainer_dense_sha256_denylist.csv"
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
        "Expected 60 frozen dense hashes."
    )


cache = SparseTrainingCache(
    CACHE_ROOT,
    development_cases,
)


prefit_rows = []


def fg_set(sample):

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

    return {
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


def bg_set(sample):

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

    return {
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


for case_id in development_cases:

    loaded = {}


    for condition in CONDITIONS:

        sample = cache.get(
            case_id,
            condition,
        )

        validate_cache_sample(
            sample,
            dense_sha256_denylist,
        )

        loaded[
            condition
        ] = sample


    natural_fg = fg_set(
        loaded[
            "component_natural_50"
        ]
    )

    pixel_fg = fg_set(
        loaded[
            "pixel_dropout_matched_50"
        ]
    )

    fixed_component_fg = fg_set(
        loaded[
            "component_fixed_50"
        ]
    )

    fixed_complete_fg = fg_set(
        loaded[
            "complete_fixed_50"
        ]
    )


    if len(
        natural_fg
    ) != len(
        pixel_fg
    ):

        raise RuntimeError(
            f"Primary Gate-B FG budget mismatch: {case_id}"
        )


    if len(
        fixed_component_fg
    ) != len(
        fixed_complete_fg
    ):

        raise RuntimeError(
            f"Fixed Gate-B FG budget mismatch: {case_id}"
        )


    if bg_set(
        loaded[
            "component_natural_50"
        ]
    ) != bg_set(
        loaded[
            "pixel_dropout_matched_50"
        ]
    ):

        raise RuntimeError(
            f"Primary Gate-B BG mismatch: {case_id}"
        )


    if bg_set(
        loaded[
            "component_fixed_50"
        ]
    ) != bg_set(
        loaded[
            "complete_fixed_50"
        ]
    ):

        raise RuntimeError(
            f"Fixed Gate-B BG mismatch: {case_id}"
        )


    prefit_rows.append(
        {
            "case_id":
                case_id,

            "natural_fg":
                len(
                    natural_fg
                ),

            "pixel_matched_fg":
                len(
                    pixel_fg
                ),

            "primary_fg_equal":
                True,

            "component_fixed_fg":
                len(
                    fixed_component_fg
                ),

            "complete_fixed_fg":
                len(
                    fixed_complete_fg
                ),

            "fixed_fg_equal":
                True,

            "primary_bg_equal":
                True,

            "fixed_bg_equal":
                True,
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


print(
    "\n✓ Primary matched pair FG/BG budgets : 4/4"
)

print(
    "✓ Fixed matched pair FG/BG budgets   : 4/4"
)

print(
    "✓ Final-CV volumes accessed           : 0"
)

print(
    "✓ Dense arrays accessed               : 0"
)


# ==========================================================================================
# 5. FREEZE GATE-B TRAINING CONFIG BEFORE FIRST OPTIMIZER STEP
# ==========================================================================================

heading(
    "STEP 2/7 — FREEZE AND AUTHORIZE DEVELOPMENT-ONLY FITS"
)


config_path = (
    REPO
    / "configs/gate_b_problem_validation.yaml"
)


config_text = """
experiment:
  name: gate_b_problem_validation
  purpose: component_omission_vs_equal_count_pixel_sparsity
  evaluation_population: permanent_development_only
  final_outer_cv_access: prohibited

conditions:
  - complete
  - pixel_dropout_matched_50
  - component_natural_50
  - complete_fixed_50
  - component_fixed_50

decisive_pairs:
  primary:
    control: pixel_dropout_matched_50
    omission: component_natural_50
  secondary_fixed:
    control: complete_fixed_50
    omission: component_fixed_50

fit:
  seed: 17
  epochs: 30
  microbatches_per_epoch: 100
  batch_size: 1
  gradient_accumulation: 2
  patch_zyx: [48, 128, 128]
  optimizer: AdamW
  learning_rate: 0.0003
  weight_decay: 0.0001
  amp: true
  augmentation: none
  checkpoint_selection: final_epoch_only

model:
  family: CORALungResidualUNet
  widths: [16, 32, 64, 128]
  embedding_dimension: 64

loss:
  partial_bce_weight: 1.0
  partial_dice_weight: 1.0
  unknown_label: -1

firewall:
  trainer_boundary_guard_every_microbatch: true
  dense_mask_access_during_fit: prohibited
  dense_checksum_denylist: required

interpretation:
  gate_b_is_development_problem_validation: true
  independent_patient_generalization_claim: prohibited
"""


if first_authorization:

    write_text(
        config_path,
        config_text,
    )


    prefit_manifest_path = (
        REPO
        / "data/manifests/"
        "gate_b_development_prefit_budget_audit.csv"
    )


    prefit_df.to_csv(
        prefit_manifest_path,
        index=False,
    )


    # Capture this exact executable cell BEFORE any optimizer update.
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
            "CORA-LUNG — CODE BLOCK 08A"
            in cell
        ):

            source_path = (
                REPO
                / "scripts/code_blocks/"
                "block08a_gate_b_weak_only_fits.py"
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


    authorization = {
        "project":
            "CORA-Lung",

        "block":
            "08A",

        "status":
            "AUTHORIZED",

        "authorized_at_utc":
            NOW_ISO,

        "starting_commit":
            current_head,

        "development_cases":
            development_cases,

        "sealed_final_cases_count":
            16,

        "conditions":
            CONDITIONS,

        "primary_pair":
            list(
                PRIMARY_PAIR
            ),

        "fixed_pair":
            list(
                FIXED_PAIR
            ),

        "seed":
            SEED,

        "epochs":
            EPOCHS,

        "microbatches_per_epoch":
            MICROBATCHES_PER_EPOCH,

        "accumulation_steps":
            ACCUMULATION_STEPS,

        "dense_training_access":
            False,

        "trainer_boundary_firewall":
            "REQUIRED_EVERY_MICROBATCH",

        "checkpoint_policy":
            "FINAL_EPOCH_ONLY",

        "source_capture":
            source_capture,
    }


    write_json(
        REPO
        / "experiments/audits/"
        "block08a_gate_b_training_authorization.json",
        authorization,
    )


    state.update(
        {
            "last_attempted_block":
                "08A",

            "current_stage":
                "gate_b_problem_validation_authorized",

            "current_gate":
                "B",

            "gate_b":
                "IN_PROGRESS",

            "pilot_training_authorized":
                True,

            "training_authorized":
                False,

            "model_training_started":
                False,

            "optimizer_steps_performed":
                0,

            "next_action":
                (
                    "Run the five frozen development-only "
                    "weak-supervision Gate-B fits."
                ),

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
            "configs/gate_b_problem_validation.yaml",
            "data/manifests/"
            "gate_b_development_prefit_budget_audit.csv",
            "experiments/audits/"
            "block08a_gate_b_training_authorization.json",
        ],
        cwd=REPO,
    )


    source_repo_path = (
        REPO
        / "scripts/code_blocks/"
        "block08a_gate_b_weak_only_fits.py"
    )


    if source_repo_path.exists():

        sh(
            [
                "git",
                "add",
                "scripts/code_blocks/"
                "block08a_gate_b_weak_only_fits.py",
            ],
            cwd=REPO,
        )


    sh(
        [
            "git",
            "commit",
            "-m",
            "train: authorize frozen development-only Gate-B fits",
        ],
        cwd=REPO,
    )


    git_push_with_secret()


    authorization_commit = sh(
        [
            "git",
            "rev-parse",
            "HEAD",
        ],
        cwd=REPO,
    ).stdout.strip()


    print(
        f"✓ Training authorization commit : "
        f"{authorization_commit[:12]}"
    )


else:

    authorization_commit = current_head


    if not config_path.exists():

        raise RuntimeError(
            "Gate-B frozen config missing on resume."
        )


    print(
        f"✓ Resuming authorized Gate-B fits at "
        f"{authorization_commit[:12]}"
    )


# Hard guarantee: repository must be clean before optimizer step 1.
post_authorization_dirty = sh(
    [
        "git",
        "status",
        "--porcelain",
    ],
    cwd=REPO,
).stdout.strip()


if post_authorization_dirty:

    raise RuntimeError(
        "Repository became dirty before fitting."
    )


config_hash = sha256_file(
    config_path
)

cache_hash = sha256_file(
    CACHE_MANIFEST
)


# ==========================================================================================
# 6. TRAINING UTILITIES
# ==========================================================================================

heading(
    "STEP 3/7 — INITIALIZE RESUMABLE TRAINING PROGRAM"
)


RUN_ROOT.mkdir(
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
        "Gate-B scientific fits require the Kaggle GPU."
    )


amp_enabled = True


print(
    f"✓ Device                         : "
    f"{torch.cuda.get_device_name(0)}"
)

print(
    f"✓ AMP                            : "
    f"{amp_enabled}"
)

print(
    f"✓ Conditions                     : "
    f"{len(CONDITIONS)}"
)

print(
    f"✓ Optimizer steps / condition    : "
    f"{EPOCHS * MICROBATCHES_PER_EPOCH // ACCUMULATION_STEPS}"
)

print(
    f"✓ Total planned optimizer steps  : "
    f"{len(CONDITIONS) * EPOCHS * MICROBATCHES_PER_EPOCH // ACCUMULATION_STEPS}"
)


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


def load_condition_samples(
    condition,
):

    samples = {}


    for case_id in development_cases:

        sample = cache.get(
            case_id,
            condition,
        )


        validate_cache_sample(
            sample,
            dense_sha256_denylist,
        )


        samples[
            case_id
        ] = sample


    return samples


def final_model_payload(
    model,
    *,
    condition,
    run_id,
    final_epoch,
    optimizer_steps,
):

    return {
        "run_id":
            run_id,

        "condition":
            condition,

        "seed":
            SEED,

        "final_epoch":
            int(
                final_epoch
            ),

        "optimizer_steps":
            int(
                optimizer_steps
            ),

        "config_sha256":
            config_hash,

        "cache_manifest_sha256":
            cache_hash,

        "authorization_commit":
            authorization_commit,

        "model_state_dict":
            model.state_dict(),
    }


# ==========================================================================================
# 7. FIT FIVE IDENTICAL BACKBONES
# ==========================================================================================

heading(
    "STEP 4/7 — RUN GATE-B WEAK-ONLY FITS"
)


fit_summaries = []

all_epoch_rows = []


for condition_index, condition in enumerate(
    CONDITIONS,
    start=1,
):

    print(
        "\n"
        + "#" * 118
    )

    print(
        f"FIT {condition_index}/{len(CONDITIONS)} — "
        f"{condition}"
    )

    print(
        "#" * 118
    )


    run_id = (
        "gateb_pl_"
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


    run_spec_path = (
        run_dir
        / "run_spec.json"
    )


    run_spec = {
        "run_id":
            run_id,

        "condition":
            condition,

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

        "learning_rate":
            LR,

        "weight_decay":
            WEIGHT_DECAY,

        "base_channels":
            BASE_CHANNELS,

        "embedding_dim":
            EMBEDDING_DIM,

        "config_sha256":
            config_hash,

        "cache_manifest_sha256":
            cache_hash,

        "authorization_commit":
            authorization_commit,

        "dense_training_access":
            False,

        "checkpoint_policy":
            "final_epoch_only",
    }


    if run_spec_path.exists():

        existing_spec = json.loads(
            run_spec_path.read_text(
                encoding="utf-8"
            )
        )


        if existing_spec != run_spec:

            raise RuntimeError(
                f"Run specification mismatch for {run_id}."
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


    # ------------------------------------------------------------------
    # Already completed? Validate lineage and skip.
    # ------------------------------------------------------------------

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
                f"Invalid completed-run manifest for {run_id}."
            )


        if (
            final_manifest.get(
                "config_sha256"
            )
            != config_hash
            or final_manifest.get(
                "cache_manifest_sha256"
            )
            != cache_hash
        ):

            raise RuntimeError(
                f"Completed run lineage mismatch: {run_id}"
            )


        print(
            "✓ Existing frozen final model detected; skipping fit."
        )


        epoch_csv = (
            run_dir
            / "epoch_log.csv"
        )


        epoch_df = pd.read_csv(
            epoch_csv
        )


        all_epoch_rows.extend(
            epoch_df.to_dict(
                "records"
            )
        )


        fit_summaries.append(
            final_manifest
        )


        continue


    # ------------------------------------------------------------------
    # Every condition starts from identical initialization seed.
    # ------------------------------------------------------------------

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
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )


    scaler = make_scaler()


    samples = load_condition_samples(
        condition
    )


    latest_checkpoint = (
        run_dir
        / "latest_checkpoint.pt"
    )


    start_epoch = 0

    optimizer_steps = 0


    if latest_checkpoint.exists():

        payload = load_training_checkpoint(
            latest_checkpoint,
            expected_config_hash=config_hash,
            expected_cache_manifest_hash=cache_hash,
        )


        if payload[
            "run_id"
        ] != run_id:

            raise RuntimeError(
                f"Checkpoint run ID mismatch: {run_id}"
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
            f"✓ Resume checkpoint: epoch "
            f"{start_epoch}/{EPOCHS}, "
            f"optimizer steps={optimizer_steps}"
        )


    epoch_log_path = (
        run_dir
        / "epoch_log.csv"
    )


    if epoch_log_path.exists():

        existing_epoch_df = pd.read_csv(
            epoch_log_path
        )


        epoch_rows = existing_epoch_df.to_dict(
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


        optimizer.zero_grad(
            set_to_none=True
        )


        loss_sum = 0.0
        bce_sum = 0.0
        dice_sum = 0.0

        labelled_sum = 0
        fg_sum = 0

        zero_label_microbatches = 0

        sampling_counts = {
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
                f"{condition} | "
                f"epoch {epoch + 1:02d}/{EPOCHS}"
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


            # ----------------------------------------------------------
            # REQUIRED optimizer-boundary firewall.
            # This runs on every scientific training microbatch.
            # ----------------------------------------------------------

            guard_optimizer_batch(
                patch,
                dense_sha256_denylist,
            )


            sampling_source = patch[
                "sampling_source"
            ]


            sampling_counts[
                sampling_source
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


            fg_count = int(
                (
                    target
                    == 1
                ).sum().item()
            )


            labelled_sum += labelled_count
            fg_sum += fg_count


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


            if (
                (
                    microbatch_index
                    + 1
                )
                % ACCUMULATION_STEPS
                == 0
            ):

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
            )


            del image
            del target
            del output
            del raw_loss
            del scaled_loss
            del bce
            del dice


        # 100 microbatches is exactly divisible by accumulation=2.
        if (
            MICROBATCHES_PER_EPOCH
            % ACCUMULATION_STEPS
            != 0
        ):

            raise RuntimeError(
                "Unexpected partial accumulation batch."
            )


        torch.cuda.synchronize()


        epoch_seconds = (
            time.perf_counter()
            - epoch_start
        )


        epoch_row = {
            "run_id":
                run_id,

            "condition":
                condition,

            "seed":
                SEED,

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
                labelled_sum,

            "foreground_voxels_seen":
                fg_sum,

            "zero_label_microbatches":
                zero_label_microbatches,

            "foreground_centered":
                sampling_counts[
                    "foreground"
                ],

            "background_centered":
                sampling_counts[
                    "background"
                ],

            "random_centered":
                sampling_counts[
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
                        1024 ** 3
                    )
                ),

            "peak_reserved_gb":
                float(
                    torch.cuda.max_memory_reserved(
                        device
                    )
                    / (
                        1024 ** 3
                    )
                ),
        }


        # Remove any previous row for this epoch if resuming/repeating.
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
            latest_checkpoint,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            epoch=epoch,
            global_step=optimizer_steps,
            config_hash=config_hash,
            cache_manifest_hash=cache_hash,
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

                "completed_epoch":
                    epoch,

                "optimizer_steps":
                    optimizer_steps,

                "latest_checkpoint":
                    latest_checkpoint.name,

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
            f"  Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"loss={epoch_row['mean_loss']:.5f} | "
            f"BCE={epoch_row['mean_partial_bce']:.5f} | "
            f"DiceLoss={epoch_row['mean_partial_dice']:.5f} | "
            f"steps={optimizer_steps} | "
            f"{epoch_seconds:.1f}s"
        )


    # ------------------------------------------------------------------
    # Freeze FINAL checkpoint/model — no validation-based selection.
    # ------------------------------------------------------------------

    expected_steps = (
        EPOCHS
        * MICROBATCHES_PER_EPOCH
        // ACCUMULATION_STEPS
    )


    if optimizer_steps != expected_steps:

        raise RuntimeError(
            f"Unexpected optimizer-step count for {condition}: "
            f"{optimizer_steps} != {expected_steps}"
        )


    final_payload = final_model_payload(
        model,
        condition=condition,
        run_id=run_id,
        final_epoch=EPOCHS - 1,
        optimizer_steps=optimizer_steps,
    )


    tmp_final = (
        run_dir
        / "final_model.pt.tmp"
    )


    torch.save(
        final_payload,
        tmp_final,
    )


    os.replace(
        tmp_final,
        final_model_path,
    )


    final_model_sha = sha256_file(
        final_model_path
    )


    latest_checkpoint_sha = sha256_file(
        latest_checkpoint
    )


    fit_seconds = (
        time.perf_counter()
        - fit_start
    )


    final_epoch_df = pd.DataFrame(
        epoch_rows
    ).sort_values(
        "epoch"
    )


    final_epoch_row = (
        final_epoch_df.iloc[
            -1
        ].to_dict()
    )


    final_manifest = {
        "status":
            "COMPLETE",

        "run_id":
            run_id,

        "condition":
            condition,

        "seed":
            SEED,

        "epochs":
            EPOCHS,

        "microbatches_per_epoch":
            MICROBATCHES_PER_EPOCH,

        "optimizer_steps":
            optimizer_steps,

        "final_epoch":
            EPOCHS
            - 1,

        "checkpoint_selection":
            "FINAL_EPOCH_ONLY",

        "dense_training_access":
            False,

        "development_cases":
            development_cases,

        "final_outer_cv_cases_used":
            0,

        "config_sha256":
            config_hash,

        "cache_manifest_sha256":
            cache_hash,

        "authorization_commit":
            authorization_commit,

        "final_model_file":
            str(
                final_model_path
            ),

        "final_model_sha256":
            final_model_sha,

        "latest_checkpoint_sha256":
            latest_checkpoint_sha,

        "final_mean_loss":
            float(
                final_epoch_row[
                    "mean_loss"
                ]
            ),

        "final_mean_partial_bce":
            float(
                final_epoch_row[
                    "mean_partial_bce"
                ]
            ),

        "final_mean_partial_dice":
            float(
                final_epoch_row[
                    "mean_partial_dice"
                ]
            ),

        "fit_seconds_this_session":
            float(
                fit_seconds
            ),

        "peak_allocated_gb":
            float(
                torch.cuda.max_memory_allocated(
                    device
                )
                / (
                    1024 ** 3
                )
            ),

        "peak_reserved_gb":
            float(
                torch.cuda.max_memory_reserved(
                    device
                )
                / (
                    1024 ** 3
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
        f"✓ Frozen final model: "
        f"{final_model_sha[:12]}"
    )


    del model
    del optimizer
    del scaler
    del samples

    gc.collect()

    torch.cuda.empty_cache()


# ==========================================================================================
# 8. VERIFY ALL FIVE FINAL CHECKPOINTS
# ==========================================================================================

heading(
    "STEP 5/7 — VERIFY SEALED FINAL FITS"
)


if len(
    fit_summaries
) != len(
    CONDITIONS
):

    raise RuntimeError(
        "Not all Gate-B fits produced final manifests."
    )


summary_df = pd.DataFrame(
    fit_summaries
)


if set(
    summary_df[
        "condition"
    ].astype(
        str
    )
) != set(
    CONDITIONS
):

    raise RuntimeError(
        "Final fit-condition set is incomplete."
    )


if not (
    summary_df[
        "status"
    ]
    == "COMPLETE"
).all():

    raise RuntimeError(
        "At least one Gate-B fit is incomplete."
    )


if not (
    summary_df[
        "optimizer_steps"
    ]
    == (
        EPOCHS
        * MICROBATCHES_PER_EPOCH
        // ACCUMULATION_STEPS
    )
).all():

    raise RuntimeError(
        "Optimizer-step count differs across conditions."
    )


if not (
    summary_df[
        "checkpoint_selection"
    ]
    == "FINAL_EPOCH_ONLY"
).all():

    raise RuntimeError(
        "A non-final checkpoint was selected."
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


total_optimizer_steps = int(
    summary_df[
        "optimizer_steps"
    ].sum()
)


print(
    f"✓ Frozen final models       : "
    f"{len(summary_df)}/5"
)

print(
    f"✓ Steps per condition       : "
    f"{summary_df['optimizer_steps'].iloc[0]}"
)

print(
    f"✓ Total optimizer steps     : "
    f"{total_optimizer_steps}"
)

print(
    "✓ Checkpoint selection      : FINAL EPOCH ONLY"
)

print(
    "✓ Dense labels during fit   : NONE"
)

print(
    "✓ Final outer-CV cases used : 0"
)


# ==========================================================================================
# 9. COMMIT TRAINING-ONLY AUDIT — NO CHECKPOINT BINARIES
# ==========================================================================================

heading(
    "STEP 6/7 — FREEZE TRAINING-ONLY AUDIT"
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
    / "figures/audit"
)


summary_repo_path = (
    manifest_dir
    / "gate_b_weak_only_fit_summary.csv"
)


summary_df[
    [
        "run_id",
        "condition",
        "seed",
        "epochs",
        "microbatches_per_epoch",
        "optimizer_steps",
        "checkpoint_selection",
        "dense_training_access",
        "final_outer_cv_cases_used",
        "config_sha256",
        "cache_manifest_sha256",
        "authorization_commit",
        "final_model_sha256",
        "latest_checkpoint_sha256",
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


epoch_repo_path = (
    manifest_dir
    / "gate_b_weak_only_epoch_log.csv"
)


epoch_df.to_csv(
    epoch_repo_path,
    index=False,
)


# Training curves are diagnostics only, NOT Gate-B outcome evidence.
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
        linewidth=1.5,
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
    "Gate-B Development Fits: Weak-Only Training Diagnostics\n"
    "No Dense Labels Used for Fitting or Checkpoint Selection",
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
    / "fig11_gate_b_weak_only_training_curves.png",
    dpi=600,
    bbox_inches="tight",
)


fig.savefig(
    figure_dir
    / "fig11_gate_b_weak_only_training_curves.pdf",
    bbox_inches="tight",
)


plt.close(
    fig
)


fit_audit = {
    "project":
        "CORA-Lung",

    "block":
        "08A",

    "status":
        "PASS",

    "completed_at_utc":
        datetime.now(
            timezone.utc
        ).strftime(
            "%Y-%m-%dT%H:%M:%SZ"
        ),

    "gate":
        "B",

    "stage":
        "WEAK_ONLY_FITS_COMPLETE_DENSE_EVAL_PENDING",

    "development_cases":
        development_cases,

    "final_outer_cv_cases_used":
        0,

    "dense_mask_arrays_accessed":
        0,

    "conditions":
        CONDITIONS,

    "primary_pair":
        list(
            PRIMARY_PAIR
        ),

    "fixed_pair":
        list(
            FIXED_PAIR
        ),

    "seed":
        SEED,

    "epochs":
        EPOCHS,

    "microbatches_per_epoch":
        MICROBATCHES_PER_EPOCH,

    "accumulation_steps":
        ACCUMULATION_STEPS,

    "optimizer_steps_per_condition":
        int(
            EPOCHS
            * MICROBATCHES_PER_EPOCH
            // ACCUMULATION_STEPS
        ),

    "total_optimizer_steps":
        total_optimizer_steps,

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
        "BLOCK_08B_ONLY",

    "replay_model":
        "NOT_IMPLEMENTED",

    "independent_generalization_claim":
        False,
}


write_json(
    audit_dir
    / "block08a_gate_b_weak_only_fits.json",
    fit_audit,
)


# ==========================================================================================
# 10. UPDATE PROJECT STATE AND PUSH
# ==========================================================================================

heading(
    "STEP 7/7 — SEAL FITS AND SYNCHRONIZE"
)


state = json.loads(
    state_path.read_text(
        encoding="utf-8"
    )
)


state.update(
    {
        "last_attempted_block":
            "08A",

        "last_completed_block":
            "08A",

        "last_completed_block_name":
            "gate_b_weak_only_fits",

        "current_stage":
            "gate_b_weak_only_fits_complete_dense_eval_pending",

        "current_gate":
            "B",

        "gate_b":
            "IN_PROGRESS",

        "gate_b_weak_only_fits":
            "PASS",

        "pilot_training_authorized":
            False,

        "training_authorized":
            False,

        "model_training_started":
            True,

        "optimizer_steps_performed":
            total_optimizer_steps,

        "dense_development_evaluation":
            "NOT_RUN",

        "next_action":
            (
                "Run Block 08B: frozen-checkpoint dense development "
                "evaluation and Gate-B problem-validation decision."
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


# Refresh repository manifest, excluding itself.
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
            "08A",

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


sh(
    [
        "git",
        "add",
        "PROJECT_STATE.json",
        "REPOSITORY_MANIFEST.json",
        "data/manifests/gate_b_weak_only_fit_summary.csv",
        "data/manifests/gate_b_weak_only_epoch_log.csv",
        "experiments/audits/block08a_gate_b_weak_only_fits.json",
        "figures/audit/fig11_gate_b_weak_only_training_curves.png",
        "figures/audit/fig11_gate_b_weak_only_training_curves.pdf",
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
        "No completed Gate-B fit audit to commit."
    )


print(
    status
)


sh(
    [
        "git",
        "commit",
        "-m",
        "train: seal Gate-B weak-only development fits",
    ],
    cwd=REPO,
)


git_push_with_secret()


final_commit = sh(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    cwd=REPO,
).stdout.strip()


print(
    "\n"
    + "=" * 118
)

print(
    "CORA-LUNG CODE BLOCK 08A — FINAL WEAK-ONLY FIT REPORT"
)

print(
    "=" * 118
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
    "Primary causal pair                : "
    "pixel_dropout_matched_50 vs component_natural_50"
)

print(
    "Secondary fixed-budget pair        : "
    "complete_fixed_50 vs component_fixed_50"
)

print(
    "Seed                               :",
    SEED,
)

print(
    "Epochs / condition                 :",
    EPOCHS,
)

print(
    "Microbatches / epoch               :",
    MICROBATCHES_PER_EPOCH,
)

print(
    "Optimizer steps / condition        :",
    EPOCHS
    * MICROBATCHES_PER_EPOCH
    // ACCUMULATION_STEPS,
)

print(
    "Total scientific optimizer steps   :",
    total_optimizer_steps,
)

print(
    "Trainer firewall every microbatch  : PASS"
)

print(
    "Checkpoint selection               : FINAL EPOCH ONLY"
)

print(
    "All five final models sealed       : PASS"
)

print(
    "Gate B                             : IN PROGRESS"
)

print(
    "Dense development evaluation       : NOT RUN"
)

print(
    "Replay / CORA mechanism            : NOT RUN"
)

print(
    "Authorization commit               :",
    authorization_commit[
        :12
    ],
)

print(
    "Final audit commit                 :",
    final_commit[
        :12
    ],
)

print(
    "GitHub synchronization             : PASS"
)

print()
print(
    "NEXT: Send me this COMPLETE report."
)

print(
    "=" * 118
)