# ==========================================================================================
# COVA-3D — BLOCK 09E-N1-LOCK
# Prospective Numerical Amendment N1:
# Stable FP16 Loss Scaling + Overflow-Safe Retry
#
# EXPECTED START COMMIT
# ---------------------
# a4db51c79ecf
#
# TRIGGER
# -------
# The first attempted 09E-SANITY-FIT failed before optimizer.step() because
# gradients were non-finite after GradScaler.unscale_().
#
# Diagnostic:
#
#   scale 65536 : NON-FINITE gradients
#   scale 32768 : NON-FINITE
#   scale 16384 : NON-FINITE
#   scale  8192 : NON-FINITE
#   scale  4096 : FINITE
#   ...
#   scale     1 : FINITE
#
# Forward losses remained finite.
#
# INTERPRETATION
# --------------
# This is an FP16 loss-scale overflow, not an intrinsically non-finite
# model/loss computation.
#
# N1 CHANGES ONLY NUMERICAL AMP HANDLING:
#
#   • AMP dtype remains float16.
#   • initial loss scale = 4096.
#   • growth factor = 2.
#   • backoff factor = 0.5.
#   • growth interval = 1,000,000 successful steps, effectively preventing
#     upward scale growth during the frozen 1,000-step sanity fit and
#     2,400-step final fits.
#   • if an overflow is detected, NO optimizer update is counted.
#   • the SAME logical accumulation pair is recomputed after scale backoff.
#   • if gradients remain non-finite at scale <= 1, STOP.
#
# UNCHANGED
# ---------
#   architecture
#   partial BCE + partial Dice
#   AdamW
#   LR and LR schedule
#   gradient clipping threshold
#   patch size
#   patient schedule
#   annotation coordinates
#   augmentation
#   accumulation = 2
#   20 sanity epochs
#   100 microbatches/epoch
#   exactly 1,000 SUCCESSFUL sanity optimizer updates
#   final 72-run protocol
#
# THIS BLOCK PERFORMS:
#   ✓ a three-scale reproduction on the exact first accumulation pair
#   ✓ protocol amendment freeze
#   ✓ Git commit / push
#
# THIS BLOCK DOES NOT:
#   ✗ call optimizer.step()
#   ✗ access dense masks
#   ✗ access final-CV data/outcomes
#   ✗ train any model
#
# ==========================================================================================

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import gc
import hashlib
import importlib
import json
import os
import random
import subprocess
import sys
import textwrap

import numpy as np
import pandas as pd
import torch
import yaml

from kaggle_secrets import UserSecretsClient


# ==========================================================================================
# 0. CONSTANTS
# ==========================================================================================

REPO = Path(
    "/kaggle/working/CORA-LUNG"
)

EXPECTED_START_COMMIT = (
    "a4db51c79ecf"
)

BLOCK = (
    "09E-N1-LOCK"
)

AMENDMENT_ID = (
    "N1_FP16_STABLE_LOSS_SCALING"
)

BASELINE_ID = (
    "COVA3D_NNUNET_STYLE_PARTIAL_V1"
)

EFFECTIVE_PROTOCOL_BEFORE = (
    "COVA3D_1.0+A1+A1.1+A1.2"
)

EFFECTIVE_PROTOCOL_AFTER = (
    "COVA3D_1.0+A1+A1.1+A1.2+N1"
)

CONDITION = (
    "C100_COH"
)

SEED = (
    17
)

PATCH_ZYX = np.asarray(
    [
        48,
        128,
        128,
    ],
    dtype=np.int32,
)

FEATURES = (
    32,
    64,
    128,
    256,
    320,
)

ACCUMULATION_STEPS = (
    2
)

UNKNOWN_LABEL = (
    -1
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

AMP_MINIMUM_SCALE = (
    1.0
)

AMP_MAX_RETRIES_PER_LOGICAL_STEP = (
    16
)

PROBE_SCALES = [
    65536.0,
    4096.0,
    1.0,
]

FLIP_X_PROBABILITY = (
    0.50
)

FLIP_Y_PROBABILITY = (
    0.50
)

FLIP_Z_PROBABILITY = (
    0.00
)

INTENSITY_SCALE_RANGE = (
    0.90,
    1.10,
)

INTENSITY_SHIFT_RANGE = (
    -0.10,
    0.10,
)

NOISE_SIGMA_RANGE = (
    0.00,
    0.03,
)

RUN_ROOT_FAILED = Path(
    "/kaggle/working/cova3d_sanity_fit_v1_0"
)

CACHE_ROOT = Path(
    "/kaggle/working/cova3d_sanity_ct_cache_v1_0"
)

FAILED_RUN_STATE = (
    RUN_ROOT_FAILED
    / "run_state.json"
)

FAILED_EPOCH_LOG = (
    RUN_ROOT_FAILED
    / "epoch_log.csv"
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
    payload,
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
            payload,
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
        int(
            seed
        )
    )

    np.random.seed(
        int(
            seed
        )
    )

    torch.manual_seed(
        int(
            seed
        )
    )

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(
            int(
                seed
            )
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
        "/tmp/cova3d_git_askpass_09e_n1.sh"
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


def augment_patch(
    image,
    target,
    *,
    epoch,
    microbatch_index,
    device,
):

    rng = np.random.default_rng(
        stable_seed(
            BASELINE_ID,
            SEED,
            "sanity_augmentation",
            epoch,
            microbatch_index,
        )
    )

    flip_x = bool(
        rng.random()
        < FLIP_X_PROBABILITY
    )

    flip_y = bool(
        rng.random()
        < FLIP_Y_PROBABILITY
    )

    flip_z = bool(
        rng.random()
        < FLIP_Z_PROBABILITY
    )

    scale = float(
        rng.uniform(
            *INTENSITY_SCALE_RANGE
        )
    )

    shift = float(
        rng.uniform(
            *INTENSITY_SHIFT_RANGE
        )
    )

    sigma = float(
        rng.uniform(
            *NOISE_SIGMA_RANGE
        )
    )

    if flip_x:

        image = torch.flip(
            image,
            dims=[
                3
            ],
        )

        target = torch.flip(
            target,
            dims=[
                3
            ],
        )

    if flip_y:

        image = torch.flip(
            image,
            dims=[
                2
            ],
        )

        target = torch.flip(
            target,
            dims=[
                2
            ],
        )

    if flip_z:

        image = torch.flip(
            image,
            dims=[
                1
            ],
        )

        target = torch.flip(
            target,
            dims=[
                1
            ],
        )

    image = (
        image
        .unsqueeze(
            0
        )
        .to(
            device=device,
            dtype=torch.float32,
        )
    )

    target = (
        target
        .unsqueeze(
            0
        )
        .to(
            device=device,
            dtype=torch.int8,
        )
    )

    image = (
        image
        * scale
        + shift
    )

    if sigma > 0:

        generator = torch.Generator(
            device=device
        )

        generator.manual_seed(
            stable_seed(
                BASELINE_ID,
                SEED,
                "sanity_noise",
                epoch,
                microbatch_index,
            )
        )

        noise = torch.randn(
            image.shape,
            generator=generator,
            device=device,
            dtype=image.dtype,
        )

        image = (
            image
            + noise
            * sigma
        )

    return (
        torch.clamp(
            image,
            -1.0,
            1.0,
        ),
        target,
    )


def gradient_is_finite(model):

    tensors = (
        0
    )

    nonfinite_tensors = (
        0
    )

    nonfinite_values = (
        0
    )

    for parameter in model.parameters():

        if parameter.grad is None:

            continue

        tensors += 1

        finite = torch.isfinite(
            parameter.grad
        )

        if not bool(
            finite.all()
        ):

            nonfinite_tensors += 1

            nonfinite_values += int(
                (
                    ~finite
                ).sum().item()
            )

    return {
        "finite":
            (
                nonfinite_tensors
                == 0
            ),

        "gradient_tensors":
            tensors,

        "nonfinite_tensors":
            nonfinite_tensors,

        "nonfinite_values":
            nonfinite_values,
    }


# ==========================================================================================
# 2. VERIFY REPOSITORY + FAILED ATTEMPT PROVENANCE
# ==========================================================================================

heading(
    "COVA-3D BLOCK 09E-N1-LOCK — FP16 NUMERICAL AMENDMENT"
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
        "Unexpected Git HEAD.\n"
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
        "Repository must be clean before N1.\n"
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
) != "09E-LOCK":

    raise RuntimeError(
        "Expected 09E-LOCK as last completed block."
    )


if int(
    cova_state.get(
        "optimizer_steps_in_cova3d",
        -1,
    )
) != 0:

    raise RuntimeError(
        "Repository state no longer records zero optimizer steps."
    )


if cova_state.get(
    "factorial_training_authorized"
) is not False:

    raise RuntimeError(
        "Factorial training was unexpectedly authorized."
    )


failed_state = {}


if FAILED_RUN_STATE.exists():

    failed_state = json.loads(
        FAILED_RUN_STATE.read_text(
            encoding="utf-8"
        )
    )


persisted_steps = int(
    failed_state.get(
        "optimizer_steps",
        0,
    )
)


persisted_completed_epoch = int(
    failed_state.get(
        "completed_epoch",
        -1,
    )
)


persisted_epoch_rows = (
    len(
        pd.read_csv(
            FAILED_EPOCH_LOG
        )
    )
    if FAILED_EPOCH_LOG.exists()
    else 0
)


if persisted_steps != 0:

    raise RuntimeError(
        "The aborted run unexpectedly persisted optimizer updates."
    )


if persisted_completed_epoch != -1:

    raise RuntimeError(
        "The aborted run unexpectedly persisted a completed epoch."
    )


if persisted_epoch_rows != 0:

    raise RuntimeError(
        "The aborted run unexpectedly persisted completed epoch rows."
    )


print(
    "✓ Starting commit                     :",
    head[:12],
)

print(
    "✓ Repository state                    : CLEAN"
)

print(
    "✓ Failed attempt persisted steps      : 0"
)

print(
    "✓ Failed attempt completed epochs     : 0"
)

print(
    "✓ Dense outcomes opened               : NO"
)

print(
    "✓ Factorial training                  : LOCKED"
)


# ==========================================================================================
# 3. VERIFY EXACT FIRST ACCUMULATION PAIR
# ==========================================================================================

heading(
    "STEP 1/7 — REPRODUCE CRITICAL AMP DIAGNOSTIC"
)


if not torch.cuda.is_available():

    raise RuntimeError(
        "GPU required."
    )


cache_manifest_path = (
    CACHE_ROOT
    / "manifest.csv"
)


if not cache_manifest_path.exists():

    raise RuntimeError(
        "Development CT cache is missing. Keep/reopen the original Kaggle "
        "session or tell me and I will provide a standalone reconstruction."
    )


baseline_config = yaml.safe_load(
    (
        REPO
        / "configs/"
        "cova3d_baseline_training_v1_0.yaml"
    ).read_text(
        encoding="utf-8"
    )
)


dev_cases = sorted(
    str(
        case_id
    )
    for case_id in baseline_config[
        "sanity_run"
    ][
        "fit_cases"
    ]
)


cache_df = pd.read_csv(
    cache_manifest_path
)


artifact_df = pd.read_csv(
    REPO
    / "data/manifests/"
    "cova3d_training_grid_candidate_artifact_manifest_v1_2.csv"
)


artifact_df = artifact_df[
    (
        artifact_df[
            "case_id"
        ].astype(
            str
        ).isin(
            dev_cases
        )
    )
    & (
        artifact_df[
            "condition_id"
        ].astype(
            str
        )
        == CONDITION
    )
]


if len(
    artifact_df
) != 4:

    raise RuntimeError(
        "Expected four sanity sparse artifacts."
    )


cache_lookup = {
    str(
        row[
            "case_id"
        ]
    ):
        row
    for _, row
    in cache_df.iterrows()
}


artifact_lookup = {
    str(
        row[
            "case_id"
        ]
    ):
        row
    for _, row
    in artifact_df.iterrows()
}


samples = {}


for case_id in dev_cases:

    image_path = Path(
        cache_lookup[
            case_id
        ][
            "image_file"
        ]
    )


    with np.load(
        image_path,
        allow_pickle=False,
    ) as data:

        image = np.asarray(
            data[
                "ct_zyx"
            ],
            dtype=np.float32,
        )


    annotation_path = (
        REPO
        / str(
            artifact_lookup[
                case_id
            ][
                "artifact_file"
            ]
        )
    )


    with np.load(
        annotation_path,
        allow_pickle=False,
    ) as data:

        samples[
            case_id
        ] = {
            "case_id":
                case_id,

            "condition":
                CONDITION,

            "image":
                image,

            "supervision_voxel_zyx":
                np.asarray(
                    data[
                        "supervision_voxel_zyx"
                    ],
                    dtype=np.int32,
                ),

            "supervision_label":
                np.asarray(
                    data[
                        "supervision_label"
                    ],
                    dtype=np.int8,
                ),

            "fg_membership_voxel_zyx":
                np.asarray(
                    data[
                        "fg_membership_voxel_zyx"
                    ],
                    dtype=np.int32,
                ),

            "fg_membership_group_id":
                np.asarray(
                    data[
                        "fg_membership_group_id"
                    ],
                    dtype=np.int32,
                ),
        }


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
    "cora_lung.data.pilot_dataset",
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


from cora_lung.data.pilot_dataset import (
    extract_patch,
)


initialization_seed = stable_seed(
    BASELINE_ID,
    "sanity_init",
    SEED,
)


patch_sampler_seed = stable_seed(
    BASELINE_ID,
    "sanity_patch_sampler",
    SEED,
)


schedule = []


for case_id in dev_cases:

    schedule.extend(
        [
            case_id
        ]
        * 25
    )


rng = np.random.default_rng(
    stable_seed(
        BASELINE_ID,
        SEED,
        "sanity_patient_schedule",
        0,
    )
)


rng.shuffle(
    schedule
)


prepared = []


for microbatch_index in [
    0,
    1,
]:

    case_id = schedule[
        microbatch_index
    ]


    patch = extract_patch(
        samples[
            case_id
        ],
        PATCH_ZYX,
        seed=patch_sampler_seed,
        sample_index=microbatch_index,
    )


    prepared.append(
        {
            "case_id":
                case_id,

            "microbatch_index":
                microbatch_index,

            "sampling_source":
                str(
                    patch[
                        "sampling_source"
                    ]
                ),

            "image":
                patch[
                    "image"
                ].clone(),

            "target":
                patch[
                    "target"
                ].clone(),
        }
    )


device = torch.device(
    "cuda"
)


def probe_scale(scale_value):

    reset_seed(
        initialization_seed
    )


    model = COVA3DNNUNetStyle(
        in_channels=1,
        out_channels=1,
        features=FEATURES,
    ).to(
        device
    )


    model.train()


    model.zero_grad(
        set_to_none=True
    )


    losses = []


    for item in prepared:

        image, target = augment_patch(
            item[
                "image"
            ].clone(),
            item[
                "target"
            ].clone(),
            epoch=0,
            microbatch_index=item[
                "microbatch_index"
            ],
            device=device,
        )


        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        ):

            logits = model(
                image
            )


            bce = partial_bce(
                logits,
                target,
            )


            dice = partial_dice(
                logits,
                target,
            )


            loss = (
                bce
                + dice
            )


        if not bool(
            torch.isfinite(
                loss
            )
        ):

            raise RuntimeError(
                "Forward loss itself became non-finite."
            )


        losses.append(
            float(
                loss.detach().cpu()
            )
        )


        (
            loss
            / ACCUMULATION_STEPS
            * float(
                scale_value
            )
        ).backward()


        del image
        del target
        del logits
        del bce
        del dice
        del loss


    grad = gradient_is_finite(
        model
    )


    result = {
        "scale":
            float(
                scale_value
            ),

        "mean_loss":
            float(
                np.mean(
                    losses
                )
            ),

        **grad,
    }


    del model

    torch.cuda.empty_cache()

    gc.collect()


    return result


probe_rows = []


for scale_value in PROBE_SCALES:

    result = probe_scale(
        scale_value
    )


    probe_rows.append(
        result
    )


    print(
        "scale="
        + str(
            int(
                scale_value
            )
        ).rjust(
            6
        )
        + " | gradients="
        + (
            "FINITE"
            if result[
                "finite"
            ]
            else "NONFINITE"
        )
        + " | nonfinite tensors="
        + str(
            result[
                "nonfinite_tensors"
            ]
        )
    )


probe_df = pd.DataFrame(
    probe_rows
)


probe_65536 = probe_df[
    np.isclose(
        probe_df[
            "scale"
        ],
        65536.0,
    )
].iloc[
    0
]


probe_4096 = probe_df[
    np.isclose(
        probe_df[
            "scale"
        ],
        4096.0,
    )
].iloc[
    0
]


probe_1 = probe_df[
    np.isclose(
        probe_df[
            "scale"
        ],
        1.0,
    )
].iloc[
    0
]


if bool(
    probe_65536[
        "finite"
    ]
):

    raise RuntimeError(
        "Expected scale 65536 to reproduce overflow."
    )


if not bool(
    probe_4096[
        "finite"
    ]
):

    raise RuntimeError(
        "Scale 4096 did not reproduce as finite."
    )


if not bool(
    probe_1[
        "finite"
    ]
):

    raise RuntimeError(
        "Scale 1 gradients are non-finite; N1 must not be frozen."
    )


print()
print(
    "✓ Scale 65536                         : OVERFLOW REPRODUCED"
)

print(
    "✓ Scale 4096                          : FINITE"
)

print(
    "✓ Scale 1                             : FINITE"
)

print(
    "✓ optimizer.step() calls              : 0"
)


# ==========================================================================================
# 4. FREEZE N1 NUMERICAL POLICY
# ==========================================================================================

heading(
    "STEP 2/7 — FREEZE NUMERICAL AMENDMENT N1"
)


amendment = {
    "project":
        "COVA-3D",

    "amendment_id":
        AMENDMENT_ID,

    "status":
        "PROSPECTIVELY_FROZEN_BEFORE_FIRST_SUCCESSFUL_COVA_OPTIMIZER_STEP",

    "parent_commit":
        head,

    "parent_baseline":
        BASELINE_ID,

    "protocol_before":
        EFFECTIVE_PROTOCOL_BEFORE,

    "protocol_after":
        EFFECTIVE_PROTOCOL_AFTER,

    "trigger":
        {
            "block":
                "09E-SANITY-FIT_ATTEMPT_1",

            "failure":
                "FP16_SCALED_GRADIENT_OVERFLOW_BEFORE_FIRST_OPTIMIZER_STEP",

            "persisted_optimizer_steps":
                0,

            "persisted_completed_epochs":
                0,

            "dense_outcomes_accessed":
                False,

            "first_accumulation_pair":
                [
                    0,
                    1,
                ],

            "forward_loss_finite":
                True,
        },

    "diagnostic":
        {
            "full_observed_scale_ladder":
                {
                    "65536":
                        "NONFINITE",

                    "32768":
                        "NONFINITE",

                    "16384":
                        "NONFINITE",

                    "8192":
                        "NONFINITE",

                    "4096":
                        "FINITE",

                    "2048":
                        "FINITE",

                    "1024":
                        "FINITE",

                    "512":
                        "FINITE",

                    "256":
                        "FINITE",

                    "128":
                        "FINITE",

                    "64":
                        "FINITE",

                    "32":
                        "FINITE",

                    "16":
                        "FINITE",

                    "8":
                        "FINITE",

                    "4":
                        "FINITE",

                    "2":
                        "FINITE",

                    "1":
                        "FINITE",
                },

            "independent_lock_reproduction":
                {
                    str(
                        int(
                            row[
                                "scale"
                            ]
                        )
                    ):
                        (
                            "FINITE"
                            if bool(
                                row[
                                    "finite"
                                ]
                            )
                            else "NONFINITE"
                        )
                    for _, row in probe_df.iterrows()
                },

            "largest_observed_finite_scale":
                4096.0,

            "scale_1_finite":
                True,

            "diagnosis":
                "TRANSIENT_FP16_LOSS_SCALE_OVERFLOW",
        },

    "amp_policy":
        {
            "autocast":
                True,

            "dtype":
                "float16",

            "initial_scale":
                AMP_INITIAL_SCALE,

            "growth_factor":
                AMP_GROWTH_FACTOR,

            "backoff_factor":
                AMP_BACKOFF_FACTOR,

            "growth_interval_successful_steps":
                AMP_GROWTH_INTERVAL,

            "effective_upward_growth_during_frozen_runs":
                False,

            "minimum_permitted_scale":
                AMP_MINIMUM_SCALE,

            "overflow_detection":
                (
                    "After scaler.unscale_(optimizer), inspect all parameter "
                    "gradients for finite values before gradient clipping."
                ),

            "overflow_action":
                (
                    "Call scaler.step(optimizer), allowing GradScaler to "
                    "skip the weight update; call scaler.update() to back off "
                    "the scale; zero gradients; recompute the exact same "
                    "logical accumulation pair."
                ),

            "successful_optimizer_step_definition":
                (
                    "A logical update is counted only when gradients are "
                    "finite after unscale and the optimizer weight update "
                    "actually occurs."
                ),

            "retry_same_logical_pair":
                True,

            "max_retries_per_logical_step":
                AMP_MAX_RETRIES_PER_LOGICAL_STEP,

            "abort_condition":
                (
                    "If non-finite gradients persist at scale <=1 or retry "
                    "limit is exceeded, stop as genuine numerical instability."
                ),

            "gradient_clip_only_after_finite_unscale":
                True,

            "gradient_clip_norm":
                12.0,
        },

    "scientifically_unchanged":
        {
            "architecture":
                True,

            "model_parameters":
                True,

            "loss":
                True,

            "optimizer":
                True,

            "base_learning_rate":
                True,

            "learning_rate_schedule":
                True,

            "weight_decay":
                True,

            "gradient_clip_threshold":
                True,

            "patch_size":
                True,

            "annotation_protocol":
                True,

            "patient_schedule":
                True,

            "augmentation":
                True,

            "gradient_accumulation":
                True,

            "sanity_epochs":
                True,

            "sanity_microbatches":
                True,

            "successful_sanity_optimizer_step_target":
                1000,

            "final_factorial_protocol":
                True,
        },

    "factorial_training_authorized":
        False,

    "dense_outcome_evaluation_authorized":
        False,

    "next_block":
        "09E-SANITY-FIT-R1",

    "frozen_at_utc":
        NOW_ISO,
}


amendment_path = (
    REPO
    / "configs/"
    "cova3d_numerical_amendment_N1_amp.yaml"
)


amendment_path.write_text(
    yaml.safe_dump(
        amendment,
        sort_keys=False,
        allow_unicode=True,
    ),
    encoding="utf-8",
)


write_text(
    REPO
    / "docs/"
    "cova3d_numerical_amendment_N1_amp.md",
    """
    # COVA-3D Numerical Amendment N1

    ## Trigger

    The first COVA-3D sanity-fit attempt stopped on its first accumulated
    gradient pair before the first optimizer update.

    Forward losses were finite, but FP16 scaled gradients were non-finite at
    the default loss scale of 65536.

    A pre-outcome numerical diagnostic found non-finite gradients at scales
    65536, 32768, 16384 and 8192. Gradients were finite from scale 4096 down
    to scale 1.

    Therefore the failure was classified as loss-scale overflow rather than
    intrinsic model or loss instability.

    No dense development outcome was opened. No final-CV outcome was opened.
    No successful COVA optimizer update preceded this amendment.

    ## Frozen N1 policy

    FP16 autocast is retained.

    GradScaler starts at 4096.

    The growth factor is 2 and the backoff factor is 0.5.

    The growth interval is 1,000,000 successful optimizer steps, preventing
    upward loss-scale growth during the frozen sanity and final runs.

    After gradients are unscaled, their finiteness is checked before
    gradient clipping.

    If overflow occurs, GradScaler skips the update and backs off the scale.
    The same logical two-microbatch accumulation pair is then recomputed.

    A logical optimizer step is counted only after a finite, successful
    weight update.

    If gradients remain non-finite at scale 1, training stops.

    ## Scientific protocol

    N1 does not alter the COVA-3D architecture, sparse loss, annotation
    conditions, optimizer, learning rate, scheduler, sampling, augmentation,
    patch size, accumulation factor, epoch count, checkpoint rule, or final
    factorial experiment.

    Factorial training remains locked.
    """
)


print(
    "✓ Amendment ID                        :",
    AMENDMENT_ID,
)

print(
    "✓ Initial FP16 scale                   :",
    AMP_INITIAL_SCALE,
)

print(
    "✓ Scale growth during planned runs     : DISABLED"
)

print(
    "✓ Overflow policy                      : BACKOFF + SAME-PAIR RETRY"
)

print(
    "✓ Successful sanity update target      : 1000"
)

print(
    "✓ Scientific model/training variables  : UNCHANGED"
)


# ==========================================================================================
# 5. ARCHIVE AMP DIAGNOSTIC
# ==========================================================================================

heading(
    "STEP 3/7 — ARCHIVE PRE-OUTCOME NUMERICAL DIAGNOSTIC"
)


audit_dir = (
    REPO
    / "experiments/audits"
)


manifest_dir = (
    REPO
    / "data/manifests"
)


audit_dir.mkdir(
    parents=True,
    exist_ok=True,
)


manifest_dir.mkdir(
    parents=True,
    exist_ok=True,
)


probe_path = (
    manifest_dir
    / "cova3d_amp_N1_lock_probe.csv"
)


probe_df.to_csv(
    probe_path,
    index=False,
)


diagnostic_audit = {
    "project":
        "COVA-3D",

    "block":
        "09E-SANITY-AMP-DIAG",

    "status":
        "DIAGNOSIS_COMPLETE",

    "diagnosis":
        "TRANSIENT_AMP_LOSS_SCALE_OVERFLOW",

    "repository_modified_during_original_diagnostic":
        False,

    "optimizer_step_calls_during_diagnostic":
        0,

    "dense_masks_accessed":
        0,

    "final_outer_cv_access":
        0,

    "failed_runtime":
        {
            "persisted_completed_epochs":
                persisted_epoch_rows,

            "persisted_optimizer_steps":
                persisted_steps,

            "scientific_updates_to_retain":
                0,
        },

    "reported_full_scale_ladder":
        {
            "65536":
                False,

            "32768":
                False,

            "16384":
                False,

            "8192":
                False,

            "4096":
                True,

            "2048":
                True,

            "1024":
                True,

            "512":
                True,

            "256":
                True,

            "128":
                True,

            "64":
                True,

            "32":
                True,

            "16":
                True,

            "8":
                True,

            "4":
                True,

            "2":
                True,

            "1":
                True,
        },

    "lock_block_reproduction":
        probe_rows,

    "conclusion":
        (
            "Forward loss and unscaled gradients are finite. Default AMP "
            "loss scaling is too large for the frozen first accumulation pair."
        ),

    "created_at_utc":
        NOW_ISO,
}


diagnostic_audit_path = (
    audit_dir
    / "block09e_sanity_amp_diagnostic.json"
)


write_json(
    diagnostic_audit_path,
    diagnostic_audit,
)


# ==========================================================================================
# 6. UPDATE PROJECT STATE — RETRY AUTHORIZED, FACTORIAL STILL LOCKED
# ==========================================================================================

heading(
    "STEP 4/7 — UPDATE STATE"
)


cova_state.update(
    {
        "last_completed_block":
            BLOCK,

        "current_stage":
            "SANITY_NUMERICAL_POLICY_CORRECTED",

        "effective_protocol":
            EFFECTIVE_PROTOCOL_AFTER,

        "numerical_amendment":
            "N1",

        "numerical_amendment_status":
            "FROZEN",

        "sanity_fit_attempt_1":
            "ABORTED_BEFORE_FIRST_SUCCESSFUL_OPTIMIZER_STEP",

        "sanity_fit_attempt_1_failure":
            "FP16_LOSS_SCALE_OVERFLOW",

        "sanity_fit_attempt_1_persisted_optimizer_steps":
            0,

        "optimizer_steps_in_cova3d":
            0,

        "amp_initial_scale":
            AMP_INITIAL_SCALE,

        "amp_growth_interval":
            AMP_GROWTH_INTERVAL,

        "amp_overflow_retry_same_pair":
            True,

        "sanity_training_authorized":
            True,

        "sanity_gate_status":
            "NOT_RUN",

        "dense_outcomes_opened_in_cova3d":
            False,

        "factorial_training_authorized":
            False,

        "method_development_authorized":
            False,

        "final_outer_cv_outcomes_authorized":
            False,

        "final_outer_cv_access_in_cova3d":
            0,

        "next_block":
            "09E-SANITY-FIT-R1",

        "next_action":
            (
                "Restart the frozen sanity fit from initialization under "
                "numerical amendment N1. Do not resume the aborted attempt."
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
            "cova3d_amp_numerical_amendment_N1",

        "current_stage":
            "cova3d_sanity_retry_authorized",

        "current_gate":
            "COVA_SANITY_GATE",

        "cova3d_effective_protocol":
            EFFECTIVE_PROTOCOL_AFTER,

        "cova3d_numerical_amendment":
            "N1",

        "cova3d_optimizer_steps":
            0,

        "cova3d_sanity_training_authorized":
            True,

        "cova3d_sanity_gate":
            "NOT_RUN",

        "cova3d_factorial_training_authorized":
            False,

        "cova3d_method_development_authorized":
            False,

        "cova3d_dense_outcomes_opened":
            False,

        "cova3d_final_outer_cv_access":
            0,

        "training_authorized":
            False,

        "pilot_training_authorized":
            False,

        "next_action":
            "Run 09E-SANITY-FIT-R1 only.",

        "updated_at_utc":
            NOW_ISO,
    }
)


write_json(
    project_state_path,
    project_state,
)


# ==========================================================================================
# 7. TEST AMENDMENT
# ==========================================================================================

heading(
    "STEP 5/7 — REGRESSION TESTS"
)


test_source = r'''
from pathlib import Path
import json

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_n1_is_frozen():

    cfg = yaml.safe_load(
        (
            ROOT
            / "configs/"
            "cova3d_numerical_amendment_N1_amp.yaml"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        cfg[
            "status"
        ]
        == "PROSPECTIVELY_FROZEN_BEFORE_FIRST_SUCCESSFUL_COVA_OPTIMIZER_STEP"
    )

    assert (
        cfg[
            "amp_policy"
        ][
            "initial_scale"
        ]
        == 4096.0
    )

    assert (
        cfg[
            "amp_policy"
        ][
            "growth_interval_successful_steps"
        ]
        == 1000000
    )

    assert (
        cfg[
            "amp_policy"
        ][
            "retry_same_logical_pair"
        ]
        is True
    )

    assert (
        cfg[
            "scientifically_unchanged"
        ][
            "successful_sanity_optimizer_step_target"
        ]
        == 1000
    )

    assert (
        cfg[
            "factorial_training_authorized"
        ]
        is False
    )


def test_state_after_n1():

    state = json.loads(
        (
            ROOT
            / "COVA3D_STATE.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        state[
            "last_completed_block"
        ]
        == "09E-N1-LOCK"
    )

    assert (
        state[
            "numerical_amendment"
        ]
        == "N1"
    )

    assert (
        state[
            "optimizer_steps_in_cova3d"
        ]
        == 0
    )

    assert (
        state[
            "sanity_training_authorized"
        ]
        is True
    )

    assert (
        state[
            "factorial_training_authorized"
        ]
        is False
    )

    assert (
        state[
            "dense_outcomes_opened_in_cova3d"
        ]
        is False
    )
'''


test_path = (
    REPO
    / "tests/"
    "test_cova3d_numerical_amendment_n1.py"
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

        "tests/test_cova3d_baseline_lock.py",
        "tests/test_cova3d_numerical_amendment_n1.py",

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
        "N1 regression tests failed."
    )


print(
    "✓ N1 regression tests                  : PASS"
)


# ==========================================================================================
# 8. SOURCE CAPTURE
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
        "COVA-3D — BLOCK 09E-N1-LOCK"
        in cell
    ):

        source_path = (
            REPO
            / "scripts/code_blocks/"
            "block09e_n1_lock_amp_numerics.py"
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


amendment_audit = {
    "project":
        "COVA-3D",

    "block":
        BLOCK,

    "status":
        "PASS",

    "amendment_id":
        AMENDMENT_ID,

    "parent_commit":
        head,

    "protocol_after":
        EFFECTIVE_PROTOCOL_AFTER,

    "failed_attempt_retained_optimizer_steps":
        0,

    "dense_outcomes_accessed_before_amendment":
        False,

    "probe": {
        str(
            int(
                row[
                    "scale"
                ]
            )
        ):
            bool(
                row[
                    "finite"
                ]
            )
        for _, row in probe_df.iterrows()
    },

    "amp_initial_scale":
        AMP_INITIAL_SCALE,

    "amp_growth_interval":
        AMP_GROWTH_INTERVAL,

    "overflow_same_pair_retry":
        True,

    "successful_sanity_optimizer_step_target":
        1000,

    "factorial_training_authorized":
        False,

    "source_capture":
        source_capture,

    "next_block":
        "09E-SANITY-FIT-R1",

    "generated_at_utc":
        NOW_ISO,
}


amendment_audit_path = (
    REPO
    / "experiments/audits/"
    "block09e_n1_lock_amp_numerics.json"
)


write_json(
    amendment_audit_path,
    amendment_audit,
)


# ==========================================================================================
# 9. NORMALIZE TEXT BEFORE REPOSITORY MANIFEST
# ==========================================================================================

heading(
    "STEP 6/7 — FREEZE REPOSITORY PROVENANCE"
)


text_paths = [
    amendment_path,
    diagnostic_audit_path,
    amendment_audit_path,
    cova_state_path,
    project_state_path,
    probe_path,
    test_path,
    REPO
    / "docs/"
    "cova3d_numerical_amendment_N1_amp.md",
]


source_path = (
    REPO
    / "scripts/code_blocks/"
    "block09e_n1_lock_amp_numerics.py"
)


if source_path.exists():

    text_paths.append(
        source_path
    )


for path in text_paths:

    path = Path(
        path
    )

    text = path.read_text(
        encoding="utf-8"
    )

    path.write_text(
        text.rstrip()
        + "\n",
        encoding="utf-8",
    )


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
            and not any(
                part in EXCLUDED_DIRS
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
            EFFECTIVE_PROTOCOL_AFTER,

        "numerical_amendment":
            "N1",

        "amp_initial_scale":
            AMP_INITIAL_SCALE,

        "sanity_training_authorized":
            True,

        "sanity_gate":
            "NOT_RUN",

        "cova_optimizer_steps":
            0,

        "dense_outcomes_opened":
            False,

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
# 10. COMMIT + PUSH
# ==========================================================================================

heading(
    "STEP 7/7 — COMMIT N1"
)


git_paths = [
    "COVA3D_STATE.json",
    "PROJECT_STATE.json",
    "REPOSITORY_MANIFEST.json",

    "configs/"
    "cova3d_numerical_amendment_N1_amp.yaml",

    "data/manifests/"
    "cova3d_amp_N1_lock_probe.csv",

    "docs/"
    "cova3d_numerical_amendment_N1_amp.md",

    "experiments/audits/"
    "block09e_sanity_amp_diagnostic.json",

    "experiments/audits/"
    "block09e_n1_lock_amp_numerics.json",

    "tests/"
    "test_cova3d_numerical_amendment_n1.py",
]


if source_path.exists():

    git_paths.append(
        "scripts/code_blocks/"
        "block09e_n1_lock_amp_numerics.py"
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
    "✓ git diff --cached --check            : PASS"
)


staged = sh(
    [
        "git",
        "diff",
        "--cached",
        "--name-only",
    ],
    cwd=REPO,
).stdout.splitlines()


if any(
    path.endswith(
        ".pt"
    )
    for path in staged
):

    raise RuntimeError(
        "Runtime model/checkpoint must not be committed."
    )


sh(
    [
        "git",
        "commit",
        "-m",
        "protocol: freeze COVA-3D AMP numerical amendment N1",
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
        "Repository is not clean after N1 commit:\n"
        + final_status
    )


# ==========================================================================================
# 11. FINAL REPORT
# ==========================================================================================

heading(
    "COVA-3D BLOCK 09E-N1-LOCK — FINAL REPORT"
)


print(
    "Starting commit                       :",
    head[:12],
)

print(
    "N1 commit                             :",
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
    "FAILURE CLASSIFICATION"
)

print(
    "----------------------"
)

print(
    "Failed attempt successful updates     : 0"
)

print(
    "Failed attempt completed epochs       : 0"
)

print(
    "Forward computation                   : FINITE"
)

print(
    "Scale 65536 gradients                 : NONFINITE"
)

print(
    "Scale 4096 gradients                  : FINITE"
)

print(
    "Scale 1 gradients                     : FINITE"
)

print(
    "Diagnosis                             : TRANSIENT FP16 LOSS-SCALE OVERFLOW"
)

print()

print(
    "NUMERICAL AMENDMENT N1"
)

print(
    "----------------------"
)

print(
    "AMP dtype                             : FP16"
)

print(
    "Initial loss scale                    :",
    AMP_INITIAL_SCALE,
)

print(
    "Growth factor                         :",
    AMP_GROWTH_FACTOR,
)

print(
    "Backoff factor                        :",
    AMP_BACKOFF_FACTOR,
)

print(
    "Growth interval                       :",
    AMP_GROWTH_INTERVAL,
)

print(
    "Upward growth during frozen runs      : EFFECTIVELY DISABLED"
)

print(
    "Overflow action                       : SAME-PAIR RETRY AFTER BACKOFF"
)

print(
    "Minimum permitted scale               :",
    AMP_MINIMUM_SCALE,
)

print(
    "Max retries / logical step            :",
    AMP_MAX_RETRIES_PER_LOGICAL_STEP,
)

print()

print(
    "SCIENTIFIC PROTOCOL"
)

print(
    "-------------------"
)

print(
    "Architecture changed                  : NO"
)

print(
    "Sparse loss changed                   : NO"
)

print(
    "Optimizer changed                     : NO"
)

print(
    "LR / schedule changed                 : NO"
)

print(
    "Annotations changed                   : NO"
)

print(
    "Sampling changed                      : NO"
)

print(
    "Augmentation changed                  : NO"
)

print(
    "Epochs changed                        : NO"
)

print(
    "Successful optimizer-step target      : 1000"
)

print()

print(
    "AUTHORIZATION"
)

print(
    "-------------"
)

print(
    "Sanity retry authorized               : YES"
)

print(
    "Dense sanity evaluation authorized    : NO — wait for completed retry"
)

print(
    "Factorial training authorized         : NO"
)

print(
    "COVA successful optimizer steps       : 0"
)

print(
    "Final outer-CV access                 : 0"
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
    "Do NOT rerun the old sanity-fit cell."
)

print(
    "Next I will give you 09E-SANITY-FIT-R1."
)

print(
    "It will start again from the original frozen initialization, use the "
    "same 2,000 logical microbatches, and implement N1 correctly."
)

print(
    "=" * 128
)
