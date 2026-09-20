# ==========================================================================================
# COVA-3D — BLOCK 09E-N2-LOCK
# Prospective Numerical Amendment N2:
# FP32 Sparse-Dice Arithmetic Under FP16 Model Autocast
#
# EXPECTED START COMMIT
# ---------------------
# b51e3667b543
#
# EXACT-STATE DIAGNOSIS
# ---------------------
# Failed R1:
#   epoch                  : 5
#   logical pair           : 33
#   microbatches           : 66, 67
#   successful updates     : 233
#   persisted epoch-boundary updates : 200
#   AMP scale              : 1
#
# Exact failed-state ablation:
#
#   AMP original BCE            -> FINITE gradients
#   AMP original Dice           -> NON-FINITE gradients
#   AMP original combined       -> NON-FINITE gradients
#
#   AMP forward + FP32 Dice     -> FINITE gradients
#   AMP forward + FP32 combined -> FINITE gradients
#   full FP32 combined          -> FINITE gradients
#
# Root numerical event:
#
#   all-background sparse microbatch
#   Dice denominator FP16 ≈ 1e-6
#   reciprocal FP16 = inf
#   reciprocal FP32 = finite
#
# N2 CHANGES ONLY:
# ----------------
# Sparse Dice arithmetic is evaluated explicitly in FP32.
#
# The network forward remains AMP FP16.
# Partial BCE remains exactly as frozen.
# Dice equation remains exactly the same.
# Dice epsilon remains exactly 1e-6.
# BCE weight remains 1.
# Dice weight remains 1.
#
# N1 also remains active:
#   initial loss scale = 4096
#   overflow backoff
#   exact same-pair retry
#   successful updates only are counted
#
# NOTHING ELSE CHANGES.
#
# THIS BLOCK:
#   ✓ verifies the exact failed state again
#   ✓ adds an explicit FP32 sparse-Dice implementation
#   ✓ verifies the new implementation on the exact failed pair
#   ✓ freezes N2 prospectively
#   ✓ records all 233 R1 updates as NON-RETAINED
#   ✓ commits/pushes protocol metadata and implementation
#
# THIS BLOCK DOES NOT:
#   ✗ call optimizer.step()
#   ✗ train a model
#   ✗ open dense lesion masks
#   ✗ access final-CV data/outcomes
#
# NEXT AFTER PASS:
#   09E-SANITY-FIT-R2
#   clean restart from ORIGINAL initialization.
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
    "b51e3667b543"
)

BLOCK = (
    "09E-N2-LOCK"
)

AMENDMENT_ID = (
    "N2_FP32_SPARSE_DICE_ARITHMETIC"
)

BASELINE_ID = (
    "COVA3D_NNUNET_STYLE_PARTIAL_V1"
)

PROTOCOL_BEFORE = (
    "COVA3D_1.0+A1+A1.1+A1.2+N1"
)

PROTOCOL_AFTER = (
    "COVA3D_1.0+A1+A1.1+A1.2+N1+N2"
)

CONDITION = (
    "C100_COH"
)

SEED = (
    17
)

FEATURES = (
    32,
    64,
    128,
    256,
    320,
)

PATCH_ZYX = np.asarray(
    [
        48,
        128,
        128,
    ],
    dtype=np.int32,
)

ACCUMULATION_STEPS = (
    2
)

UNKNOWN_LABEL = (
    -1
)

DICE_EPS = (
    1e-6
)

AMP_INITIAL_SCALE = (
    4096.0
)

AMP_MINIMUM_SCALE = (
    1.0
)

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

EXPECTED_FAILED_EPOCH_ZERO_BASED = (
    4
)

EXPECTED_FAILED_PAIR = (
    33
)

EXPECTED_LIVE_SUCCESSFUL_STEPS = (
    233
)

EXPECTED_PERSISTED_SUCCESSFUL_STEPS = (
    200
)

EXPECTED_INCOMPLETE_EPOCH_STEPS = (
    33
)

EXPECTED_FAILED_SCALE = (
    1.0
)

R1_ROOT = Path(
    "/kaggle/working/cova3d_sanity_fit_R1_v1_0"
)

R1_STATE_PATH = (
    R1_ROOT
    / "run_state.json"
)

R1_EPOCH_LOG_PATH = (
    R1_ROOT
    / "epoch_log.csv"
)

NOW_ISO = datetime.now(
    timezone.utc
).strftime(
    "%Y-%m-%dT%H:%M:%SZ"
)


# ==========================================================================================
# 1. GENERAL HELPERS
# ==========================================================================================

def heading(text):

    print(
        "\n"
        + "=" * 132
    )

    print(
        text
    )

    print(
        "=" * 132
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
        "/tmp/cova3d_git_askpass_09e_n2.sh"
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


def gradients_report(model):

    gradient_tensors = 0

    nonfinite_tensors = 0

    nonfinite_values = 0

    first_bad_parameter = None

    maximum_abs_finite_gradient = 0.0


    for name, parameter in model.named_parameters():

        if parameter.grad is None:

            continue


        gradient_tensors += 1


        gradient = parameter.grad.detach()


        finite = torch.isfinite(
            gradient
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


            if first_bad_parameter is None:

                first_bad_parameter = (
                    name
                )


        finite_values = gradient[
            finite
        ]


        if finite_values.numel() > 0:

            maximum_abs_finite_gradient = max(
                maximum_abs_finite_gradient,
                float(
                    finite_values.abs().max().item()
                ),
            )


    return {
        "finite":
            (
                nonfinite_tensors
                == 0
            ),

        "gradient_tensors":
            gradient_tensors,

        "nonfinite_tensors":
            nonfinite_tensors,

        "nonfinite_values":
            nonfinite_values,

        "first_bad_parameter":
            first_bad_parameter,

        "maximum_abs_finite_gradient":
            maximum_abs_finite_gradient,
    }


def augment_patch_exact(
    image,
    target,
    *,
    epoch_value,
    microbatch_index,
    device,
):

    rng = np.random.default_rng(
        stable_seed(
            BASELINE_ID,
            SEED,
            "sanity_augmentation",
            epoch_value,
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


    intensity_scale = float(
        rng.uniform(
            *INTENSITY_SCALE_RANGE
        )
    )


    intensity_shift = float(
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
        * intensity_scale
        + intensity_shift
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
                epoch_value,
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


# ==========================================================================================
# 2. VERIFY REPOSITORY + LIVE FAILED R1 STATE
# ==========================================================================================

heading(
    "COVA-3D BLOCK 09E-N2-LOCK — FP32 SPARSE-DICE NUMERICAL AMENDMENT"
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
        "Repository must be clean before N2.\n"
        + dirty
    )


required_globals = [
    "model",
    "scaler",
    "epoch",
    "pair_index",
    "schedule",
    "samples",
    "successful_optimizer_steps",
]


missing = [
    key
    for key in required_globals
    if key not in globals()
]


if missing:

    raise RuntimeError(
        "Exact failed R1 state is no longer available.\n"
        + "Missing variables: "
        + ", ".join(
            missing
        )
        + "\nDo not restart training. Send me this error."
    )


live_model = globals()[
    "model"
]


live_scaler = globals()[
    "scaler"
]


live_epoch = int(
    globals()[
        "epoch"
    ]
)


live_pair_index = int(
    globals()[
        "pair_index"
    ]
)


live_schedule = globals()[
    "schedule"
]


live_samples = globals()[
    "samples"
]


live_successful_steps = int(
    globals()[
        "successful_optimizer_steps"
    ]
)


live_scale = float(
    live_scaler.get_scale()
)


if live_epoch != EXPECTED_FAILED_EPOCH_ZERO_BASED:

    raise RuntimeError(
        "Unexpected failed epoch."
    )


if live_pair_index != EXPECTED_FAILED_PAIR:

    raise RuntimeError(
        "Unexpected failed logical pair."
    )


if live_successful_steps != EXPECTED_LIVE_SUCCESSFUL_STEPS:

    raise RuntimeError(
        "Unexpected successful-update count at failure."
    )


if not np.isclose(
    live_scale,
    EXPECTED_FAILED_SCALE,
):

    raise RuntimeError(
        "Unexpected AMP scale at failure."
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
) != "09E-N1-LOCK":

    raise RuntimeError(
        "Expected 09E-N1-LOCK as the previous committed block."
    )


if int(
    cova_state.get(
        "optimizer_steps_in_cova3d",
        -1,
    )
) != 0:

    raise RuntimeError(
        "Committed state must still contain zero retained COVA updates."
    )


if cova_state.get(
    "dense_outcomes_opened_in_cova3d"
) is not False:

    raise RuntimeError(
        "Dense outcomes were unexpectedly opened."
    )


if cova_state.get(
    "factorial_training_authorized"
) is not False:

    raise RuntimeError(
        "Factorial training must remain locked."
    )


runtime_state = {}


if R1_STATE_PATH.exists():

    runtime_state = json.loads(
        R1_STATE_PATH.read_text(
            encoding="utf-8"
        )
    )


persisted_steps = int(
    runtime_state.get(
        "successful_optimizer_steps",
        0,
    )
)


persisted_completed_epoch = int(
    runtime_state.get(
        "completed_epoch",
        -1,
    )
)


if persisted_steps != EXPECTED_PERSISTED_SUCCESSFUL_STEPS:

    raise RuntimeError(
        "Unexpected persisted R1 update count."
    )


if persisted_completed_epoch != 3:

    raise RuntimeError(
        "Expected four completed R1 epochs."
    )


discarded_incomplete_steps = (
    live_successful_steps
    - persisted_steps
)


if discarded_incomplete_steps != EXPECTED_INCOMPLETE_EPOCH_STEPS:

    raise RuntimeError(
        "Unexpected incomplete-epoch update count."
    )


print(
    "✓ Starting commit                      :",
    head[:12],
)

print(
    "✓ Repository state                     : CLEAN"
)

print(
    "✓ R1 failure epoch                     : 5"
)

print(
    "✓ R1 failure pair                      : 33"
)

print(
    "✓ R1 successful updates executed       :",
    live_successful_steps,
)

print(
    "✓ R1 epoch-boundary updates persisted  :",
    persisted_steps,
)

print(
    "✓ R1 incomplete-epoch updates          :",
    discarded_incomplete_steps,
)

print(
    "✓ R1 updates retained scientifically   : 0"
)

print(
    "✓ Dense outcomes                       : SEALED"
)

print(
    "✓ Final outer-CV                       : SEALED"
)


# ==========================================================================================
# 3. SNAPSHOT EXACT FAILED MODEL WEIGHTS
# ==========================================================================================

heading(
    "STEP 1/8 — SNAPSHOT EXACT FAILED MODEL STATE"
)


failed_state_cpu = {
    key:
        value.detach().cpu().clone()
    for key, value
    in live_model.state_dict().items()
}


if not all(
    bool(
        torch.isfinite(
            tensor
        ).all()
    )
    for tensor in failed_state_cpu.values()
    if torch.is_floating_point(
        tensor
    )
):

    raise RuntimeError(
        "Failed-state model parameters themselves are non-finite."
    )


print(
    "✓ Exact failed model parameters        : FINITE"
)

print(
    "✓ Failed gradients themselves          : NON-FINITE"
)

print(
    "✓ Snapshot source                      : PRE-FAILED-UPDATE WEIGHTS"
)


# ==========================================================================================
# 4. MODIFY LOSS MODULE — ADD N2 FUNCTION WITHOUT ALTERING LEGACY FUNCTIONS
# ==========================================================================================

heading(
    "STEP 2/8 — ADD EXPLICIT FP32 SPARSE-DICE IMPLEMENTATION"
)


partial_path = (
    REPO
    / "src/cora_lung/losses/partial.py"
)


partial_text = partial_path.read_text(
    encoding="utf-8"
)


function_marker = (
    "def partial_dice_fp32("
)


if function_marker not in partial_text:

    addition = r'''

def partial_dice_fp32(
    logits,
    target,
    eps=1e-6,
):
    """Sparse Dice loss with explicit FP32 arithmetic.

    This preserves the exact sparse Dice equation and epsilon while avoiding
    FP16 reciprocal overflow for very small denominators.

    The cast remains differentiable, so gradients propagate back through the
    mixed-precision model forward.
    """

    mask = labelled_mask(
        target
    )

    if not torch.any(mask):
        return (
            logits.float().sum()
            * 0.0
        )

    # Explicitly disable autocast for the numerically sensitive Dice
    # arithmetic. The network forward may still have produced FP16 logits.
    with torch.autocast(
        device_type=logits.device.type,
        enabled=False,
    ):

        logits_fp32 = logits.float()

        p = torch.sigmoid(
            logits_fp32[
                mask
            ]
        )

        y = target[
            mask
        ].float()

        numerator = (
            2.0
            * torch.sum(
                p
                * y
            )
            + float(
                eps
            )
        )

        denominator = (
            torch.sum(
                p
            )
            + torch.sum(
                y
            )
            + float(
                eps
            )
        )

        return (
            1.0
            - numerator
            / denominator
        )


def partial_segmentation_loss_n2(
    logits,
    target,
):
    """COVA-3D N2 sparse objective.

    BCE is unchanged.
    Only sparse Dice arithmetic is forced to FP32.
    """

    return (
        partial_bce(
            logits,
            target,
        )
        + partial_dice_fp32(
            logits,
            target,
        )
    )
'''

    partial_path.write_text(
        partial_text.rstrip()
        + addition
        + "\n",
        encoding="utf-8",
    )


print(
    "✓ Legacy partial_bce                    : UNCHANGED"
)

print(
    "✓ Legacy partial_dice                   : UNCHANGED"
)

print(
    "✓ New partial_dice_fp32                 : ADDED"
)

print(
    "✓ New partial_segmentation_loss_n2      : ADDED"
)

print(
    "✓ Dice epsilon                          :",
    DICE_EPS,
)


# ==========================================================================================
# 5. REIMPORT UPDATED LOSS MODULE + RECONSTRUCT EXACT FAILED PAIR
# ==========================================================================================

heading(
    "STEP 3/8 — RECONSTRUCT EXACT FAILED PAIR"
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
    partial_dice_fp32,
    partial_segmentation_loss_n2,
)


from cora_lung.data.pilot_dataset import (
    extract_patch,
)


if not torch.cuda.is_available():

    raise RuntimeError(
        "GPU required."
    )


device = torch.device(
    "cuda"
)


patch_sampler_seed = stable_seed(
    BASELINE_ID,
    "sanity_patch_sampler",
    SEED,
)


failed_microbatch_indices = [
    (
        live_pair_index
        * ACCUMULATION_STEPS
        + offset
    )
    for offset in range(
        ACCUMULATION_STEPS
    )
]


prepared_pair = []


for microbatch_index in failed_microbatch_indices:

    case_id = str(
        live_schedule[
            microbatch_index
        ]
    )


    patch = extract_patch(
        live_samples[
            case_id
        ],
        PATCH_ZYX,
        seed=patch_sampler_seed,
        sample_index=(
            live_epoch
            * 100
            + microbatch_index
        ),
    )


    image, target = augment_patch_exact(
        patch[
            "image"
        ].clone(),
        patch[
            "target"
        ].clone(),
        epoch_value=live_epoch,
        microbatch_index=microbatch_index,
        device=device,
    )


    prepared_pair.append(
        {
            "microbatch_index":
                microbatch_index,

            "case_id":
                case_id,

            "sampling_source":
                str(
                    patch[
                        "sampling_source"
                    ]
                ),

            "image_cpu":
                image.detach().cpu(),

            "target_cpu":
                target.detach().cpu(),

            "labelled":
                int(
                    (
                        target
                        != UNKNOWN_LABEL
                    ).sum().item()
                ),

            "foreground":
                int(
                    (
                        target
                        == 1
                    ).sum().item()
                ),

            "background":
                int(
                    (
                        target
                        == 0
                    ).sum().item()
                ),
        }
    )


    del image
    del target
    del patch


pair_fg = int(
    sum(
        item[
            "foreground"
        ]
        for item in prepared_pair
    )
)


pair_bg = int(
    sum(
        item[
            "background"
        ]
        for item in prepared_pair
    )
)


for item in prepared_pair:

    print(
        "microbatch="
        + str(
            item[
                "microbatch_index"
            ]
        )
        + " | case="
        + item[
            "case_id"
        ]
        + " | source="
        + item[
            "sampling_source"
        ]
        + " | labelled="
        + str(
            item[
                "labelled"
            ]
        )
        + " | FG="
        + str(
            item[
                "foreground"
            ]
        )
        + " | BG="
        + str(
            item[
                "background"
            ]
        )
    )


if pair_fg != 29:

    raise RuntimeError(
        "Failed-pair foreground count did not reproduce."
    )


if pair_bg != 29:

    raise RuntimeError(
        "Failed-pair background count did not reproduce."
    )


print()
print(
    "✓ Pair total FG                        :",
    pair_fg,
)

print(
    "✓ Pair total BG                        :",
    pair_bg,
)


# ==========================================================================================
# 6. EXACT-STATE N2 VALIDATION
# ==========================================================================================

heading(
    "STEP 4/8 — VALIDATE N2 ON EXACT FAILED MODEL STATE"
)


def instantiate_failed_snapshot():

    test_model = COVA3DNNUNetStyle(
        in_channels=1,
        out_channels=1,
        features=FEATURES,
    ).to(
        device
    )


    test_model.load_state_dict(
        failed_state_cpu,
        strict=True,
    )


    test_model.train()


    test_model.zero_grad(
        set_to_none=True
    )


    return test_model


def run_variant(
    variant_name,
    *,
    use_fp32_dice,
    manual_gradient_scale,
):

    test_model = instantiate_failed_snapshot()


    losses = []

    bces = []

    dices = []


    for item in prepared_pair:

        image = item[
            "image_cpu"
        ].to(
            device=device,
            dtype=torch.float32,
        )


        target = item[
            "target_cpu"
        ].to(
            device=device,
            dtype=torch.int8,
        )


        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=True,
        ):

            logits = test_model(
                image
            )


            bce = partial_bce(
                logits,
                target,
            )


            if not use_fp32_dice:

                dice = partial_dice(
                    logits,
                    target,
                )


        if use_fp32_dice:

            dice = partial_dice_fp32(
                logits,
                target,
            )


        raw_loss = (
            bce
            + dice
        )


        if not bool(
            torch.isfinite(
                raw_loss
            )
        ):

            loss_finite = (
                False
            )

        else:

            loss_finite = (
                True
            )


        losses.append(
            float(
                raw_loss.detach().cpu()
            )
        )


        bces.append(
            float(
                bce.detach().cpu()
            )
        )


        dices.append(
            float(
                dice.detach().cpu()
            )
        )


        scaled = (
            raw_loss
            / ACCUMULATION_STEPS
            * float(
                manual_gradient_scale
            )
        )


        scaled.backward()


        del image
        del target
        del logits
        del bce
        del dice
        del raw_loss
        del scaled


    grad = gradients_report(
        test_model
    )


    result = {
        "variant":
            variant_name,

        "fp32_dice":
            bool(
                use_fp32_dice
            ),

        "manual_gradient_scale":
            float(
                manual_gradient_scale
            ),

        "loss_finite":
            bool(
                np.isfinite(
                    losses
                ).all()
            ),

        "mean_loss":
            float(
                np.mean(
                    losses
                )
            ),

        "mean_bce":
            float(
                np.mean(
                    bces
                )
            ),

        "mean_dice":
            float(
                np.mean(
                    dices
                )
            ),

        **grad,
    }


    del test_model

    torch.cuda.empty_cache()

    gc.collect()


    return result


validation_rows = []


for specification in [
    (
        "OLD_FP16_DICE_SCALE1",
        False,
        1.0,
    ),

    (
        "N2_FP32_DICE_SCALE1",
        True,
        1.0,
    ),

    (
        "N2_FP32_DICE_SCALE4096",
        True,
        4096.0,
    ),
]:

    result = run_variant(
        specification[
            0
        ],
        use_fp32_dice=specification[
            1
        ],
        manual_gradient_scale=specification[
            2
        ],
    )


    validation_rows.append(
        result
    )


    print(
        result[
            "variant"
        ].ljust(
            30
        )
        + " | loss="
        + (
            "FINITE"
            if result[
                "loss_finite"
            ]
            else "NONFINITE"
        )
        + " | gradients="
        + (
            "FINITE"
            if result[
                "finite"
            ]
            else "NONFINITE"
        )
        + " | bad tensors="
        + str(
            result[
                "nonfinite_tensors"
            ]
        )
    )


validation_df = pd.DataFrame(
    validation_rows
)


validation_lookup = {
    row[
        "variant"
    ]:
        row
    for row in validation_rows
}


if validation_lookup[
    "OLD_FP16_DICE_SCALE1"
][
    "finite"
]:

    raise RuntimeError(
        "Original FP16 Dice failure did not reproduce at scale 1."
    )


if not validation_lookup[
    "N2_FP32_DICE_SCALE1"
][
    "finite"
]:

    raise RuntimeError(
        "N2 failed to produce finite gradients at scale 1."
    )


print()
print(
    "✓ Original FP16 Dice @ scale 1         : NON-FINITE"
)

print(
    "✓ N2 FP32 Dice @ scale 1               : FINITE"
)


if validation_lookup[
    "N2_FP32_DICE_SCALE4096"
][
    "finite"
]:

    print(
        "✓ N2 FP32 Dice @ scale 4096            : FINITE"
    )

else:

    print(
        "✓ N2 FP32 Dice @ scale 4096            : OVERFLOW — N1 BACKOFF WILL HANDLE"
    )


# ==========================================================================================
# 7. VERIFY NUMERICAL ROOT EVENT
# ==========================================================================================

heading(
    "STEP 5/8 — VERIFY FP16 RECIPROCAL OVERFLOW ROOT EVENT"
)


all_background_items = [
    item
    for item in prepared_pair
    if item[
        "foreground"
    ]
    == 0
]


if len(
    all_background_items
) != 1:

    raise RuntimeError(
        "Expected exactly one all-background microbatch in failed pair."
    )


item = all_background_items[
    0
]


diagnostic_model = instantiate_failed_snapshot()


image = item[
    "image_cpu"
].to(
    device=device,
    dtype=torch.float32,
)


target = item[
    "target_cpu"
].to(
    device=device,
    dtype=torch.int8,
)


with torch.no_grad():

    with torch.autocast(
        device_type="cuda",
        dtype=torch.float16,
        enabled=True,
    ):

        logits = diagnostic_model(
            image
        )


mask = (
    target
    != UNKNOWN_LABEL
)


labelled_logits = logits[
    mask
]


p16 = torch.sigmoid(
    labelled_logits
)


p32 = torch.sigmoid(
    labelled_logits.float()
)


y16 = target[
    mask
].to(
    dtype=p16.dtype
)


y32 = target[
    mask
].float()


denominator16 = (
    p16.sum()
    + y16.sum()
    + DICE_EPS
)


denominator32 = (
    p32.sum()
    + y32.sum()
    + DICE_EPS
)


reciprocal16 = (
    1.0
    / denominator16
)


reciprocal32 = (
    1.0
    / denominator32
)


if bool(
    torch.isfinite(
        reciprocal16
    )
):

    raise RuntimeError(
        "Expected FP16 reciprocal overflow did not reproduce."
    )


if not bool(
    torch.isfinite(
        reciprocal32
    )
):

    raise RuntimeError(
        "FP32 reciprocal is unexpectedly non-finite."
    )


root_event = {
    "microbatch_index":
        int(
            item[
                "microbatch_index"
            ]
        ),

    "case_id":
        item[
            "case_id"
        ],

    "labelled":
        int(
            item[
                "labelled"
            ]
        ),

    "foreground":
        int(
            item[
                "foreground"
            ]
        ),

    "background":
        int(
            item[
                "background"
            ]
        ),

    "dice_denominator_fp16":
        float(
            denominator16.float().cpu()
        ),

    "dice_denominator_fp32":
        float(
            denominator32.cpu()
        ),

    "reciprocal_fp16_finite":
        bool(
            torch.isfinite(
                reciprocal16
            )
        ),

    "reciprocal_fp32":
        float(
            reciprocal32.cpu()
        ),

    "reciprocal_fp32_finite":
        bool(
            torch.isfinite(
                reciprocal32
            )
        ),
}


print(
    "✓ All-background microbatch            :",
    root_event[
        "microbatch_index"
    ],
)

print(
    "✓ Labelled FG/BG                       :",
    str(
        root_event[
            "foreground"
        ]
    )
    + "/"
    + str(
        root_event[
            "background"
        ]
    ),
)

print(
    "✓ Dice denominator FP16                :",
    root_event[
        "dice_denominator_fp16"
    ],
)

print(
    "✓ Reciprocal FP16 finite               :",
    root_event[
        "reciprocal_fp16_finite"
    ],
)

print(
    "✓ Dice denominator FP32                :",
    root_event[
        "dice_denominator_fp32"
    ],
)

print(
    "✓ Reciprocal FP32                      :",
    root_event[
        "reciprocal_fp32"
    ],
)

print(
    "✓ Reciprocal FP32 finite               :",
    root_event[
        "reciprocal_fp32_finite"
    ],
)


del diagnostic_model
del image
del target
del logits
del labelled_logits
del p16
del p32
del y16
del y32
del denominator16
del denominator32
del reciprocal16
del reciprocal32

torch.cuda.empty_cache()

gc.collect()


# ==========================================================================================
# 8. FREEZE N2 PROTOCOL AMENDMENT
# ==========================================================================================

heading(
    "STEP 6/8 — FREEZE NUMERICAL AMENDMENT N2"
)


n2 = {
    "project":
        "COVA-3D",

    "amendment_id":
        AMENDMENT_ID,

    "status":
        "PROSPECTIVELY_FROZEN_BEFORE_CLEAN_SANITY_RESTART_R2",

    "parent_commit":
        head,

    "baseline_id":
        BASELINE_ID,

    "protocol_before":
        PROTOCOL_BEFORE,

    "protocol_after":
        PROTOCOL_AFTER,

    "trigger":
        {
            "run":
                "09E-SANITY-FIT-R1",

            "failed_epoch_1based":
                live_epoch
                + 1,

            "failed_pair_index":
                live_pair_index,

            "failed_microbatches":
                failed_microbatch_indices,

            "amp_scale_at_failure":
                live_scale,

            "successful_optimizer_updates_executed":
                live_successful_steps,

            "epoch_boundary_updates_persisted":
                persisted_steps,

            "incomplete_epoch_updates":
                discarded_incomplete_steps,

            "updates_retained_for_scientific_run":
                0,

            "dense_outcomes_accessed":
                False,

            "final_outer_cv_access":
                0,
        },

    "exact_state_diagnosis":
        {
            "AMP_original_combined":
                "NONFINITE_GRADIENTS",

            "AMP_original_BCE":
                "FINITE_GRADIENTS",

            "AMP_original_Dice":
                "NONFINITE_GRADIENTS",

            "AMP_forward_FP32_loss_combined":
                "FINITE_GRADIENTS",

            "AMP_forward_FP32_loss_BCE":
                "FINITE_GRADIENTS",

            "AMP_forward_FP32_loss_Dice":
                "FINITE_GRADIENTS",

            "full_FP32_combined":
                "FINITE_GRADIENTS",

            "full_FP32_BCE":
                "FINITE_GRADIENTS",

            "full_FP32_Dice":
                "FINITE_GRADIENTS",

            "diagnosis":
                "FP16_SPARSE_DICE_ARITHMETIC_INSTABILITY",
        },

    "root_event":
        root_event,

    "N2_policy":
        {
            "network_forward_autocast":
                True,

            "network_forward_dtype":
                "float16",

            "partial_BCE":
                "UNCHANGED",

            "partial_Dice_formula":
                "UNCHANGED",

            "partial_Dice_epsilon":
                DICE_EPS,

            "partial_Dice_arithmetic_dtype":
                "float32",

            "partial_Dice_autocast":
                False,

            "BCE_weight":
                1.0,

            "Dice_weight":
                1.0,

            "combined_objective":
                "partial_BCE + partial_Dice_fp32",

            "N1_remains_active":
                True,

            "N1_initial_scale":
                AMP_INITIAL_SCALE,

            "N1_minimum_scale":
                AMP_MINIMUM_SCALE,

            "N1_same_pair_retry":
                True,
        },

    "scientifically_unchanged":
        {
            "annotation_protocol":
                True,

            "coverage_factor":
                True,

            "geometry_factor":
                True,

            "network_architecture":
                True,

            "network_parameter_count":
                True,

            "loss_equation":
                True,

            "loss_weights":
                True,

            "Dice_epsilon":
                True,

            "optimizer":
                True,

            "learning_rate":
                True,

            "learning_rate_schedule":
                True,

            "weight_decay":
                True,

            "gradient_clip":
                True,

            "patch_size":
                True,

            "sampling_schedule":
                True,

            "augmentation":
                True,

            "epochs":
                True,

            "microbatches_per_epoch":
                True,

            "successful_optimizer_step_target":
                1000,

            "checkpoint_rule":
                True,

            "final_factorial_protocol":
                True,
        },

    "restart_policy":
        {
            "R1_weights_reused":
                False,

            "R1_optimizer_state_reused":
                False,

            "R1_scaler_state_reused":
                False,

            "R1_updates_retained":
                0,

            "restart":
                "ORIGINAL_FROZEN_INITIALIZATION",

            "next_run":
                "09E-SANITY-FIT-R2",
        },

    "factorial_training_authorized":
        False,

    "dense_sanity_evaluation_authorized":
        False,

    "final_outer_cv_outcomes_authorized":
        False,

    "next_block":
        "09E-SANITY-FIT-R2",

    "frozen_at_utc":
        NOW_ISO,
}


n2_path = (
    REPO
    / "configs/"
    "cova3d_numerical_amendment_N2_fp32_dice.yaml"
)


n2_path.write_text(
    yaml.safe_dump(
        n2,
        sort_keys=False,
        allow_unicode=True,
    ),
    encoding="utf-8",
)


write_text(
    REPO
    / "docs/"
    "cova3d_numerical_amendment_N2_fp32_dice.md",
    """
    # COVA-3D Numerical Amendment N2

    ## Trigger

    The N1-corrected sanity run reached epoch 5 and 233 successful optimizer
    updates before gradients became non-finite at AMP scale 1.

    The exact failed model state was preserved and diagnosed before any dense
    outcome was opened.

    The failed logical pair contained two microbatches. One was an
    all-background sparse patch with only two labelled background voxels.

    ## Exact-state diagnosis

    BCE alone produced finite gradients.

    Sparse Dice alone produced non-finite gradients in the original FP16
    arithmetic.

    The combined original objective therefore produced non-finite gradients.

    Recomputing the sparse Dice arithmetic in FP32 while retaining the AMP
    FP16 network forward produced finite gradients.

    Full FP32 training also produced finite gradients.

    The failure was therefore localized to FP16 sparse-Dice arithmetic.

    ## Numerical mechanism

    For the all-background sparse patch, the learned probabilities at the two
    labelled voxels were extremely small.

    The FP16 Dice denominator approached the epsilon term of 1e-6.

    Its reciprocal exceeded the representable finite FP16 range and became
    infinity.

    The corresponding FP32 reciprocal remained finite.

    ## Frozen N2 policy

    The network forward remains under FP16 autocast.

    Partial BCE is unchanged.

    Sparse Dice uses the exact same equation and epsilon, but its arithmetic
    is explicitly performed in FP32 with autocast disabled.

    The total objective remains:

        partial BCE + partial Dice

    with equal weights.

    N1 remains active for ordinary mixed-precision loss-scale overflow.

    ## Restart

    The R1 model is retired.

    Its 233 executed optimizer updates are recorded for provenance but none
    are retained in the clean scientific sanity run.

    R2 will restart from the original frozen initialization.

    No dense development outcome has yet been opened.

    Final factorial training remains locked.
    """
)


# ==========================================================================================
# 9. ARCHIVE DIAGNOSTIC + UPDATE STATE
# ==========================================================================================

heading(
    "STEP 7/8 — ARCHIVE DIAGNOSTIC AND UPDATE STATE"
)


manifest_dir = (
    REPO
    / "data/manifests"
)


audit_dir = (
    REPO
    / "experiments/audits"
)


manifest_dir.mkdir(
    parents=True,
    exist_ok=True,
)


audit_dir.mkdir(
    parents=True,
    exist_ok=True,
)


validation_path = (
    manifest_dir
    / "cova3d_N2_exact_failed_state_validation.csv"
)


validation_df.to_csv(
    validation_path,
    index=False,
)


diagnostic_audit = {
    "project":
        "COVA-3D",

    "block":
        "09E-N1-FAIL-DIAG",

    "status":
        "DIAGNOSIS_COMPLETE",

    "repository_modified_during_diagnostic":
        False,

    "optimizer_steps_during_diagnostic":
        0,

    "dense_masks_accessed":
        0,

    "final_outer_cv_access":
        0,

    "failure_location":
        {
            "epoch_1based":
                5,

            "pair_index":
                33,

            "microbatches":
                [
                    66,
                    67,
                ],

            "successful_updates_before_failure":
                233,

            "persisted_successful_updates":
                200,

            "incomplete_epoch_updates":
                33,

            "amp_scale":
                1.0,

            "pair_foreground_labels":
                29,

            "pair_background_labels":
                29,
        },

    "precision_loss_matrix":
        {
            "AMP_original_combined":
                "NONFINITE",

            "AMP_original_BCE":
                "FINITE",

            "AMP_original_Dice":
                "NONFINITE",

            "AMP_forward_FP32_loss_combined":
                "FINITE",

            "AMP_forward_FP32_loss_BCE":
                "FINITE",

            "AMP_forward_FP32_loss_Dice":
                "FINITE",

            "FULL_FP32_combined":
                "FINITE",

            "FULL_FP32_BCE":
                "FINITE",

            "FULL_FP32_Dice":
                "FINITE",
        },

    "root_event":
        root_event,

    "diagnosis":
        "FP16_DICE_ARITHMETIC_INSTABILITY",

    "conclusion":
        (
            "Sparse Dice arithmetic must be executed in FP32. "
            "The network forward may remain AMP FP16."
        ),

    "created_at_utc":
        NOW_ISO,
}


diagnostic_audit_path = (
    audit_dir
    / "block09e_n1_failed_state_diagnostic.json"
)


write_json(
    diagnostic_audit_path,
    diagnostic_audit,
)


n2_audit = {
    "project":
        "COVA-3D",

    "block":
        BLOCK,

    "status":
        "PASS",

    "amendment":
        AMENDMENT_ID,

    "parent_commit":
        head,

    "protocol_after":
        PROTOCOL_AFTER,

    "R1_successful_updates_executed":
        live_successful_steps,

    "R1_successful_updates_retained":
        0,

    "R1_dense_outcomes_accessed":
        False,

    "legacy_Dice_scale1_finite":
        bool(
            validation_lookup[
                "OLD_FP16_DICE_SCALE1"
            ][
                "finite"
            ]
        ),

    "N2_Dice_scale1_finite":
        bool(
            validation_lookup[
                "N2_FP32_DICE_SCALE1"
            ][
                "finite"
            ]
        ),

    "N2_Dice_scale4096_finite":
        bool(
            validation_lookup[
                "N2_FP32_DICE_SCALE4096"
            ][
                "finite"
            ]
        ),

    "root_event":
        root_event,

    "network_forward_dtype":
        "AMP_FP16",

    "Dice_arithmetic_dtype":
        "FP32",

    "Dice_formula_changed":
        False,

    "Dice_epsilon_changed":
        False,

    "BCE_changed":
        False,

    "factorial_training_authorized":
        False,

    "next_block":
        "09E-SANITY-FIT-R2",

    "generated_at_utc":
        NOW_ISO,
}


n2_audit_path = (
    audit_dir
    / "block09e_n2_lock_fp32_dice.json"
)


write_json(
    n2_audit_path,
    n2_audit,
)


# ------------------------------------------------------------------------------------------
# Transparent accounting:
#
# 233 successful optimizer updates were physically executed in R1.
# They are retained as FAILURE-PROVENANCE ONLY.
#
# The clean frozen sanity run begins again from the original initialization,
# so retained scientific sanity-run optimizer steps remain zero at this lock.
# ------------------------------------------------------------------------------------------

previous_discarded = int(
    cova_state.get(
        "optimizer_steps_executed_nonretained",
        0,
    )
)


total_nonretained = (
    previous_discarded
    + live_successful_steps
)


cova_state.update(
    {
        "last_completed_block":
            BLOCK,

        "current_stage":
            "SANITY_NUMERICAL_POLICY_N2_FROZEN",

        "effective_protocol":
            PROTOCOL_AFTER,

        "numerical_amendment":
            "N1+N2",

        "N2_status":
            "FROZEN",

        "sanity_fit_R1_status":
            "ABORTED_FP16_DICE_ARITHMETIC",

        "sanity_fit_R1_successful_updates_executed":
            live_successful_steps,

        "sanity_fit_R1_epoch_boundary_updates":
            persisted_steps,

        "sanity_fit_R1_incomplete_epoch_updates":
            discarded_incomplete_steps,

        "sanity_fit_R1_updates_retained":
            0,

        "optimizer_steps_executed_nonretained":
            total_nonretained,

        "optimizer_steps_retained_for_current_frozen_run":
            0,

        "optimizer_steps_in_cova3d":
            0,

        "optimizer_steps_in_cova3d_semantics":
            "retained_clean_frozen_run_only",

        "sparse_Dice_arithmetic":
            "FP32",

        "sparse_Dice_epsilon":
            DICE_EPS,

        "network_forward_precision":
            "AMP_FP16",

        "N1_active":
            True,

        "N2_active":
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
            "09E-SANITY-FIT-R2",

        "next_action":
            (
                "Restart the frozen C100_COH sanity fit from the original "
                "initialization using N1 AMP handling and N2 FP32 sparse-Dice "
                "arithmetic. Do not reuse R1 weights or optimizer state."
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
            "cova3d_numerical_amendment_N2_fp32_dice",

        "current_stage":
            "cova3d_sanity_R2_authorized",

        "current_gate":
            "COVA_SANITY_GATE",

        "cova3d_effective_protocol":
            PROTOCOL_AFTER,

        "cova3d_numerical_amendment":
            "N1+N2",

        "cova3d_R1_successful_updates_executed":
            live_successful_steps,

        "cova3d_R1_updates_retained":
            0,

        "cova3d_optimizer_steps_executed_nonretained":
            total_nonretained,

        "cova3d_optimizer_steps":
            0,

        "cova3d_sanity_training_authorized":
            True,

        "cova3d_sanity_gate":
            "NOT_RUN",

        "cova3d_dense_outcomes_opened":
            False,

        "cova3d_final_outer_cv_access":
            0,

        "cova3d_factorial_training_authorized":
            False,

        "cova3d_method_development_authorized":
            False,

        "training_authorized":
            False,

        "pilot_training_authorized":
            False,

        "next_action":
            "Run 09E-SANITY-FIT-R2 only.",

        "updated_at_utc":
            NOW_ISO,
    }
)


write_json(
    project_state_path,
    project_state,
)


# ==========================================================================================
# 10. REGRESSION TESTS
# ==========================================================================================

test_source = r'''
from pathlib import Path
import json

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_partial_dice_fp32_matches_reference():

    from cora_lung.losses.partial import (
        partial_dice,
        partial_dice_fp32,
    )

    logits = torch.tensor(
        [
            -16.0,
            -15.5,
            2.0,
            -1.0,
        ],
        dtype=torch.float16,
        requires_grad=True,
    )

    target = torch.tensor(
        [
            0,
            0,
            1,
            -1,
        ],
        dtype=torch.int8,
    )

    observed = partial_dice_fp32(
        logits,
        target,
    )

    reference = partial_dice(
        logits.float(),
        target,
    )

    assert observed.dtype == torch.float32

    assert torch.allclose(
        observed,
        reference,
        atol=1e-7,
        rtol=1e-7,
    )


def test_partial_dice_fp32_unknown_only_zero_gradient():

    from cora_lung.losses.partial import (
        partial_dice_fp32,
    )

    logits = torch.randn(
        12,
        dtype=torch.float16,
        requires_grad=True,
    )

    target = torch.full(
        (
            12,
        ),
        -1,
        dtype=torch.int8,
    )

    loss = partial_dice_fp32(
        logits,
        target,
    )

    assert float(
        loss.detach()
    ) == 0.0

    loss.backward()

    assert torch.all(
        logits.grad
        == 0
    )


def test_n2_protocol_is_frozen():

    cfg = yaml.safe_load(
        (
            ROOT
            / "configs/"
            "cova3d_numerical_amendment_N2_fp32_dice.yaml"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        cfg[
            "status"
        ]
        == "PROSPECTIVELY_FROZEN_BEFORE_CLEAN_SANITY_RESTART_R2"
    )

    assert (
        cfg[
            "N2_policy"
        ][
            "network_forward_dtype"
        ]
        == "float16"
    )

    assert (
        cfg[
            "N2_policy"
        ][
            "partial_Dice_arithmetic_dtype"
        ]
        == "float32"
    )

    assert (
        cfg[
            "N2_policy"
        ][
            "partial_Dice_epsilon"
        ]
        == 1e-6
    )

    assert (
        cfg[
            "N2_policy"
        ][
            "partial_BCE"
        ]
        == "UNCHANGED"
    )

    assert (
        cfg[
            "restart_policy"
        ][
            "R1_updates_retained"
        ]
        == 0
    )

    assert (
        cfg[
            "factorial_training_authorized"
        ]
        is False
    )


def test_n2_state():

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
        == "09E-N2-LOCK"
    )

    assert (
        state[
            "effective_protocol"
        ]
        == "COVA3D_1.0+A1+A1.1+A1.2+N1+N2"
    )

    assert (
        state[
            "sanity_fit_R1_successful_updates_executed"
        ]
        == 233
    )

    assert (
        state[
            "sanity_fit_R1_updates_retained"
        ]
        == 0
    )

    assert (
        state[
            "optimizer_steps_retained_for_current_frozen_run"
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
    "test_cova3d_numerical_amendment_n2.py"
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
        "tests/test_cova3d_numerical_amendment_n1.py",
        "tests/test_cova3d_numerical_amendment_n2.py",

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
        "N2 regression tests failed."
    )


print(
    "✓ N2 regression tests                  : PASS"
)


# ==========================================================================================
# 11. SOURCE CAPTURE + REPOSITORY MANIFEST
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
        "COVA-3D — BLOCK 09E-N2-LOCK"
        in cell
    ):

        source_path = (
            REPO
            / "scripts/code_blocks/"
            "block09e_n2_lock_fp32_sparse_dice.py"
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


n2_audit[
    "source_capture"
] = source_capture


write_json(
    n2_audit_path,
    n2_audit,
)


heading(
    "STEP 8/8 — FREEZE REPOSITORY PROVENANCE AND COMMIT"
)


text_paths = [
    partial_path,
    n2_path,
    diagnostic_audit_path,
    n2_audit_path,
    cova_state_path,
    project_state_path,
    validation_path,
    test_path,
    REPO
    / "docs/"
    "cova3d_numerical_amendment_N2_fp32_dice.md",
]


source_path = (
    REPO
    / "scripts/code_blocks/"
    "block09e_n2_lock_fp32_sparse_dice.py"
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
            PROTOCOL_AFTER,

        "numerical_amendments":
            [
                "N1",
                "N2",
            ],

        "R1_successful_updates_executed":
            live_successful_steps,

        "R1_updates_retained":
            0,

        "clean_R2_retained_optimizer_steps":
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
# 12. COMMIT + PUSH
# ==========================================================================================

git_paths = [
    "COVA3D_STATE.json",
    "PROJECT_STATE.json",
    "REPOSITORY_MANIFEST.json",

    "configs/"
    "cova3d_numerical_amendment_N2_fp32_dice.yaml",

    "data/manifests/"
    "cova3d_N2_exact_failed_state_validation.csv",

    "docs/"
    "cova3d_numerical_amendment_N2_fp32_dice.md",

    "experiments/audits/"
    "block09e_n1_failed_state_diagnostic.json",

    "experiments/audits/"
    "block09e_n2_lock_fp32_dice.json",

    "src/cora_lung/losses/"
    "partial.py",

    "tests/"
    "test_cova3d_numerical_amendment_n2.py",
]


if source_path.exists():

    git_paths.append(
        "scripts/code_blocks/"
        "block09e_n2_lock_fp32_sparse_dice.py"
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


staged_files = sh(
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
    for path in staged_files
):

    raise RuntimeError(
        "Runtime checkpoint/model must not enter normal Git."
    )


print(
    "✓ Runtime checkpoints staged            : NO"
)


sh(
    [
        "git",
        "commit",
        "-m",
        "protocol: freeze COVA-3D FP32 sparse-Dice amendment N2",
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
        "Repository is not clean after N2 commit:\n"
        + final_status
    )


# ==========================================================================================
# 13. FINAL REPORT
# ==========================================================================================

heading(
    "COVA-3D BLOCK 09E-N2-LOCK — FINAL REPORT"
)


print(
    "Starting commit                        :",
    head[:12],
)

print(
    "N2 commit                              :",
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
    "R1 FAILURE PROVENANCE"
)

print(
    "---------------------"
)

print(
    "Failure epoch                          : 5"
)

print(
    "Failure logical pair                   : 33"
)

print(
    "Successful updates executed            :",
    live_successful_steps,
)

print(
    "Epoch-boundary updates persisted       :",
    persisted_steps,
)

print(
    "Incomplete-epoch updates               :",
    discarded_incomplete_steps,
)

print(
    "R1 updates retained                    : 0"
)

print(
    "Dense outcomes opened                  : NO"
)

print()

print(
    "ROOT CAUSE"
)

print(
    "----------"
)

print(
    "AMP BCE-only gradients                 : FINITE"
)

print(
    "AMP FP16-Dice gradients                : NON-FINITE"
)

print(
    "AMP FP32-Dice gradients                : FINITE"
)

print(
    "Full FP32 combined gradients           : FINITE"
)

print(
    "Root event                             : FP16 DICE RECIPROCAL OVERFLOW"
)

print(
    "All-background denominator FP16        :",
    root_event[
        "dice_denominator_fp16"
    ],
)

print(
    "FP16 reciprocal finite                :",
    root_event[
        "reciprocal_fp16_finite"
    ],
)

print(
    "FP32 reciprocal                        :",
    root_event[
        "reciprocal_fp32"
    ],
)

print()

print(
    "NUMERICAL AMENDMENT N2"
)

print(
    "----------------------"
)

print(
    "Network forward                        : AMP FP16"
)

print(
    "Partial BCE                            : UNCHANGED"
)

print(
    "Sparse Dice arithmetic                 : FP32"
)

print(
    "Dice equation                          : UNCHANGED"
)

print(
    "Dice epsilon                           :",
    DICE_EPS,
)

print(
    "BCE/Dice weights                       : 1 / 1"
)

print(
    "N1 loss-scaling policy                 : ACTIVE"
)

print()

print(
    "EXACT FAILED-STATE VALIDATION"
)

print(
    "-----------------------------"
)

print(
    "Old FP16 Dice @ scale 1                : NON-FINITE"
)

print(
    "N2 FP32 Dice @ scale 1                 : FINITE"
)

print(
    "N2 FP32 Dice @ scale 4096              :",
    (
        "FINITE"
        if validation_lookup[
            "N2_FP32_DICE_SCALE4096"
        ][
            "finite"
        ]
        else "OVERFLOW — N1 BACKOFF"
    ),
)

print()

print(
    "SCIENTIFIC PROTOCOL"
)

print(
    "-------------------"
)

print(
    "Architecture changed                   : NO"
)

print(
    "Annotations changed                    : NO"
)

print(
    "Loss equation changed                  : NO"
)

print(
    "Loss weights changed                   : NO"
)

print(
    "Optimizer changed                      : NO"
)

print(
    "LR / scheduler changed                 : NO"
)

print(
    "Sampling changed                       : NO"
)

print(
    "Augmentation changed                   : NO"
)

print(
    "Epoch count changed                    : NO"
)

print(
    "Successful update target               : 1000"
)

print()

print(
    "AUTHORIZATION"
)

print(
    "-------------"
)

print(
    "Clean R2 sanity restart authorized     : YES"
)

print(
    "Original initialization required       : YES"
)

print(
    "R1 weights reusable                    : NO"
)

print(
    "Retained clean-run optimizer steps     : 0"
)

print(
    "Executed non-retained R1 steps         :",
    total_nonretained,
)

print(
    "Dense sanity evaluation authorized     : NO"
)

print(
    "Factorial training authorized          : NO"
)

print(
    "Final outer-CV access                  : 0"
)

print()

print(
    "Regression tests                       : PASS"
)

print(
    "Exact source captured                  :",
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
    "Do NOT rerun R1."
)

print(
    "Next block will be 09E-SANITY-FIT-R2."
)

print(
    "R2 will restart from the original frozen initialization with N1 + N2, "
    "run exactly 1,000 successful optimizer updates, and still keep all dense "
    "development masks sealed."
)

print(
    "=" * 132
)
