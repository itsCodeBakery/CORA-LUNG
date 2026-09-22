# ==========================================================================================
# COVA-3D — BLOCK 10C-FACTORIAL-BATCH-03-PRED-FREEZE
#
# BATCH
# -----
# fold0_seed29
#
# INPUT
# -----
# 6 completed final-epoch models
# 4 held-out image-only CT crops
#
# OUTPUT
# ------
# 24 frozen probability maps:
#
#       6 conditions × 4 held-out cases
#
# INFERENCE — ALREADY FROZEN BEFORE FINAL OUTCOMES
# ------------------------------------------------
# patch              : 48 × 128 × 128
# overlap            : 0.50
# weighting          : Gaussian
# sigma scale        : 0.125
# blend domain       : logits
# probability        : sigmoid after merged logits
# TTA                : NO
#
# FIREWALL
# --------
# NO infection masks
# NO lung masks
# NO final dense outcomes
# NO threshold search
# NO Dice / IoU / FROC
# NO optimizer.step()
# NO training
#
# Runtime probability NPZs are intentionally NOT committed to normal Git.
# Their hashes + provenance ARE committed.
#
# NEXT AFTER PASS
# ---------------
# Batch 02 = fold0_seed29
# ==========================================================================================


from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import gc
import hashlib
import importlib
import json
import os
import subprocess
import sys
import textwrap
import time

import numpy as np
import pandas as pd
import torch
import yaml

from tqdm.auto import tqdm
from kaggle_secrets import UserSecretsClient


# ==========================================================================================
# 0. CONSTANTS
# ==========================================================================================

REPO = Path(
    "/kaggle/working/CORA-LUNG"
)

EXPECTED_HEAD = (
    "852d17901310"
)

BLOCK = (
    "10C-FACTORIAL-BATCH-03-PRED-FREEZE"
)

NEXT_BLOCK = (
    "10B-FACTORIAL-BATCH-04"
)

CURRENT_BATCH = (
    "fold0_seed43"
)

NEXT_BATCH = (
    "fold1_seed17"
)

OUTER_FOLD = (
    0
)

SEED = (
    43
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

OVERLAP = (
    0.50
)

GAUSSIAN_SIGMA_SCALE = (
    0.125
)

FEATURES = (
    32,
    64,
    128,
    256,
    320,
)

EXPECTED_PARAMETER_COUNT = (
    19_166_529
)

EXPECTED_PREDICTIONS = (
    24
)

EFFECTIVE_PROTOCOL = (
    "COVA3D_1.0+A1+A1.1+A1.2+A1.3+N1+N2"
)

NOW_ISO = datetime.now(
    timezone.utc
).strftime(
    "%Y-%m-%dT%H:%M:%SZ"
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

FIT_SUMMARY_PATH = (
    REPO
    / "data/manifests/"
    "cova3d_factorial_batch03_fit_summary_v1_0.csv"
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

FINAL16_CACHE_MANIFEST_PATH = (
    REPO
    / "data/manifests/"
    "cova3d_final16_CT_cache_manifest_v1_0.csv"
)

CT_CACHE_ROOT = Path(
    "/kaggle/working/cova3d_final16_ct_cache_v1_0"
)

BATCH_ROOT = Path(
    "/kaggle/working/cova3d_factorial_v1_0/"
    "batch03_fold0_seed29"
)

PRED_ROOT = (
    BATCH_ROOT
    / "predictions"
)

PRED_MANIFEST_REPO = (
    REPO
    / "data/manifests/"
    "cova3d_factorial_batch03_prediction_freeze_v1_0.csv"
)

INFERENCE_LOCK_PATH = (
    REPO
    / "configs/"
    "cova3d_factorial_inference_implementation_lock_v1_0.yaml"
)

AUDIT_PATH = (
    REPO
    / "experiments/audits/"
    "block10c_factorial_batch03_prediction_freeze.json"
)

TEST_PATH = (
    REPO
    / "tests/"
    "test_cova3d_factorial_batch03_prediction_freeze.py"
)

SOURCE_PATH = (
    REPO
    / "scripts/code_blocks/"
    "block10c_factorial_batch03_prediction_freeze.py"
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


def git(
    *args,
    env=None,
):

    return sh(
        [
            "git",
            *args,
        ],
        env=env,
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
    payload,
):

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
    content,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        textwrap.dedent(
            content
        ).strip()
        + "\n",
        encoding="utf-8",
    )


def make_git_auth():

    token = UserSecretsClient().get_secret(
        "pushCora"
    )

    if not token:

        raise RuntimeError(
            "Kaggle secret pushCora unavailable."
        )

    token = token.strip()

    askpass = Path(
        "/tmp/cova3d_git_askpass_batch03_predictions.sh"
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


def stable_sigmoid(x):

    x = np.asarray(
        x,
        dtype=np.float32,
    )

    output = np.empty_like(
        x
    )

    positive = (
        x >= 0
    )

    output[
        positive
    ] = (
        1.0
        / (
            1.0
            + np.exp(
                -x[
                    positive
                ]
            )
        )
    )

    negative_exp = np.exp(
        x[
            ~positive
        ]
    )

    output[
        ~positive
    ] = (
        negative_exp
        / (
            1.0
            + negative_exp
        )
    )

    return output


# ==========================================================================================
# 3. FROZEN INFERENCE HELPERS
# ==========================================================================================

def axis_starts(
    length,
    patch,
):

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
                    - OVERLAP
                )
            )
        ),
    )

    values = list(
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

    if values[
        -1
    ] != last:

        values.append(
            last
        )

    return values


def gaussian_weight():

    vectors = []

    for size in PATCH_ZYX:

        coordinates = np.arange(
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
                size
            )
            * GAUSSIAN_SIGMA_SCALE,
        )

        vector = np.exp(
            -0.5
            * (
                (
                    coordinates
                    - center
                )
                / sigma
            )
            ** 2
        )

        vectors.append(
            vector
        )

    weight = (
        vectors[
            0
        ][
            :,
            None,
            None
        ]
        * vectors[
            1
        ][
            None,
            :,
            None
        ]
        * vectors[
            2
        ][
            None,
            None,
            :
        ]
    )

    return np.maximum(
        weight,
        1e-3,
    ).astype(
        np.float32
    )


@torch.no_grad()
def infer_volume(
    model,
    image,
    device,
    *,
    description,
):

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    original_shape = tuple(
        int(x)
        for x in image.shape
    )

    padded_shape = tuple(
        max(
            original_shape[
                axis
            ],
            PATCH_ZYX[
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

    image_pad = np.pad(
        image,
        pad_width,
        mode="constant",
        constant_values=-1.0,
    )

    z_starts = axis_starts(
        padded_shape[
            0
        ],
        PATCH_ZYX[
            0
        ],
    )

    y_starts = axis_starts(
        padded_shape[
            1
        ],
        PATCH_ZYX[
            1
        ],
    )

    x_starts = axis_starts(
        padded_shape[
            2
        ],
        PATCH_ZYX[
            2
        ],
    )

    weight = gaussian_weight()

    logit_sum = np.zeros(
        padded_shape,
        dtype=np.float32,
    )

    weight_sum = np.zeros(
        padded_shape,
        dtype=np.float32,
    )

    total_patches = (
        len(
            z_starts
        )
        * len(
            y_starts
        )
        * len(
            x_starts
        )
    )

    model.eval()

    progress = tqdm(
        total=total_patches,
        desc=description,
        leave=False,
    )

    for z in z_starts:

        for y in y_starts:

            for x in x_starts:

                patch = image_pad[
                    z:
                    z
                    + PATCH_ZYX[
                        0
                    ],

                    y:
                    y
                    + PATCH_ZYX[
                        1
                    ],

                    x:
                    x
                    + PATCH_ZYX[
                        2
                    ],
                ]

                tensor = torch.from_numpy(
                    patch[
                        None,
                        None,
                        ...
                    ].astype(
                        np.float32
                    )
                ).to(
                    device
                )

                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=True,
                ):

                    logits = model(
                        tensor
                    )

                logits_np = (
                    logits[
                        0,
                        0
                    ]
                    .float()
                    .cpu()
                    .numpy()
                )

                if not np.isfinite(
                    logits_np
                ).all():

                    raise RuntimeError(
                        "Non-finite inference logits."
                    )

                logit_sum[
                    z:
                    z
                    + PATCH_ZYX[
                        0
                    ],

                    y:
                    y
                    + PATCH_ZYX[
                        1
                    ],

                    x:
                    x
                    + PATCH_ZYX[
                        2
                    ],
                ] += (
                    logits_np
                    * weight
                )

                weight_sum[
                    z:
                    z
                    + PATCH_ZYX[
                        0
                    ],

                    y:
                    y
                    + PATCH_ZYX[
                        1
                    ],

                    x:
                    x
                    + PATCH_ZYX[
                        2
                    ],
                ] += weight

                del (
                    tensor,
                    logits,
                )

                progress.update(
                    1
                )

    progress.close()

    if np.any(
        weight_sum
        <= 0
    ):

        raise RuntimeError(
            "Sliding-window coverage failure."
        )

    merged_logits = (
        logit_sum
        / weight_sum
    )

    merged_logits = merged_logits[
        :original_shape[
            0
        ],
        :original_shape[
            1
        ],
        :original_shape[
            2
        ],
    ]

    probability = stable_sigmoid(
        merged_logits
    ).astype(
        np.float32
    )

    if not np.isfinite(
        probability
    ).all():

        raise RuntimeError(
            "Non-finite probability map."
        )

    return (
        probability,
        total_patches,
    )


# ==========================================================================================
# 4. PREFLIGHT
# ==========================================================================================

heading(
    "COVA-3D 10C — BATCH-03 IMAGE-ONLY PREDICTION FREEZE"
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
        "Repository must be clean before prediction freeze:\n"
        + dirty
    )


state = json.loads(
    STATE_PATH.read_text(
        encoding="utf-8"
    )
)


if state.get(
    "last_completed_block"
) != "10B-FACTORIAL-BATCH-03":

    raise RuntimeError(
        "Batch-03 fit is not the latest completed block."
    )


if state.get(
    "factorial_batch03_fit_status"
) != "COMPLETE":

    raise RuntimeError(
        "Batch-03 fit status is not COMPLETE."
    )


if state.get(
    "next_block"
) != BLOCK:

    raise RuntimeError(
        "10C is not the next authorized block."
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
        "Final outer-CV outcome access is not zero."
    )


if not torch.cuda.is_available():

    raise RuntimeError(
        "CUDA GPU is required."
    )


# Ensure restarted sessions also have a local Git identity.
if not git(
    "config",
    "--get",
    "user.name",
):

    git(
        "config",
        "user.name",
        "COVA-3D Kaggle Runner",
    )


if not git(
    "config",
    "--get",
    "user.email",
):

    git(
        "config",
        "user.email",
        "cova3d@local.invalid",
    )


print(
    "✓ Git HEAD                             :",
    head[:12],
)

print(
    "✓ Batch-03 fits                        : COMPLETE"
)

print(
    "✓ Final dense outcomes                 : SEALED"
)

print(
    "✓ Final outer-CV outcome access        : 0"
)

print(
    "✓ GPU                                  :",
    torch.cuda.get_device_name(
        0
    ),
)


# ==========================================================================================
# 5. VERIFY FROZEN INFERENCE IMPLEMENTATION
# ==========================================================================================

heading(
    "STEP 1/7 — VERIFY FROZEN INFERENCE CONTRACT"
)


sanity_lock_path = (
    REPO
    / "configs/"
    "cova3d_sanity_inference_implementation_lock_v1_0.yaml"
)


sanity_lock = yaml.safe_load(
    sanity_lock_path.read_text(
        encoding="utf-8"
    )
)


frozen_inf = sanity_lock[
    "inference"
]


if frozen_inf[
    "patch_zyx"
] != [
    48,
    128,
    128,
]:

    raise RuntimeError(
        "Frozen inference patch changed."
    )


if float(
    frozen_inf[
        "overlap"
    ]
) != OVERLAP:

    raise RuntimeError(
        "Frozen inference overlap changed."
    )


if float(
    frozen_inf[
        "gaussian_sigma_scale"
    ]
) != GAUSSIAN_SIGMA_SCALE:

    raise RuntimeError(
        "Frozen Gaussian sigma changed."
    )


if frozen_inf[
    "blend_domain"
] != "logits":

    raise RuntimeError(
        "Frozen blend domain changed."
    )


if frozen_inf[
    "TTA"
] is not False:

    raise RuntimeError(
        "Frozen inference unexpectedly enables TTA."
    )


factorial_inference_lock = {
    "project":
        "COVA-3D",

    "effective_protocol":
        EFFECTIVE_PROTOCOL,

    "status":
        "FROZEN_BEFORE_FIRST_FINAL_FACTORIAL_DENSE_OUTCOME",

    "source":
        (
            "Exact implementation contract inherited from the "
            "pre-outcome sanity inference lock."
        ),

    "inference":
        {
            "patch_zyx":
                list(
                    PATCH_ZYX
                ),

            "overlap":
                OVERLAP,

            "window_weighting":
                "gaussian",

            "gaussian_sigma_scale":
                GAUSSIAN_SIGMA_SCALE,

            "minimum_gaussian_weight":
                1e-3,

            "blend_domain":
                "logits",

            "probability":
                "stable_sigmoid_after_merged_logits",

            "TTA":
                False,

            "output_storage_dtype":
                "float32",
        },

    "dense_outcomes_opened_at_freeze":
        False,

    "final_outer_cv_outcome_access_at_freeze":
        0,

    "frozen_at_utc":
        NOW_ISO,
}


INFERENCE_LOCK_PATH.parent.mkdir(
    parents=True,
    exist_ok=True,
)


INFERENCE_LOCK_PATH.write_text(
    yaml.safe_dump(
        factorial_inference_lock,
        sort_keys=False,
    ),
    encoding="utf-8",
)


print(
    "✓ Patch                                : 48 × 128 × 128"
)

print(
    "✓ Overlap                              : 0.50"
)

print(
    "✓ Gaussian sigma scale                 : 0.125"
)

print(
    "✓ Blend domain                         : LOGITS"
)

print(
    "✓ TTA                                  : NO"
)

print(
    "✓ Probability storage                  : FLOAT32"
)


# ==========================================================================================
# 6. RESOLVE MODELS + FOUR HELD-OUT CASES
# ==========================================================================================

heading(
    "STEP 2/7 — VERIFY SIX MODELS AND FOUR HELD-OUT CTs"
)


fit_df = pd.read_csv(
    FIT_SUMMARY_PATH
)


if len(
    fit_df
) != 6:

    raise RuntimeError(
        "Expected six Batch-03 fit rows."
    )


if set(
    fit_df[
        "condition"
    ]
) != set(
    CONDITIONS
):

    raise RuntimeError(
        "Unexpected Batch-03 condition set."
    )


execution_df = pd.read_csv(
    EXECUTION_REGISTRY_PATH
)


batch_runs = execution_df[
    execution_df[
        "paired_batch_id"
    ]
    == CURRENT_BATCH
].copy()


if len(
    batch_runs
) != 6:

    raise RuntimeError(
        "Execution registry does not contain six Batch-03 runs."
    )


if batch_runs[
    "heldout_cases"
].nunique() != 1:

    raise RuntimeError(
        "Held-out case set differs across paired conditions."
    )


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
    heldout_cases
) != 4:

    raise RuntimeError(
        "Expected exactly four held-out cases."
    )


cache_manifest = pd.read_csv(
    FINAL16_CACHE_MANIFEST_PATH
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


heldout_images = {}


for case_id in heldout_cases:

    if case_id not in cache_lookup:

        raise RuntimeError(
            "Missing frozen CT manifest row for "
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
            "Runtime held-out CT cache missing:\n"
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


    expected = cache_lookup[
        case_id
    ]


    expected_shape = (
        int(
            expected[
                "shape_z"
            ]
        ),
        int(
            expected[
                "shape_y"
            ]
        ),
        int(
            expected[
                "shape_x"
            ]
        ),
    )


    if image.shape != expected_shape:

        raise RuntimeError(
            "Held-out CT shape mismatch for "
            + case_id
        )


    if image.dtype != np.float16:

        raise RuntimeError(
            "Held-out CT cache is not float16."
        )


    if not np.isfinite(
        image
    ).all():

        raise RuntimeError(
            "Non-finite held-out CT."
        )


    heldout_images[
        case_id
    ] = image


for _, row in fit_df.iterrows():

    model_path = Path(
        str(
            row[
                "final_model_path"
            ]
        )
    )


    if not model_path.exists():

        raise RuntimeError(
            "Batch-03 checkpoint missing:\n"
            + str(
                model_path
            )
        )


    if sha256_file(
        model_path
    ) != str(
        row[
            "final_model_sha256"
        ]
    ):

        raise RuntimeError(
            "Checkpoint SHA mismatch for "
            + str(
                row[
                    "condition"
                ]
            )
        )


print(
    "✓ Final checkpoints                    : 6 /6"
)

print(
    "✓ Model SHA verification               : 6 /6 PASS"
)

print(
    "✓ Held-out CTs                         : 4 /4"
)

print(
    "Held-out cases:"
)


for case_id in heldout_cases:

    print(
        "  ",
        case_id,
    )


print(
    "✓ Dense masks opened                   : 0"
)


# ==========================================================================================
# 7. LOAD MODEL CLASS
# ==========================================================================================

heading(
    "STEP 3/7 — LOAD FROZEN NETWORK"
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


importlib.invalidate_caches()


from cora_lung.models.cova3d_nnunet_style import (
    COVA3DNNUNetStyle,
)


device = torch.device(
    "cuda:0"
)


test_model = COVA3DNNUNetStyle(
    in_channels=1,
    out_channels=1,
    features=FEATURES,
)


parameter_count = sum(
    parameter.numel()
    for parameter in test_model.parameters()
)


if parameter_count != EXPECTED_PARAMETER_COUNT:

    raise RuntimeError(
        "Frozen parameter count changed."
    )


del test_model


print(
    "✓ Model class                          : COVA3DNNUNetStyle"
)

print(
    "✓ Parameter count                      :",
    parameter_count,
)


# ==========================================================================================
# 8. GENERATE 24 IMAGE-ONLY PREDICTIONS
# ==========================================================================================

heading(
    "STEP 4/7 — FREEZE 24 HELD-OUT IMAGE-ONLY PREDICTIONS"
)


PRED_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


prediction_rows = []


for model_index, condition in enumerate(
    CONDITIONS,
    start=1,
):

    model_row = fit_df[
        fit_df[
            "condition"
        ]
        == condition
    ]


    if len(
        model_row
    ) != 1:

        raise RuntimeError(
            "Could not resolve model row for "
            + condition
        )


    model_row = model_row.iloc[
        0
    ]


    model_path = Path(
        str(
            model_row[
                "final_model_path"
            ]
        )
    )


    model_sha = str(
        model_row[
            "final_model_sha256"
        ]
    )


    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )


    if checkpoint.get(
        "condition"
    ) != condition:

        raise RuntimeError(
            "Checkpoint condition mismatch."
        )


    if int(
        checkpoint.get(
            "outer_fold",
            -1,
        )
    ) != OUTER_FOLD:

        raise RuntimeError(
            "Checkpoint outer-fold mismatch."
        )


    if int(
        checkpoint.get(
            "seed",
            -1,
        )
    ) != SEED:

        raise RuntimeError(
            "Checkpoint seed mismatch."
        )


    if int(
        checkpoint.get(
            "successful_optimizer_steps",
            -1,
        )
    ) != 2400:

        raise RuntimeError(
            "Checkpoint optimizer-step count mismatch."
        )


    if checkpoint.get(
        "checkpoint_selection"
    ) != "FINAL_EPOCH_ONLY":

        raise RuntimeError(
            "Unexpected checkpoint-selection policy."
        )


    model = COVA3DNNUNetStyle(
        in_channels=1,
        out_channels=1,
        features=FEATURES,
    ).to(
        device
    )


    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True,
    )


    model.eval()


    del checkpoint


    condition_root = (
        PRED_ROOT
        / condition
    )


    condition_root.mkdir(
        parents=True,
        exist_ok=True,
    )


    print()
    print(
        "-" * 132
    )

    print(
        f"[{model_index}/6] {condition}"
    )

    print(
        "Model SHA                              :",
        model_sha,
    )

    print(
        "-" * 132
    )


    for case_index, case_id in enumerate(
        heldout_cases,
        start=1,
    ):

        output_path = (
            condition_root
            / (
                case_id
                + ".npz"
            )
        )


        # ------------------------------------------------------------------
        # Safe same-session resume:
        # reuse only if a previously frozen prediction has an accompanying
        # row that will be regenerated from the file itself below.
        # ------------------------------------------------------------------

        if output_path.exists():

            try:

                with np.load(
                    output_path,
                    allow_pickle=False,
                ) as existing:

                    if set(
                        existing.files
                    ) != {
                        "probability_zyx"
                    }:

                        raise RuntimeError(
                            "bad schema"
                        )


                    probability = np.asarray(
                        existing[
                            "probability_zyx"
                        ],
                        dtype=np.float32,
                    )


                expected_shape = heldout_images[
                    case_id
                ].shape


                if (
                    probability.shape
                    != expected_shape
                    or not np.isfinite(
                        probability
                    ).all()
                ):

                    raise RuntimeError(
                        "invalid existing prediction"
                    )


                reused = (
                    True
                )

                inference_seconds = (
                    np.nan
                )

                patch_count = (
                    -1
                )


            except Exception:

                output_path.unlink(
                    missing_ok=True
                )

                reused = (
                    False
                )


        else:

            reused = (
                False
            )


        if not reused:

            torch.cuda.synchronize()

            started = time.perf_counter()


            probability, patch_count = infer_volume(
                model,
                heldout_images[
                    case_id
                ],
                device,
                description=(
                    condition
                    + " / "
                    + case_id
                ),
            )


            torch.cuda.synchronize()


            inference_seconds = (
                time.perf_counter()
                - started
            )


            np.savez_compressed(
                output_path,
                probability_zyx=
                    probability.astype(
                        np.float32
                    ),
            )


        # Re-open exact persisted prediction.
        with np.load(
            output_path,
            allow_pickle=False,
        ) as persisted:

            probability = np.asarray(
                persisted[
                    "probability_zyx"
                ],
                dtype=np.float32,
            )


        if probability.shape != heldout_images[
            case_id
        ].shape:

            raise RuntimeError(
                "Persisted prediction shape mismatch."
            )


        if not np.isfinite(
            probability
        ).all():

            raise RuntimeError(
                "Persisted prediction contains non-finite values."
            )


        probability64 = probability.astype(
            np.float64
        )


        prediction_rows.append(
            {
                "batch_id":
                    CURRENT_BATCH,

                "run_id":
                    str(
                        model_row[
                            "run_id"
                        ]
                    ),

                "condition":
                    condition,

                "outer_fold":
                    OUTER_FOLD,

                "seed":
                    SEED,

                "case_id":
                    case_id,

                "role":
                    "heldout_final_outer_cv",

                "model_sha256":
                    model_sha,

                "prediction_file":
                    str(
                        output_path
                    ),

                "prediction_sha256":
                    sha256_file(
                        output_path
                    ),

                "shape_z":
                    int(
                        probability.shape[
                            0
                        ]
                    ),

                "shape_y":
                    int(
                        probability.shape[
                            1
                        ]
                    ),

                "shape_x":
                    int(
                        probability.shape[
                            2
                        ]
                    ),

                "storage_dtype":
                    "float32",

                "probability_min":
                    float(
                        probability64.min()
                    ),

                "probability_max":
                    float(
                        probability64.max()
                    ),

                "probability_mean":
                    float(
                        probability64.mean()
                    ),

                "probability_std":
                    float(
                        probability64.std()
                    ),

                "patch_count":
                    int(
                        patch_count
                    ),

                "inference_seconds":
                    float(
                        inference_seconds
                    ),

                "runtime_prediction_reused":
                    bool(
                        reused
                    ),

                "dense_mask_accessed":
                    False,

                "final_outcome_access":
                    False,
            }
        )


        print(
            f"  [{case_index}/4] {case_id:28s} "
            f"mean={probability64.mean():.6f}  "
            f"std={probability64.std():.6f}  "
            + (
                "REUSED"
                if reused
                else f"{inference_seconds:.2f}s"
            )
        )


    del model

    gc.collect()

    torch.cuda.empty_cache()


prediction_df = pd.DataFrame(
    prediction_rows
)


if len(
    prediction_df
) != EXPECTED_PREDICTIONS:

    raise RuntimeError(
        "Expected exactly 24 frozen prediction rows."
    )


if prediction_df[
    [
        "condition",
        "case_id",
    ]
].duplicated().any():

    raise RuntimeError(
        "Duplicate condition/case prediction pair."
    )


if prediction_df[
    "prediction_sha256"
].isna().any():

    raise RuntimeError(
        "Missing prediction SHA."
    )


if not (
    prediction_df[
        "dense_mask_accessed"
    ]
    == False
).all():

    raise RuntimeError(
        "Dense-mask firewall violation."
    )


PRED_MANIFEST_REPO.parent.mkdir(
    parents=True,
    exist_ok=True,
)


prediction_df.to_csv(
    PRED_MANIFEST_REPO,
    index=False,
)


print()
print(
    "✓ Frozen probability maps              : 24 /24"
)

print(
    "✓ Conditions                           : 6"
)

print(
    "✓ Held-out cases / model               : 4"
)

print(
    "✓ Prediction SHA records               : 24 /24"
)

print(
    "✓ Dense masks opened                   : 0"
)


# ==========================================================================================
# 9. VERIFY FROZEN PREDICTIONS FROM DISK
# ==========================================================================================

heading(
    "STEP 5/7 — VERIFY 24 PERSISTED PREDICTION ARTIFACTS"
)


verified = (
    0
)


for _, row in tqdm(
    prediction_df.iterrows(),
    total=len(
        prediction_df
    ),
    desc="Prediction SHA",
):

    path = Path(
        str(
            row[
                "prediction_file"
            ]
        )
    )


    if not path.exists():

        raise RuntimeError(
            "Frozen prediction file missing:\n"
            + str(
                path
            )
        )


    if sha256_file(
        path
    ) != str(
        row[
            "prediction_sha256"
        ]
    ):

        raise RuntimeError(
            "Frozen prediction SHA mismatch."
        )


    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        if set(
            data.files
        ) != {
            "probability_zyx"
        }:

            raise RuntimeError(
                "Unexpected prediction schema."
            )


        array = np.asarray(
            data[
                "probability_zyx"
            ]
        )


    if array.dtype != np.float32:

        raise RuntimeError(
            "Frozen prediction is not float32."
        )


    if not np.isfinite(
        array
    ).all():

        raise RuntimeError(
            "Frozen prediction contains non-finite values."
        )


    if (
        float(
            array.min()
        )
        < 0.0
        or float(
            array.max()
        )
        > 1.0
    ):

        raise RuntimeError(
            "Frozen probability outside [0,1]."
        )


    verified += (
        1
    )


print(
    "✓ Prediction artifacts verified        :",
    str(
        verified
    )
    + "/24",
)

print(
    "✓ Storage dtype                        : FLOAT32"
)

print(
    "✓ Finite probability arrays            : 24 /24"
)

print(
    "✓ Probability range [0,1]              : 24 /24"
)


# ==========================================================================================
# 10. UPDATE EXECUTION / BATCH REGISTRIES
# ==========================================================================================

heading(
    "STEP 6/7 — ADVANCE FACTORIAL EXECUTION STATE"
)


batch_mask = (
    execution_df[
        "paired_batch_id"
    ]
    == CURRENT_BATCH
)


if int(
    batch_mask.sum()
) != 6:

    raise RuntimeError(
        "Could not identify six Batch-03 execution rows."
    )


execution_df.loc[
    batch_mask,
    "execution_status",
] = (
    "PREDICTIONS_FROZEN"
)


execution_df.to_csv(
    EXECUTION_REGISTRY_PATH,
    index=False,
)


batch_registry = pd.read_csv(
    BATCH_REGISTRY_PATH
)


batch_mask_registry = (
    batch_registry[
        "batch_id"
    ]
    == CURRENT_BATCH
)


if int(
    batch_mask_registry.sum()
) != 1:

    raise RuntimeError(
        "Could not identify Batch-03 registry row."
    )


batch_registry.loc[
    batch_mask_registry,
    "execution_status",
] = (
    "PREDICTIONS_FROZEN"
)


batch_registry.to_csv(
    BATCH_REGISTRY_PATH,
    index=False,
)


audit = {
    "project":
        "COVA-3D",

    "block":
        BLOCK,

    "status":
        "PREDICTIONS_FROZEN",

    "effective_protocol":
        EFFECTIVE_PROTOCOL,

    "batch_id":
        CURRENT_BATCH,

    "outer_fold":
        OUTER_FOLD,

    "seed":
        SEED,

    "models":
        6,

    "heldout_cases_per_model":
        4,

    "prediction_maps":
        24,

    "inference":
        factorial_inference_lock[
            "inference"
        ],

    "prediction_SHA_verified":
        24,

    "firewall":
        {
            "infection_masks_accessed":
                0,

            "lung_masks_accessed":
                0,

            "final_dense_outcomes_accessed":
                0,

            "optimizer_steps":
                0,

            "training":
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

    "factorial_progress":
        {
            "fits_completed":
                18,

            "fits_total":
                72,

            "prediction_frozen_runs":
                18,

            "prediction_maps_frozen":
                72,

            "training_batches_completed":
                3,

            "training_batches_total":
                12,
        },

    "next_batch":
        NEXT_BATCH,

    "next_block":
        NEXT_BLOCK,

    "generated_at_utc":
        NOW_ISO,
}


write_json(
    AUDIT_PATH,
    audit,
)


state[
    "last_completed_block"
] = (
    BLOCK
)

state[
    "current_stage"
] = (
    "FACTORIAL_BATCH03_PREDICTIONS_FROZEN_BATCH04_PENDING"
)

state[
    "factorial_runs_completed"
] = (
    18
)

state[
    "factorial_batches_completed"
] = (
    3
)

state[
    "factorial_prediction_runs_frozen"
] = (
    18
)

state[
    "factorial_prediction_maps_frozen"
] = (
    72
)

state[
    "factorial_batch03_prediction_status"
] = (
    "FROZEN"
)

state[
    "factorial_batch03_prediction_manifest"
] = str(
    PRED_MANIFEST_REPO.relative_to(
        REPO
    )
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
    "next_factorial_batch"
] = (
    NEXT_BATCH
)

state[
    "next_block"
] = (
    NEXT_BLOCK
)

state[
    "next_action"
] = (
    "Train the six frozen factorial conditions for fold1_seed17. "
    "Do not open final dense outcomes."
)

state[
    "updated_at_utc"
] = (
    NOW_ISO
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
] = (
    BLOCK
)

project_state[
    "current_stage"
] = (
    "cova3d_factorial_batch04_pending"
)

project_state[
    "cova3d_factorial_runs_completed"
] = (
    18
)

project_state[
    "cova3d_factorial_prediction_runs_frozen"
] = (
    18
)

project_state[
    "cova3d_factorial_prediction_maps_frozen"
] = (
    72
)

project_state[
    "cova3d_final_outer_cv_access"
] = (
    0
)

project_state[
    "next_action"
] = (
    "Run 10B-FACTORIAL-BATCH-04."
)

project_state[
    "updated_at_utc"
] = (
    NOW_ISO
)


write_json(
    PROJECT_STATE_PATH,
    project_state,
)


print(
    "✓ Batch-03 execution status            : PREDICTIONS_FROZEN"
)

print(
    "✓ Prediction-frozen runs               : 18 /72"
)

print(
    "✓ Frozen probability maps              : 72"
)

print(
    "✓ Next batch                           :",
    NEXT_BATCH,
)

print(
    "✓ Final dense outcomes                 : SEALED"
)


# ==========================================================================================
# 11. REGRESSION TEST
# ==========================================================================================

heading(
    "STEP 7/7 — REGRESSION / COMMIT / PUSH"
)


write_text(
    TEST_PATH,
    r'''
from pathlib import Path
import json

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_batch03_prediction_manifest():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_factorial_batch03_prediction_freeze_v1_0.csv"
    )

    assert len(frame) == 24

    assert frame["condition"].nunique() == 6
    assert frame["case_id"].nunique() == 4

    assert (
        frame.groupby("condition")["case_id"].count()
        == 4
    ).all()

    assert (frame["outer_fold"] == 0).all()
    assert (frame["seed"] == 43).all()

    assert (frame["storage_dtype"] == "float32").all()
    assert (~frame["dense_mask_accessed"]).all()
    assert (~frame["final_outcome_access"]).all()

    assert frame["prediction_sha256"].notna().all()


def test_factorial_inference_lock():

    cfg = yaml.safe_load(
        (
            ROOT
            / "configs/"
            "cova3d_factorial_inference_implementation_lock_v1_0.yaml"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        cfg["status"]
        == "FROZEN_BEFORE_FIRST_FINAL_FACTORIAL_DENSE_OUTCOME"
    )

    assert cfg["inference"]["patch_zyx"] == [48, 128, 128]
    assert cfg["inference"]["overlap"] == 0.5
    assert cfg["inference"]["gaussian_sigma_scale"] == 0.125
    assert cfg["inference"]["blend_domain"] == "logits"
    assert cfg["inference"]["TTA"] is False
    assert cfg["inference"]["output_storage_dtype"] == "float32"


def test_batch03_prediction_audit():

    audit = json.loads(
        (
            ROOT
            / "experiments/audits/"
            "block10c_factorial_batch03_prediction_freeze.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert audit["status"] == "PREDICTIONS_FROZEN"
    assert audit["batch_id"] == "fold0_seed43"
    assert audit["models"] == 6
    assert audit["prediction_maps"] == 24
    assert audit["prediction_SHA_verified"] == 24

    assert audit["firewall"]["infection_masks_accessed"] == 0
    assert audit["firewall"]["lung_masks_accessed"] == 0
    assert audit["firewall"]["final_dense_outcomes_accessed"] == 0
    assert audit["firewall"]["optimizer_steps"] == 0
    assert audit["firewall"]["training"] is False


def test_batch03_prediction_state():

    state = json.loads(
        (
            ROOT
            / "COVA3D_STATE.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        state["last_completed_block"]
        == "10C-FACTORIAL-BATCH-03-PRED-FREEZE"
    )

    assert state["factorial_runs_completed"] == 18
    assert state["factorial_prediction_runs_frozen"] == 18
    assert state["factorial_prediction_maps_frozen"] == 72

    assert state["final_outer_cv_outcomes_authorized"] is False
    assert state["final_outer_cv_access_in_cova3d"] == 0

    assert state["next_factorial_batch"] == "fold1_seed17"
    assert state["next_block"] == "10B-FACTORIAL-BATCH-04"


def test_registry_advanced():

    execution = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_final_factorial_execution_registry_v1_0.csv"
    )

    rows = execution[
        execution["paired_batch_id"]
        == "fold0_seed43"
    ]

    assert len(rows) == 6

    assert (
        rows["execution_status"]
        == "PREDICTIONS_FROZEN"
    ).all()


    batch = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_final_factorial_batch_registry_v1_0.csv"
    )

    row = batch[
        batch["batch_id"]
        == "fold0_seed43"
    ]

    assert len(row) == 1

    assert (
        row.iloc[0]["execution_status"]
        == "PREDICTIONS_FROZEN"
    )
'''
)


pytest_env = os.environ.copy()

pytest_env[
    "PYTHONPATH"
] = str(
    REPO
    / "src"
)


tests = sh(
    [
        sys.executable,
        "-m",
        "pytest",

        "tests/"
        "test_cova3d_factorial_batch03_prediction_freeze.py",

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
        "Batch-03 prediction-freeze regression tests failed."
    )


# ==========================================================================================
# 12. SOURCE CAPTURE
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
    AUDIT_PATH,
    audit,
)


# ==========================================================================================
# 13. NORMALIZE TEXT BEFORE REPOSITORY MANIFEST
# ==========================================================================================

text_paths = [
    PRED_MANIFEST_REPO,
    INFERENCE_LOCK_PATH,
    AUDIT_PATH,
    TEST_PATH,
    EXECUTION_REGISTRY_PATH,
    BATCH_REGISTRY_PATH,
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
# 14. REPOSITORY MANIFEST — LAST TEXT WRITE
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
            EFFECTIVE_PROTOCOL,

        "factorial_runs_completed":
            18,

        "factorial_prediction_runs_frozen":
            18,

        "factorial_prediction_maps_frozen":
            72,

        "factorial_batches_completed":
            3,

        "final_outer_cv_outcome_access":
            0,

        "next_batch":
            NEXT_BATCH,

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
# 15. COMMIT / PUSH
# ==========================================================================================

git_paths = [
    "COVA3D_STATE.json",
    "PROJECT_STATE.json",
    "REPOSITORY_MANIFEST.json",

    "configs/"
    "cova3d_factorial_inference_implementation_lock_v1_0.yaml",

    "data/manifests/"
    "cova3d_factorial_batch03_prediction_freeze_v1_0.csv",

    "data/manifests/"
    "cova3d_final_factorial_execution_registry_v1_0.csv",

    "data/manifests/"
    "cova3d_final_factorial_batch_registry_v1_0.csv",

    "experiments/audits/"
    "block10c_factorial_batch03_prediction_freeze.json",

    "tests/"
    "test_cova3d_factorial_batch03_prediction_freeze.py",
]


if SOURCE_PATH.exists():

    git_paths.append(
        "scripts/code_blocks/"
        "block10c_factorial_batch03_prediction_freeze.py"
    )


git(
    "add",
    *git_paths,
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


staged = git(
    "diff",
    "--cached",
    "--name-only",
).splitlines()


if any(
    path.endswith(
        ".npz"
    )
    or path.endswith(
        ".pt"
    )
    for path in staged
):

    raise RuntimeError(
        "Runtime prediction/checkpoint was accidentally staged."
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
    "experiment: freeze COVA-3D factorial batch 03 predictions",
)


final_commit = git(
    "rev-parse",
    "HEAD",
)


token, askpass, git_env = make_git_auth()


try:

    push_result = sh(
        [
            "git",
            "push",
            "origin",
            "main",
        ],
        env=git_env,
        check=False,
    )


    if push_result.returncode != 0:

        raise RuntimeError(
            "GitHub push failed:\n"
            + (
                push_result.stderr
                or ""
            ).replace(
                token,
                "***TOKEN_REDACTED***",
            )
        )


finally:

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
        "Repository not clean after prediction freeze:\n"
        + final_status
    )


# ==========================================================================================
# FINAL REPORT
# ==========================================================================================

heading(
    "COVA-3D 10C — BATCH-03 PREDICTION FREEZE FINAL REPORT"
)


print(
    "Starting commit                        :",
    head[:12],
)

print(
    "Prediction-freeze commit               :",
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
    "BATCH-03 IMAGE-ONLY PREDICTIONS"
)

print(
    "-------------------------------"
)

print(
    "Models                                 : 6"
)

print(
    "Held-out cases / model                 : 4"
)

print(
    "Frozen probability maps                : 24 /24"
)

print(
    "Prediction SHA verification            : 24 /24 PASS"
)

print(
    "Storage dtype                          : float32"
)

print(
    "Inference                              : Gaussian logit blending"
)

print(
    "TTA                                    : NO"
)

print()

print(
    "FIREWALL"
)

print(
    "--------"
)

print(
    "Infection masks opened                 : 0"
)

print(
    "Lung masks opened                      : 0"
)

print(
    "Final dense outcomes opened            : 0"
)

print(
    "optimizer.step() calls                 : 0"
)

print(
    "Training performed                     : NO"
)

print(
    "Dice / IoU / FROC                      : NOT COMPUTED"
)

print(
    "Threshold search                       : NO"
)

print()

print(
    "PRIMARY FACTORIAL PROGRESS"
)

print(
    "--------------------------"
)

print(
    "Fits completed                         : 18 /72"
)

print(
    "Prediction-frozen runs                 : 18 /72"
)

print(
    "Frozen probability maps                : 72"
)

print(
    "Training batches completed             : 3 /12"
)

print()

print(
    "NEXT"
)

print(
    "----"
)

print(
    "Next batch                             :",
    NEXT_BATCH,
)

print(
    "Next block                             :",
    NEXT_BLOCK,
)

print(
    "Final outer-CV outcome access          : 0"
)

print()

print(
    "Exact source captured                  :",
    source_capture,
)

print()
print(
    "STATUS                                 : BATCH03_PREDICTIONS_SAFELY_FROZEN"
)

print()

print(
    "IMPORTANT:"
)

print(
    "Keep the current Kaggle session running until you send me this report."
)

print(
    "Do NOT open any infection mask."
)

print(
    "After I audit this result, Batch 03 will be backed up. Batch 04 is fold1_seed17."
)

print(
    "=" * 132
)
