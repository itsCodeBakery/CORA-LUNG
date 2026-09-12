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
import textwrap
import time
import gc

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm.auto import tqdm
from kaggle_secrets import UserSecretsClient

import torch


# ==========================================================================================
# CORA-LUNG — BLOCK 07C-R1
# Dtype precision repair + resume interrupted Gate-B pilot harness
#
# NO SCIENTIFIC TRAINING
# NO optimizer.step()
# GATE B REMAINS NOT RUN
# ==========================================================================================

PROJECT = "CORA-Lung"

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
    "/kaggle/working/cora_runs"
)

PATCH_ZYX = (
    48,
    128,
    128,
)

BASE_CHANNELS = 16

EMBEDDING_DIM = 64

GLOBAL_SEED = 17

FUTURE_LR = 3e-4

FUTURE_WEIGHT_DECAY = 1e-4

GPU_MEMORY_LIMIT_GB = 14.5

NOW_ISO = datetime.now(
    timezone.utc
).strftime(
    "%Y-%m-%dT%H:%M:%SZ"
)


# ==========================================================================================
# HELPERS
# ==========================================================================================

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
                    8 * 1024 * 1024
                ),
            b"",
        ):

            h.update(
                chunk
            )

    return h.hexdigest()


def stable_seed(
    *parts,
):

    digest = hashlib.sha256(
        "|".join(
            map(
                str,
                parts,
            )
        ).encode(
            "utf-8"
        )
    ).digest()

    return int.from_bytes(
        digest[:8],
        "little",
    ) % (
        2**32
        - 1
    )


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


def cuda_gb(
    value,
):

    return float(
        value
    ) / (
        1024 ** 3
    )


random.seed(
    GLOBAL_SEED
)

np.random.seed(
    GLOBAL_SEED
)

torch.manual_seed(
    GLOBAL_SEED
)

if torch.cuda.is_available():

    torch.cuda.manual_seed_all(
        GLOBAL_SEED
    )

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


print(
    "=" * 118
)

print(
    "CORA-LUNG BLOCK 07C-R1 — DTYPE REPAIR + HARNESS RESUME"
)

print(
    "=" * 118
)


# ==========================================================================================
# 1. VERIFY INTERRUPTED BLOCK-07 STATE
# ==========================================================================================

state = json.loads(
    (
        REPO
        / "PROJECT_STATE.json"
    ).read_text(
        encoding="utf-8"
    )
)


if (
    state.get(
        "last_completed_block"
    )
    != "06"
):

    raise RuntimeError(
        "Expected accepted Block 06."
    )


if (
    state.get(
        "training_grid_cache"
    )
    != "PASS"
):

    raise RuntimeError(
        "Training cache is not PASS."
    )


if not CACHE_MANIFEST.exists():

    raise RuntimeError(
        "v1.1 cache missing."
    )


starting_commit = sh(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    cwd=REPO,
).stdout.strip()


required_files = [
    "configs/experiment_gate_b_pilot.yaml",
    "src/cora_lung/data/pilot_dataset.py",
    "src/cora_lung/losses/partial.py",
    "src/cora_lung/engine/checkpointing.py",
    "src/cora_lung/engine/pilot_contract.py",
    "src/cora_lung/engine/registry.py",
    "src/cora_lung/models/resunet3d.py",
    "tests/test_checkpoint_resume.py",
    "tests/test_partial_losses.py",
    "tests/test_patch_sampler.py",
    "tests/test_pilot_training_contract.py",
    "tests/test_resunet3d.py",
]


missing = [
    path
    for path in required_files
    if not (
        REPO
        / path
    ).exists()
]


if missing:

    raise RuntimeError(
        "Interrupted Block-07 files missing:\n"
        + "\n".join(
            missing
        )
    )


print(
    "✓ Interrupted Block-07 files are intact."
)

print(
    "✓ Starting commit:",
    starting_commit[:12],
)


# ==========================================================================================
# 2. PATCH ONLY THE BCE DTYPE BUG
# ==========================================================================================

partial_path = (
    REPO
    / "src/cora_lung/losses/partial.py"
)


partial_text = partial_path.read_text(
    encoding="utf-8"
)


old_expression = (
    "target[mask].float()"
)

new_expression = (
    "target[mask].to(dtype=logits.dtype)"
)


if partial_text.count(
    old_expression
) == 1:

    partial_text = partial_text.replace(
        old_expression,
        new_expression,
        1,
    )

    partial_path.write_text(
        partial_text,
        encoding="utf-8",
    )

    print(
        "✓ Repaired BCE target dtype."
    )


elif (
    partial_text.count(
        old_expression
    )
    == 0
    and partial_text.count(
        new_expression
    )
    == 1
):

    print(
        "✓ BCE dtype repair already present."
    )


else:

    raise RuntimeError(
        "Could not safely identify the BCE target cast."
    )


# ==========================================================================================
# 3. ADD FLOAT64 REGRESSION TEST
# ==========================================================================================

test_path = (
    REPO
    / "tests/test_partial_losses.py"
)


test_text = test_path.read_text(
    encoding="utf-8"
)


if (
    "test_partial_bce_preserves_float64_precision"
    not in test_text
):

    test_text += """

def test_partial_bce_preserves_float64_precision():
    logits = torch.tensor(
        [[[[[0.13, -0.71, 1.19, -0.44]]]]],
        dtype=torch.float64,
        requires_grad=True,
    )

    target = torch.tensor(
        [[[[[1, 0, 1, 0]]]]],
        dtype=torch.int8,
    )

    loss = partial_bce(
        logits,
        target,
    )

    assert loss.dtype == torch.float64


def test_query_decomposition_float64_precision_regression():
    logits = torch.tensor(
        [[[[[0.13, -0.71, 1.19, -0.44]]]]],
        dtype=torch.float64,
        requires_grad=True,
    )

    target = torch.tensor(
        [[[[[1, 0, 1, 0]]]]],
        dtype=torch.int8,
    )

    query_mask = torch.zeros_like(
        target,
        dtype=torch.bool,
    )

    query_mask[
        0,
        0,
        0,
        0,
        0,
    ] = True

    split = query_holdout_bce_exact(
        logits,
        target,
        query_mask,
        query_lambda=1.0,
    )

    assert split["ordinary_bce"].dtype == torch.float64

    assert split["decomposed_bce"].dtype == torch.float64

    assert torch.allclose(
        split["ordinary_bce"],
        split["decomposed_bce"],
        atol=1e-12,
        rtol=1e-12,
    )
"""

    test_path.write_text(
        test_text,
        encoding="utf-8",
    )

    print(
        "✓ Added float64 regression tests."
    )


# ==========================================================================================
# 4. RUN FULL HARNESS TEST SUITE
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


pytest_targets = [
    "tests/test_firewall.py",
    "tests/test_scribble_budget.py",
    "tests/test_physical_geometry.py",
    "tests/test_training_cache_firewall.py",
    "tests/test_training_grid_equalization.py",
    "tests/test_partial_losses.py",
    "tests/test_patch_sampler.py",
    "tests/test_resunet3d.py",
    "tests/test_checkpoint_resume.py",
    "tests/test_pilot_training_contract.py",
]


pytest_result = sh(
    [
        sys.executable,
        "-m",
        "pytest",
        *pytest_targets,
        "-q",
    ],
    cwd=REPO,
    env=pytest_env,
    check=False,
)


print(
    pytest_result.stdout
)


if pytest_result.stderr.strip():

    print(
        pytest_result.stderr
    )


if pytest_result.returncode != 0:

    raise RuntimeError(
        "Harness tests failed."
    )


print(
    "✓ Full scientific/unit test suite: PASS"
)


# ==========================================================================================
# 5. FRESH IMPORTS
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
    "cora_lung.data.pilot_dataset",
    "cora_lung.losses.partial",
    "cora_lung.models.resunet3d",
    "cora_lung.engine.checkpointing",
    "cora_lung.engine.registry",
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
    query_group_mask,
    query_holdout_bce_exact,
)

from cora_lung.models.resunet3d import (
    CORALungResidualUNet,
)

from cora_lung.engine.checkpointing import (
    save_training_checkpoint,
    load_training_checkpoint,
)

from cora_lung.engine.registry import (
    ExperimentRegistry,
)


# ==========================================================================================
# 6. LOAD FROZEN DEVELOPMENT CASES
# ==========================================================================================

registry_df = pd.read_csv(
    REPO
    / "data/splits/primary_volume_split_registry.csv"
)


development_cases = sorted(
    registry_df.loc[
        registry_df[
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
        "Expected 4 permanent development volumes."
    )


development_cache = SparseTrainingCache(
    CACHE_ROOT,
    development_cases,
)


# ==========================================================================================
# 7. REPEAT REAL PATCH AUDIT
# ==========================================================================================

conditions = [
    "complete",
    "component_natural_50",
    "pixel_dropout_matched_50",
    "component_fixed_50",
    "complete_fixed_50",
]


patch_rows = []

counter = 0


for case_id in development_cases:

    for condition in conditions:

        sample = development_cache.get(
            case_id,
            condition,
        )

        for local_index in range(
            6
        ):

            patch = extract_patch(
                sample,
                PATCH_ZYX,
                seed=GLOBAL_SEED,
                sample_index=(
                    counter
                    * 100
                    + local_index
                ),
            )

            target_np = (
                patch[
                    "target"
                ].numpy()
            )

            patch_rows.append(
                {
                    "case_id":
                        case_id,

                    "condition":
                        condition,

                    "sample_index":
                        local_index,

                    "sampling_source":
                        patch[
                            "sampling_source"
                        ],

                    "labelled_voxels":
                        int(
                            (
                                target_np
                                != -1
                            ).sum()
                        ),

                    "foreground_voxels":
                        int(
                            (
                                target_np
                                == 1
                            ).sum()
                        ),

                    "background_voxels":
                        int(
                            (
                                target_np
                                == 0
                            ).sum()
                        ),

                    "foreground_groups":
                        int(
                            torch.unique(
                                patch[
                                    "membership_group_id"
                                ]
                            ).numel()
                            if patch[
                                "membership_group_id"
                            ].numel()
                            else 0
                        ),
                }
            )

        counter += 1


patch_df = pd.DataFrame(
    patch_rows
)


patches_with_labels = int(
    (
        patch_df[
            "labelled_voxels"
        ]
        > 0
    ).sum()
)


patches_with_fg = int(
    (
        patch_df[
            "foreground_voxels"
        ]
        > 0
    ).sum()
)


if patches_with_labels == 0:

    raise RuntimeError(
        "No sparse labels found in patch audit."
    )


# ==========================================================================================
# 8. REAL QUERY-HOLDOUT VALUE + GRADIENT VERIFICATION
# ==========================================================================================

query_rows = []


for case_id in development_cases:

    sample = development_cache.get(
        case_id,
        "complete",
    )


    chosen_patch = None


    for attempt in range(
        50
    ):

        patch = extract_patch(
            sample,
            PATCH_ZYX,
            seed=GLOBAL_SEED,
            sample_index=stable_seed(
                case_id,
                "query_test",
                attempt,
            ),
        )


        if (
            patch[
                "membership_group_id"
            ].numel()
            > 0
        ):

            chosen_patch = patch

            break


    if chosen_patch is None:

        raise RuntimeError(
            f"No query patch found for {case_id}."
        )


    query_group_id = int(
        torch.unique(
            chosen_patch[
                "membership_group_id"
            ]
        )[
            0
        ].item()
    )


    target = (
        chosen_patch[
            "target"
        ]
        .unsqueeze(
            0
        )
        .to(
            dtype=torch.int8
        )
    )


    query_mask = query_group_mask(
        target,
        chosen_patch[
            "membership_voxel_zyx"
        ],
        chosen_patch[
            "membership_group_id"
        ],
        query_group_id,
    )


    logits_a = torch.randn(
        (
            1,
            1,
            *PATCH_ZYX,
        ),
        dtype=torch.float64,
        requires_grad=True,
    )


    ordinary = partial_bce(
        logits_a,
        target,
    )


    grad_a = torch.autograd.grad(
        ordinary,
        logits_a,
    )[
        0
    ]


    logits_b = (
        logits_a.detach()
        .clone()
        .requires_grad_()
    )


    split = query_holdout_bce_exact(
        logits_b,
        target,
        query_mask,
        query_lambda=1.0,
    )


    grad_b = torch.autograd.grad(
        split[
            "decomposed_bce"
        ],
        logits_b,
    )[
        0
    ]


    query_rows.append(
        {
            "case_id":
                case_id,

            "query_group_id":
                query_group_id,

            "query_voxels":
                int(
                    query_mask.sum()
                ),

            "ordinary_bce_dtype":
                str(
                    ordinary.dtype
                ),

            "value_abs_error":
                float(
                    torch.abs(
                        ordinary.detach()
                        - split[
                            "decomposed_bce"
                        ].detach()
                    )
                ),

            "max_gradient_abs_error":
                float(
                    torch.max(
                        torch.abs(
                            grad_a
                            - grad_b
                        )
                    )
                ),
        }
    )


query_df = pd.DataFrame(
    query_rows
)


print(
    query_df.to_string(
        index=False
    )
)


query_value_pass = bool(
    (
        query_df[
            "value_abs_error"
        ]
        <= 1e-12
    ).all()
)


query_gradient_pass = bool(
    (
        query_df[
            "max_gradient_abs_error"
        ]
        <= 1e-12
    ).all()
)


if not query_value_pass:

    raise RuntimeError(
        "Corrected BCE value equivalence failed."
    )


if not query_gradient_pass:

    raise RuntimeError(
        "Corrected BCE gradient equivalence failed."
    )


# ==========================================================================================
# 9. GPU FORWARD/BACKWARD DRY RUN
# ==========================================================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


model = CORALungResidualUNet(
    in_channels=1,
    base_channels=BASE_CHANNELS,
    embedding_dim=EMBEDDING_DIM,
).to(
    device
)


total_parameters = int(
    sum(
        p.numel()
        for p in model.parameters()
    )
)


trainable_parameters = int(
    sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )
)


profile_sample = development_cache.get(
    development_cases[
        0
    ],
    "component_natural_50",
)


profile_patch = None


for attempt in range(
    40
):

    candidate = extract_patch(
        profile_sample,
        PATCH_ZYX,
        seed=GLOBAL_SEED,
        sample_index=(
            900000
            + attempt
        ),
    )


    if int(
        (
            candidate[
                "target"
            ]
            != -1
        ).sum()
    ) > 0:

        profile_patch = candidate

        break


if profile_patch is None:

    raise RuntimeError(
        "No labelled GPU profile patch."
    )


image = (
    profile_patch[
        "image"
    ]
    .unsqueeze(
        0
    )
    .to(
        device=device,
        dtype=torch.float32,
    )
)


target = (
    profile_patch[
        "target"
    ]
    .unsqueeze(
        0
    )
    .to(
        device=device,
        dtype=torch.int8,
    )
)


amp_enabled = bool(
    device.type
    == "cuda"
)


peak_allocated_gb = 0.0
peak_reserved_gb = 0.0


if device.type == "cuda":

    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats(
        device
    )


model.zero_grad(
    set_to_none=True
)


start = time.perf_counter()


with torch.autocast(
    device_type=device.type,
    dtype=(
        torch.float16
        if device.type == "cuda"
        else torch.bfloat16
    ),
    enabled=amp_enabled,
):

    output = model(
        image
    )

    dry_loss = (
        partial_bce(
            output[
                "logits"
            ],
            target,
        )
        + partial_dice(
            output[
                "logits"
            ],
            target,
        )
    )


dry_loss.backward()


if device.type == "cuda":

    torch.cuda.synchronize()

    peak_allocated_gb = cuda_gb(
        torch.cuda.max_memory_allocated(
            device
        )
    )

    peak_reserved_gb = cuda_gb(
        torch.cuda.max_memory_reserved(
            device
        )
    )


dry_run_seconds = (
    time.perf_counter()
    - start
)


dry_run_loss = float(
    dry_loss.detach()
    .cpu()
)


memory_gate_pass = bool(
    device.type != "cuda"
    or peak_reserved_gb
    <= GPU_MEMORY_LIMIT_GB
)


if not memory_gate_pass:

    raise RuntimeError(
        "GPU memory gate failed."
    )


model.zero_grad(
    set_to_none=True
)


del image
del target
del output

gc.collect()


if torch.cuda.is_available():

    torch.cuda.empty_cache()


# ==========================================================================================
# 10. CHECKPOINT / REGISTRY DRY TEST
# ==========================================================================================

dry_run_dir = (
    RUN_ROOT
    / "_harness_dry_run"
)


if dry_run_dir.exists():

    shutil.rmtree(
        dry_run_dir
    )


dry_run_dir.mkdir(
    parents=True,
    exist_ok=True,
)


config_hash = sha256_file(
    REPO
    / "configs/experiment_gate_b_pilot.yaml"
)


cache_hash = sha256_file(
    CACHE_MANIFEST
)


optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=FUTURE_LR,
    weight_decay=FUTURE_WEIGHT_DECAY,
)


checkpoint_path = (
    dry_run_dir
    / "checkpoint.pt"
)


checkpoint_sha = save_training_checkpoint(
    checkpoint_path,
    model=model,
    optimizer=optimizer,
    scaler=None,
    epoch=0,
    global_step=0,
    config_hash=config_hash,
    cache_manifest_hash=cache_hash,
    run_id="HARNESS_DRY_RUN",
)


loaded = load_training_checkpoint(
    checkpoint_path,
    expected_config_hash=config_hash,
    expected_cache_manifest_hash=cache_hash,
)


checkpoint_pass = bool(
    loaded[
        "run_id"
    ]
    == "HARNESS_DRY_RUN"
    and loaded[
        "epoch"
    ]
    == 0
    and loaded[
        "global_step"
    ]
    == 0
)


registry = ExperimentRegistry(
    dry_run_dir
    / "registry.json"
)


registry.register(
    "HARNESS_DRY_RUN",
    {
        "purpose":
            "Block-07 infrastructure validation",

        "scientific_training":
            False,

        "optimizer_step_performed":
            False,
    },
)


registry.update(
    "HARNESS_DRY_RUN",
    status="VALIDATED",
    checkpoint_sha256=checkpoint_sha,
)


registry_payload = json.loads(
    (
        dry_run_dir
        / "registry.json"
    ).read_text(
        encoding="utf-8"
    )
)


registry_pass = bool(
    len(
        registry_payload[
            "runs"
        ]
    )
    == 1
    and registry_payload[
        "runs"
    ][
        0
    ][
        "status"
    ]
    == "VALIDATED"
)


if not (
    checkpoint_pass
    and registry_pass
):

    raise RuntimeError(
        "Checkpoint / registry validation failed."
    )


# ==========================================================================================
# 11. SAVE AUDITS
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

docs_dir = (
    REPO
    / "docs"
)


for directory in [
    manifest_dir,
    audit_dir,
    figure_dir,
    docs_dir,
]:

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )


patch_df.to_csv(
    manifest_dir
    / "gate_b_harness_patch_sampler_audit.csv",
    index=False,
)


query_df.to_csv(
    manifest_dir
    / "gate_b_query_holdout_equivalence.csv",
    index=False,
)


pd.DataFrame(
    [
        {
            "device":
                str(
                    device
                ),

            "patch_z":
                PATCH_ZYX[
                    0
                ],

            "patch_y":
                PATCH_ZYX[
                    1
                ],

            "patch_x":
                PATCH_ZYX[
                    2
                ],

            "base_channels":
                BASE_CHANNELS,

            "embedding_dim":
                EMBEDDING_DIM,

            "parameters":
                total_parameters,

            "amp_enabled":
                amp_enabled,

            "peak_allocated_gb":
                peak_allocated_gb,

            "peak_reserved_gb":
                peak_reserved_gb,

            "forward_backward_seconds":
                dry_run_seconds,

            "loss":
                dry_run_loss,

            "optimizer_step_performed":
                False,

            "memory_gate_pass":
                memory_gate_pass,
        }
    ]
).to_csv(
    manifest_dir
    / "gate_b_harness_gpu_profile.csv",
    index=False,
)


value_error = float(
    query_df[
        "value_abs_error"
    ].max()
)


gradient_error = float(
    query_df[
        "max_gradient_abs_error"
    ].max()
)


audit = {
    "project":
        PROJECT,

    "block":
        "07",

    "corrective_block":
        "07C-R1",

    "status":
        "PASS",

    "generated_at_utc":
        NOW_ISO,

    "scientific_training_performed":
        False,

    "optimizer_steps_performed":
        0,

    "precision_correction": {
        "initial_value_error":
            5.9604644775390625e-08,

        "initial_gradient_error":
            0.0,

        "root_cause":
            "target.float() forced float32 BCE during float64 audit",

        "correction":
            "target inherits logits dtype",

        "tolerance_relaxed":
            False,
    },

    "patch_sampler": {
        "patch_zyx":
            list(
                PATCH_ZYX
            ),

        "audited_patches":
            int(
                len(
                    patch_df
                )
            ),

        "patches_with_labels":
            patches_with_labels,

        "patches_with_foreground":
            patches_with_fg,
    },

    "loss_contract": {
        "query_value_equivalence":
            "PASS",

        "max_value_abs_error":
            value_error,

        "query_gradient_equivalence":
            "PASS",

        "max_gradient_abs_error":
            gradient_error,
    },

    "model": {
        "family":
            "residual_3d_unet",

        "base_channels":
            BASE_CHANNELS,

        "embedding_dim":
            EMBEDDING_DIM,

        "parameters":
            total_parameters,
    },

    "gpu_profile": {
        "device":
            str(
                device
            ),

        "amp":
            amp_enabled,

        "peak_allocated_gb":
            peak_allocated_gb,

        "peak_reserved_gb":
            peak_reserved_gb,

        "forward_backward_seconds":
            dry_run_seconds,

        "memory_limit_gb":
            GPU_MEMORY_LIMIT_GB,

        "memory_gate":
            "PASS",

        "optimizer_step":
            False,
    },

    "checkpointing": {
        "atomic_checkpoint":
            "PASS",

        "registry":
            "PASS",

        "primary_checkpoint":
            "final_epoch",

        "dense_validation_selection":
            False,
    },

    "training_authorized":
        False,
}


write_json(
    audit_dir
    / "block07_gate_b_pilot_harness.json",
    audit,
)


# ==========================================================================================
# 12. SIMPLE PROTOCOL NOTE
# ==========================================================================================

protocol_lines = [
    "# CORA-Lung Gate-B Pilot Harness Protocol",
    "",
    "Scientific training in Block 07: **No**",
    "Optimizer updates in Block 07: **0**",
    "",
    "## Numerical precision correction",
    "",
    "The initial query-BCE verification used `target.float()`.",
    "This forced the BCE result to float32 even when the audit logits were float64.",
    "",
    "Observed initial scalar discrepancy:",
    "`5.9604644775390625e-08`",
    "",
    "Observed initial gradient discrepancy:",
    "`0.0`",
    "",
    "Before any model fitting, the cast was corrected to:",
    "`target.to(dtype=logits.dtype)`",
    "",
    "The verification tolerance was not relaxed.",
    "",
    "Final maximum BCE value error:",
    "`{}`".format(
        "{:.18e}".format(
            value_error
        )
    ),
    "",
    "Final maximum BCE gradient error:",
    "`{}`".format(
        "{:.18e}".format(
            gradient_error
        )
    ),
    "",
    "## GPU dry run",
    "",
    "Peak allocated memory: **{:.3f} GB**".format(
        peak_allocated_gb
    ),
    "",
    "Peak reserved memory: **{:.3f} GB**".format(
        peak_reserved_gb
    ),
    "",
    "Optimizer step performed: **No**",
]


write_text(
    docs_dir
    / "gate_b_pilot_harness_protocol.md",
    "\n".join(
        protocol_lines
    ),
)


# ==========================================================================================
# 13. QA FIGURE
# ==========================================================================================

labels = [
    "Unit\nTests",
    "BCE Value\nEquivalence",
    "BCE Gradient\nEquivalence",
    "Sparse Patch\nSampling",
    "Checkpoint\nResume",
    "GPU Memory\nGate",
]


values = [
    1,
    int(
        query_value_pass
    ),
    int(
        query_gradient_pass
    ),
    int(
        patches_with_labels
        > 0
    ),
    int(
        checkpoint_pass
        and registry_pass
    ),
    int(
        memory_gate_pass
    ),
]


fig, ax = plt.subplots(
    figsize=(
        11,
        5.8,
    )
)


x = np.arange(
    len(
        labels
    )
)


ax.bar(
    x,
    values,
)


ax.set_xticks(
    x
)


ax.set_xticklabels(
    labels,
    fontweight="bold",
)


ax.set_yticks(
    [
        0,
        1,
    ]
)


ax.set_yticklabels(
    [
        "FAIL",
        "PASS",
    ],
    fontweight="bold",
)


ax.set_ylim(
    0,
    1.15,
)


ax.set_ylabel(
    "Verification Status",
    fontweight="bold",
)


ax.set_title(
    "CORA-Lung Gate-B Pilot Harness Verification Before Model Fitting\n"
    "Sparse-Loss Arithmetic, Sampling, Resume Safety, and GPU Feasibility",
    fontweight="bold",
)


ax.grid(
    axis="y",
    alpha=0.25,
)


fig.tight_layout()


fig.savefig(
    figure_dir
    / "fig09_gate_b_pilot_harness_verification.png",
    dpi=600,
    bbox_inches="tight",
)


fig.savefig(
    figure_dir
    / "fig09_gate_b_pilot_harness_verification.pdf",
    bbox_inches="tight",
)


plt.close(
    fig
)


# ==========================================================================================
# 14. CAPTURE REPAIR SOURCE
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
        "BLOCK 07C-R1"
        in cell
    ):

        source_path = (
            REPO
            / "scripts/code_blocks/block07c_r1_dtype_repair_resume.py"
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
# 15. UPDATE PROJECT STATE
# ==========================================================================================

state.update(
    {
        "last_attempted_block":
            "07",

        "last_completed_block":
            "07",

        "last_completed_block_name":
            "gate_b_pilot_training_harness",

        "current_stage":
            "gate_b_pilot_harness_verified",

        "current_gate":
            "PRE_GATE_B_PILOT",

        "gate_a":
            "PASS",

        "gate_b":
            "NOT_RUN",

        "gate_d_memory_dry_run":
            "PASS",

        "gate_b_pilot_harness":
            "PASS",

        "pilot_training_authorized":
            False,

        "training_authorized":
            False,

        "model_training_started":
            False,

        "optimizer_steps_performed":
            0,

        "next_action":
            (
                "Audit Block 07C-R1. "
                "Then authorize development-only "
                "Gate-B omission-deficit pilot."
            ),

        "updated_at_utc":
            NOW_ISO,
    }
)


write_json(
    REPO
    / "PROJECT_STATE.json",
    state,
)


# ==========================================================================================
# 16. REFRESH REPOSITORY MANIFEST — EXCLUDE SELF
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
            "07",

        "corrective_block":
            "07C-R1",

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
# 17. COMMIT + PUSH
# ==========================================================================================

secret = UserSecretsClient().get_secret(
    "pushCora"
)


if not secret:

    raise RuntimeError(
        "Kaggle secret pushCora unavailable."
    )


secret = secret.strip()


askpass = Path(
    "/tmp/cora_git_askpass_07cr1.sh"
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


git_env = os.environ.copy()


git_env[
    "GITHUB_TOKEN"
] = secret


git_env[
    "GIT_ASKPASS"
] = str(
    askpass
)


git_env[
    "GIT_TERMINAL_PROMPT"
] = "0"


sh(
    [
        "git",
        "add",
        ".",
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
        "No changes to commit."
    )


sh(
    [
        "git",
        "commit",
        "-m",
        "train: freeze Gate-B harness with dtype-exact BCE",
    ],
    cwd=REPO,
)


push = sh(
    [
        "git",
        "push",
        "origin",
        "main",
    ],
    cwd=REPO,
    env=git_env,
    check=False,
)


if push.returncode != 0:

    raise RuntimeError(
        "Git push failed:\n"
        + (
            push.stderr
            or ""
        ).replace(
            secret,
            "***TOKEN_REDACTED***",
        )
    )


current_commit = sh(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    cwd=REPO,
).stdout.strip()


try:

    askpass.unlink(
        missing_ok=True
    )

except Exception:
    pass


# ==========================================================================================
# 18. FINAL REPORT
# ==========================================================================================

print(
    "\n"
    + "=" * 118
)

print(
    "CORA-LUNG CODE BLOCK 07C-R1 — FINAL REPORT"
)

print(
    "=" * 118
)


print(
    "Initial BCE value error              :",
    "{:.18e}".format(
        5.9604644775390625e-08
    ),
)

print(
    "Initial BCE gradient error           :",
    "{:.18e}".format(
        0.0
    ),
)

print(
    "Correction                           : target.to(dtype=logits.dtype)"
)

print(
    "Tolerance relaxed                    : NO"
)

print(
    "Final maximum BCE value error        :",
    "{:.18e}".format(
        value_error
    ),
)

print(
    "Final maximum BCE gradient error     :",
    "{:.18e}".format(
        gradient_error
    ),
)

print(
    "BCE value equivalence @ 1e-12        :",
    (
        "PASS"
        if query_value_pass
        else "FAIL"
    ),
)

print(
    "BCE gradient equivalence @ 1e-12     :",
    (
        "PASS"
        if query_gradient_pass
        else "FAIL"
    ),
)

print(
    "Development volumes                  :",
    len(
        development_cases
    ),
)

print(
    "Final outer-fold volumes used        : 0"
)

print(
    "Dense masks accessed                 : NO"
)

print(
    "Scientific training                  : NO"
)

print(
    "Optimizer steps                      : 0"
)

print(
    "Real patches audited                 :",
    len(
        patch_df
    ),
)

print(
    "Patches with sparse labels           :",
    str(
        patches_with_labels
    )
    + "/"
    + str(
        len(
            patch_df
        )
    ),
)

print(
    "Patches with foreground              :",
    str(
        patches_with_fg
    )
    + "/"
    + str(
        len(
            patch_df
        )
    ),
)

print(
    "Trainable parameters                 :",
    "{:,}".format(
        trainable_parameters
    ),
)

print(
    "Device                               :",
    device,
)

print(
    "AMP enabled                          :",
    amp_enabled,
)

print(
    "Dry-run loss                         :",
    "{:.6f}".format(
        dry_run_loss
    ),
)

print(
    "Forward/backward time                :",
    "{:.3f} sec".format(
        dry_run_seconds
    ),
)

print(
    "Peak allocated memory                :",
    "{:.3f} GB".format(
        peak_allocated_gb
    ),
)

print(
    "Peak reserved memory                 :",
    "{:.3f} GB".format(
        peak_reserved_gb
    ),
)

print(
    "Gate-D dry-run                       :",
    (
        "PASS"
        if memory_gate_pass
        else "FAIL"
    ),
)

print(
    "Atomic checkpoint                    : PASS"
)

print(
    "Experiment registry                  : PASS"
)

print(
    "Full scientific/unit tests           : PASS"
)

print(
    "Exact repair source captured         :",
    source_capture,
)

print(
    "Block 07 pilot harness               : PASS"
)

print(
    "Gate B                               : NOT RUN"
)

print(
    "Model training                       : NOT STARTED"
)

print(
    "Pilot training authorized            : NO"
)

print(
    "Starting commit                      :",
    starting_commit[:12],
)

print(
    "Current commit                       :",
    current_commit[:12],
)

print(
    "GitHub synchronization               : PASS"
)

print(
    "\nNEXT: Send me this COMPLETE report."
)

print(
    "=" * 118
)