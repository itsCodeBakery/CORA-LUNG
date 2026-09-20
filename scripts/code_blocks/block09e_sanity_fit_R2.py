# ==========================================================================================
# COVA-3D — BLOCK 09E-SANITY-FIT-R2-FINALIZE
# Finalize Completed R2 After Historical N2 Regression-Test Scope Failure
#
# EXPECTED COMMITTED HEAD
# -----------------------
# aa01153258c9
#
# WHY THE PREVIOUS BLOCK STOPPED
# ------------------------------
# R2 training/state generation completed before pytest.
#
# Historical test:
#
#   tests/test_cova3d_numerical_amendment_n2.py::test_n2_state
#
# incorrectly requires CURRENT mutable state to remain forever at:
#
#   last_completed_block == "09E-N2-LOCK"
#   retained optimizer steps == 0
#   sanity_training_authorized == True
#
# After a legitimate completed R2 fit, CURRENT state must instead record:
#
#   last_completed_block == "09E-SANITY-FIT-R2"
#   retained optimizer steps == 1000
#   sanity_training_authorized == False
#
# This is the same historical-test scoping issue previously repaired for N1.
#
# THIS BLOCK:
#   ✓ verifies that R2 really completed
#   ✓ verifies exactly 20 epochs / 1000 successful updates
#   ✓ verifies final model checksum
#   ✓ verifies dense masks remained sealed
#   ✓ reproduces the expected historical N2 test failure
#   ✓ repairs ONLY the historical N2 test scope
#   ✓ reruns partial-loss + baseline + N1 + N2 + R2 regression tests
#   ✓ captures R2 source from notebook history
#   ✓ regenerates repository manifest
#   ✓ commits/pushes the ALREADY COMPLETED R2 experiment
#
# THIS BLOCK DOES NOT:
#   ✗ train
#   ✗ call optimizer.step()
#   ✗ reopen CT arrays
#   ✗ open dense masks
#   ✗ access final outer-CV
#   ✗ change N1
#   ✗ change N2
#   ✗ change R2 model weights
#   ✗ change annotations / architecture / optimizer / loss
#
# ==========================================================================================


from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import hashlib
import importlib
import json
import os
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

EXPECTED_HEAD = (
    "aa01153258c9"
)

BLOCK = (
    "09E-SANITY-FIT-R2-FINALIZE"
)

R2_BLOCK = (
    "09E-SANITY-FIT-R2"
)

EFFECTIVE_PROTOCOL = (
    "COVA3D_1.0+A1+A1.1+A1.2+N1+N2"
)

EXPECTED_R2_STEPS = (
    1000
)

EXPECTED_R2_EPOCHS = (
    20
)

R2_ROOT = Path(
    "/kaggle/working/cova3d_sanity_fit_R2_v1_0"
)

R2_STATE_PATH = (
    R2_ROOT
    / "run_state.json"
)

R2_FINAL_MODEL_PATH = (
    R2_ROOT
    / "final_model.pt"
)

R2_RUNTIME_EPOCH_LOG_PATH = (
    R2_ROOT
    / "epoch_log.csv"
)

R2_REPO_EPOCH_LOG_PATH = (
    REPO
    / "data/manifests/"
    "cova3d_sanity_fit_R2_epoch_log_v1_0.csv"
)

R2_AUDIT_PATH = (
    REPO
    / "experiments/audits/"
    "block09e_sanity_fit_R2.json"
)

R2_TEST_PATH = (
    REPO
    / "tests/"
    "test_cova3d_sanity_fit_R2.py"
)

N2_TEST_PATH = (
    REPO
    / "tests/"
    "test_cova3d_numerical_amendment_n2.py"
)

N2_CONFIG_PATH = (
    REPO
    / "configs/"
    "cova3d_numerical_amendment_N2_fp32_dice.yaml"
)

N2_AUDIT_PATH = (
    REPO
    / "experiments/audits/"
    "block09e_n2_lock_fp32_dice.json"
)

COVA_STATE_PATH = (
    REPO
    / "COVA3D_STATE.json"
)

PROJECT_STATE_PATH = (
    REPO
    / "PROJECT_STATE.json"
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
    *,
    env=None,
    check=True,
):

    result = subprocess.run(
        cmd,
        cwd=str(
            REPO
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
        "/tmp/cova3d_git_askpass_09e_r2_finalize.sh"
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

    return token, askpass, env


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
# 2. VERIFY COMMITTED HEAD + EXPECTED R2 WORKING TREE
# ==========================================================================================

heading(
    "COVA-3D BLOCK 09E-SANITY-FIT-R2-FINALIZE — VERIFY COMPLETED R2"
)


head = sh(
    [
        "git",
        "rev-parse",
        "HEAD",
    ]
).stdout.strip()


if not head.startswith(
    EXPECTED_HEAD
):

    raise RuntimeError(
        "Unexpected committed HEAD.\n"
        + "Expected: "
        + EXPECTED_HEAD
        + "\nObserved: "
        + head
    )


print(
    "✓ Committed HEAD                       :",
    head[:12],
)


print()
print(
    "Current working tree:"
)

status_before = sh(
    [
        "git",
        "status",
        "--short",
    ]
).stdout


print(
    status_before.rstrip()
    if status_before.strip()
    else "CLEAN — unexpected if R2 had completed before pytest."
)


required_repo_files = [
    COVA_STATE_PATH,
    PROJECT_STATE_PATH,
    R2_REPO_EPOCH_LOG_PATH,
    R2_AUDIT_PATH,
    R2_TEST_PATH,
]


missing_repo_files = [
    str(
        path.relative_to(
            REPO
        )
    )
    for path in required_repo_files
    if not path.exists()
]


if missing_repo_files:

    raise RuntimeError(
        "Expected completed-R2 repository files are missing:\n"
        + "\n".join(
            missing_repo_files
        )
    )


required_runtime_files = [
    R2_STATE_PATH,
    R2_FINAL_MODEL_PATH,
    R2_RUNTIME_EPOCH_LOG_PATH,
]


missing_runtime_files = [
    str(
        path
    )
    for path in required_runtime_files
    if not path.exists()
]


if missing_runtime_files:

    raise RuntimeError(
        "R2 runtime did not reach FIT_COMPLETE.\nMissing:\n"
        + "\n".join(
            missing_runtime_files
        )
        + "\n\nDo NOT rerun training yet. Send me this output."
    )


# ==========================================================================================
# 3. VERIFY R2 RUNTIME COMPLETION
# ==========================================================================================

heading(
    "STEP 1/9 — VERIFY R2 RUNTIME / MODEL CHECKSUM"
)


runtime_state = json.loads(
    R2_STATE_PATH.read_text(
        encoding="utf-8"
    )
)


if runtime_state.get(
    "status"
) != "FIT_COMPLETE":

    raise RuntimeError(
        "R2 runtime status is not FIT_COMPLETE."
    )


if int(
    runtime_state.get(
        "successful_optimizer_steps",
        -1,
    )
) != EXPECTED_R2_STEPS:

    raise RuntimeError(
        "R2 runtime does not contain exactly 1000 successful optimizer steps."
    )


if int(
    runtime_state.get(
        "completed_epoch",
        -1,
    )
) != (
    EXPECTED_R2_EPOCHS
    - 1
):

    raise RuntimeError(
        "R2 runtime does not contain exactly 20 completed epochs."
    )


runtime_model_sha = sha256_file(
    R2_FINAL_MODEL_PATH
)


if runtime_model_sha != str(
    runtime_state.get(
        "final_model_sha256"
    )
):

    raise RuntimeError(
        "R2 final-model checksum does not match run_state.json."
    )


runtime_epoch_df = (
    pd.read_csv(
        R2_RUNTIME_EPOCH_LOG_PATH
    )
    .sort_values(
        "epoch"
    )
    .reset_index(
        drop=True
    )
)


if len(
    runtime_epoch_df
) != EXPECTED_R2_EPOCHS:

    raise RuntimeError(
        "R2 runtime epoch log does not contain 20 epochs."
    )


if int(
    runtime_epoch_df.iloc[
        -1
    ][
        "successful_optimizer_steps_total"
    ]
) != EXPECTED_R2_STEPS:

    raise RuntimeError(
        "R2 runtime epoch log final step is not 1000."
    )


print(
    "✓ R2 runtime status                    : FIT_COMPLETE"
)

print(
    "✓ Completed epochs                     : 20"
)

print(
    "✓ Successful optimizer steps           : 1000"
)

print(
    "✓ Runtime final-model SHA256            :",
    runtime_model_sha,
)


# ==========================================================================================
# 4. VERIFY WORKING-TREE R2 SCIENTIFIC STATE
# ==========================================================================================

heading(
    "STEP 2/9 — VERIFY R2 SCIENTIFIC STATE / FIREWALL"
)


cova_state = json.loads(
    COVA_STATE_PATH.read_text(
        encoding="utf-8"
    )
)


project_state = json.loads(
    PROJECT_STATE_PATH.read_text(
        encoding="utf-8"
    )
)


r2_audit = json.loads(
    R2_AUDIT_PATH.read_text(
        encoding="utf-8"
    )
)


if cova_state.get(
    "last_completed_block"
) != R2_BLOCK:

    raise RuntimeError(
        "Working-tree COVA state does not record completed R2."
    )


if cova_state.get(
    "effective_protocol"
) != EFFECTIVE_PROTOCOL:

    raise RuntimeError(
        "R2 effective protocol mismatch."
    )


if cova_state.get(
    "sanity_fit_R2_status"
) != "COMPLETE":

    raise RuntimeError(
        "R2 state is not COMPLETE."
    )


if int(
    cova_state.get(
        "sanity_fit_R2_successful_optimizer_steps",
        -1,
    )
) != EXPECTED_R2_STEPS:

    raise RuntimeError(
        "R2 state does not record exactly 1000 successful steps."
    )


if int(
    cova_state.get(
        "optimizer_steps_retained_for_current_frozen_run",
        -1,
    )
) != EXPECTED_R2_STEPS:

    raise RuntimeError(
        "Current frozen-run retained step count is not 1000."
    )


if cova_state.get(
    "sanity_gate_status"
) != "EVALUATION_PENDING":

    raise RuntimeError(
        "Sanity gate should still be EVALUATION_PENDING."
    )


if cova_state.get(
    "dense_outcomes_opened_in_cova3d"
) is not False:

    raise RuntimeError(
        "Dense outcomes were unexpectedly opened."
    )


if int(
    cova_state.get(
        "final_outer_cv_access_in_cova3d",
        -1,
    )
) != 0:

    raise RuntimeError(
        "Final outer-CV was unexpectedly accessed."
    )


if cova_state.get(
    "factorial_training_authorized"
) is not False:

    raise RuntimeError(
        "Factorial training must remain locked."
    )


if r2_audit.get(
    "status"
) != "FIT_COMPLETE_EVALUATION_PENDING":

    raise RuntimeError(
        "R2 audit does not report FIT_COMPLETE_EVALUATION_PENDING."
    )


if int(
    r2_audit.get(
        "successful_optimizer_steps",
        -1,
    )
) != EXPECTED_R2_STEPS:

    raise RuntimeError(
        "R2 audit step count mismatch."
    )


if str(
    r2_audit.get(
        "final_model_sha256"
    )
) != runtime_model_sha:

    raise RuntimeError(
        "R2 audit final-model checksum mismatch."
    )


firewall = r2_audit[
    "firewall"
]


if int(
    firewall[
        "dense_lesion_masks_accessed"
    ]
) != 0:

    raise RuntimeError(
        "R2 audit reports dense lesion-mask access."
    )


if int(
    firewall[
        "dense_lung_masks_accessed"
    ]
) != 0:

    raise RuntimeError(
        "R2 audit reports dense lung-mask access."
    )


if int(
    firewall[
        "final_outer_cv_access"
    ]
) != 0:

    raise RuntimeError(
        "R2 audit reports final outer-CV access."
    )


print(
    "✓ R2 working-tree state                : COMPLETE"
)

print(
    "✓ Retained clean R2 steps              : 1000"
)

print(
    "✓ Sanity gate                          : EVALUATION PENDING"
)

print(
    "✓ Dense lesion masks accessed          : 0"
)

print(
    "✓ Dense lung masks accessed            : 0"
)

print(
    "✓ Final outer-CV access                : 0"
)

print(
    "✓ Factorial training                   : LOCKED"
)


# ==========================================================================================
# 5. VERIFY REPOSITORY EPOCH LOG MATCHES RUNTIME LOG
# ==========================================================================================

heading(
    "STEP 3/9 — VERIFY TRAINING LOG CONSISTENCY"
)


repo_epoch_df = (
    pd.read_csv(
        R2_REPO_EPOCH_LOG_PATH
    )
    .sort_values(
        "epoch"
    )
    .reset_index(
        drop=True
    )
)


if len(
    repo_epoch_df
) != EXPECTED_R2_EPOCHS:

    raise RuntimeError(
        "Repository R2 epoch log does not contain 20 epochs."
    )


shared_columns = [
    column
    for column in runtime_epoch_df.columns
    if column in repo_epoch_df.columns
]


if not runtime_epoch_df[
    shared_columns
].equals(
    repo_epoch_df[
        shared_columns
    ]
):

    raise RuntimeError(
        "Repository and runtime R2 epoch logs differ."
    )


first3 = float(
    repo_epoch_df.head(
        3
    )[
        "mean_loss"
    ].mean()
)


last3 = float(
    repo_epoch_df.tail(
        3
    )[
        "mean_loss"
    ].mean()
)


loss_ratio = float(
    last3
    / first3
)


total_zero = int(
    repo_epoch_df[
        "zero_label_microbatches"
    ].sum()
)


total_microbatches = (
    EXPECTED_R2_EPOCHS
    * 100
)


zero_fraction = float(
    total_zero
    / total_microbatches
)


total_overflows = int(
    repo_epoch_df[
        "amp_overflow_events"
    ].sum()
)


epochs_with_overflow = int(
    (
        repo_epoch_df[
            "amp_overflow_events"
        ]
        > 0
    ).sum()
)


minimum_amp_scale = float(
    repo_epoch_df[
        "amp_scale_min"
    ].min()
)


maximum_amp_scale = float(
    repo_epoch_df[
        "amp_scale_max"
    ].max()
)


final_amp_scale = float(
    repo_epoch_df.iloc[
        -1
    ][
        "amp_scale_end"
    ]
)


source_totals = {
    "foreground":
        int(
            repo_epoch_df[
                "foreground_centered"
            ].sum()
        ),

    "background":
        int(
            repo_epoch_df[
                "background_centered"
            ].sum()
        ),

    "uniform":
        int(
            repo_epoch_df[
                "uniform_crop"
            ].sum()
        ),
}


print(
    "✓ Runtime / repository logs            : IDENTICAL"
)

print(
    "First-3 mean loss                      :",
    f"{first3:.6f}",
)

print(
    "Last-3 mean loss                       :",
    f"{last3:.6f}",
)

print(
    "Last3 / First3 ratio                   :",
    f"{loss_ratio:.4f}",
)

print(
    "Zero-label fraction                    :",
    f"{zero_fraction:.4f}",
)

print(
    "Patch-source FG/BG/U                   :",
    source_totals[
        "foreground"
    ],
    "/",
    source_totals[
        "background"
    ],
    "/",
    source_totals[
        "uniform"
    ],
)

print(
    "AMP overflow events                    :",
    total_overflows,
)

print(
    "Epochs with overflow                   :",
    epochs_with_overflow,
    "/20",
)

print(
    "AMP min / max / final                  :",
    minimum_amp_scale,
    "/",
    maximum_amp_scale,
    "/",
    final_amp_scale,
)


# ==========================================================================================
# 6. REPRODUCE EXPECTED HISTORICAL N2 TEST FAILURE
# ==========================================================================================

heading(
    "STEP 4/9 — REPRODUCE HISTORICAL N2 TEST-SCOPE FAILURE"
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


pre_fix = sh(
    [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_cova3d_numerical_amendment_n2.py",
        "-vv",
        "--tb=short",
        "-p",
        "no:cacheprovider",
    ],
    env=pytest_env,
    check=False,
)


print(
    pre_fix.stdout.rstrip()
)


if pre_fix.stderr.strip():

    print()
    print(
        pre_fix.stderr.rstrip()
    )


if pre_fix.returncode == 0:

    raise RuntimeError(
        "Historical N2 test unexpectedly passes after completed R2. "
        "Do not modify it; send me this output."
    )


failure_text = (
    (
        pre_fix.stdout
        or ""
    )
    + "\n"
    + (
        pre_fix.stderr
        or ""
    )
)


if "test_n2_state" not in failure_text:

    raise RuntimeError(
        "The N2 failure is not the expected mutable-state assertion.\n"
        "Do not repair automatically. Send me the complete output."
    )


print()
print(
    "✓ Expected failing historical test     : test_n2_state"
)

print(
    "✓ N2 numerical implementation implicated: NO"
)

print(
    "✓ Completed R2 training implicated     : NO"
)


# ==========================================================================================
# 7. REPAIR ONLY HISTORICAL N2 TEST SCOPE
# ==========================================================================================

heading(
    "STEP 5/9 — REPAIR HISTORICAL N2 TEST SCOPE"
)


n2_test_source = r'''
from pathlib import Path
import json

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_n2():

    return yaml.safe_load(
        (
            ROOT
            / "configs/"
            "cova3d_numerical_amendment_N2_fp32_dice.yaml"
        ).read_text(
            encoding="utf-8"
        )
    )


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

    cfg = load_n2()

    assert (
        cfg[
            "status"
        ]
        == "PROSPECTIVELY_FROZEN_BEFORE_CLEAN_SANITY_RESTART_R2"
    )

    assert (
        cfg[
            "amendment_id"
        ]
        == "N2_FP32_SPARSE_DICE_ARITHMETIC"
    )

    assert (
        cfg[
            "protocol_before"
        ]
        == "COVA3D_1.0+A1+A1.1+A1.2+N1"
    )

    assert (
        cfg[
            "protocol_after"
        ]
        == "COVA3D_1.0+A1+A1.1+A1.2+N1+N2"
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
            "restart_policy"
        ][
            "restart"
        ]
        == "ORIGINAL_FROZEN_INITIALIZATION"
    )

    assert (
        cfg[
            "factorial_training_authorized"
        ]
        is False
    )


def test_n2_historical_audit_is_immutable():

    """Validate N2's frozen historical artifact, not mutable CURRENT state."""

    audit = json.loads(
        (
            ROOT
            / "experiments/audits/"
            "block09e_n2_lock_fp32_dice.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        audit[
            "status"
        ]
        == "PASS"
    )

    assert (
        audit[
            "amendment"
        ]
        == "N2_FP32_SPARSE_DICE_ARITHMETIC"
    )

    assert (
        audit[
            "R1_successful_updates_executed"
        ]
        == 233
    )

    assert (
        audit[
            "R1_successful_updates_retained"
        ]
        == 0
    )

    assert (
        audit[
            "legacy_Dice_scale1_finite"
        ]
        is False
    )

    assert (
        audit[
            "N2_Dice_scale1_finite"
        ]
        is True
    )

    assert (
        audit[
            "Dice_arithmetic_dtype"
        ]
        == "FP32"
    )

    assert (
        audit[
            "Dice_formula_changed"
        ]
        is False
    )

    assert (
        audit[
            "Dice_epsilon_changed"
        ]
        is False
    )

    assert (
        audit[
            "BCE_changed"
        ]
        is False
    )

    assert (
        audit[
            "factorial_training_authorized"
        ]
        is False
    )
'''


write_text(
    N2_TEST_PATH,
    n2_test_source,
)


print(
    "✓ Functional N2 numerical tests        : RETAINED"
)

print(
    "✓ Frozen N2 protocol test              : RETAINED"
)

print(
    "✓ Mutable CURRENT-state test           : REMOVED"
)

print(
    "✓ Immutable N2 audit test              : ADDED"
)

print(
    "✓ Scientific protocol changed          : NO"
)


# ==========================================================================================
# 8. RUN COMPLETE REGRESSION SUITE INCLUDING R2
# ==========================================================================================

heading(
    "STEP 6/9 — RUN COMPLETE N1 + N2 + R2 REGRESSION SUITE"
)


combined = sh(
    [
        sys.executable,
        "-m",
        "pytest",

        "tests/test_partial_losses.py",
        "tests/test_cova3d_baseline_lock.py",
        "tests/test_cova3d_numerical_amendment_n1.py",
        "tests/test_cova3d_numerical_amendment_n2.py",
        "tests/test_cova3d_sanity_fit_R2.py",

        "-q",
        "-p",
        "no:cacheprovider",
    ],
    env=pytest_env,
    check=False,
)


print(
    combined.stdout.rstrip()
)


if combined.stderr.strip():

    print()
    print(
        combined.stderr.rstrip()
    )


if combined.returncode != 0:

    raise RuntimeError(
        "Combined regression suite still fails after N2 historical-test repair.\n"
        "Do NOT rerun R2 training."
    )


print()
print(
    "✓ Complete regression suite            : PASS"
)


# ==========================================================================================
# 9. CAPTURE R2 / FINALIZER SOURCE
# ==========================================================================================

heading(
    "STEP 7/9 — CAPTURE EXACT R2 SOURCE"
)


scripts_dir = (
    REPO
    / "scripts/code_blocks"
)


scripts_dir.mkdir(
    parents=True,
    exist_ok=True,
)


r2_source_path = (
    scripts_dir
    / "block09e_sanity_fit_R2.py"
)


finalizer_source_path = (
    scripts_dir
    / "block09e_sanity_fit_R2_finalize.py"
)


r2_source_capture = (
    "NOT_AVAILABLE"
)


finalizer_source_capture = (
    "NOT_AVAILABLE"
)


try:

    ip = get_ipython()

    history = list(
        ip.history_manager.input_hist_raw
    )


    prior_history = (
        history[
            :-1
        ]
        if len(
            history
        )
        else []
    )


    for cell in reversed(
        prior_history
    ):

        if (
            "# COVA-3D — BLOCK 09E-SANITY-FIT-R2"
            in cell
            and "Clean Sanity Restart Under Frozen Numerical Amendments N1 + N2"
            in cell
        ):

            r2_source_path.write_text(
                cell.rstrip()
                + "\n",
                encoding="utf-8",
            )

            r2_source_capture = (
                "PASS"
            )

            break


    current_cell = history[
        -1
    ]


    if (
        "# COVA-3D — BLOCK 09E-SANITY-FIT-R2-FINALIZE"
        in current_cell
    ):

        finalizer_source_path.write_text(
            current_cell.rstrip()
            + "\n",
            encoding="utf-8",
        )

        finalizer_source_capture = (
            "PASS"
        )


except Exception:

    pass


print(
    "R2 original source capture             :",
    r2_source_capture,
)

print(
    "R2 finalizer source capture            :",
    finalizer_source_capture,
)


# ==========================================================================================
# 10. UPDATE R2 AUDIT + STATE
# ==========================================================================================

heading(
    "STEP 8/9 — FINALIZE R2 PROVENANCE"
)


r2_audit = json.loads(
    R2_AUDIT_PATH.read_text(
        encoding="utf-8"
    )
)


r2_audit[
    "source_capture"
] = (
    r2_source_capture
)


r2_audit[
    "regression_suite_status"
] = (
    "PASS_AFTER_HISTORICAL_N2_TEST_SCOPE_FIX"
)


r2_audit[
    "regression_scope_fix"
] = {
    "scientific_protocol_changed":
        False,

    "N2_policy_changed":
        False,

    "R2_training_changed":
        False,

    "reason":
        (
            "Historical N2 test incorrectly asserted mutable current project "
            "state. It was replaced with immutable N2 configuration/audit "
            "assertions."
        ),

    "finalized_at_utc":
        NOW_ISO,
}


write_json(
    R2_AUDIT_PATH,
    r2_audit,
)


scope_audit_path = (
    REPO
    / "experiments/audits/"
    "block09e_R2_regression_scope_fix.json"
)


write_json(
    scope_audit_path,
    {
        "project":
            "COVA-3D",

        "block":
            BLOCK,

        "status":
            "PASS",

        "parent_commit":
            head,

        "completed_R2_verified":
            True,

        "R2_epochs":
            EXPECTED_R2_EPOCHS,

        "R2_successful_optimizer_steps":
            EXPECTED_R2_STEPS,

        "R2_final_model_sha256":
            runtime_model_sha,

        "original_regression_failure":
            "HISTORICAL_N2_MUTABLE_CURRENT_STATE_ASSERTION",

        "failing_test":
            "test_n2_state",

        "repair":
            (
                "Test immutable N2 configuration and audit instead of mutable "
                "current state."
            ),

        "scientific_changes":
            0,

        "model_changes":
            0,

        "loss_changes":
            0,

        "optimizer_changes":
            0,

        "annotation_changes":
            0,

        "additional_training_steps":
            0,

        "dense_masks_accessed":
            0,

        "final_outer_cv_access":
            0,

        "combined_regression_suite":
            "PASS",

        "R2_source_capture":
            r2_source_capture,

        "finalizer_source_capture":
            finalizer_source_capture,

        "generated_at_utc":
            NOW_ISO,
    },
)


cova_state = json.loads(
    COVA_STATE_PATH.read_text(
        encoding="utf-8"
    )
)


cova_state[
    "R2_regression_tests"
] = (
    "PASS"
)


cova_state[
    "R2_regression_scope_fix"
] = (
    "HISTORICAL_N2_MUTABLE_STATE_ASSERTION_REMOVED"
)


cova_state[
    "sanity_gate_status"
] = (
    "EVALUATION_PENDING"
)


cova_state[
    "factorial_training_authorized"
] = (
    False
)


cova_state[
    "next_block"
] = (
    "09E-SANITY-EVAL"
)


cova_state[
    "next_action"
] = (
    "Before factorial training, evaluate the frozen R2 sanity model only "
    "under the authorized development protocol. Final outer-CV remains sealed."
)


cova_state[
    "updated_at_utc"
] = (
    NOW_ISO
)


write_json(
    COVA_STATE_PATH,
    cova_state,
)


project_state = json.loads(
    PROJECT_STATE_PATH.read_text(
        encoding="utf-8"
    )
)


project_state[
    "cova3d_R2_regression_tests"
] = (
    "PASS"
)


project_state[
    "next_action"
] = (
    "Run the next locked COVA-3D sanity-evaluation step only."
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


# ==========================================================================================
# 11. FINAL TEST AFTER STATE WRITES
# ==========================================================================================

final_tests = sh(
    [
        sys.executable,
        "-m",
        "pytest",

        "tests/test_partial_losses.py",
        "tests/test_cova3d_baseline_lock.py",
        "tests/test_cova3d_numerical_amendment_n1.py",
        "tests/test_cova3d_numerical_amendment_n2.py",
        "tests/test_cova3d_sanity_fit_R2.py",

        "-q",
        "-p",
        "no:cacheprovider",
    ],
    env=pytest_env,
    check=False,
)


print(
    final_tests.stdout.rstrip()
)


if final_tests.stderr.strip():

    print()
    print(
        final_tests.stderr.rstrip()
    )


if final_tests.returncode != 0:

    raise RuntimeError(
        "Final regression suite failed after provenance finalization."
    )


print()
print(
    "✓ Post-state regression suite          : PASS"
)


# ==========================================================================================
# 12. NORMALIZE TEXT BEFORE MANIFEST
# ==========================================================================================

text_paths = [
    COVA_STATE_PATH,
    PROJECT_STATE_PATH,
    R2_REPO_EPOCH_LOG_PATH,
    R2_AUDIT_PATH,
    R2_TEST_PATH,
    N2_TEST_PATH,
    scope_audit_path,
]


for optional in [
    r2_source_path,
    finalizer_source_path,
]:

    if optional.exists():

        text_paths.append(
            optional
        )


for path in text_paths:

    path = Path(
        path
    )

    content = path.read_text(
        encoding="utf-8"
    )

    path.write_text(
        content.rstrip()
        + "\n",
        encoding="utf-8",
    )


# ==========================================================================================
# 13. REGENERATE REPOSITORY MANIFEST LAST
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
            R2_BLOCK,

        "finalization_block":
            BLOCK,

        "active_track":
            "COVA3D",

        "effective_protocol":
            EFFECTIVE_PROTOCOL,

        "numerical_amendments":
            [
                "N1",
                "N2",
            ],

        "R1_updates_executed_nonretained":
            233,

        "R1_updates_retained":
            0,

        "R2_successful_optimizer_steps":
            EXPECTED_R2_STEPS,

        "R2_final_model_sha256":
            runtime_model_sha,

        "R2_regression_suite":
            "PASS",

        "dense_outcomes_opened":
            False,

        "sanity_gate":
            "EVALUATION_PENDING",

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
# 14. COMMIT / PUSH COMPLETED R2
# ==========================================================================================

heading(
    "STEP 9/9 — COMMIT COMPLETED R2"
)


git_paths = [
    "COVA3D_STATE.json",
    "PROJECT_STATE.json",
    "REPOSITORY_MANIFEST.json",

    "data/manifests/"
    "cova3d_sanity_fit_R2_epoch_log_v1_0.csv",

    "experiments/audits/"
    "block09e_sanity_fit_R2.json",

    "experiments/audits/"
    "block09e_R2_regression_scope_fix.json",

    "figures/audit/"
    "fig_cova3d_sanity_R2_training_curve_v1_0.png",

    "figures/audit/"
    "fig_cova3d_sanity_R2_training_curve_v1_0.pdf",

    "figures/audit/"
    "fig_cova3d_sanity_R2_amp_scale_v1_0.png",

    "figures/audit/"
    "fig_cova3d_sanity_R2_amp_scale_v1_0.pdf",

    "tests/"
    "test_cova3d_numerical_amendment_n2.py",

    "tests/"
    "test_cova3d_sanity_fit_R2.py",
]


for optional_relative in [
    "scripts/code_blocks/block09e_sanity_fit_R2.py",
    "scripts/code_blocks/block09e_sanity_fit_R2_finalize.py",
]:

    if (
        REPO
        / optional_relative
    ).exists():

        git_paths.append(
            optional_relative
        )


sh(
    [
        "git",
        "add",
        *git_paths,
    ]
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
        "git diff --cached --check failed:\n"
        + (
            diff_check.stdout
            or ""
        )
        + (
            diff_check.stderr
            or ""
        )
    )


staged_files = sh(
    [
        "git",
        "diff",
        "--cached",
        "--name-only",
    ]
).stdout.splitlines()


if any(
    item.endswith(
        ".pt"
    )
    for item in staged_files
):

    raise RuntimeError(
        "Runtime checkpoint/model must not enter normal Git."
    )


print(
    "✓ git diff --cached --check            : PASS"
)

print(
    "✓ Runtime model staged                 : NO"
)


unstaged_tracked = sh(
    [
        "git",
        "diff",
        "--name-only",
    ]
).stdout.strip()


if unstaged_tracked:

    raise RuntimeError(
        "Tracked unstaged changes remain before commit:\n"
        + unstaged_tracked
    )


sh(
    [
        "git",
        "commit",
        "-m",
        "experiment: finalize clean COVA-3D sanity fit R2",
    ]
)


final_commit = sh(
    [
        "git",
        "rev-parse",
        "HEAD",
    ]
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
    ]
).stdout.strip()


if final_status:

    raise RuntimeError(
        "Repository is not clean after R2 finalization:\n"
        + final_status
    )


# ==========================================================================================
# 15. FINAL REPORT
# ==========================================================================================

heading(
    "COVA-3D BLOCK 09E-SANITY-FIT-R2-FINALIZE — FINAL REPORT"
)


print(
    "Starting commit                        :",
    head[:12],
)

print(
    "R2 finalization commit                 :",
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
    "R2 COMPLETION"
)

print(
    "-------------"
)

print(
    "Epochs                                 : 20"
)

print(
    "Successful optimizer steps             : 1000"
)

print(
    "Final model SHA256                     :",
    runtime_model_sha,
)

print(
    "R1 weights reused                      : NO"
)

print(
    "R1 updates retained                    : 0"
)

print()

print(
    "TRAINING DIAGNOSTICS"
)

print(
    "--------------------"
)

print(
    "First-3 mean loss                      :",
    f"{first3:.6f}",
)

print(
    "Last-3 mean loss                       :",
    f"{last3:.6f}",
)

print(
    "Last3 / First3 ratio                   :",
    f"{loss_ratio:.4f}",
)

print(
    "Zero-label fraction                    :",
    f"{zero_fraction:.4f}",
)

print(
    "Patch-source FG/BG/U                   :",
    source_totals[
        "foreground"
    ],
    "/",
    source_totals[
        "background"
    ],
    "/",
    source_totals[
        "uniform"
    ],
)

print(
    "AMP overflow events                    :",
    total_overflows,
)

print(
    "Epochs with overflow                   :",
    epochs_with_overflow,
    "/20",
)

print(
    "AMP scale min / max / final            :",
    minimum_amp_scale,
    "/",
    maximum_amp_scale,
    "/",
    final_amp_scale,
)

print()

print(
    "REGRESSION FAILURE"
)

print(
    "------------------"
)

print(
    "R2 training failure                    : NO"
)

print(
    "N2 numerical failure                   : NO"
)

print(
    "Historical N2 test-scope defect        : YES"
)

print(
    "Repair                                 : immutable N2 artifact assertions"
)

print(
    "Complete regression suite              : PASS"
)

print()

print(
    "FIREWALL"
)

print(
    "--------"
)

print(
    "Dense lesion masks accessed            : 0"
)

print(
    "Dense lung masks accessed              : 0"
)

print(
    "Final outer-CV access                  : 0"
)

print()

print(
    "AUTHORIZATION"
)

print(
    "-------------"
)

print(
    "Sanity fit R2                          : COMPLETE"
)

print(
    "Sanity gate                            : EVALUATION PENDING"
)

print(
    "Factorial training authorized          : NO"
)

print(
    "Final outer-CV authorized              : NO"
)

print()

print(
    "SOURCE CAPTURE"
)

print(
    "--------------"
)

print(
    "R2 original block                      :",
    r2_source_capture,
)

print(
    "R2 finalizer                           :",
    finalizer_source_capture,
)

print()

print(
    "NEXT:"
)

print(
    "Send me this COMPLETE report."
)

print(
    "Do NOT rerun R2 training."
)

print(
    "Do NOT begin the six-condition factorial experiment."
)

print(
    "=" * 132
)
