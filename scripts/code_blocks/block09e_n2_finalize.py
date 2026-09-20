# ==========================================================================================
# COVA-3D — BLOCK 09E-N2-FINALIZE
# Repair Historical N1 Regression-Test Scope + Finalize Already-Validated N2
#
# EXPECTED COMMITTED HEAD
# -----------------------
# b51e3667b543
#
# PURPOSE
# -------
# N2 itself has already been validated.
#
# The previous N2 block stopped because the combined regression suite included
# an OLD N1 test that incorrectly required CURRENT mutable project state to
# remain permanently at:
#
#     last_completed_block == "09E-N1-LOCK"
#     numerical_amendment  == "N1"
#
# That assertion becomes invalid after a legitimate later amendment N2.
#
# THIS BLOCK:
#   • verifies the existing uncommitted N2 working tree
#   • reproduces the expected historical N1 test failure
#   • changes ONLY the historical N1 test scope
#   • tests immutable N1 protocol instead of mutable current state
#   • reruns partial-loss + baseline + N1 + N2 tests
#   • performs direct N2 numerical smoke validation
#   • preserves N2 implementation exactly
#   • refreshes repository provenance
#   • commits and pushes N2
#
# THIS BLOCK DOES NOT:
#   ✗ train
#   ✗ call optimizer.step()
#   ✗ access CT arrays
#   ✗ access dense masks
#   ✗ access final outer-CV
#   ✗ change N1 numerical policy
#   ✗ change N2 numerical policy
#   ✗ change model architecture
#   ✗ change annotation protocol
#   ✗ change loss equation
#
# NEXT AFTER PASS:
#   09E-SANITY-FIT-R2
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

import torch

from kaggle_secrets import UserSecretsClient


# ==========================================================================================
# 0. CONSTANTS
# ==========================================================================================

REPO = Path(
    "/kaggle/working/CORA-LUNG"
)

EXPECTED_HEAD = (
    "b51e3667b543"
)

BLOCK = (
    "09E-N2-FINALIZE"
)

EXPECTED_PROTOCOL = (
    "COVA3D_1.0+A1+A1.1+A1.2+N1+N2"
)

N1_TEST_PATH = (
    REPO
    / "tests/"
    "test_cova3d_numerical_amendment_n1.py"
)

N2_TEST_PATH = (
    REPO
    / "tests/"
    "test_cova3d_numerical_amendment_n2.py"
)

PARTIAL_LOSS_PATH = (
    REPO
    / "src/cora_lung/losses/"
    "partial.py"
)

N2_CONFIG_PATH = (
    REPO
    / "configs/"
    "cova3d_numerical_amendment_N2_fp32_dice.yaml"
)

N2_DIAG_PATH = (
    REPO
    / "experiments/audits/"
    "block09e_n1_failed_state_diagnostic.json"
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
            "Kaggle secret 'pushCora' unavailable."
        )

    token = token.strip()

    askpass = Path(
        "/tmp/cova3d_git_askpass_09e_n2_finalize.sh"
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
# 2. VERIFY UNCOMMITTED N2 WORKING TREE
# ==========================================================================================

heading(
    "COVA-3D BLOCK 09E-N2-FINALIZE — VERIFY WORKING TREE"
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


status_before = sh(
    [
        "git",
        "status",
        "--short",
    ]
).stdout


print(
    "✓ Committed HEAD                       :",
    head[:12],
)

print()
print(
    "Existing N2 working-tree changes:"
)
print(
    status_before.rstrip()
)


required_paths = [
    COVA_STATE_PATH,
    PROJECT_STATE_PATH,
    PARTIAL_LOSS_PATH,
    N2_CONFIG_PATH,
    N2_DIAG_PATH,
    N2_AUDIT_PATH,
    N2_TEST_PATH,
]


missing = [
    str(
        path.relative_to(
            REPO
        )
    )
    for path in required_paths
    if not path.exists()
]


if missing:

    raise RuntimeError(
        "Expected N2 files are missing:\n"
        + "\n".join(
            missing
        )
    )


partial_text = PARTIAL_LOSS_PATH.read_text(
    encoding="utf-8"
)


if "def partial_dice_fp32(" not in partial_text:

    raise RuntimeError(
        "N2 partial_dice_fp32 implementation is absent."
    )


if "def partial_segmentation_loss_n2(" not in partial_text:

    raise RuntimeError(
        "N2 combined loss implementation is absent."
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


if cova_state.get(
    "last_completed_block"
) != "09E-N2-LOCK":

    raise RuntimeError(
        "Expected uncommitted N2 state is missing."
    )


if cova_state.get(
    "effective_protocol"
) != EXPECTED_PROTOCOL:

    raise RuntimeError(
        "Unexpected N2 effective protocol."
    )


if int(
    cova_state.get(
        "sanity_fit_R1_successful_updates_executed",
        -1,
    )
) != 233:

    raise RuntimeError(
        "R1 executed-update provenance mismatch."
    )


if int(
    cova_state.get(
        "sanity_fit_R1_updates_retained",
        -1,
    )
) != 0:

    raise RuntimeError(
        "R1 updates must not be retained."
    )


if int(
    cova_state.get(
        "optimizer_steps_retained_for_current_frozen_run",
        -1,
    )
) != 0:

    raise RuntimeError(
        "Clean R2 retained optimizer-step count must remain zero."
    )


if cova_state.get(
    "N1_active"
) is not True:

    raise RuntimeError(
        "N1 must remain active."
    )


if cova_state.get(
    "N2_active"
) is not True:

    raise RuntimeError(
        "N2 must be active."
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
        "Factorial training was unexpectedly authorized."
    )


print()
print(
    "✓ N2 implementation                    : PRESENT"
)

print(
    "✓ N1 active                            : YES"
)

print(
    "✓ N2 active                            : YES"
)

print(
    "✓ R1 updates executed                  : 233"
)

print(
    "✓ R1 updates retained                  : 0"
)

print(
    "✓ Clean R2 retained updates            : 0"
)

print(
    "✓ Dense outcomes                       : SEALED"
)

print(
    "✓ Factorial training                   : LOCKED"
)


# ==========================================================================================
# 3. PRE-FIX N1 FAILURE REPRODUCTION
# ==========================================================================================

heading(
    "STEP 1/8 — REPRODUCE HISTORICAL N1 TEST-SCOPE FAILURE"
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
        "tests/test_cova3d_numerical_amendment_n1.py",
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
        "Historical N1 test unexpectedly passes after N2. "
        "Do not alter the test; send me this output."
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


if "test_state_after_n1" not in failure_text:

    raise RuntimeError(
        "The failing N1 test is not the expected mutable-state test."
    )


print()
print(
    "✓ Expected N1 scope failure reproduced  : YES"
)

print(
    "✓ N1 numerical policy implicated        : NO"
)


# ==========================================================================================
# 4. REPAIR N1 TEST SCOPE ONLY
# ==========================================================================================

heading(
    "STEP 2/8 — REPAIR HISTORICAL N1 TEST SCOPE"
)


n1_test_source = r'''
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_n1():

    return yaml.safe_load(
        (
            ROOT
            / "configs/"
            "cova3d_numerical_amendment_N1_amp.yaml"
        ).read_text(
            encoding="utf-8"
        )
    )


def test_n1_is_frozen():

    cfg = load_n1()

    assert (
        cfg[
            "status"
        ]
        == "PROSPECTIVELY_FROZEN_BEFORE_FIRST_SUCCESSFUL_COVA_OPTIMIZER_STEP"
    )

    assert (
        cfg[
            "amendment_id"
        ]
        == "N1_FP16_STABLE_LOSS_SCALING"
    )

    assert (
        cfg[
            "protocol_before"
        ]
        == "COVA3D_1.0+A1+A1.1+A1.2"
    )

    assert (
        cfg[
            "protocol_after"
        ]
        == "COVA3D_1.0+A1+A1.1+A1.2+N1"
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
            "amp_policy"
        ][
            "gradient_clip_only_after_finite_unscale"
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


def test_n1_historical_provenance_is_immutable():

    """Validate the historical N1 artifact, not mutable CURRENT project state."""

    cfg = load_n1()

    assert (
        cfg[
            "diagnostic"
        ][
            "diagnosis"
        ]
        == "TRANSIENT_FP16_LOSS_SCALE_OVERFLOW"
    )

    assert (
        cfg[
            "diagnostic"
        ][
            "largest_observed_finite_scale"
        ]
        == 4096.0
    )

    assert (
        cfg[
            "diagnostic"
        ][
            "scale_1_finite"
        ]
        is True
    )

    assert (
        cfg[
            "scientifically_unchanged"
        ][
            "architecture"
        ]
        is True
    )

    assert (
        cfg[
            "scientifically_unchanged"
        ][
            "loss"
        ]
        is True
    )

    assert (
        cfg[
            "scientifically_unchanged"
        ][
            "annotation_protocol"
        ]
        is True
    )
'''


write_text(
    N1_TEST_PATH,
    n1_test_source,
)


print(
    "✓ Immutable N1 config still tested      : YES"
)

print(
    "✓ Mutable CURRENT-state assertion       : REMOVED"
)

print(
    "✓ N1 scientific policy changed         : NO"
)

print(
    "✓ N2 scientific policy changed         : NO"
)


# ==========================================================================================
# 5. COMBINED REGRESSION SUITE
# ==========================================================================================

heading(
    "STEP 3/8 — RUN FULL RELEVANT REGRESSION SUITE"
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
        "Combined N1/N2 regression suite still fails."
    )


print()
print(
    "✓ Combined regression suite            : PASS"
)


# ==========================================================================================
# 6. DIRECT NUMERICAL N2 SMOKE TEST
# ==========================================================================================

heading(
    "STEP 4/8 — DIRECT N2 NUMERICAL SMOKE TEST"
)


sys.modules.pop(
    "cora_lung.losses.partial",
    None,
)


importlib.invalidate_caches()


from cora_lung.losses.partial import (
    partial_dice,
    partial_dice_fp32,
    partial_segmentation_loss_n2,
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


if observed.dtype != torch.float32:

    raise RuntimeError(
        "N2 Dice output is not FP32."
    )


if not torch.allclose(
    observed,
    reference,
    atol=1e-7,
    rtol=1e-7,
):

    raise RuntimeError(
        "N2 Dice differs from frozen FP32 reference equation."
    )


observed.backward()


if not bool(
    torch.isfinite(
        logits.grad
    ).all()
):

    raise RuntimeError(
        "N2 direct smoke-test gradients are non-finite."
    )


absolute_difference = abs(
    float(
        observed.detach()
    )
    - float(
        reference.detach()
    )
)


print(
    "✓ Observed N2 Dice                     :",
    float(
        observed.detach()
    ),
)

print(
    "✓ FP32 reference Dice                  :",
    float(
        reference.detach()
    ),
)

print(
    "✓ Absolute difference                  :",
    absolute_difference,
)

print(
    "✓ N2 output dtype                      :",
    observed.dtype,
)

print(
    "✓ N2 gradient finite                   : YES"
)

print(
    "✓ optimizer.step() calls               : 0"
)


# ==========================================================================================
# 7. SOURCE CAPTURE
# ==========================================================================================

heading(
    "STEP 5/8 — CAPTURE NOTEBOOK SOURCES"
)


scripts_dir = (
    REPO
    / "scripts/code_blocks"
)


scripts_dir.mkdir(
    parents=True,
    exist_ok=True,
)


n2_source_path = (
    scripts_dir
    / "block09e_n2_lock_fp32_sparse_dice.py"
)


diag_source_path = (
    scripts_dir
    / "block09e_n2_test_diag.py"
)


finalize_source_path = (
    scripts_dir
    / "block09e_n2_finalize.py"
)


n2_source_capture = (
    "NOT_AVAILABLE"
)


diag_source_capture = (
    "NOT_AVAILABLE"
)


finalize_source_capture = (
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
            "# COVA-3D — BLOCK 09E-N2-LOCK"
            in cell
            and "Prospective Numerical Amendment N2"
            in cell
        ):

            n2_source_path.write_text(
                cell.rstrip()
                + "\n",
                encoding="utf-8",
            )

            n2_source_capture = (
                "PASS"
            )

            break


    for cell in reversed(
        prior_history
    ):

        if (
            "# COVA-3D — BLOCK 09E-N2-TEST-DIAG"
            in cell
            and "Diagnose N2 Regression-Test Failure"
            in cell
        ):

            diag_source_path.write_text(
                cell.rstrip()
                + "\n",
                encoding="utf-8",
            )

            diag_source_capture = (
                "PASS"
            )

            break


    current_cell = history[
        -1
    ]


    if (
        "# COVA-3D — BLOCK 09E-N2-FINALIZE"
        in current_cell
    ):

        finalize_source_path.write_text(
            current_cell.rstrip()
            + "\n",
            encoding="utf-8",
        )

        finalize_source_capture = (
            "PASS"
        )


except Exception:

    pass


print(
    "Original N2 source capture             :",
    n2_source_capture,
)

print(
    "N2 test diagnostic source capture      :",
    diag_source_capture,
)

print(
    "Finalize source capture                :",
    finalize_source_capture,
)


# ==========================================================================================
# 8. UPDATE N2 AUDIT / WRITE TEST-SCOPE AUDIT
# ==========================================================================================

heading(
    "STEP 6/8 — FINALIZE N2 PROVENANCE"
)


n2_audit = json.loads(
    N2_AUDIT_PATH.read_text(
        encoding="utf-8"
    )
)


if n2_audit.get(
    "status"
) != "PASS":

    raise RuntimeError(
        "Existing N2 scientific audit is not PASS."
    )


n2_audit[
    "source_capture"
] = (
    n2_source_capture
)


n2_audit[
    "regression_suite_final_status"
] = (
    "PASS_AFTER_HISTORICAL_N1_TEST_SCOPE_FIX"
)


n2_audit[
    "regression_test_scope_fix"
] = {
    "scientific_protocol_changed":
        False,

    "N1_policy_changed":
        False,

    "N2_policy_changed":
        False,

    "reason":
        (
            "Historical N1 regression test asserted mutable current project "
            "state. The repaired test validates immutable N1 configuration "
            "and provenance instead."
        ),

    "finalized_at_utc":
        NOW_ISO,
}


write_json(
    N2_AUDIT_PATH,
    n2_audit,
)


scope_audit_path = (
    REPO
    / "experiments/audits/"
    "block09e_n2_regression_scope_fix.json"
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

        "root_cause_of_original_regression_failure":
            (
                "Historical N1 test incorrectly required mutable current "
                "project state to remain at N1 after N2."
            ),

        "pre_fix_N1_return_code":
            int(
                pre_fix.returncode
            ),

        "pre_fix_failure_test":
            "test_state_after_n1",

        "repair":
            (
                "Replace current-state assertions with immutable N1 "
                "configuration/provenance assertions."
            ),

        "scientific_changes":
            0,

        "model_changes":
            0,

        "loss_changes":
            0,

        "annotation_changes":
            0,

        "optimizer_changes":
            0,

        "optimizer_steps":
            0,

        "dense_masks_accessed":
            0,

        "final_outer_cv_access":
            0,

        "combined_regression_suite":
            "PASS",

        "direct_N2_smoke_test":
            {
                "observed_dtype":
                    str(
                        observed.dtype
                    ),

                "observed_value":
                    float(
                        observed.detach()
                    ),

                "reference_value":
                    float(
                        reference.detach()
                    ),

                "absolute_difference":
                    absolute_difference,

                "gradient_finite":
                    True,
            },

        "source_capture":
            {
                "N2":
                    n2_source_capture,

                "test_diagnostic":
                    diag_source_capture,

                "finalization":
                    finalize_source_capture,
            },

        "next_block":
            "09E-SANITY-FIT-R2",

        "generated_at_utc":
            NOW_ISO,
    },
)


# ==========================================================================================
# 9. FINALIZE STATE
# ==========================================================================================

cova_state = json.loads(
    COVA_STATE_PATH.read_text(
        encoding="utf-8"
    )
)


cova_state[
    "N2_regression_tests"
] = (
    "PASS"
)


cova_state[
    "N2_regression_scope_fix"
] = (
    "HISTORICAL_N1_MUTABLE_STATE_ASSERTION_REMOVED"
)


cova_state[
    "next_block"
] = (
    "09E-SANITY-FIT-R2"
)


cova_state[
    "next_action"
] = (
    "Restart clean C100_COH sanity fit from original initialization "
    "under N1 + N2. Do not reuse R1 state."
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
    "cova3d_N2_regression_tests"
] = (
    "PASS"
)


project_state[
    "next_action"
] = (
    "Run 09E-SANITY-FIT-R2 only."
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
# 10. FINAL TEST SUITE AFTER STATE WRITES
# ==========================================================================================

heading(
    "STEP 7/8 — FINAL REGRESSION CHECK"
)


final_tests = sh(
    [
        sys.executable,
        "-m",
        "pytest",

        "tests/test_partial_losses.py",
        "tests/test_cova3d_baseline_lock.py",
        "tests/test_cova3d_numerical_amendment_n1.py",
        "tests/test_cova3d_numerical_amendment_n2.py",

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
        "Final regression suite failed after N2 state finalization."
    )


print()
print(
    "✓ Final regression suite               : PASS"
)


# ==========================================================================================
# 11. NORMALIZE TEXT BEFORE HASH MANIFEST
# ==========================================================================================

text_paths = [
    N1_TEST_PATH,
    N2_TEST_PATH,
    PARTIAL_LOSS_PATH,
    N2_CONFIG_PATH,
    N2_DIAG_PATH,
    N2_AUDIT_PATH,
    scope_audit_path,
    COVA_STATE_PATH,
    PROJECT_STATE_PATH,
]


for optional in [
    n2_source_path,
    diag_source_path,
    finalize_source_path,
]:

    if optional.exists():

        text_paths.append(
            optional
        )


for path in text_paths:

    path = Path(
        path
    )

    if not path.exists():

        continue


    content = path.read_text(
        encoding="utf-8"
    )


    path.write_text(
        content.rstrip()
        + "\n",
        encoding="utf-8",
    )


# ==========================================================================================
# 12. REGENERATE REPOSITORY MANIFEST LAST
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
            "09E-N2-LOCK",

        "finalization_block":
            BLOCK,

        "active_track":
            "COVA3D",

        "effective_protocol":
            EXPECTED_PROTOCOL,

        "numerical_amendments":
            [
                "N1",
                "N2",
            ],

        "N2_status":
            "FROZEN_AND_REGRESSION_VERIFIED",

        "N2_regression_suite":
            "PASS",

        "R1_successful_updates_executed":
            233,

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
# 13. COMMIT / PUSH
# ==========================================================================================

heading(
    "STEP 8/8 — COMMIT FINALIZED N2"
)


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

    "experiments/audits/"
    "block09e_n2_regression_scope_fix.json",

    "src/cora_lung/losses/"
    "partial.py",

    "tests/"
    "test_cova3d_numerical_amendment_n1.py",

    "tests/"
    "test_cova3d_numerical_amendment_n2.py",
]


for optional_relative in [
    "scripts/code_blocks/block09e_n2_lock_fp32_sparse_dice.py",
    "scripts/code_blocks/block09e_n2_test_diag.py",
    "scripts/code_blocks/block09e_n2_finalize.py",
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


print(
    "✓ git diff --cached --check             : PASS"
)


staged = sh(
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
    for item in staged
):

    raise RuntimeError(
        "Runtime model/checkpoint entered normal Git."
    )


print(
    "✓ Runtime checkpoints staged            : NO"
)


# Ensure tracked files have no remaining unstaged edits.
unstaged = sh(
    [
        "git",
        "diff",
        "--name-only",
    ]
).stdout.strip()


if unstaged:

    raise RuntimeError(
        "Tracked unstaged changes remain:\n"
        + unstaged
    )


sh(
    [
        "git",
        "commit",
        "-m",
        "protocol: finalize COVA-3D FP32 sparse-Dice amendment N2",
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
        "Repository is not clean after N2 finalization:\n"
        + final_status
    )


# ==========================================================================================
# 14. FINAL REPORT
# ==========================================================================================

heading(
    "COVA-3D BLOCK 09E-N2-FINALIZE — FINAL REPORT"
)


print(
    "Starting committed HEAD                :",
    head[:12],
)

print(
    "Final N2 commit                        :",
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
    "ORIGINAL REGRESSION FAILURE"
)

print(
    "---------------------------"
)

print(
    "N2 implementation failure              : NO"
)

print(
    "N2 numerical failure                   : NO"
)

print(
    "Historical N1 test-scope defect        : YES"
)

print(
    "Repair                                 : immutable N1 artifact test"
)

print()

print(
    "N2 NUMERICS"
)

print(
    "-----------"
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
    "Dice epsilon                           : 1e-6"
)

print(
    "FP32-reference absolute difference     :",
    absolute_difference,
)

print(
    "Gradient finite                        : YES"
)

print()

print(
    "R1 PROVENANCE"
)

print(
    "-------------"
)

print(
    "Successful updates executed            : 233"
)

print(
    "Successful updates retained            : 0"
)

print(
    "Clean R2 retained updates              : 0"
)

print(
    "Dense outcomes opened                  : NO"
)

print(
    "Final outer-CV access                  : 0"
)

print()

print(
    "REGRESSION"
)

print(
    "----------"
)

print(
    "Original partial-loss tests            : PASS"
)

print(
    "Baseline tests                         : PASS"
)

print(
    "N1 immutable-policy tests              : PASS"
)

print(
    "N2 tests                               : PASS"
)

print(
    "Combined suite                         : PASS"
)

print()

print(
    "SOURCE CAPTURE"
)

print(
    "--------------"
)

print(
    "Original N2 block                      :",
    n2_source_capture,
)

print(
    "N2 diagnostic                          :",
    diag_source_capture,
)

print(
    "N2 finalization                        :",
    finalize_source_capture,
)

print()

print(
    "AUTHORIZATION"
)

print(
    "-------------"
)

print(
    "Effective protocol                     :",
    EXPECTED_PROTOCOL,
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
    "Dense sanity evaluation authorized     : NO"
)

print(
    "Factorial training authorized          : NO"
)

print()

print(
    "NEXT:"
)

print(
    "Send me this COMPLETE report."
)

print(
    "Do NOT start training manually."
)

print(
    "Next block will be 09E-SANITY-FIT-R2."
)

print(
    "=" * 132
)
