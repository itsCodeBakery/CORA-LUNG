# ==========================================================================================
# COVA-3D — BLOCK 09E-SANITY-FIT-R2-FINALIZE-V2
# Finalize the Already-Completed R2 Fit
#
# FIXES TWO NON-SCIENTIFIC VALIDATION ISSUES:
#
#   1. Runtime/repository epoch-log comparison:
#        old : pandas.DataFrame.equals()        [bit-exact]
#        new : semantic comparison
#              - integers exact
#              - strings exact
#              - floats rtol=1e-12, atol=1e-12
#
#      Observed mismatch:
#        epoch_seconds, one cell,
#        absolute difference = 3.55e-15
#
#   2. Historical N2 regression test:
#        old : requires mutable CURRENT state to remain at N2
#        new : validates immutable N2 config/audit
#
# R2 IS NOT RETRAINED.
#
# EXPECTED COMMITTED HEAD:
#   aa01153258c9
#
# NEXT AFTER SUCCESS:
#   09E-GEOM-OP-DIAG
#
# IMPORTANT:
#   Dense development outcomes remain SEALED.
#   Factorial training remains LOCKED.
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
    "09E-SANITY-FIT-R2-FINALIZE-V2"
)

SCIENTIFIC_BLOCK = (
    "09E-SANITY-FIT-R2"
)

EFFECTIVE_PROTOCOL = (
    "COVA3D_1.0+A1+A1.1+A1.2+N1+N2"
)

NEXT_BLOCK = (
    "09E-GEOM-OP-DIAG"
)

FLOAT_RTOL = (
    1e-12
)

FLOAT_ATOL = (
    1e-12
)

EXPECTED_EPOCHS = (
    20
)

EXPECTED_SUCCESSFUL_STEPS = (
    1000
)

R2_ROOT = Path(
    "/kaggle/working/cova3d_sanity_fit_R2_v1_0"
)

RUNTIME_LOG = (
    R2_ROOT
    / "epoch_log.csv"
)

RUNTIME_STATE = (
    R2_ROOT
    / "run_state.json"
)

FINAL_MODEL = (
    R2_ROOT
    / "final_model.pt"
)

REPO_LOG = (
    REPO
    / "data/manifests/"
    "cova3d_sanity_fit_R2_epoch_log_v1_0.csv"
)

R2_AUDIT_PATH = (
    REPO
    / "experiments/audits/"
    "block09e_sanity_fit_R2.json"
)

N2_AUDIT_PATH = (
    REPO
    / "experiments/audits/"
    "block09e_n2_lock_fp32_dice.json"
)

N2_CONFIG_PATH = (
    REPO
    / "configs/"
    "cova3d_numerical_amendment_N2_fp32_dice.yaml"
)

N2_TEST_PATH = (
    REPO
    / "tests/"
    "test_cova3d_numerical_amendment_n2.py"
)

R2_TEST_PATH = (
    REPO
    / "tests/"
    "test_cova3d_sanity_fit_R2.py"
)

COVA_STATE_PATH = (
    REPO
    / "COVA3D_STATE.json"
)

PROJECT_STATE_PATH = (
    REPO
    / "PROJECT_STATE.json"
)

FIGURE_PATHS = [
    REPO
    / "figures/audit/"
    "fig_cova3d_sanity_R2_training_curve_v1_0.png",

    REPO
    / "figures/audit/"
    "fig_cova3d_sanity_R2_training_curve_v1_0.pdf",

    REPO
    / "figures/audit/"
    "fig_cova3d_sanity_R2_amp_scale_v1_0.png",

    REPO
    / "figures/audit/"
    "fig_cova3d_sanity_R2_amp_scale_v1_0.pdf",
]

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
        "/tmp/cova3d_git_askpass_r2_finalize_v2.sh"
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


def semantic_frame_compare(
    left,
    right,
    *,
    rtol=1e-12,
    atol=1e-12,
):

    if list(
        left.columns
    ) != list(
        right.columns
    ):

        return {
            "pass":
                False,

            "reason":
                "COLUMN_MISMATCH",

            "details":
                [],
        }


    if left.shape != right.shape:

        return {
            "pass":
                False,

            "reason":
                "SHAPE_MISMATCH",

            "details":
                [],
        }


    details = []


    for column in left.columns:

        a = left[
            column
        ]

        b = right[
            column
        ]


        a_numeric = pd.api.types.is_numeric_dtype(
            a
        )

        b_numeric = pd.api.types.is_numeric_dtype(
            b
        )


        if a_numeric != b_numeric:

            return {
                "pass":
                    False,

                "reason":
                    "DTYPE_CLASS_MISMATCH",

                "details":
                    [
                        {
                            "column":
                                column,

                            "left_dtype":
                                str(
                                    a.dtype
                                ),

                            "right_dtype":
                                str(
                                    b.dtype
                                ),
                        }
                    ],
            }


        if a_numeric:

            aa = a.to_numpy(
                dtype=np.float64
            )

            bb = b.to_numpy(
                dtype=np.float64
            )


            close = np.isclose(
                aa,
                bb,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            )


            exact = (
                np.equal(
                    aa,
                    bb,
                )
                | (
                    np.isnan(
                        aa
                    )
                    & np.isnan(
                        bb
                    )
                )
            )


            exact_mismatches = int(
                (
                    ~exact
                ).sum()
            )


            tolerance_mismatches = int(
                (
                    ~close
                ).sum()
            )


            finite = (
                np.isfinite(
                    aa
                )
                & np.isfinite(
                    bb
                )
            )


            if finite.any():

                max_abs_difference = float(
                    np.max(
                        np.abs(
                            aa[
                                finite
                            ]
                            - bb[
                                finite
                            ]
                        )
                    )
                )

            else:

                max_abs_difference = (
                    0.0
                )


            details.append(
                {
                    "column":
                        column,

                    "kind":
                        "numeric",

                    "exact_mismatches":
                        exact_mismatches,

                    "tolerance_mismatches":
                        tolerance_mismatches,

                    "max_abs_difference":
                        max_abs_difference,
                }
            )


            if tolerance_mismatches > 0:

                return {
                    "pass":
                        False,

                    "reason":
                        "NUMERIC_VALUE_MISMATCH",

                    "details":
                        details,
                }


        else:

            aa = a.astype(
                str
            ).to_numpy()

            bb = b.astype(
                str
            ).to_numpy()


            mismatches = int(
                (
                    aa
                    != bb
                ).sum()
            )


            details.append(
                {
                    "column":
                        column,

                    "kind":
                        "non_numeric",

                    "exact_mismatches":
                        mismatches,
                }
            )


            if mismatches > 0:

                return {
                    "pass":
                        False,

                    "reason":
                        "NON_NUMERIC_VALUE_MISMATCH",

                    "details":
                        details,
                }


    return {
        "pass":
            True,

        "reason":
            "SEMANTICALLY_EQUAL",

        "details":
            details,
    }


# ==========================================================================================
# 2. VERIFY COMMITTED HEAD / R2 ARTIFACTS
# ==========================================================================================

heading(
    "COVA-3D 09E-SANITY-FIT-R2-FINALIZE-V2 — VERIFY COMPLETED R2"
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


for path in [
    RUNTIME_LOG,
    RUNTIME_STATE,
    FINAL_MODEL,
    REPO_LOG,
    R2_AUDIT_PATH,
    R2_TEST_PATH,
    COVA_STATE_PATH,
    PROJECT_STATE_PATH,
]:

    if not path.exists():

        raise RuntimeError(
            "Required R2 artifact missing:\n"
            + str(
                path
            )
        )


for path in FIGURE_PATHS:

    if not path.exists():

        raise RuntimeError(
            "Expected R2 figure missing:\n"
            + str(
                path
            )
        )


print(
    "✓ Required runtime artifacts           : PRESENT"
)

print(
    "✓ Required repository artifacts        : PRESENT"
)

print(
    "✓ R2 publication/audit figures         : PRESENT"
)


# ==========================================================================================
# 3. VERIFY RUNTIME COMPLETION / FINAL MODEL
# ==========================================================================================

heading(
    "STEP 1/9 — VERIFY R2 RUNTIME"
)


runtime_state = json.loads(
    RUNTIME_STATE.read_text(
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
        "completed_epoch",
        -1,
    )
) != (
    EXPECTED_EPOCHS
    - 1
):

    raise RuntimeError(
        "R2 runtime does not contain 20 completed epochs."
    )


if int(
    runtime_state.get(
        "successful_optimizer_steps",
        -1,
    )
) != EXPECTED_SUCCESSFUL_STEPS:

    raise RuntimeError(
        "R2 runtime does not contain exactly 1000 successful optimizer steps."
    )


model_sha = sha256_file(
    FINAL_MODEL
)


if model_sha != str(
    runtime_state.get(
        "final_model_sha256"
    )
):

    raise RuntimeError(
        "R2 final-model checksum does not match run_state."
    )


r2_audit = json.loads(
    R2_AUDIT_PATH.read_text(
        encoding="utf-8"
    )
)


if model_sha != str(
    r2_audit.get(
        "final_model_sha256"
    )
):

    raise RuntimeError(
        "R2 final-model checksum does not match R2 audit."
    )


print(
    "✓ Runtime status                       : FIT_COMPLETE"
)

print(
    "✓ Completed epochs                     : 20"
)

print(
    "✓ Successful optimizer steps           : 1000"
)

print(
    "✓ Final-model SHA256                   :",
    model_sha,
)


# ==========================================================================================
# 4. TOLERANCE-AWARE EPOCH-LOG VALIDATION
# ==========================================================================================

heading(
    "STEP 2/9 — SEMANTIC RUNTIME / REPOSITORY LOG COMPARISON"
)


runtime_df = (
    pd.read_csv(
        RUNTIME_LOG
    )
    .sort_values(
        "epoch"
    )
    .reset_index(
        drop=True
    )
)


repo_df = (
    pd.read_csv(
        REPO_LOG
    )
    .sort_values(
        "epoch"
    )
    .reset_index(
        drop=True
    )
)


comparison = semantic_frame_compare(
    runtime_df,
    repo_df,
    rtol=FLOAT_RTOL,
    atol=FLOAT_ATOL,
)


if not comparison[
    "pass"
]:

    print(
        json.dumps(
            comparison,
            indent=2,
        )
    )

    raise RuntimeError(
        "Runtime and repository R2 logs differ beyond frozen "
        "round-trip tolerance."
    )


numeric_exact_differences = [
    row
    for row in comparison[
        "details"
    ]
    if (
        row.get(
            "kind"
        )
        == "numeric"
        and int(
            row.get(
                "exact_mismatches",
                0,
            )
        )
        > 0
    )
]


print(
    "✓ Shapes                               : IDENTICAL"
)

print(
    "✓ Columns                              : IDENTICAL"
)

print(
    "✓ Non-numeric fields                   : EXACT"
)

print(
    "✓ Numeric fields within tolerance      : PASS"
)

print(
    "✓ Float rtol                           :",
    FLOAT_RTOL,
)

print(
    "✓ Float atol                           :",
    FLOAT_ATOL,
)


if numeric_exact_differences:

    print()
    print(
        "Exact-bit differences tolerated:"
    )

    print(
        pd.DataFrame(
            numeric_exact_differences
        ).to_string(
            index=False
        )
    )


max_semantic_abs_difference = max(
    [
        float(
            row.get(
                "max_abs_difference",
                0.0,
            )
        )
        for row in comparison[
            "details"
        ]
        if row.get(
            "kind"
        )
        == "numeric"
    ]
    or [
        0.0
    ]
)


print()
print(
    "✓ Maximum absolute numeric difference  :",
    max_semantic_abs_difference,
)


if max_semantic_abs_difference > FLOAT_ATOL:

    # rtol can legitimately permit larger magnitude-dependent differences,
    # but this run is known to differ only at ~3.55e-15.
    print(
        "NOTE: maximum difference exceeds atol but remains within rtol+atol."
    )


# ==========================================================================================
# 5. VERIFY SCIENTIFIC INVARIANTS
# ==========================================================================================

heading(
    "STEP 3/9 — VERIFY SCIENTIFIC INVARIANTS"
)


if len(
    repo_df
) != EXPECTED_EPOCHS:

    raise RuntimeError(
        "Expected 20 R2 epoch rows."
    )


if repo_df[
    "epoch"
].astype(
    int
).tolist() != list(
    range(
        EXPECTED_EPOCHS
    )
):

    raise RuntimeError(
        "R2 epoch sequence mismatch."
    )


if int(
    repo_df.iloc[
        -1
    ][
        "successful_optimizer_steps_total"
    ]
) != EXPECTED_SUCCESSFUL_STEPS:

    raise RuntimeError(
        "R2 final successful-step count mismatch."
    )


if int(
    repo_df[
        "logical_microbatches"
    ].sum()
) != 2000:

    raise RuntimeError(
        "R2 total logical microbatch count mismatch."
    )


if int(
    repo_df[
        "logical_optimizer_steps"
    ].sum()
) != 1000:

    raise RuntimeError(
        "R2 logical optimizer-step count mismatch."
    )


total_overflows = int(
    repo_df[
        "amp_overflow_events"
    ].sum()
)


epochs_with_overflow = int(
    (
        repo_df[
            "amp_overflow_events"
        ]
        > 0
    ).sum()
)


source_totals = {
    "foreground":
        int(
            repo_df[
                "foreground_centered"
            ].sum()
        ),

    "background":
        int(
            repo_df[
                "background_centered"
            ].sum()
        ),

    "uniform":
        int(
            repo_df[
                "uniform_crop"
            ].sum()
        ),
}


if sum(
    source_totals.values()
) != 2000:

    raise RuntimeError(
        "R2 patch-source count does not total 2000."
    )


for patient_column in [
    "coronacases_004_samples",
    "coronacases_008_samples",
    "radiopaedia_14_85914_0_samples",
    "radiopaedia_27_86410_0_samples",
]:

    if not (
        repo_df[
            patient_column
        ]
        == 25
    ).all():

        raise RuntimeError(
            "Patient-uniform schedule mismatch."
        )


first3 = float(
    repo_df.head(
        3
    )[
        "mean_loss"
    ].mean()
)


last3 = float(
    repo_df.tail(
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
    repo_df[
        "zero_label_microbatches"
    ].sum()
)


zero_fraction = float(
    total_zero
    / 2000
)


minimum_amp_scale = float(
    repo_df[
        "amp_scale_min"
    ].min()
)


maximum_amp_scale = float(
    repo_df[
        "amp_scale_max"
    ].max()
)


final_amp_scale = float(
    repo_df.iloc[
        -1
    ][
        "amp_scale_end"
    ]
)


training_loss_gate_pass = bool(
    loss_ratio
    <= 0.95
)


zero_label_gate_pass = bool(
    zero_fraction
    <= 0.30
)


print(
    "✓ Epochs                               : 20"
)

print(
    "✓ Successful optimizer steps           : 1000"
)

print(
    "✓ Logical microbatches                 : 2000"
)

print(
    "✓ Logical optimizer steps              : 1000"
)

print(
    "✓ Patch-source FG/BG/U                 :",
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
    "✓ AMP overflow events                  :",
    total_overflows,
)

print(
    "✓ Epochs with overflow                 :",
    epochs_with_overflow,
)

print(
    "✓ AMP min / max / final                :",
    minimum_amp_scale,
    "/",
    maximum_amp_scale,
    "/",
    final_amp_scale,
)

print()
print(
    "First-3 mean loss                      :",
    f"{first3:.12f}",
)

print(
    "Last-3 mean loss                       :",
    f"{last3:.12f}",
)

print(
    "Last3 / First3 ratio                   :",
    f"{loss_ratio:.6f}",
)

print(
    "Training-loss sanity subcriterion      :",
    "PASS"
    if training_loss_gate_pass
    else "FAIL",
)

print(
    "Zero-label fraction                    :",
    f"{zero_fraction:.6f}",
)

print(
    "Zero-label sanity subcriterion         :",
    "PASS"
    if zero_label_gate_pass
    else "FAIL",
)


# ==========================================================================================
# 6. VERIFY FIREWALL / CURRENT R2 STATE
# ==========================================================================================

heading(
    "STEP 4/9 — VERIFY OUTCOME FIREWALL"
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
) != SCIENTIFIC_BLOCK:

    raise RuntimeError(
        "Working-tree state does not record completed R2."
    )


if cova_state.get(
    "effective_protocol"
) != EFFECTIVE_PROTOCOL:

    raise RuntimeError(
        "Effective protocol mismatch."
    )


if int(
    cova_state.get(
        "optimizer_steps_retained_for_current_frozen_run",
        -1,
    )
) != EXPECTED_SUCCESSFUL_STEPS:

    raise RuntimeError(
        "R2 retained-step state mismatch."
    )


if cova_state.get(
    "dense_outcomes_opened_in_cova3d"
) is not False:

    raise RuntimeError(
        "Dense outcomes unexpectedly opened."
    )


if int(
    cova_state.get(
        "final_outer_cv_access_in_cova3d",
        -1,
    )
) != 0:

    raise RuntimeError(
        "Final outer-CV unexpectedly accessed."
    )


if cova_state.get(
    "factorial_training_authorized"
) is not False:

    raise RuntimeError(
        "Factorial training unexpectedly authorized."
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
        "R2 audit reports final-CV access."
    )


print(
    "✓ Dense lesion masks accessed          : 0"
)

print(
    "✓ Dense lung masks accessed            : 0"
)

print(
    "✓ Dense development evaluation         : NOT RUN"
)

print(
    "✓ Final outer-CV access                : 0"
)

print(
    "✓ Factorial training                   : LOCKED"
)


# ==========================================================================================
# 7. REPAIR HISTORICAL N2 TEST SCOPE
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
            "factorial_training_authorized"
        ]
        is False
    )


def test_n2_historical_audit_is_immutable():

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
    "✓ N2 functional tests retained         : YES"
)

print(
    "✓ N2 immutable protocol tested         : YES"
)

print(
    "✓ N2 immutable audit tested            : YES"
)

print(
    "✓ Mutable-current-state assertion      : REMOVED"
)

print(
    "✓ Scientific N2 policy changed         : NO"
)


# ==========================================================================================
# 8. RUN FULL REGRESSION SUITE
# ==========================================================================================

heading(
    "STEP 6/9 — RUN COMPLETE REGRESSION SUITE"
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


tests = sh(
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
    tests.stdout.rstrip()
)


if tests.stderr.strip():

    print()
    print(
        tests.stderr.rstrip()
    )


if tests.returncode != 0:

    raise RuntimeError(
        "R2 final regression suite failed.\n"
        "Do NOT rerun training."
    )


print()
print(
    "✓ Full regression suite                : PASS"
)


# ==========================================================================================
# 9. SOURCE CAPTURE
# ==========================================================================================

heading(
    "STEP 7/9 — CAPTURE R2 / FINALIZER SOURCE"
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


finalize_source_path = (
    scripts_dir
    / "block09e_sanity_fit_R2_finalize_v2.py"
)


r2_source_capture = (
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


    prior = (
        history[
            :-1
        ]
        if history
        else []
    )


    for cell in reversed(
        prior
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


    current = history[
        -1
    ]


    if (
        "# COVA-3D — BLOCK 09E-SANITY-FIT-R2-FINALIZE-V2"
        in current
    ):

        finalize_source_path.write_text(
            current.rstrip()
            + "\n",
            encoding="utf-8",
        )

        finalize_source_capture = (
            "PASS"
        )


except Exception:

    pass


print(
    "R2 source capture                      :",
    r2_source_capture,
)

print(
    "Finalizer source capture               :",
    finalize_source_capture,
)


# ==========================================================================================
# 10. FREEZE R2 FINALIZATION / LOG-AUDIT PROVENANCE
# ==========================================================================================

heading(
    "STEP 8/9 — FREEZE FINALIZATION PROVENANCE"
)


log_audit_path = (
    REPO
    / "experiments/audits/"
    "block09e_R2_log_semantic_equivalence.json"
)


exact_difference_rows = [
    row
    for row in comparison[
        "details"
    ]
    if int(
        row.get(
            "exact_mismatches",
            0,
        )
    )
    > 0
]


write_json(
    log_audit_path,
    {
        "project":
            "COVA-3D",

        "block":
            BLOCK,

        "status":
            "PASS",

        "comparison":
            "TOLERANCE_AWARE_SEMANTIC",

        "rtol":
            FLOAT_RTOL,

        "atol":
            FLOAT_ATOL,

        "runtime_csv_sha256":
            sha256_file(
                RUNTIME_LOG
            ),

        "repository_csv_sha256":
            sha256_file(
                REPO_LOG
            ),

        "raw_files_identical":
            (
                RUNTIME_LOG.read_bytes()
                == REPO_LOG.read_bytes()
            ),

        "semantic_equality":
            True,

        "max_absolute_numeric_difference":
            max_semantic_abs_difference,

        "exact_difference_columns":
            exact_difference_rows,

        "scientific_invariants":
            {
                "epochs":
                    EXPECTED_EPOCHS,

                "successful_optimizer_steps":
                    EXPECTED_SUCCESSFUL_STEPS,

                "logical_microbatches":
                    2000,

                "logical_optimizer_steps":
                    1000,

                "AMP_overflows":
                    total_overflows,

                "foreground_centered":
                    source_totals[
                        "foreground"
                    ],

                "background_centered":
                    source_totals[
                        "background"
                    ],

                "uniform":
                    source_totals[
                        "uniform"
                    ],

                "first3_mean_loss":
                    first3,

                "last3_mean_loss":
                    last3,

                "last3_div_first3":
                    loss_ratio,

                "final_amp_scale":
                    final_amp_scale,
            },

        "conclusion":
            (
                "Runtime and repository epoch logs are scientifically "
                "equivalent. The only observed discrepancy is CSV "
                "floating-point round-trip precision."
            ),

        "generated_at_utc":
            NOW_ISO,
    },
)


r2_audit = json.loads(
    R2_AUDIT_PATH.read_text(
        encoding="utf-8"
    )
)


r2_audit[
    "log_semantic_equivalence"
] = {
    "status":
        "PASS",

    "comparison":
        "numeric tolerance plus exact categorical/integer semantics",

    "rtol":
        FLOAT_RTOL,

    "atol":
        FLOAT_ATOL,

    "max_absolute_difference":
        max_semantic_abs_difference,
}


r2_audit[
    "regression_suite_status"
] = (
    "PASS"
)


r2_audit[
    "historical_N2_test_scope_fix"
] = (
    "PASS"
)


r2_audit[
    "training_loss_subcriterion"
] = {
    "criterion":
        "last3_first3_ratio <= 0.95",

    "observed":
        loss_ratio,

    "pass":
        training_loss_gate_pass,
}


r2_audit[
    "zero_label_subcriterion"
] = {
    "criterion":
        "zero_label_fraction <= 0.30",

    "observed":
        zero_fraction,

    "pass":
        zero_label_gate_pass,
}


r2_audit[
    "source_capture"
] = (
    r2_source_capture
)


r2_audit[
    "next_block"
] = (
    NEXT_BLOCK
)


r2_audit[
    "dense_sanity_evaluation_deferred"
] = (
    True
)


r2_audit[
    "dense_sanity_evaluation_deferral_reason"
] = (
    "Close the pre-outcome geometry-operationality audit for DIS versus FRG "
    "before opening dense development outcomes."
)


r2_audit[
    "finalized_at_utc"
] = (
    NOW_ISO
)


write_json(
    R2_AUDIT_PATH,
    r2_audit,
)


scope_audit_path = (
    REPO
    / "experiments/audits/"
    "block09e_R2_historical_N2_test_scope_fix.json"
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

        "scientific_changes":
            0,

        "additional_optimizer_steps":
            0,

        "dense_masks_accessed":
            0,

        "final_outer_cv_access":
            0,

        "defect":
            (
                "Historical N2 regression test required mutable CURRENT "
                "project state to remain at N2 after R2."
            ),

        "repair":
            (
                "Historical N2 test now validates immutable N2 configuration "
                "and audit artifacts."
            ),

        "full_regression_suite":
            "PASS",

        "generated_at_utc":
            NOW_ISO,
    },
)


# ==========================================================================================
# 11. UPDATE CURRENT STATE
# ==========================================================================================

cova_state = json.loads(
    COVA_STATE_PATH.read_text(
        encoding="utf-8"
    )
)


cova_state.update(
    {
        "last_completed_block":
            SCIENTIFIC_BLOCK,

        "last_finalization_block":
            BLOCK,

        "current_stage":
            "SANITY_FIT_R2_COMPLETE_GEOMETRY_OPERATIONALITY_AUDIT_PENDING",

        "R2_regression_tests":
            "PASS",

        "R2_log_semantic_equivalence":
            "PASS",

        "R2_training_loss_subcriterion":
            (
                "PASS"
                if training_loss_gate_pass
                else "FAIL"
            ),

        "R2_zero_label_subcriterion":
            (
                "PASS"
                if zero_label_gate_pass
                else "FAIL"
            ),

        "sanity_gate_status":
            "EVALUATION_PENDING",

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
            NEXT_BLOCK,

        "next_action":
            (
                "Run a structural, no-outcome geometry-operationality audit "
                "of the frozen six-cell sparse annotations before opening "
                "dense development outcomes."
            ),

        "updated_at_utc":
            NOW_ISO,
    }
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


project_state.update(
    {
        "last_completed_block":
            SCIENTIFIC_BLOCK,

        "last_finalization_block":
            BLOCK,

        "current_stage":
            "cova3d_geometry_operationality_audit_pending",

        "cova3d_R2_regression_tests":
            "PASS",

        "cova3d_R2_log_semantic_equivalence":
            "PASS",

        "cova3d_sanity_gate":
            "EVALUATION_PENDING",

        "cova3d_dense_outcomes_opened":
            False,

        "cova3d_factorial_training_authorized":
            False,

        "cova3d_final_outer_cv_access":
            0,

        "next_action":
            (
                "Run 09E-GEOM-OP-DIAG before dense sanity evaluation."
            ),

        "updated_at_utc":
            NOW_ISO,
    }
)


write_json(
    PROJECT_STATE_PATH,
    project_state,
)


# ==========================================================================================
# 12. RUN FINAL REGRESSION SUITE AFTER STATE WRITE
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


print()
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
        "Final regression suite failed after R2 finalization."
    )


print()
print(
    "✓ Post-state regression suite          : PASS"
)


# ==========================================================================================
# 13. NORMALIZE TEXT
# ==========================================================================================

text_paths = [
    N2_TEST_PATH,
    R2_TEST_PATH,
    REPO_LOG,
    R2_AUDIT_PATH,
    log_audit_path,
    scope_audit_path,
    COVA_STATE_PATH,
    PROJECT_STATE_PATH,
]


for optional in [
    r2_source_path,
    finalize_source_path,
]:

    if optional.exists():

        text_paths.append(
            optional
        )


for path in text_paths:

    content = Path(
        path
    ).read_text(
        encoding="utf-8"
    )

    Path(
        path
    ).write_text(
        content.rstrip()
        + "\n",
        encoding="utf-8",
    )


# ==========================================================================================
# 14. REPOSITORY MANIFEST — GENERATED LAST
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

        "completed_scientific_block":
            SCIENTIFIC_BLOCK,

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

        "R1_successful_updates_executed_nonretained":
            233,

        "R1_updates_retained":
            0,

        "R2_successful_optimizer_steps":
            EXPECTED_SUCCESSFUL_STEPS,

        "R2_final_model_sha256":
            model_sha,

        "R2_log_semantic_equivalence":
            "PASS",

        "R2_regression_suite":
            "PASS",

        "training_loss_subcriterion":
            (
                "PASS"
                if training_loss_gate_pass
                else "FAIL"
            ),

        "zero_label_subcriterion":
            (
                "PASS"
                if zero_label_gate_pass
                else "FAIL"
            ),

        "dense_outcomes_opened":
            False,

        "sanity_gate":
            "EVALUATION_PENDING",

        "factorial_training_authorized":
            False,

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

heading(
    "STEP 9/9 — COMMIT ALREADY-COMPLETED R2"
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
    "block09e_R2_log_semantic_equivalence.json",

    "experiments/audits/"
    "block09e_R2_historical_N2_test_scope_fix.json",

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
    "scripts/code_blocks/block09e_sanity_fit_R2_finalize_v2.py",
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
    path.endswith(
        ".pt"
    )
    for path in staged_files
):

    raise RuntimeError(
        "Runtime model/checkpoint must not enter Git."
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
        "Tracked unstaged files remain:\n"
        + unstaged_tracked
    )


print(
    "✓ git diff --cached --check            : PASS"
)

print(
    "✓ Runtime model staged                 : NO"
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
# 16. FINAL REPORT
# ==========================================================================================

heading(
    "COVA-3D 09E-SANITY-FIT-R2-FINALIZE-V2 — FINAL REPORT"
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
    "R2 SCIENTIFIC RUN"
)

print(
    "-----------------"
)

print(
    "Epochs                                 : 20"
)

print(
    "Successful optimizer steps             : 1000"
)

print(
    "Final-model SHA256                     :",
    model_sha,
)

print(
    "R1 weights retained                    : NO"
)

print(
    "R1 updates retained                    : 0"
)

print()

print(
    "LOG CONSISTENCY"
)

print(
    "---------------"
)

print(
    "Raw CSV files identical                :",
    RUNTIME_LOG.read_bytes()
    == REPO_LOG.read_bytes(),
)

print(
    "Semantic equality                      : PASS"
)

print(
    "Float rtol / atol                      :",
    FLOAT_RTOL,
    "/",
    FLOAT_ATOL,
)

print(
    "Maximum absolute numeric difference    :",
    max_semantic_abs_difference,
)

print(
    "Scientific invariant mismatch          : NONE"
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
    f"{first3:.12f}",
)

print(
    "Last-3 mean loss                       :",
    f"{last3:.12f}",
)

print(
    "Last3 / First3                         :",
    f"{loss_ratio:.6f}",
)

print(
    "Training-loss subcriterion             :",
    "PASS"
    if training_loss_gate_pass
    else "FAIL",
)

print(
    "Zero-label fraction                    :",
    f"{zero_fraction:.6f}",
)

print(
    "Zero-label subcriterion                :",
    "PASS"
    if zero_label_gate_pass
    else "FAIL",
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
    "AMP scale min / max / final            :",
    minimum_amp_scale,
    "/",
    maximum_amp_scale,
    "/",
    final_amp_scale,
)

print()

print(
    "REGRESSION"
)

print(
    "----------"
)

print(
    "Historical N2 mutable-state defect     : FIXED"
)

print(
    "Full relevant regression suite         : PASS"
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

print(
    "Factorial training authorized          : NO"
)

print()

print(
    "NEXT"
)

print(
    "----"
)

print(
    "Next block                             :",
    NEXT_BLOCK,
)

print(
    "Dense sanity evaluation                : DEFERRED"
)

print(
    "Reason                                 : verify operational distinction "
    "of frozen geometry interventions before opening outcomes"
)

print()

print(
    "Send me this COMPLETE report."
)

print(
    "Do NOT retrain R2."
)

print(
    "Do NOT open dense masks yet."
)

print(
    "=" * 132
)
