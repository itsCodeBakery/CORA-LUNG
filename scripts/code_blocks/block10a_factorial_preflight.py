# ==========================================================================================
# COVA-3D — BLOCK 10A-FACTORIAL-PREFLIGHT
#
# PURPOSE
# -------
# Prepare the real 72-fit primary factorial experiment.
#
# COMPLETED BEFORE THIS BLOCK
# ---------------------------
# ✓ A1.3 annotation methodology frozen
# ✓ 120 v1.3 sparse supervision artifacts frozen
# ✓ R2 sanity fit complete
# ✓ locked sanity gate PASS
# ✓ factorial FITTING authorized
#
# THIS BLOCK
# ----------
# 1. Verify the frozen 72-run factorial registry.
# 2. Verify paired design across all six conditions.
# 3. Verify all required v1.3 sparse artifacts.
# 4. Build CT-ONLY cache for the 16 final outer-CV cases.
# 5. Do NOT open infection masks or lung masks.
# 6. Create 12 paired execution batches:
#
#       4 folds × 3 seeds = 12 batches
#       6 conditions / batch
#
# 7. Freeze execution registry/config.
# 8. Commit + push.
#
# NO TRAINING in this block.
# NO optimizer.step().
# NO final dense outcomes.
#
# NEXT AFTER PASS
# ---------------
# 10B-FACTORIAL-BATCH-01
# = fold 0 / seed 17 / six conditions
# ==========================================================================================


from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone

import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap

import nibabel as nib
import numpy as np
import pandas as pd
import yaml

from nibabel.processing import resample_to_output
from tqdm.auto import tqdm
from kaggle_secrets import UserSecretsClient


# ==========================================================================================
# 0. CONSTANTS
# ==========================================================================================

REPO = Path(
    "/kaggle/working/CORA-LUNG"
)

EXPECTED_HEAD = (
    "07174465d9c5"
)

BLOCK = (
    "10A-FACTORIAL-PREFLIGHT"
)

NEXT_BLOCK = (
    "10B-FACTORIAL-BATCH-01"
)

EFFECTIVE_PROTOCOL = (
    "COVA3D_1.0+A1+A1.1+A1.2+A1.3+N1+N2"
)

DATASET_ROOT = Path(
    "/kaggle/input/datasets/andrewmvd/covid19-ct-scans"
)

RUN_REGISTRY_PATH = (
    REPO
    / "data/manifests/"
    "cova3d_final_factorial_run_registry_v1_0.csv"
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

GEOMETRY_PATH = (
    REPO
    / "data/manifests/"
    "cova3d_training_grid_geometry_v1_2.csv"
)

SPLIT_PATH = (
    REPO
    / "data/splits/"
    "primary_volume_split_registry.csv"
)

BASELINE_CONFIG_PATH = (
    REPO
    / "configs/"
    "cova3d_baseline_training_v1_0.yaml"
)

EXECUTION_LOCK_PATH = (
    REPO
    / "configs/"
    "cova3d_factorial_execution_lock_v1_0.yaml"
)

STATE_PATH = (
    REPO
    / "COVA3D_STATE.json"
)

PROJECT_STATE_PATH = (
    REPO
    / "PROJECT_STATE.json"
)

R2_EPOCH_LOG_PATH = (
    REPO
    / "data/manifests/"
    "cova3d_sanity_fit_R2_epoch_log_v1_0.csv"
)

CACHE_ROOT = Path(
    "/kaggle/working/cova3d_final16_ct_cache_v1_0"
)

CACHE_MANIFEST_RUNTIME = (
    CACHE_ROOT
    / "manifest.csv"
)

CACHE_MANIFEST_REPO = (
    REPO
    / "data/manifests/"
    "cova3d_final16_CT_cache_manifest_v1_0.csv"
)

AUDIT_PATH = (
    REPO
    / "experiments/audits/"
    "block10a_factorial_preflight.json"
)

TEST_PATH = (
    REPO
    / "tests/"
    "test_cova3d_factorial_preflight.py"
)

TARGET_SPACING_XYZ = (
    1.5,
    1.5,
    3.0,
)

TARGET_SPACING_ZYX = (
    3.0,
    1.5,
    1.5,
)

HU_MIN = (
    -1250.0
)

HU_MAX = (
    250.0
)

CONDITIONS = [
    "C50_COH",
    "C50_DIS",
    "C50_FRG",
    "C100_COH",
    "C100_DIS",
    "C100_FRG",
]

FOLDS = [
    0,
    1,
    2,
    3,
]

SEEDS = [
    17,
    29,
    43,
]

EXPECTED_RUNS = (
    72
)

EXPECTED_BATCHES = (
    12
)

RUNS_PER_BATCH = (
    6
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


def normalize_raw_hu(image):

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    image = np.clip(
        image,
        HU_MIN,
        HU_MAX,
    )

    output = (
        2.0
        * (
            image
            - HU_MIN
        )
        / (
            HU_MAX
            - HU_MIN
        )
        - 1.0
    )

    return np.clip(
        output,
        -1.0,
        1.0,
    ).astype(
        np.float32
    )


def normalize_prewindowed(image):

    image = np.asarray(
        image,
        dtype=np.float32,
    )

    image = np.clip(
        image,
        0.0,
        255.0,
    )

    output = (
        2.0
        * image
        / 255.0
        - 1.0
    )

    return np.clip(
        output,
        -1.0,
        1.0,
    ).astype(
        np.float32
    )


def parse_cases(value):

    value = str(
        value
    ).strip()

    if not value:

        return []

    return [
        item
        for item in value.split(
            ";"
        )
        if item
    ]


def git_auth():

    token = (
        UserSecretsClient()
        .get_secret(
            "pushCora"
        )
        .strip()
    )

    askpass = Path(
        "/tmp/cova3d_git_askpass_factorial_preflight.sh"
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


def push(
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

        raise RuntimeError(
            "Git push failed:\n"
            + (
                result.stderr
                or ""
            ).replace(
                token,
                "***TOKEN_REDACTED***",
            )
        )


# ==========================================================================================
# 2. VERIFY PROJECT STATE
# ==========================================================================================

heading(
    "COVA-3D 10A-FACTORIAL-PREFLIGHT — VERIFY STATE"
)


head = git(
    "rev-parse",
    "HEAD",
)


if not head.startswith(
    EXPECTED_HEAD
):

    raise RuntimeError(
        "Unexpected HEAD.\n"
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
        "Repository must be clean before factorial preflight:\n"
        + dirty
    )


state = json.loads(
    STATE_PATH.read_text(
        encoding="utf-8"
    )
)


project_state = json.loads(
    PROJECT_STATE_PATH.read_text(
        encoding="utf-8"
    )
)


if state.get(
    "last_completed_block"
) != "09E-SANITY-EVAL":

    raise RuntimeError(
        "Expected completed sanity evaluation."
    )


if state.get(
    "sanity_gate_status"
) != "PASS":

    raise RuntimeError(
        "Sanity gate is not PASS."
    )


if state.get(
    "factorial_training_authorized"
) is not True:

    raise RuntimeError(
        "Factorial fitting is not authorized."
    )


if state.get(
    "method_development_authorized"
) is not False:

    raise RuntimeError(
        "Method development must remain locked."
    )


if state.get(
    "final_outer_cv_outcomes_authorized"
) is not False:

    raise RuntimeError(
        "Final outer-CV outcomes must remain sealed."
    )


if int(
    state.get(
        "final_outer_cv_access_in_cova3d",
        -1,
    )
) != 0:

    raise RuntimeError(
        "Final outer-CV outcome access must remain zero."
    )


if state.get(
    "effective_protocol"
) != EFFECTIVE_PROTOCOL:

    raise RuntimeError(
        "Effective protocol mismatch."
    )


print(
    "✓ Git HEAD                             :",
    head[:12],
)

print(
    "✓ Sanity gate                          : PASS"
)

print(
    "✓ Factorial fitting                    : AUTHORIZED"
)

print(
    "✓ Method development                   : LOCKED"
)

print(
    "✓ Final outer-CV outcomes              : SEALED"
)

print(
    "✓ Final outer-CV outcome access        : 0"
)


# ==========================================================================================
# 3. VERIFY FROZEN 72-RUN REGISTRY
# ==========================================================================================

heading(
    "STEP 1/8 — VERIFY FROZEN 72-RUN FACTORIAL REGISTRY"
)


run_df = pd.read_csv(
    RUN_REGISTRY_PATH
)


if len(
    run_df
) != EXPECTED_RUNS:

    raise RuntimeError(
        "Expected exactly 72 frozen factorial runs."
    )


if run_df[
    "run_id"
].nunique() != EXPECTED_RUNS:

    raise RuntimeError(
        "Factorial run IDs are not unique."
    )


if set(
    run_df[
        "condition"
    ]
) != set(
    CONDITIONS
):

    raise RuntimeError(
        "Unexpected factorial condition set."
    )


if set(
    run_df[
        "outer_fold"
    ].astype(
        int
    )
) != set(
    FOLDS
):

    raise RuntimeError(
        "Unexpected outer-fold set."
    )


if set(
    run_df[
        "seed"
    ].astype(
        int
    )
) != set(
    SEEDS
):

    raise RuntimeError(
        "Unexpected seed set."
    )


condition_counts = (
    run_df[
        "condition"
    ]
    .value_counts()
    .to_dict()
)


for condition in CONDITIONS:

    if condition_counts.get(
        condition,
        0,
    ) != 12:

        raise RuntimeError(
            "Each condition must contain exactly 12 runs."
        )


if not (
    run_df[
        "epochs"
    ].astype(
        int
    )
    == 40
).all():

    raise RuntimeError(
        "Frozen final epochs must be 40."
    )


if not (
    run_df[
        "microbatches_per_epoch"
    ].astype(
        int
    )
    == 120
).all():

    raise RuntimeError(
        "Frozen microbatches/epoch must be 120."
    )


if not (
    run_df[
        "accumulation_steps"
    ].astype(
        int
    )
    == 2
).all():

    raise RuntimeError(
        "Frozen accumulation must be 2."
    )


if not (
    run_df[
        "optimizer_steps"
    ].astype(
        int
    )
    == 2400
).all():

    raise RuntimeError(
        "Frozen optimizer steps/run must be 2400."
    )


if not (
    run_df[
        "checkpoint_selection"
    ]
    == "FINAL_EPOCH_ONLY"
).all():

    raise RuntimeError(
        "Checkpoint policy changed."
    )


if not (
    run_df[
        "dense_training_labels"
    ]
    == False
).all():

    raise RuntimeError(
        "Dense training labels are unexpectedly enabled."
    )


if not (
    run_df[
        "dense_heldout_outcomes_open_during_fit"
    ]
    == False
).all():

    raise RuntimeError(
        "Held-out outcomes must remain closed during fit."
    )


print(
    "✓ Frozen runs                          : 72"
)

print(
    "✓ Conditions                           : 6 × 12 runs"
)

print(
    "✓ Outer folds                          : 4"
)

print(
    "✓ Seeds                                : 17, 29, 43"
)

print(
    "✓ Optimizer steps / run                : 2400"
)

print(
    "✓ Dense labels during fit              : NO"
)

print(
    "✓ Held-out outcomes during fit         : CLOSED"
)


# ==========================================================================================
# 4. VERIFY SPLITS AND PAIRED DESIGN
# ==========================================================================================

heading(
    "STEP 2/8 — VERIFY OUTER-CV + PAIRED FACTORIAL DESIGN"
)


split_df = pd.read_csv(
    SPLIT_PATH
)


final_df = split_df[
    split_df[
        "role"
    ]
    == "final_outer_cv"
].copy()


dev_df = split_df[
    split_df[
        "role"
    ]
    == "permanent_development"
].copy()


final_cases = sorted(
    final_df[
        "case_id"
    ].astype(
        str
    ).tolist()
)


dev_cases = sorted(
    dev_df[
        "case_id"
    ].astype(
        str
    ).tolist()
)


if len(
    final_cases
) != 16:

    raise RuntimeError(
        "Expected exactly 16 final outer-CV cases."
    )


if len(
    dev_cases
) != 4:

    raise RuntimeError(
        "Expected exactly four permanent-development cases."
    )


batch_rows = []


for fold in FOLDS:

    expected_heldout = sorted(
        final_df.loc[
            final_df[
                "outer_fold"
            ].astype(
                int
            )
            == fold,
            "case_id",
        ]
        .astype(
            str
        )
        .tolist()
    )


    expected_training = sorted(
        set(
            final_cases
        )
        - set(
            expected_heldout
        )
    )


    if len(
        expected_heldout
    ) != 4:

        raise RuntimeError(
            f"Fold {fold} does not have four held-out cases."
        )


    if len(
        expected_training
    ) != 12:

        raise RuntimeError(
            f"Fold {fold} does not have twelve training cases."
        )


    for seed in SEEDS:

        group = run_df[
            (
                run_df[
                    "outer_fold"
                ].astype(
                    int
                )
                == fold
            )
            & (
                run_df[
                    "seed"
                ].astype(
                    int
                )
                == seed
            )
        ].copy()


        if len(
            group
        ) != RUNS_PER_BATCH:

            raise RuntimeError(
                "Each fold×seed batch must contain six conditions."
            )


        if set(
            group[
                "condition"
            ]
        ) != set(
            CONDITIONS
        ):

            raise RuntimeError(
                "Batch does not contain all six conditions."
            )


        # Same initialization seed across all conditions.
        if group[
            "initialization_seed"
        ].nunique() != 1:

            raise RuntimeError(
                "Conditions do not share the same initialization seed."
            )


        # Same split across all conditions.
        if group[
            "training_cases"
        ].nunique() != 1:

            raise RuntimeError(
                "Training cases differ across paired conditions."
            )


        if group[
            "heldout_cases"
        ].nunique() != 1:

            raise RuntimeError(
                "Held-out cases differ across paired conditions."
            )


        registry_training = sorted(
            parse_cases(
                group.iloc[
                    0
                ][
                    "training_cases"
                ]
            )
        )


        registry_heldout = sorted(
            parse_cases(
                group.iloc[
                    0
                ][
                    "heldout_cases"
                ]
            )
        )


        if registry_training != expected_training:

            raise RuntimeError(
                f"Fold {fold}, seed {seed}: training split mismatch."
            )


        if registry_heldout != expected_heldout:

            raise RuntimeError(
                f"Fold {fold}, seed {seed}: held-out split mismatch."
            )


        if set(
            registry_training
        ) & set(
            registry_heldout
        ):

            raise RuntimeError(
                "Training/held-out split overlap."
            )


        if set(
            registry_training
        ) & set(
            dev_cases
        ):

            raise RuntimeError(
                "Permanent-development case leaked into final training."
            )


        if set(
            registry_heldout
        ) & set(
            dev_cases
        ):

            raise RuntimeError(
                "Permanent-development case leaked into final evaluation."
            )


        batch_id = (
            f"fold{fold}_seed{seed}"
        )


        batch_rows.append(
            {
                "batch_order":
                    len(
                        batch_rows
                    )
                    + 1,

                "batch_id":
                    batch_id,

                "outer_fold":
                    fold,

                "seed":
                    seed,

                "initialization_seed":
                    int(
                        group.iloc[
                            0
                        ][
                            "initialization_seed"
                        ]
                    ),

                "conditions":
                    ";".join(
                        CONDITIONS
                    ),

                "run_count":
                    6,

                "training_cases":
                    ";".join(
                        registry_training
                    ),

                "heldout_cases":
                    ";".join(
                        registry_heldout
                    ),

                "training_case_count":
                    12,

                "heldout_case_count":
                    4,

                "optimizer_steps_per_run":
                    2400,

                "total_optimizer_steps_in_batch":
                    14400,

                "execution_status":
                    "PENDING",
            }
        )


batch_df = pd.DataFrame(
    batch_rows
)


if len(
    batch_df
) != EXPECTED_BATCHES:

    raise RuntimeError(
        "Expected 12 paired batches."
    )


print(
    "✓ Final cases                          : 16"
)

print(
    "✓ Permanent-development cases excluded : 4 /4"
)

print(
    "✓ Paired fold×seed batches             : 12"
)

print(
    "✓ Runs / paired batch                  : 6"
)

print(
    "✓ Paired initialization                : PASS"
)

print(
    "✓ Paired train/heldout splits          : PASS"
)


# ==========================================================================================
# 5. VERIFY ALL V1.3 SPARSE ARTIFACTS
# ==========================================================================================

heading(
    "STEP 3/8 — VERIFY FINAL V1.3 SPARSE SUPERVISION"
)


sparse_df = pd.read_csv(
    SPARSE_MANIFEST_PATH
)


if len(
    sparse_df
) != 120:

    raise RuntimeError(
        "Expected 120 v1.3 sparse artifacts."
    )


sparse_lookup = {
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
    for _, row in sparse_df.iterrows()
}


required_pairs = []


for case_id in final_cases:

    for condition in CONDITIONS:

        required_pairs.append(
            (
                case_id,
                condition,
            )
        )


if len(
    required_pairs
) != 96:

    raise RuntimeError(
        "Expected 96 final-case × condition sparse artifacts."
    )


for case_id, condition in tqdm(
    required_pairs,
    desc="Sparse artifact SHA",
):

    key = (
        case_id,
        condition,
    )


    if key not in sparse_lookup:

        raise RuntimeError(
            "Missing sparse artifact: "
            + str(
                key
            )
        )


    row = sparse_lookup[
        key
    ]


    path = (
        REPO
        / str(
            row[
                "artifact_file"
            ]
        )
    )


    if not path.exists():

        raise RuntimeError(
            "Sparse artifact file missing:\n"
            + str(
                path
            )
        )


    if sha256_file(
        path
    ) != str(
        row[
            "artifact_file_sha256"
        ]
    ):

        raise RuntimeError(
            "Sparse artifact SHA mismatch:\n"
            + str(
                path
            )
        )


    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        required_keys = {
            "supervision_voxel_zyx",
            "supervision_label",
            "fg_membership_voxel_zyx",
            "fg_membership_group_id",
        }


        if set(
            data.files
        ) != required_keys:

            raise RuntimeError(
                "Unexpected sparse artifact schema."
            )


        labels = np.asarray(
            data[
                "supervision_label"
            ],
            dtype=np.int8,
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
                "Sparse artifact contains invalid labels."
            )


print(
    "✓ Required final sparse artifacts      : 96 /96"
)

print(
    "✓ SHA verification                     : 96 /96"
)

print(
    "✓ Sparse-only schema                   : PASS"
)


# ==========================================================================================
# 6. BUILD FINAL16 CT-ONLY CACHE
# ==========================================================================================

heading(
    "STEP 4/8 — BUILD FINAL16 CT-ONLY TRAINING CACHE"
)


if not DATASET_ROOT.exists():

    raise RuntimeError(
        "Primary dataset is not mounted:\n"
        + str(
            DATASET_ROOT
        )
    )


geometry_df = pd.read_csv(
    GEOMETRY_PATH
)


geometry_lookup = {
    str(
        row[
            "case_id"
        ]
    ):
        row
    for _, row in geometry_df.iterrows()
}


split_lookup = {
    str(
        row[
            "case_id"
        ]
    ):
        row
    for _, row in split_df.iterrows()
}


if CACHE_ROOT.exists():

    shutil.rmtree(
        CACHE_ROOT
    )


CACHE_ROOT.mkdir(
    parents=True,
    exist_ok=True,
)


cache_rows = []


for case_id in tqdm(
    final_cases,
    desc="Final16 CT cache",
):

    split_row = split_lookup[
        case_id
    ]


    geometry_row = geometry_lookup[
        case_id
    ]


    ct_path = (
        DATASET_ROOT
        / str(
            split_row[
                "ct_scan"
            ]
        )
    )


    if not ct_path.exists():

        raise RuntimeError(
            "Missing CT:\n"
            + str(
                ct_path
            )
        )


    source_encoding = str(
        geometry_row[
            "source_encoding"
        ]
    )


    native_img = nib.load(
        str(
            ct_path
        ),
        mmap=True,
    )


    native = np.asarray(
        native_img.dataobj,
        dtype=np.float32,
    )


    if source_encoding == "raw_hu":

        encoded = np.clip(
            native,
            HU_MIN,
            HU_MAX,
        ).astype(
            np.float32
        )

        cval = HU_MIN


    elif source_encoding == "documented_prewindowed_0_255":

        if (
            float(
                native.min()
            )
            < -1e-5
            or float(
                native.max()
            )
            > 255.00001
        ):

            raise RuntimeError(
                "Radiopaedia encoding assertion failed for "
                + case_id
            )


        encoded = np.clip(
            native,
            0.0,
            255.0,
        ).astype(
            np.float32
        )

        cval = 0.0


    else:

        raise RuntimeError(
            "Unknown source encoding: "
            + source_encoding
        )


    header = native_img.header.copy()

    header.set_data_dtype(
        np.float32
    )


    encoded_img = nib.Nifti1Image(
        encoded,
        native_img.affine,
        header=header,
    )


    canonical_img = nib.as_closest_canonical(
        encoded_img
    )


    resampled_img = resample_to_output(
        canonical_img,
        voxel_sizes=TARGET_SPACING_XYZ,
        order=1,
        mode="constant",
        cval=float(
            cval
        ),
    )


    resampled_xyz = np.asarray(
        resampled_img.dataobj,
        dtype=np.float32,
    )


    resampled_zyx = np.transpose(
        resampled_xyz,
        (
            2,
            1,
            0,
        ),
    )


    if source_encoding == "raw_hu":

        normalized = normalize_raw_hu(
            resampled_zyx
        )

    else:

        normalized = normalize_prewindowed(
            resampled_zyx
        )


    expected_full_shape = (
        int(
            geometry_row[
                "full_shape_z"
            ]
        ),
        int(
            geometry_row[
                "full_shape_y"
            ]
        ),
        int(
            geometry_row[
                "full_shape_x"
            ]
        ),
    )


    if tuple(
        normalized.shape
    ) != expected_full_shape:

        raise RuntimeError(
            f"{case_id}: full-grid shape mismatch. "
            + f"Expected {expected_full_shape}, "
            + f"observed {normalized.shape}"
        )


    origin = np.asarray(
        [
            int(
                geometry_row[
                    "crop_origin_z"
                ]
            ),
            int(
                geometry_row[
                    "crop_origin_y"
                ]
            ),
            int(
                geometry_row[
                    "crop_origin_x"
                ]
            ),
        ],
        dtype=np.int32,
    )


    crop_shape = np.asarray(
        [
            int(
                geometry_row[
                    "crop_shape_z"
                ]
            ),
            int(
                geometry_row[
                    "crop_shape_y"
                ]
            ),
            int(
                geometry_row[
                    "crop_shape_x"
                ]
            ),
        ],
        dtype=np.int32,
    )


    hi = (
        origin
        + crop_shape
    )


    crop = normalized[
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
    ]


    if tuple(
        crop.shape
    ) != tuple(
        crop_shape.tolist()
    ):

        raise RuntimeError(
            case_id
            + ": final CT crop shape mismatch."
        )


    if not np.isfinite(
        crop
    ).all():

        raise RuntimeError(
            case_id
            + ": non-finite final CT cache."
        )


    if (
        float(
            crop.min()
        )
        < -1.00001
        or float(
            crop.max()
        )
        > 1.00001
    ):

        raise RuntimeError(
            case_id
            + ": CT normalization outside [-1,1]."
        )


    # Same compact representation as corrected training cache.
    crop_float16 = crop.astype(
        np.float16
    )


    output_path = (
        CACHE_ROOT
        / (
            case_id
            + ".npz"
        )
    )


    np.savez_compressed(
        output_path,
        ct_zyx=
            crop_float16,
    )


    cache_rows.append(
        {
            "case_id":
                case_id,

            "role":
                "final_outer_cv",

            "outer_fold":
                int(
                    split_row[
                        "outer_fold"
                    ]
                ),

            "source_origin":
                str(
                    split_row[
                        "source_origin"
                    ]
                ),

            "source_encoding":
                source_encoding,

            "runtime_cache_file":
                str(
                    output_path
                ),

            "runtime_cache_file_sha256":
                sha256_file(
                    output_path
                ),

            "shape_z":
                int(
                    crop.shape[
                        0
                    ]
                ),

            "shape_y":
                int(
                    crop.shape[
                        1
                    ]
                ),

            "shape_x":
                int(
                    crop.shape[
                        2
                    ]
                ),

            "dtype":
                "float16",

            "intensity_min":
                float(
                    crop_float16.min()
                ),

            "intensity_max":
                float(
                    crop_float16.max()
                ),

            "dense_lesion_mask_accessed":
                False,

            "dense_lung_mask_accessed":
                False,
        }
    )


cache_df = pd.DataFrame(
    cache_rows
)


if len(
    cache_df
) != 16:

    raise RuntimeError(
        "Expected 16 final CT cache rows."
    )


cache_df.to_csv(
    CACHE_MANIFEST_RUNTIME,
    index=False,
)


cache_df.to_csv(
    CACHE_MANIFEST_REPO,
    index=False,
)


print(
    "✓ Final outer-CV CT cache              : 16 /16"
)

print(
    "✓ CT cache dtype                       : float16"
)

print(
    "✓ Dense lesion masks accessed          : 0"
)

print(
    "✓ Dense lung masks accessed            : 0"
)


# ==========================================================================================
# 7. CREATE EXECUTION REGISTRY
# ==========================================================================================

heading(
    "STEP 5/8 — FREEZE 12-BATCH EXECUTION REGISTRY"
)


execution_df = run_df.copy()


execution_df[
    "training_authorized"
] = True


execution_df[
    "sanity_gate_dependency"
] = "PASS_09E"


execution_df[
    "execution_status"
] = "PENDING"


execution_df[
    "final_dense_outcome_access_during_fit"
] = False


execution_df[
    "paired_batch_id"
] = execution_df.apply(
    lambda row:
        (
            "fold"
            + str(
                int(
                    row[
                        "outer_fold"
                    ]
                )
            )
            + "_seed"
            + str(
                int(
                    row[
                        "seed"
                    ]
                )
            )
        ),
    axis=1,
)


batch_order_lookup = {
    str(
        row[
            "batch_id"
        ]
    ):
        int(
            row[
                "batch_order"
            ]
        )
    for _, row in batch_df.iterrows()
}


execution_df[
    "paired_batch_order"
] = execution_df[
    "paired_batch_id"
].map(
    batch_order_lookup
)


condition_order = {
    condition:
        index
    for index, condition in enumerate(
        CONDITIONS,
        start=1,
    )
}


execution_df[
    "condition_execution_order_within_batch"
] = execution_df[
    "condition"
].map(
    condition_order
)


execution_df = execution_df.sort_values(
    [
        "paired_batch_order",
        "condition_execution_order_within_batch",
    ]
).reset_index(
    drop=True
)


if not execution_df[
    "training_authorized"
].all():

    raise RuntimeError(
        "Execution registry contains unauthorized run."
    )


if len(
    execution_df
) != 72:

    raise RuntimeError(
        "Execution registry row count changed."
    )


execution_df.to_csv(
    EXECUTION_REGISTRY_PATH,
    index=False,
)


batch_df.to_csv(
    BATCH_REGISTRY_PATH,
    index=False,
)


print(
    "✓ Authorized execution runs            : 72"
)

print(
    "✓ Paired execution batches             : 12"
)

print(
    "✓ Runs / batch                         : 6"
)

print(
    "✓ Final dense outcomes during fit      : PROHIBITED"
)


# ==========================================================================================
# 8. RUNTIME ESTIMATE FROM COMPLETED SANITY FIT
# ==========================================================================================

heading(
    "STEP 6/8 — COMPUTE EXECUTION-SCALE ESTIMATE"
)


r2_log = pd.read_csv(
    R2_EPOCH_LOG_PATH
)


median_sanity_epoch_seconds = float(
    r2_log[
        "epoch_seconds"
    ].median()
)


estimated_final_epoch_seconds = (
    median_sanity_epoch_seconds
    * (
        120.0
        / 100.0
    )
)


estimated_fit_seconds = (
    estimated_final_epoch_seconds
    * 40.0
)


estimated_batch_hours = (
    estimated_fit_seconds
    * 6.0
    / 3600.0
)


estimated_total_hours = (
    estimated_fit_seconds
    * 72.0
    / 3600.0
)


print(
    "Median sanity epoch                    :",
    f"{median_sanity_epoch_seconds:.2f} s",
)

print(
    "Estimated final-fit runtime             :",
    f"{estimated_fit_seconds / 60.0:.1f} min / model",
)

print(
    "Estimated paired-batch runtime          :",
    f"{estimated_batch_hours:.2f} h",
)

print(
    "Estimated 72-fit serial runtime         :",
    f"{estimated_total_hours:.1f} h",
)

print()

print(
    "Execution policy                       : 12 resumable paired batches"
)


# ==========================================================================================
# 9. FREEZE EXECUTION LOCK
# ==========================================================================================

heading(
    "STEP 7/8 — FREEZE FACTORIAL EXECUTION LOCK"
)


execution_lock = {
    "project":
        "COVA-3D",

    "block":
        BLOCK,

    "status":
        "FROZEN",

    "effective_protocol":
        EFFECTIVE_PROTOCOL,

    "sanity_gate":
        "PASS",

    "primary_factorial":
        {
            "conditions":
                CONDITIONS,

            "outer_folds":
                FOLDS,

            "seeds":
                SEEDS,

            "runs":
                72,

            "paired_batches":
                12,

            "runs_per_batch":
                6,
        },

    "pairing":
        {
            "same_initialization_seed_across_conditions":
                True,

            "same_training_cases_across_conditions":
                True,

            "same_heldout_cases_across_conditions":
                True,

            "same_patient_schedule_across_conditions":
                True,

            "same_augmentation_RNG_across_conditions":
                True,
        },

    "training":
        {
            "epochs":
                40,

            "microbatches_per_epoch":
                120,

            "accumulation_steps":
                2,

            "optimizer_steps_per_run":
                2400,

            "checkpoint_selection":
                "FINAL_EPOCH_ONLY",

            "performance_based_checkpoint_selection":
                False,

            "dense_training_labels":
                False,

            "heldout_dense_outcomes_during_fit":
                False,
        },

    "runtime_cache":
        {
            "cases":
                16,

            "CT_only":
                True,

            "dtype":
                "float16",

            "dense_lesion_masks":
                False,

            "dense_lung_masks":
                False,

            "manifest":
                str(
                    CACHE_MANIFEST_REPO.relative_to(
                        REPO
                    )
                ),
        },

    "sparse_supervision":
        {
            "version":
                "v1.3",

            "required_final_case_condition_artifacts":
                96,

            "verification":
                "PASS",
        },

    "outcome_firewall":
        {
            "final_outer_cv_dense_outcomes_authorized_during_fit":
                False,

            "threshold_search_during_fit":
                False,

            "method_development":
                False,
        },

    "execution_order":
        [
            str(
                value
            )
            for value in batch_df[
                "batch_id"
            ].tolist()
        ],

    "next_batch":
        "fold0_seed17",

    "frozen_at_utc":
        NOW_ISO,
}


EXECUTION_LOCK_PATH.write_text(
    yaml.safe_dump(
        execution_lock,
        sort_keys=False,
    ),
    encoding="utf-8",
)


# ==========================================================================================
# 10. WRITE AUDIT + STATE
# ==========================================================================================

audit = {
    "project":
        "COVA-3D",

    "block":
        BLOCK,

    "status":
        "PASS",

    "starting_commit":
        head,

    "sanity_gate":
        "PASS",

    "registry":
        {
            "runs":
                72,

            "conditions":
                6,

            "folds":
                4,

            "seeds":
                3,

            "paired_batches":
                12,

            "runs_per_batch":
                6,

            "paired_design_verification":
                "PASS",
        },

    "sparse_supervision":
        {
            "required_artifacts":
                96,

            "SHA_verified":
                96,

            "version":
                "v1.3",
        },

    "CT_cache":
        {
            "cases":
                16,

            "CT_only":
                True,

            "dense_lesion_masks_accessed":
                0,

            "dense_lung_masks_accessed":
                0,
        },

    "runtime_estimate":
        {
            "median_sanity_epoch_seconds":
                median_sanity_epoch_seconds,

            "estimated_minutes_per_final_fit":
                estimated_fit_seconds
                / 60.0,

            "estimated_hours_per_paired_batch":
                estimated_batch_hours,

            "estimated_hours_72_serial":
                estimated_total_hours,
        },

    "authorization":
        {
            "factorial_training":
                True,

            "method_development":
                False,

            "final_outer_cv_dense_outcomes":
                False,
        },

    "optimizer_steps_executed_in_this_block":
        0,

    "next_block":
        NEXT_BLOCK,

    "next_batch":
        "fold0_seed17",

    "generated_at_utc":
        NOW_ISO,
}


write_json(
    AUDIT_PATH,
    audit,
)


state.update(
    {
        "last_completed_block":
            BLOCK,

        "current_stage":
            "FACTORIAL_PREFLIGHT_PASS_BATCH01_PENDING",

        "factorial_preflight_status":
            "PASS",

        "factorial_training_authorized":
            True,

        "factorial_execution_registry":
            str(
                EXECUTION_REGISTRY_PATH.relative_to(
                    REPO
                )
            ),

        "factorial_batch_registry":
            str(
                BATCH_REGISTRY_PATH.relative_to(
                    REPO
                )
            ),

        "factorial_batches_total":
            12,

        "factorial_batches_completed":
            0,

        "factorial_runs_total":
            72,

        "factorial_runs_completed":
            0,

        "factorial_CT_cache_status":
            "PASS_16_OF_16",

        "factorial_sparse_artifacts_required":
            96,

        "factorial_sparse_artifacts_verified":
            96,

        "method_development_authorized":
            False,

        "final_outer_cv_outcomes_authorized":
            False,

        "final_outer_cv_access_in_cova3d":
            0,

        "next_block":
            NEXT_BLOCK,

        "next_factorial_batch":
            "fold0_seed17",

        "next_action":
            (
                "Train the six frozen factorial conditions for outer fold 0, "
                "seed 17 as one paired batch. Do not open held-out dense outcomes."
            ),

        "updated_at_utc":
            NOW_ISO,
    }
)


write_json(
    STATE_PATH,
    state,
)


project_state.update(
    {
        "last_completed_block":
            BLOCK,

        "current_stage":
            "cova3d_factorial_batch01_pending",

        "cova3d_factorial_preflight":
            "PASS",

        "cova3d_factorial_training_authorized":
            True,

        "cova3d_factorial_batches_total":
            12,

        "cova3d_factorial_batches_completed":
            0,

        "cova3d_final_outer_cv_access":
            0,

        "next_action":
            "Run 10B-FACTORIAL-BATCH-01 only.",

        "updated_at_utc":
            NOW_ISO,
    }
)


write_json(
    PROJECT_STATE_PATH,
    project_state,
)


# ==========================================================================================
# 11. TEST
# ==========================================================================================

heading(
    "STEP 8/8 — REGRESSION / COMMIT"
)


write_text(
    TEST_PATH,
    r'''
from pathlib import Path
import json

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_factorial_execution_registry():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_final_factorial_execution_registry_v1_0.csv"
    )

    assert len(frame) == 72
    assert frame["run_id"].nunique() == 72
    assert frame["training_authorized"].all()
    assert (~frame["final_dense_outcome_access_during_fit"]).all()

    assert set(frame["condition"]) == {
        "C50_COH",
        "C50_DIS",
        "C50_FRG",
        "C100_COH",
        "C100_DIS",
        "C100_FRG",
    }

    assert set(frame["outer_fold"]) == {0, 1, 2, 3}
    assert set(frame["seed"]) == {17, 29, 43}

    assert frame["paired_batch_id"].nunique() == 12


def test_factorial_batch_registry():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_final_factorial_batch_registry_v1_0.csv"
    )

    assert len(frame) == 12
    assert (frame["run_count"] == 6).all()
    assert (frame["training_case_count"] == 12).all()
    assert (frame["heldout_case_count"] == 4).all()
    assert (frame["execution_status"] == "PENDING").all()


def test_factorial_execution_lock():

    cfg = yaml.safe_load(
        (
            ROOT
            / "configs/"
            "cova3d_factorial_execution_lock_v1_0.yaml"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert cfg["status"] == "FROZEN"
    assert cfg["sanity_gate"] == "PASS"
    assert cfg["primary_factorial"]["runs"] == 72
    assert cfg["primary_factorial"]["paired_batches"] == 12
    assert cfg["training"]["optimizer_steps_per_run"] == 2400

    assert (
        cfg[
            "outcome_firewall"
        ][
            "final_outer_cv_dense_outcomes_authorized_during_fit"
        ]
        is False
    )


def test_factorial_preflight_state():

    state = json.loads(
        (
            ROOT
            / "COVA3D_STATE.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert state["last_completed_block"] == "10A-FACTORIAL-PREFLIGHT"
    assert state["factorial_preflight_status"] == "PASS"
    assert state["factorial_training_authorized"] is True
    assert state["factorial_batches_total"] == 12
    assert state["factorial_batches_completed"] == 0
    assert state["factorial_runs_completed"] == 0
    assert state["final_outer_cv_outcomes_authorized"] is False
    assert state["final_outer_cv_access_in_cova3d"] == 0
    assert state["next_block"] == "10B-FACTORIAL-BATCH-01"
'''
)


tests = sh(
    [
        sys.executable,
        "-m",
        "pytest",

        "tests/test_cova3d_factorial_preflight.py",

        "-q",
        "-p",
        "no:cacheprovider",
    ],
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
        "Factorial-preflight tests failed."
    )


# ==========================================================================================
# 12. SOURCE CAPTURE
# ==========================================================================================

source_capture = (
    "NOT_AVAILABLE"
)


try:

    cell = (
        get_ipython()
        .history_manager
        .input_hist_raw[
            -1
        ]
    )

    if (
        "# COVA-3D — BLOCK 10A-FACTORIAL-PREFLIGHT"
        in cell
    ):

        source_path = (
            REPO
            / "scripts/code_blocks/"
            "block10a_factorial_preflight.py"
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
# 13. NORMALIZE TEXT / REPOSITORY MANIFEST
# ==========================================================================================

text_paths = [
    EXECUTION_REGISTRY_PATH,
    BATCH_REGISTRY_PATH,
    CACHE_MANIFEST_REPO,
    EXECUTION_LOCK_PATH,
    AUDIT_PATH,
    TEST_PATH,
    STATE_PATH,
    PROJECT_STATE_PATH,
]


source_path = (
    REPO
    / "scripts/code_blocks/"
    "block10a_factorial_preflight.py"
)


if source_path.exists():

    text_paths.append(
        source_path
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

        "sanity_gate":
            "PASS",

        "factorial_preflight":
            "PASS",

        "factorial_runs":
            72,

        "factorial_batches":
            12,

        "factorial_runs_completed":
            0,

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
# 14. COMMIT / PUSH
# ==========================================================================================

git_paths = [
    "COVA3D_STATE.json",
    "PROJECT_STATE.json",
    "REPOSITORY_MANIFEST.json",

    "configs/"
    "cova3d_factorial_execution_lock_v1_0.yaml",

    "data/manifests/"
    "cova3d_final_factorial_execution_registry_v1_0.csv",

    "data/manifests/"
    "cova3d_final_factorial_batch_registry_v1_0.csv",

    "data/manifests/"
    "cova3d_final16_CT_cache_manifest_v1_0.csv",

    "experiments/audits/"
    "block10a_factorial_preflight.json",

    "tests/"
    "test_cova3d_factorial_preflight.py",
]


if source_path.exists():

    git_paths.append(
        "scripts/code_blocks/"
        "block10a_factorial_preflight.py"
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
        ".npz"
    )
    or path.endswith(
        ".pt"
    )
    for path in staged
):

    raise RuntimeError(
        "Runtime CT/model artifact accidentally staged."
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


print(
    "✓ git diff --cached --check            : PASS"
)

print(
    "✓ Runtime CT cache staged              : NO"
)

print(
    "✓ Model/checkpoint staged              : NO"
)


git(
    "commit",
    "-m",
    "experiment: freeze COVA-3D factorial execution preflight",
)


final_commit = git(
    "rev-parse",
    "HEAD",
)


token, askpass, git_env = git_auth()


push(
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
        "Repository not clean after factorial preflight:\n"
        + final_status
    )


# ==========================================================================================
# FINAL REPORT
# ==========================================================================================

heading(
    "COVA-3D 10A-FACTORIAL-PREFLIGHT — FINAL REPORT"
)


print(
    "Starting commit                        :",
    head[:12],
)

print(
    "Preflight commit                       :",
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
    "PRIMARY FACTORIAL"
)

print(
    "-----------------"
)

print(
    "Conditions                             : 6"
)

print(
    "Outer folds                            : 4"
)

print(
    "Seeds                                  : 3"
)

print(
    "Total fits                             : 72"
)

print(
    "Paired batches                         : 12"
)

print(
    "Fits per paired batch                  : 6"
)

print(
    "Optimizer steps / fit                  : 2400"
)

print()

print(
    "FINAL16 TRAINING DATA"
)

print(
    "---------------------"
)

print(
    "CT-only cache                          : 16 /16 PASS"
)

print(
    "Sparse v1.3 artifacts required         : 96"
)

print(
    "Sparse artifact SHA checks             : 96 /96 PASS"
)

print(
    "Permanent-development leakage          : 0"
)

print()

print(
    "PAIRING"
)

print(
    "-------"
)

print(
    "Same initialization / condition set    : PASS"
)

print(
    "Same training split across conditions  : PASS"
)

print(
    "Same held-out split across conditions  : PASS"
)

print(
    "Paired patient/augmentation policy     : FROZEN"
)

print()

print(
    "COMPUTE ESTIMATE"
)

print(
    "----------------"
)

print(
    "Estimated minutes / fit                :",
    f"{estimated_fit_seconds / 60.0:.1f}",
)

print(
    "Estimated hours / six-fit batch        :",
    f"{estimated_batch_hours:.2f}",
)

print(
    "Estimated serial hours / 72 fits       :",
    f"{estimated_total_hours:.1f}",
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
    "Final outer-CV outcomes opened         : 0"
)

print(
    "optimizer.step() calls                 : 0"
)

print()

print(
    "AUTHORIZATION"
)

print(
    "-------------"
)

print(
    "Factorial training authorized          : YES"
)

print(
    "Method development authorized          : NO"
)

print(
    "Final dense outcome evaluation         : NO"
)

print(
    "Next paired batch                      : fold0_seed17"
)

print(
    "Next block                             :",
    NEXT_BLOCK,
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
    "Send me this COMPLETE report."
)

print(
    "Do NOT open held-out dense masks."
)

print(
    "Do NOT start a different batch."
)

print(
    "After I audit this, we start the first six real factorial fits."
)

print(
    "=" * 132
)
