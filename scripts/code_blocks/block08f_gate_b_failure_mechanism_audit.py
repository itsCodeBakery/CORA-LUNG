# ==========================================================================================
# CORA-LUNG — CODE BLOCK 08F
# Gate-B Failure Mechanism Audit
#
# PURPOSE
#
# Gate B has already produced the locked result:
#
#     NO_GO
#
# This block DOES NOT reopen that decision.
#
# Instead, it asks:
#
#   1. Are predictions pathologically fragmented?
#   2. Is threshold/calibration sensitivity masking spatial capability?
#   3. Why did "complete" supervision perform worse than natural omission?
#   4. Was actual sparse-supervision exposure different across conditions?
#   5. Is the observed behavior source-specific:
#          Coronacases vs Radiopaedia?
#   6. Does component omission show a descriptive regularization-like pattern?
#   7. Are lesion probabilities spatially separated from background at all?
#
# STRICT SCIENTIFIC STATUS
#
#   • Gate B remains NO_GO.
#   • Gate-B decision rule is NOT modified.
#   • Existing checkpoints are NOT reselected.
#   • Existing predictions are NOT regenerated.
#   • No optimizer is instantiated.
#   • No optimizer.step() occurs.
#   • Gate C remains NOT RUN.
#   • Final outer-CV cases remain SEALED.
#
# DATA ALLOWED
#
#   • four permanent-development CTs
#   • four permanent-development dense infection masks
#     (already opened in Block 08E)
#   • sixteen predictions frozen before those outcomes
#   • training cache sparse annotations
#   • committed training logs
#
# THIS IS DIAGNOSTIC ONLY.
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
import gc
import random
import textwrap

import numpy as np
import pandas as pd

import nibabel as nib

from scipy import ndimage as ndi

import matplotlib.pyplot as plt

from tqdm.auto import tqdm

from sklearn.metrics import roc_auc_score

from kaggle_secrets import UserSecretsClient


# ==========================================================================================
# 0. CONSTANTS
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

EVAL_ROOT = Path(
    "/kaggle/working/cora_gate_b_eval_v1_2"
)

PREDICTION_ROOT = (
    EVAL_ROOT
    / "predictions"
)

EXPECTED_START_COMMIT = (
    "eddd9caf0377"
)

SEED = 17

PATCH_ZYX = (
    48,
    128,
    128,
)

EPOCHS = 30

MICROBATCHES_PER_EPOCH = 100

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

THRESHOLDS = np.asarray(
    [
        0.10,
        0.20,
        0.30,
        0.40,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
    ],
    dtype=np.float32,
)

REFERENCE_MIN_VOLUME_ML = 0.10

PRIMARY_IOU = 0.10

SECONDARY_IOU = 0.25

BOUNDARY_RADIUS_MM = 5.0

MAX_AUC_SAMPLES_PER_CLASS = 100_000

NOW_ISO = datetime.now(
    timezone.utc
).strftime(
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
        + "=" * 122
    )

    print(
        text
    )

    print(
        "=" * 122
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
            "Microbatch count is not divisible by development-case count."
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


def physical_ball(
    radius_mm,
    spacing_xyz,
):

    spacing_xyz = np.asarray(
        spacing_xyz,
        dtype=np.float64,
    )

    radii = np.ceil(
        float(
            radius_mm
        )
        / spacing_xyz
    ).astype(
        int
    )

    ranges = [
        np.arange(
            -radius,
            radius + 1,
            dtype=np.float64,
        )
        for radius in radii
    ]

    x, y, z = np.meshgrid(
        ranges[
            0
        ],
        ranges[
            1
        ],
        ranges[
            2
        ],
        indexing="ij",
    )

    distance_sq = (
        (
            x
            * spacing_xyz[
                0
            ]
        )
        ** 2
        + (
            y
            * spacing_xyz[
                1
            ]
        )
        ** 2
        + (
            z
            * spacing_xyz[
                2
            ]
        )
        ** 2
    )

    return (
        distance_sq
        <= float(
            radius_mm
        )
        ** 2
    )


def safe_mean(
    values,
):

    values = np.asarray(
        values
    )

    if values.size == 0:

        return float(
            "nan"
        )

    return float(
        np.mean(
            values
        )
    )


def safe_median(
    values,
):

    values = np.asarray(
        values
    )

    if values.size == 0:

        return float(
            "nan"
        )

    return float(
        np.median(
            values
        )
    )


def safe_percentile(
    values,
    q,
):

    values = np.asarray(
        values
    )

    if values.size == 0:

        return float(
            "nan"
        )

    return float(
        np.percentile(
            values,
            q,
        )
    )


def git_push():

    token = UserSecretsClient().get_secret(
        "pushCora"
    )

    if not token:

        raise RuntimeError(
            "Kaggle secret 'pushCora' unavailable."
        )

    token = token.strip()

    askpass = Path(
        "/tmp/cora_git_askpass_block08f.sh"
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


# ==========================================================================================
# 2. PRECONDITIONS
# ==========================================================================================

heading(
    "CORA-LUNG BLOCK 08F — GATE-B FAILURE MECHANISM AUDIT"
)


if not (
    REPO
    / ".git"
).exists():

    raise RuntimeError(
        "Repository missing."
    )


starting_commit = sh(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    cwd=REPO,
).stdout.strip()


if not starting_commit.startswith(
    EXPECTED_START_COMMIT
):

    raise RuntimeError(
        "Unexpected starting commit."
    )


if sh(
    [
        "git",
        "status",
        "--porcelain",
    ],
    cwd=REPO,
).stdout.strip():

    raise RuntimeError(
        "Repository must be clean before Block 08F."
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


if (
    state.get(
        "last_completed_block"
    )
    != "08E"
):

    raise RuntimeError(
        "Block 08E must be the last completed block."
    )


if (
    state.get(
        "gate_b"
    )
    != "NO_GO"
):

    raise RuntimeError(
        "Block 08F is specifically the NO-GO failure audit."
    )


if (
    state.get(
        "gate_b_outcomes_opened"
    )
    is not True
):

    raise RuntimeError(
        "Gate-B development outcomes are not recorded as opened."
    )


if (
    state.get(
        "valid_gate_b_optimizer_steps"
    )
    != 6000
):

    raise RuntimeError(
        "Expected exactly 6000 valid corrected Gate-B optimizer steps."
    )


if (
    state.get(
        "gate_c"
    )
    != "NOT_RUN"
):

    raise RuntimeError(
        "Gate C must remain untouched."
    )


gate_b_audit = json.loads(
    (
        REPO
        / "experiments/audits/"
        "block08e_gate_b_dense_evaluation.json"
    ).read_text(
        encoding="utf-8"
    )
)


if (
    gate_b_audit.get(
        "gate_b_decision"
    )
    != "NO_GO"
):

    raise RuntimeError(
        "Committed Gate-B audit is not NO_GO."
    )


if int(
    gate_b_audit.get(
        "final_outer_cv_cases_accessed",
        -1,
    )
) != 0:

    raise RuntimeError(
        "Final outer-CV access was detected."
    )


print(
    "✓ Starting commit                     :",
    starting_commit[:12],
)

print(
    "✓ Locked Gate-B result                : NO_GO"
)

print(
    "✓ Gate-B result will be modified      : NO"
)

print(
    "✓ Gate C                              : NOT RUN"
)

print(
    "✓ Final outer-CV access               : 0"
)

print(
    "✓ Optimizer steps in Block 08F        : 0"
)


# ==========================================================================================
# 3. IMPORT FROZEN EVALUATOR / SAMPLER
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
    "cora_lung.eval.gate_b",
    "cora_lung.data.pilot_dataset",
]:

    sys.modules.pop(
        module_name,
        None,
    )


importlib.invalidate_caches()


from cora_lung.eval.gate_b import (
    dice_score,
    iou_score,
    label_binary_components,
    reference_component_partition,
    component_iou_matrix,
    maximum_cardinality_iou_matching,
    evaluate_selected_candidates,
)

from cora_lung.data.pilot_dataset import (
    deterministic_patch_origin,
)


# ==========================================================================================
# 4. LOCATE FROZEN DEVELOPMENT DATA / PREDICTIONS
# ==========================================================================================

heading(
    "STEP 1/9 — VERIFY FROZEN DEVELOPMENT ARTIFACTS"
)


split_df = pd.read_csv(
    REPO
    / "data/splits/"
    "primary_volume_split_registry.csv"
)


development_df = (
    split_df[
        split_df[
            "role"
        ]
        == "permanent_development"
    ]
    .copy()
)


development_df[
    "case_id"
] = development_df[
    "case_id"
].astype(
    str
)


development_cases = sorted(
    development_df[
        "case_id"
    ].tolist()
)


if len(
    development_cases
) != 4:

    raise RuntimeError(
        "Expected exactly four permanent development cases."
    )


final_cases = split_df[
    split_df[
        "role"
    ]
    == "final_outer_cv"
][
    "case_id"
].astype(
    str
).tolist()


if len(
    final_cases
) != 16:

    raise RuntimeError(
        "Expected sixteen sealed final-CV cases."
    )


candidate_roots = [
    Path(
        "/kaggle/input/datasets/andrewmvd/covid19-ct-scans"
    ),
    Path(
        "/kaggle/input/covid19-ct-scans"
    ),
]


primary_root = None


probe_relative_ct = str(
    development_df.iloc[
        0
    ][
        "ct_scan"
    ]
)


for root in candidate_roots:

    if (
        root.exists()
        and (
            root
            / probe_relative_ct
        ).exists()
    ):

        primary_root = root

        break


if primary_root is None:

    raise RuntimeError(
        "Primary dataset root not found."
    )


prediction_freeze_df = pd.read_csv(
    REPO
    / "data/manifests/"
    "gate_b_v1_2_prediction_freeze.csv"
)


if len(
    prediction_freeze_df
) != 16:

    raise RuntimeError(
        "Expected sixteen frozen prediction hashes."
    )


prediction_hash_lookup = {
    (
        str(
            row[
                "condition"
            ]
        ),
        str(
            row[
                "case_id"
            ]
        ),
    ):
        str(
            row[
                "prediction_sha256"
            ]
        )
    for _, row
    in prediction_freeze_df.iterrows()
}


for condition in CONDITIONS:

    for case_id in development_cases:

        prediction_path = (
            PREDICTION_ROOT
            / condition
            / (
                case_id
                + ".npz"
            )
        )


        if not prediction_path.exists():

            raise RuntimeError(
                "Frozen local prediction is missing:\n"
                + str(
                    prediction_path
                )
                + "\n"
                + "Do not silently regenerate it in this diagnostic block."
            )


        observed_sha = sha256_file(
            prediction_path
        )


        expected_sha = prediction_hash_lookup[
            (
                condition,
                case_id,
            )
        ]


        if observed_sha != expected_sha:

            raise RuntimeError(
                "Frozen prediction SHA mismatch for "
                + condition
                + "/"
                + case_id
            )


print(
    "✓ Frozen prediction artifacts          : 16/16"
)

print(
    "✓ Prediction checksums                 : PASS"
)

print(
    "✓ Permanent development cases          : 4"
)

print(
    "✓ Final outer-CV cases                 : SEALED 16/16"
)


# ==========================================================================================
# 5. VERIFY DEVELOPMENT DENSE MASK HASHES
# ==========================================================================================

source_hash_df = pd.read_csv(
    REPO
    / "data/manifests/"
    "primary_file_sha256.csv"
)


source_hash_lookup = {
    str(
        row[
            "relative_path"
        ]
    ):
        str(
            row[
                "sha256"
            ]
        ).lower()
    for _, row
    in source_hash_df.iterrows()
}


case_lookup = {
    str(
        row[
            "case_id"
        ]
    ):
        row
    for _, row
    in development_df.iterrows()
}


reference_metadata = {}


for case_id in development_cases:

    row = case_lookup[
        case_id
    ]


    mask_relative = str(
        row[
            "infection_mask"
        ]
    )


    mask_path = (
        primary_root
        / mask_relative
    )


    if mask_relative not in source_hash_lookup:

        raise RuntimeError(
            "Frozen infection-mask SHA is missing."
        )


    observed_sha = sha256_file(
        mask_path
    ).lower()


    if observed_sha != source_hash_lookup[
        mask_relative
    ]:

        raise RuntimeError(
            "Infection-mask SHA mismatch for "
            + case_id
        )


    mask_img = nib.load(
        str(
            mask_path
        ),
        mmap=True,
    )


    spacing_xyz = np.linalg.norm(
        np.asarray(
            mask_img.affine,
            dtype=np.float64,
        )[
            :3,
            :3,
        ],
        axis=0,
    )


    voxel_volume_ml = float(
        abs(
            np.linalg.det(
                np.asarray(
                    mask_img.affine,
                    dtype=np.float64,
                )[
                    :3,
                    :3,
                ]
            )
        )
        / 1000.0
    )


    reference_metadata[
        case_id
    ] = {
        "mask_path":
            mask_path,

        "spacing_xyz":
            spacing_xyz,

        "voxel_volume_ml":
            voxel_volume_ml,

        "source_origin":
            str(
                row[
                    "source_origin"
                ]
            ),

        "ct_path":
            (
                primary_root
                / str(
                    row[
                        "ct_scan"
                    ]
                )
            ),
    }


print(
    "✓ Development dense-mask hashes       : PASS 4/4"
)


# ==========================================================================================
# 6. THRESHOLD / FRAGMENTATION / OUTPUT-SEPARATION AUDIT
# ==========================================================================================

heading(
    "STEP 2/9 — THRESHOLD, FRAGMENTATION, AND OUTPUT-SEPARATION AUDIT"
)


threshold_rows = []

output_rows = []


for case_id in tqdm(
    development_cases,
    desc="Cases",
):

    meta = reference_metadata[
        case_id
    ]


    mask_img = nib.load(
        str(
            meta[
                "mask_path"
            ]
        ),
        mmap=True,
    )


    reference = (
        np.asarray(
            mask_img.dataobj
        )
        > 0.5
    )


    if not reference.any():

        raise RuntimeError(
            "Development reference unexpectedly empty."
        )


    reference_partition = reference_component_partition(
        reference,
        voxel_volume_ml=meta[
            "voxel_volume_ml"
        ],
        minimum_volume_ml=REFERENCE_MIN_VOLUME_ML,
    )


    eligible_ids = np.asarray(
        reference_partition[
            "eligible_ids"
        ],
        dtype=np.int32,
    )


    small_ids = np.asarray(
        reference_partition[
            "small_ids"
        ],
        dtype=np.int32,
    )


    expected_count = int(
        gate_b_audit[
            "eligible_components_total"
        ]
    )


    # Build physical boundary zones once per case.
    struct = physical_ball(
        BOUNDARY_RADIUS_MM,
        meta[
            "spacing_xyz"
        ],
    )


    dilated_reference = ndi.binary_dilation(
        reference,
        structure=struct,
    )


    eroded_reference = ndi.binary_erosion(
        reference,
        structure=struct,
    )


    lesion_boundary = (
        reference
        & np.logical_not(
            eroded_reference
        )
    )


    perilesional_background = (
        dilated_reference
        & np.logical_not(
            reference
        )
    )


    far_background = np.logical_not(
        dilated_reference
    )


    for condition in CONDITIONS:

        prediction_path = (
            PREDICTION_ROOT
            / condition
            / (
                case_id
                + ".npz"
            )
        )


        with np.load(
            prediction_path,
            allow_pickle=False,
        ) as prediction_npz:

            probability = np.asarray(
                prediction_npz[
                    "probability_xyz"
                ],
                dtype=np.float32,
            )


        if probability.shape != reference.shape:

            raise RuntimeError(
                "Prediction/reference shape mismatch for "
                + condition
                + "/"
                + case_id
            )


        # ------------------------------------------------------------------
        # Model-output probability diagnostics.
        # ------------------------------------------------------------------

        lesion_prob = probability[
            reference
        ]


        background_prob = probability[
            np.logical_not(
                reference
            )
        ]


        lesion_boundary_prob = probability[
            lesion_boundary
        ]


        perilesional_prob = probability[
            perilesional_background
        ]


        far_background_prob = probability[
            far_background
        ]


        rng = np.random.default_rng(
            stable_seed(
                "auc",
                case_id,
                condition,
            )
        )


        positive_count = min(
            MAX_AUC_SAMPLES_PER_CLASS,
            len(
                lesion_prob
            ),
        )


        negative_count = min(
            MAX_AUC_SAMPLES_PER_CLASS,
            len(
                background_prob
            ),
        )


        if positive_count <= 0:

            auc = float(
                "nan"
            )


        else:

            positive_index = rng.choice(
                len(
                    lesion_prob
                ),
                size=positive_count,
                replace=False,
            )


            negative_index = rng.choice(
                len(
                    background_prob
                ),
                size=negative_count,
                replace=False,
            )


            auc_labels = np.concatenate(
                [
                    np.ones(
                        positive_count,
                        dtype=np.int8,
                    ),
                    np.zeros(
                        negative_count,
                        dtype=np.int8,
                    ),
                ]
            )


            auc_scores = np.concatenate(
                [
                    lesion_prob[
                        positive_index
                    ],
                    background_prob[
                        negative_index
                    ],
                ]
            )


            auc = float(
                roc_auc_score(
                    auc_labels,
                    auc_scores,
                )
            )


        balanced_brier = (
            0.5
            * safe_mean(
                (
                    lesion_prob
                    - 1.0
                )
                ** 2
            )
            + 0.5
            * safe_mean(
                background_prob
                ** 2
            )
        )


        output_rows.append(
            {
                "case_id":
                    case_id,

                "source_origin":
                    meta[
                        "source_origin"
                    ],

                "condition":
                    condition,

                "sampled_voxel_auc":
                    auc,

                "balanced_brier":
                    float(
                        balanced_brier
                    ),

                "lesion_probability_mean":
                    safe_mean(
                        lesion_prob
                    ),

                "lesion_probability_median":
                    safe_median(
                        lesion_prob
                    ),

                "lesion_probability_p90":
                    safe_percentile(
                        lesion_prob,
                        90,
                    ),

                "background_probability_mean":
                    safe_mean(
                        background_prob
                    ),

                "background_probability_median":
                    safe_median(
                        background_prob
                    ),

                "background_probability_p99":
                    safe_percentile(
                        background_prob,
                        99,
                    ),

                "lesion_minus_background_mean":
                    (
                        safe_mean(
                            lesion_prob
                        )
                        - safe_mean(
                            background_prob
                        )
                    ),

                "lesion_boundary_probability_mean":
                    safe_mean(
                        lesion_boundary_prob
                    ),

                "perilesional_background_probability_mean":
                    safe_mean(
                        perilesional_prob
                    ),

                "far_background_probability_mean":
                    safe_mean(
                        far_background_prob
                    ),

                "lesion_fraction_ge_0p5":
                    float(
                        np.mean(
                            lesion_prob
                            >= 0.5
                        )
                    ),

                "background_fraction_ge_0p5":
                    float(
                        np.mean(
                            background_prob
                            >= 0.5
                        )
                    ),
            }
        )


        # ------------------------------------------------------------------
        # Threshold sweep.
        #
        # This is DIAGNOSTIC ONLY and cannot replace the locked Gate-B R@1.
        # ------------------------------------------------------------------

        all_reference_ids = np.concatenate(
            [
                eligible_ids,
                small_ids,
            ]
        )


        for threshold in THRESHOLDS:

            binary = (
                probability
                >= float(
                    threshold
                )
            )


            prediction_labels, prediction_count = (
                label_binary_components(
                    binary
                )
            )


            prediction_labels = prediction_labels.astype(
                np.int32,
                copy=False,
            )


            prediction_sizes = np.bincount(
                prediction_labels.reshape(
                    -1
                ),
                minlength=prediction_count
                + 1,
            ).astype(
                np.int64
            )


            all_iou = component_iou_matrix(
                reference_partition[
                    "labels"
                ],
                reference_partition[
                    "sizes"
                ],
                all_reference_ids,
                prediction_labels,
                prediction_sizes,
            )


            eligible_iou = all_iou[
                :
                len(
                    eligible_ids
                ),
                :,
            ]


            small_iou = all_iou[
                len(
                    eligible_ids
                ):
                ,
                :,
            ]


            dummy_scores = np.ones(
                prediction_count,
                dtype=np.float32,
            )


            primary_eval = evaluate_selected_candidates(
                candidate_scores=dummy_scores,
                eligible_iou=eligible_iou,
                small_iou=small_iou,
                score_threshold=-np.inf,
                minimum_iou=PRIMARY_IOU,
            )


            secondary_eval = evaluate_selected_candidates(
                candidate_scores=dummy_scores,
                eligible_iou=eligible_iou,
                small_iou=small_iou,
                score_threshold=-np.inf,
                minimum_iou=SECONDARY_IOU,
            )


            any_overlap_matches = (
                maximum_cardinality_iou_matching(
                    eligible_iou,
                    minimum_iou=0.0,
                )
            )


            eligible_count = int(
                len(
                    eligible_ids
                )
            )


            any_overlap_recall = (
                len(
                    any_overlap_matches
                )
                / eligible_count
                if eligible_count
                else 0.0
            )


            false_positive_voxels = int(
                np.logical_and(
                    binary,
                    np.logical_not(
                        reference
                    ),
                ).sum()
            )


            component_volumes_ml = (
                prediction_sizes[
                    1:
                ].astype(
                    np.float64
                )
                * float(
                    meta[
                        "voxel_volume_ml"
                    ]
                )
            )


            total_predicted_volume_ml = float(
                component_volumes_ml.sum()
            )


            largest_component_ml = (
                float(
                    component_volumes_ml.max()
                )
                if len(
                    component_volumes_ml
                )
                else 0.0
            )


            largest_component_fraction = (
                largest_component_ml
                / total_predicted_volume_ml
                if total_predicted_volume_ml
                > 0
                else 0.0
            )


            tiny_fraction_0p1 = (
                float(
                    np.mean(
                        component_volumes_ml
                        < 0.10
                    )
                )
                if len(
                    component_volumes_ml
                )
                else 0.0
            )


            tiny_fraction_0p5 = (
                float(
                    np.mean(
                        component_volumes_ml
                        < 0.50
                    )
                )
                if len(
                    component_volumes_ml
                )
                else 0.0
            )


            threshold_rows.append(
                {
                    "case_id":
                        case_id,

                    "source_origin":
                        meta[
                            "source_origin"
                        ],

                    "condition":
                        condition,

                    "threshold":
                        float(
                            threshold
                        ),

                    "dice":
                        float(
                            dice_score(
                                binary,
                                reference,
                            )
                        ),

                    "iou":
                        float(
                            iou_score(
                                binary,
                                reference,
                            )
                        ),

                    "predicted_components":
                        int(
                            prediction_count
                        ),

                    "eligible_reference_components":
                        eligible_count,

                    "component_to_reference_ratio":
                        (
                            float(
                                prediction_count
                            )
                            / eligible_count
                            if eligible_count
                            else float(
                                "nan"
                            )
                        ),

                    "fp_components":
                        int(
                            primary_eval[
                                "false_positives"
                            ]
                        ),

                    "fp_volume_ml":
                        float(
                            false_positive_voxels
                            * meta[
                                "voxel_volume_ml"
                            ]
                        ),

                    "iou_0p10_recall":
                        float(
                            primary_eval[
                                "recall"
                            ]
                        ),

                    "iou_0p25_recall":
                        float(
                            secondary_eval[
                                "recall"
                            ]
                        ),

                    "any_overlap_recall":
                        float(
                            any_overlap_recall
                        ),

                    "predicted_foreground_volume_ml":
                        total_predicted_volume_ml,

                    "median_component_volume_ml":
                        safe_median(
                            component_volumes_ml
                        ),

                    "p90_component_volume_ml":
                        safe_percentile(
                            component_volumes_ml,
                            90,
                        ),

                    "largest_component_volume_ml":
                        largest_component_ml,

                    "largest_component_volume_fraction":
                        largest_component_fraction,

                    "fraction_components_lt_0p1ml":
                        tiny_fraction_0p1,

                    "fraction_components_lt_0p5ml":
                        tiny_fraction_0p5,
                }
            )


        del probability
        del lesion_prob
        del background_prob

        gc.collect()


    del reference
    del dilated_reference
    del eroded_reference
    del lesion_boundary
    del perilesional_background
    del far_background

    gc.collect()


threshold_df = pd.DataFrame(
    threshold_rows
)


output_df = pd.DataFrame(
    output_rows
)


if len(
    threshold_df
) != (
    4
    * 4
    * len(
        THRESHOLDS
    )
):

    raise RuntimeError(
        "Threshold diagnostic row count mismatch."
    )


if len(
    output_df
) != 16:

    raise RuntimeError(
        "Output diagnostic row count mismatch."
    )


print(
    "✓ Threshold evaluations               :",
    len(
        threshold_df
    ),
)

print(
    "✓ Output-separation evaluations        : 16"
)


# ==========================================================================================
# 7. VALIDATE THRESHOLD 0.5 AGAINST COMMITTED BLOCK-08E METRICS
# ==========================================================================================

heading(
    "STEP 3/9 — CROSS-CHECK DIAGNOSTIC EVALUATOR AGAINST BLOCK 08E"
)


committed_case_metrics = pd.read_csv(
    REPO
    / "data/manifests/"
    "gate_b_case_metrics.csv"
)


diagnostic_05 = threshold_df[
    np.isclose(
        threshold_df[
            "threshold"
        ],
        0.5,
    )
].copy()


merged_validation = diagnostic_05.merge(
    committed_case_metrics,
    on=[
        "condition",
        "case_id",
    ],
    suffixes=(
        "_diag",
        "_08e",
    ),
)


if len(
    merged_validation
) != 16:

    raise RuntimeError(
        "0.5 evaluator cross-check merge failed."
    )


validation_errors = {
    "dice":
        float(
            np.max(
                np.abs(
                    merged_validation[
                        "dice"
                    ]
                    - merged_validation[
                        "dice_at_0p5"
                    ]
                )
            )
        ),

    "iou":
        float(
            np.max(
                np.abs(
                    merged_validation[
                        "iou"
                    ]
                    - merged_validation[
                        "iou_at_0p5"
                    ]
                )
            )
        ),

    "fp_components":
        float(
            np.max(
                np.abs(
                    merged_validation[
                        "fp_components"
                    ]
                    - merged_validation[
                        "fp_components_at_0p5"
                    ]
                )
            )
        ),

    "fp_volume_ml":
        float(
            np.max(
                np.abs(
                    merged_validation[
                        "fp_volume_ml"
                    ]
                    - merged_validation[
                        "fp_volume_ml_at_0p5"
                    ]
                )
            )
        ),

    "iou_0p10_recall":
        float(
            np.max(
                np.abs(
                    merged_validation[
                        "iou_0p10_recall"
                    ]
                    - merged_validation[
                        "primary_iou_component_recall_at_0p5"
                    ]
                )
            )
        ),

    "iou_0p25_recall":
        float(
            np.max(
                np.abs(
                    merged_validation[
                        "iou_0p25_recall"
                    ]
                    - merged_validation[
                        "iou_0p25_component_recall"
                    ]
                )
            )
        ),

    "any_overlap_recall":
        float(
            np.max(
                np.abs(
                    merged_validation[
                        "any_overlap_recall"
                    ]
                    - merged_validation[
                        "any_overlap_component_recall"
                    ]
                )
            )
        ),
}


if max(
    validation_errors.values()
) > 1e-8:

    raise RuntimeError(
        "Diagnostic evaluator does not exactly reproduce Block-08E "
        "fixed-threshold results:\n"
        + json.dumps(
            validation_errors,
            indent=2,
        )
    )


print(
    "✓ Block-08E metric reproduction       : PASS"
)

print(
    "✓ Maximum diagnostic discrepancy      :",
    "{:.3e}".format(
        max(
            validation_errors.values()
        )
    ),
)


# ==========================================================================================
# 8. EXACT TRAINING-SUPERVISION EXPOSURE REPLAY
# ==========================================================================================

heading(
    "STEP 4/9 — REPLAY EXACT SPARSE-SUPERVISION EXPOSURE WITHOUT TRAINING"
)


cache_manifest_df = pd.read_csv(
    CACHE_MANIFEST
)


epoch_log_df = pd.read_csv(
    REPO
    / "data/manifests/"
    "gate_b_v1_2_epoch_log.csv"
)


def load_sparse_meta(
    case_id,
    condition,
):

    rows = cache_manifest_df[
        (
            cache_manifest_df[
                "case_id"
            ].astype(
                str
            )
            == str(
                case_id
            )
        )
        & (
            cache_manifest_df[
                "condition"
            ].astype(
                str
            )
            == str(
                condition
            )
        )
    ]


    if len(
        rows
    ) != 1:

        raise RuntimeError(
            "Cache-row lookup failed for "
            + str(
                case_id
            )
            + "/"
            + str(
                condition
            )
        )


    row = rows.iloc[
        0
    ]


    image_path = (
        CACHE_ROOT
        / str(
            row[
                "image_file"
            ]
        )
    )


    annotation_path = (
        CACHE_ROOT
        / str(
            row[
                "annotation_file"
            ]
        )
    )


    with np.load(
        image_path,
        allow_pickle=False,
    ) as image_npz:

        shape_zyx = np.asarray(
            image_npz[
                "crop_shape_zyx"
            ],
            dtype=np.int32,
        )


    with np.load(
        annotation_path,
        allow_pickle=False,
    ) as annotation_npz:

        coords = np.asarray(
            annotation_npz[
                "supervision_voxel_zyx"
            ],
            dtype=np.int32,
        )


        labels = np.asarray(
            annotation_npz[
                "supervision_label"
            ],
            dtype=np.int8,
        )


        membership_coords = np.asarray(
            annotation_npz[
                "fg_membership_voxel_zyx"
            ],
            dtype=np.int32,
        )


        membership_groups = np.asarray(
            annotation_npz[
                "fg_membership_group_id"
            ],
            dtype=np.int32,
        )


    return {
        "shape_zyx":
            shape_zyx,

        "coords":
            coords,

        "labels":
            labels,

        "membership_coords":
            membership_coords,

        "membership_groups":
            membership_groups,
    }


exposure_case_rows = []


for condition in CONDITIONS:

    condition_meta = {
        case_id:
            load_sparse_meta(
                case_id,
                condition,
            )
        for case_id
        in development_cases
    }


    accumulators = {}


    for case_id in development_cases:

        available_groups = {
            int(
                group_id
            )
            for group_id
            in np.unique(
                condition_meta[
                    case_id
                ][
                    "membership_groups"
                ]
            ).tolist()
        }


        accumulators[
            case_id
        ] = {
            "patches":
                0,

            "labelled_voxels_seen":
                0,

            "foreground_voxels_seen":
                0,

            "background_voxels_seen":
                0,

            "zero_label_patches":
                0,

            "foreground_centered":
                0,

            "background_centered":
                0,

            "uniform_crop":
                0,

            "group_presence_events":
                0,

            "available_groups":
                available_groups,

            "seen_groups":
                set(),
        }


    for epoch in tqdm(
        range(
            EPOCHS
        ),
        desc=(
            "Exposure "
            + condition
        ),
        leave=False,
    ):

        schedule = make_case_schedule(
            development_cases,
            epoch,
        )


        for microbatch_index, case_id in enumerate(
            schedule
        ):

            meta = condition_meta[
                case_id
            ]


            global_microbatch_index = (
                epoch
                * MICROBATCHES_PER_EPOCH
                + microbatch_index
            )


            origin, source = deterministic_patch_origin(
                meta[
                    "shape_zyx"
                ],
                PATCH_ZYX,
                meta[
                    "coords"
                ],
                meta[
                    "labels"
                ],
                seed=SEED,
                sample_index=global_microbatch_index,
            )


            end = (
                origin
                + np.asarray(
                    PATCH_ZYX,
                    dtype=np.int32,
                )
            )


            direct_inside = np.all(
                (
                    meta[
                        "coords"
                    ]
                    >= origin[
                        None,
                        :
                    ]
                )
                & (
                    meta[
                        "coords"
                    ]
                    < end[
                        None,
                        :
                    ]
                ),
                axis=1,
            )


            local_labels = meta[
                "labels"
            ][
                direct_inside
            ]


            membership_inside = np.all(
                (
                    meta[
                        "membership_coords"
                    ]
                    >= origin[
                        None,
                        :
                    ]
                )
                & (
                    meta[
                        "membership_coords"
                    ]
                    < end[
                        None,
                        :
                    ]
                ),
                axis=1,
            )


            local_groups = {
                int(
                    value
                )
                for value
                in np.unique(
                    meta[
                        "membership_groups"
                    ][
                        membership_inside
                    ]
                ).tolist()
            }


            accumulator = accumulators[
                case_id
            ]


            accumulator[
                "patches"
            ] += 1


            labelled_count = int(
                len(
                    local_labels
                )
            )


            foreground_count = int(
                (
                    local_labels
                    == 1
                ).sum()
            )


            background_count = int(
                (
                    local_labels
                    == 0
                ).sum()
            )


            accumulator[
                "labelled_voxels_seen"
            ] += labelled_count


            accumulator[
                "foreground_voxels_seen"
            ] += foreground_count


            accumulator[
                "background_voxels_seen"
            ] += background_count


            if labelled_count == 0:

                accumulator[
                    "zero_label_patches"
                ] += 1


            if source == "foreground":

                accumulator[
                    "foreground_centered"
                ] += 1


            elif source == "background":

                accumulator[
                    "background_centered"
                ] += 1


            elif source == "random":

                accumulator[
                    "uniform_crop"
                ] += 1


            else:

                raise RuntimeError(
                    "Unexpected patch-source label."
                )


            accumulator[
                "group_presence_events"
            ] += len(
                local_groups
            )


            accumulator[
                "seen_groups"
            ].update(
                local_groups
            )


    for case_id in development_cases:

        accumulator = accumulators[
            case_id
        ]


        available_count = len(
            accumulator[
                "available_groups"
            ]
        )


        seen_count = len(
            accumulator[
                "seen_groups"
            ]
        )


        exposure_case_rows.append(
            {
                "condition":
                    condition,

                "case_id":
                    case_id,

                "source_origin":
                    str(
                        case_lookup[
                            case_id
                        ][
                            "source_origin"
                        ]
                    ),

                "patches":
                    accumulator[
                        "patches"
                    ],

                "labelled_voxels_seen":
                    accumulator[
                        "labelled_voxels_seen"
                    ],

                "foreground_voxels_seen":
                    accumulator[
                        "foreground_voxels_seen"
                    ],

                "background_voxels_seen":
                    accumulator[
                        "background_voxels_seen"
                    ],

                "zero_label_patches":
                    accumulator[
                        "zero_label_patches"
                    ],

                "foreground_centered":
                    accumulator[
                        "foreground_centered"
                    ],

                "background_centered":
                    accumulator[
                        "background_centered"
                    ],

                "uniform_crop":
                    accumulator[
                        "uniform_crop"
                    ],

                "available_fg_groups":
                    available_count,

                "seen_fg_groups":
                    seen_count,

                "seen_group_fraction":
                    (
                        seen_count
                        / available_count
                        if available_count
                        else float(
                            "nan"
                        )
                    ),

                "group_presence_events":
                    accumulator[
                        "group_presence_events"
                    ],
            }
        )


exposure_case_df = pd.DataFrame(
    exposure_case_rows
)


exposure_summary_df = (
    exposure_case_df.groupby(
        "condition",
        as_index=False,
    )
    .agg(
        patches=(
            "patches",
            "sum",
        ),

        labelled_voxels_seen=(
            "labelled_voxels_seen",
            "sum",
        ),

        foreground_voxels_seen=(
            "foreground_voxels_seen",
            "sum",
        ),

        background_voxels_seen=(
            "background_voxels_seen",
            "sum",
        ),

        zero_label_patches=(
            "zero_label_patches",
            "sum",
        ),

        foreground_centered=(
            "foreground_centered",
            "sum",
        ),

        background_centered=(
            "background_centered",
            "sum",
        ),

        uniform_crop=(
            "uniform_crop",
            "sum",
        ),

        available_fg_groups=(
            "available_fg_groups",
            "sum",
        ),

        seen_fg_groups=(
            "seen_fg_groups",
            "sum",
        ),

        group_presence_events=(
            "group_presence_events",
            "sum",
        ),
    )
)


exposure_summary_df[
    "mean_labels_per_patch"
] = (
    exposure_summary_df[
        "labelled_voxels_seen"
    ]
    / exposure_summary_df[
        "patches"
    ]
)


exposure_summary_df[
    "mean_fg_per_patch"
] = (
    exposure_summary_df[
        "foreground_voxels_seen"
    ]
    / exposure_summary_df[
        "patches"
    ]
)


exposure_summary_df[
    "mean_bg_per_patch"
] = (
    exposure_summary_df[
        "background_voxels_seen"
    ]
    / exposure_summary_df[
        "patches"
    ]
)


exposure_summary_df[
    "fg_to_bg_seen_ratio"
] = (
    exposure_summary_df[
        "foreground_voxels_seen"
    ]
    / np.maximum(
        exposure_summary_df[
            "background_voxels_seen"
        ],
        1,
    )
)


exposure_summary_df[
    "group_coverage_fraction"
] = (
    exposure_summary_df[
        "seen_fg_groups"
    ]
    / exposure_summary_df[
        "available_fg_groups"
    ]
)


# ------------------------------------------------------------------------------------------
# Exact comparison with Block-08D logged exposure.
# ------------------------------------------------------------------------------------------

logged_exposure_df = (
    epoch_log_df.groupby(
        "condition",
        as_index=False,
    )
    .agg(
        labelled_voxels_seen_logged=(
            "labelled_voxels_seen",
            "sum",
        ),

        foreground_voxels_seen_logged=(
            "foreground_voxels_seen",
            "sum",
        ),

        zero_label_patches_logged=(
            "zero_label_microbatches",
            "sum",
        ),

        foreground_centered_logged=(
            "foreground_centered",
            "sum",
        ),

        background_centered_logged=(
            "background_centered",
            "sum",
        ),

        uniform_crop_logged=(
            "uniform_crop",
            "sum",
        ),
    )
)


exposure_validation = exposure_summary_df.merge(
    logged_exposure_df,
    on="condition",
)


exact_columns = [
    (
        "labelled_voxels_seen",
        "labelled_voxels_seen_logged",
    ),
    (
        "foreground_voxels_seen",
        "foreground_voxels_seen_logged",
    ),
    (
        "zero_label_patches",
        "zero_label_patches_logged",
    ),
    (
        "foreground_centered",
        "foreground_centered_logged",
    ),
    (
        "background_centered",
        "background_centered_logged",
    ),
    (
        "uniform_crop",
        "uniform_crop_logged",
    ),
]


for observed_column, logged_column in exact_columns:

    if not np.array_equal(
        exposure_validation[
            observed_column
        ].to_numpy(),
        exposure_validation[
            logged_column
        ].to_numpy(),
    ):

        raise RuntimeError(
            "Exposure replay mismatch: "
            + observed_column
        )


print(
    "✓ Exact training-exposure replay       : PASS"
)

print(
    "✓ Exposure totals match Block 08D logs : PASS"
)

print()

print(
    exposure_summary_df.to_string(
        index=False
    )
)


# ==========================================================================================
# 9. AGGREGATE THRESHOLD / OUTPUT DIAGNOSTICS
# ==========================================================================================

heading(
    "STEP 5/9 — AGGREGATE FAILURE-MECHANISM DIAGNOSTICS"
)


threshold_summary_df = (
    threshold_df.groupby(
        [
            "condition",
            "threshold",
        ],
        as_index=False,
    )
    .agg(
        macro_dice=(
            "dice",
            "mean",
        ),

        macro_iou=(
            "iou",
            "mean",
        ),

        mean_predicted_components=(
            "predicted_components",
            "mean",
        ),

        mean_component_to_reference_ratio=(
            "component_to_reference_ratio",
            "mean",
        ),

        mean_fp_components=(
            "fp_components",
            "mean",
        ),

        mean_fp_volume_ml=(
            "fp_volume_ml",
            "mean",
        ),

        macro_iou_0p10_recall=(
            "iou_0p10_recall",
            "mean",
        ),

        macro_iou_0p25_recall=(
            "iou_0p25_recall",
            "mean",
        ),

        macro_any_overlap_recall=(
            "any_overlap_recall",
            "mean",
        ),

        mean_predicted_foreground_volume_ml=(
            "predicted_foreground_volume_ml",
            "mean",
        ),

        mean_median_component_volume_ml=(
            "median_component_volume_ml",
            "mean",
        ),

        mean_largest_component_fraction=(
            "largest_component_volume_fraction",
            "mean",
        ),

        mean_fraction_components_lt_0p1ml=(
            "fraction_components_lt_0p1ml",
            "mean",
        ),

        mean_fraction_components_lt_0p5ml=(
            "fraction_components_lt_0p5ml",
            "mean",
        ),
    )
)


output_summary_df = (
    output_df.groupby(
        "condition",
        as_index=False,
    )
    .agg(
        mean_sampled_voxel_auc=(
            "sampled_voxel_auc",
            "mean",
        ),

        mean_balanced_brier=(
            "balanced_brier",
            "mean",
        ),

        mean_lesion_probability=(
            "lesion_probability_mean",
            "mean",
        ),

        mean_background_probability=(
            "background_probability_mean",
            "mean",
        ),

        mean_probability_separation=(
            "lesion_minus_background_mean",
            "mean",
        ),

        mean_lesion_boundary_probability=(
            "lesion_boundary_probability_mean",
            "mean",
        ),

        mean_perilesional_background_probability=(
            "perilesional_background_probability_mean",
            "mean",
        ),

        mean_far_background_probability=(
            "far_background_probability_mean",
            "mean",
        ),

        mean_lesion_fraction_ge_0p5=(
            "lesion_fraction_ge_0p5",
            "mean",
        ),

        mean_background_fraction_ge_0p5=(
            "background_fraction_ge_0p5",
            "mean",
        ),
    )
)


# ------------------------------------------------------------------------------------------
# Best diagnostic threshold per condition.
#
# Diagnostic only. It CANNOT modify Gate B.
# ------------------------------------------------------------------------------------------

best_threshold_rows = []


for condition in CONDITIONS:

    subset = threshold_summary_df[
        threshold_summary_df[
            "condition"
        ]
        == condition
    ].copy()


    row_05 = subset[
        np.isclose(
            subset[
                "threshold"
            ],
            0.5,
        )
    ].iloc[
        0
    ]


    best = (
        subset.sort_values(
            [
                "macro_dice",
                "macro_any_overlap_recall",
                "threshold",
            ],
            ascending=[
                False,
                False,
                False,
            ],
        )
        .iloc[
            0
        ]
    )


    dice_improvement = (
        float(
            best[
                "macro_dice"
            ]
        )
        - float(
            row_05[
                "macro_dice"
            ]
        )
    )


    best_threshold_rows.append(
        {
            "condition":
                condition,

            "dice_at_0p5":
                float(
                    row_05[
                        "macro_dice"
                    ]
                ),

            "best_diagnostic_threshold":
                float(
                    best[
                        "threshold"
                    ]
                ),

            "best_diagnostic_macro_dice":
                float(
                    best[
                        "macro_dice"
                    ]
                ),

            "dice_improvement_vs_0p5":
                dice_improvement,

            "best_diagnostic_any_overlap_recall":
                float(
                    best[
                        "macro_any_overlap_recall"
                    ]
                ),

            "best_diagnostic_predicted_components":
                float(
                    best[
                        "mean_predicted_components"
                    ]
                ),

            "threshold_sensitive_heuristic":
                bool(
                    dice_improvement
                    >= 0.05
                    and abs(
                        float(
                            best[
                                "threshold"
                            ]
                        )
                        - 0.5
                    )
                    >= 0.10
                ),
        }
    )


best_threshold_df = pd.DataFrame(
    best_threshold_rows
)


# ==========================================================================================
# 10. SOURCE-SPECIFIC AUDIT
# ==========================================================================================

primary_case_df = pd.read_csv(
    REPO
    / "data/manifests/"
    "gate_b_primary_case_differences.csv"
)


source_lookup = {
    str(
        row[
            "case_id"
        ]
    ):
        str(
            row[
                "source_origin"
            ]
        )
    for _, row
    in development_df.iterrows()
}


primary_case_df[
    "source_origin"
] = primary_case_df[
    "case_id"
].astype(
    str
).map(
    source_lookup
)


source_primary_df = (
    primary_case_df.groupby(
        "source_origin",
        as_index=False,
    )
    .agg(
        cases=(
            "case_id",
            "count",
        ),

        mean_pixel_matched_R_at_1=(
            "pixel_matched_R_at_1",
            "mean",
        ),

        mean_component_omission_R_at_1=(
            "component_omission_R_at_1",
            "mean",
        ),

        mean_delta_R_at_1=(
            "delta_R_at_1",
            "mean",
        ),
    )
)


source_threshold_05_df = (
    diagnostic_05.groupby(
        [
            "source_origin",
            "condition",
        ],
        as_index=False,
    )
    .agg(
        mean_dice=(
            "dice",
            "mean",
        ),

        mean_iou=(
            "iou",
            "mean",
        ),

        mean_components=(
            "predicted_components",
            "mean",
        ),

        mean_fp_components=(
            "fp_components",
            "mean",
        ),

        mean_any_overlap_recall=(
            "any_overlap_recall",
            "mean",
        ),
    )
)


print(
    "Primary source-specific R@1 differences:"
)

print()

print(
    source_primary_df.to_string(
        index=False
    )
)


# ==========================================================================================
# 11. DESCRIPTIVE MECHANISM FLAGS
# ==========================================================================================

heading(
    "STEP 6/9 — COMPUTE DESCRIPTIVE MECHANISM FLAGS"
)


threshold_05_summary = threshold_summary_df[
    np.isclose(
        threshold_summary_df[
            "threshold"
        ],
        0.5,
    )
].set_index(
    "condition"
)


fragmentation_ratios = {
    condition:
        float(
            threshold_05_summary.loc[
                condition,
                "mean_component_to_reference_ratio",
            ]
        )
    for condition in CONDITIONS
}


severe_fragmentation_conditions = [
    condition
    for condition, ratio
    in fragmentation_ratios.items()
    if ratio >= 10.0
]


threshold_sensitive_conditions = (
    best_threshold_df[
        best_threshold_df[
            "threshold_sensitive_heuristic"
        ]
    ][
        "condition"
    ].tolist()
)


natural_dice = float(
    threshold_05_summary.loc[
        PRIMARY_OMISSION,
        "macro_dice",
    ]
)


pixel_dice = float(
    threshold_05_summary.loc[
        PRIMARY_CONTROL,
        "macro_dice",
    ]
)


complete_dice = float(
    threshold_05_summary.loc[
        "complete",
        "macro_dice",
    ]
)


natural_components = float(
    threshold_05_summary.loc[
        PRIMARY_OMISSION,
        "mean_predicted_components",
    ]
)


pixel_components = float(
    threshold_05_summary.loc[
        PRIMARY_CONTROL,
        "mean_predicted_components",
    ]
)


complete_components = float(
    threshold_05_summary.loc[
        "complete",
        "mean_predicted_components",
    ]
)


natural_fp_volume = float(
    threshold_05_summary.loc[
        PRIMARY_OMISSION,
        "mean_fp_volume_ml",
    ]
)


pixel_fp_volume = float(
    threshold_05_summary.loc[
        PRIMARY_CONTROL,
        "mean_fp_volume_ml",
    ]
)


complete_fp_volume = float(
    threshold_05_summary.loc[
        "complete",
        "mean_fp_volume_ml",
    ]
)


regularization_like_pattern = bool(
    natural_dice
    > pixel_dice
    and natural_dice
    > complete_dice
    and natural_components
    < pixel_components
    and natural_components
    < complete_components
    and natural_fp_volume
    < pixel_fp_volume
    and natural_fp_volume
    < complete_fp_volume
)


# This is deliberately NOT a scientific defect verdict.
#
# It only indicates whether the diagnostics reveal substantial baseline
# pathology that merits protocol review.
baseline_pathology_flag = bool(
    len(
        severe_fragmentation_conditions
    )
    >= 2
)


mechanism_flags = {
    "severe_fragmentation_heuristic_definition":
        "mean predicted-component/reference-component ratio >= 10 at threshold 0.5",

    "severe_fragmentation_conditions":
        severe_fragmentation_conditions,

    "threshold_sensitivity_heuristic_definition":
        (
            "best diagnostic macro Dice improves by >=0.05 over threshold 0.5 "
            "and best threshold differs from 0.5 by >=0.10"
        ),

    "threshold_sensitive_conditions":
        threshold_sensitive_conditions,

    "regularization_like_pattern_definition":
        (
            "natural omission has higher Dice, fewer components, and lower FP volume "
            "than both complete and matched-pixel conditions at threshold 0.5"
        ),

    "regularization_like_pattern":
        regularization_like_pattern,

    "baseline_pathology_flag":
        baseline_pathology_flag,

    "implementation_defect_proven":
        False,

    "root_cause_status":
        "SCIENTIFIC_REVIEW_REQUIRED",
}


print(
    "Fragmentation ratio @0.5:"
)


for condition in CONDITIONS:

    print(
        "  {:29s} {:.2f}x".format(
            condition,
            fragmentation_ratios[
                condition
            ],
        )
    )


print()

print(
    "Severe fragmentation conditions      :",
    (
        ", ".join(
            severe_fragmentation_conditions
        )
        if severe_fragmentation_conditions
        else "NONE"
    ),
)

print(
    "Threshold-sensitive conditions       :",
    (
        ", ".join(
            threshold_sensitive_conditions
        )
        if threshold_sensitive_conditions
        else "NONE"
    ),
)

print(
    "Regularization-like pattern          :",
    (
        "YES"
        if regularization_like_pattern
        else "NO"
    ),
)

print(
    "Implementation defect proven         : NO"
)

print(
    "Root-cause verdict                   : SCIENTIFIC_REVIEW_REQUIRED"
)


# ==========================================================================================
# 12. VISUAL FAILURE AUDIT
# ==========================================================================================

heading(
    "STEP 7/9 — GENERATE DEVELOPMENT-ONLY FAILURE OVERLAYS"
)


diagnostic_figure_dir = (
    REPO
    / "figures/diagnostics/"
    "gate_b_failure"
)


diagnostic_figure_dir.mkdir(
    parents=True,
    exist_ok=True,
)


condition_display = {
    "complete":
        "Complete",

    "pixel_dropout_matched_50":
        "Matched Pixel",

    "component_natural_50":
        "Natural Omission",

    "component_fixed_50":
        "Fixed Omission",
}


for case_id in tqdm(
    development_cases,
    desc="Overlay figures",
):

    meta = reference_metadata[
        case_id
    ]


    mask_img = nib.load(
        str(
            meta[
                "mask_path"
            ]
        ),
        mmap=True,
    )


    reference = (
        np.asarray(
            mask_img.dataobj
        )
        > 0.5
    )


    ct_img = nib.load(
        str(
            meta[
                "ct_path"
            ]
        ),
        mmap=True,
    )


    ct = np.asarray(
        ct_img.dataobj,
        dtype=np.float32,
    )


    source_origin = str(
        meta[
            "source_origin"
        ]
    )


    if source_origin.lower() == "radiopaedia":

        display_ct = np.clip(
            ct,
            0.0,
            255.0,
        ) / 255.0


    else:

        display_ct = (
            np.clip(
                ct,
                -1250.0,
                250.0,
            )
            + 1250.0
        ) / 1500.0


    lesion_area_per_z = reference.sum(
        axis=(
            0,
            1,
        )
    )


    z_gt = int(
        np.argmax(
            lesion_area_per_z
        )
    )


    # Choose second slice from the largest pixel-vs-natural model disagreement.
    with np.load(
        PREDICTION_ROOT
        / PRIMARY_CONTROL
        / (
            case_id
            + ".npz"
        ),
        allow_pickle=False,
    ) as npz:

        pixel_probability = np.asarray(
            npz[
                "probability_xyz"
            ],
            dtype=np.float32,
        )


    with np.load(
        PREDICTION_ROOT
        / PRIMARY_OMISSION
        / (
            case_id
            + ".npz"
        ),
        allow_pickle=False,
    ) as npz:

        natural_probability = np.asarray(
            npz[
                "probability_xyz"
            ],
            dtype=np.float32,
        )


    disagreement_per_z = np.abs(
        pixel_probability
        - natural_probability
    ).sum(
        axis=(
            0,
            1,
        )
    )


    z_difference = int(
        np.argmax(
            disagreement_per_z
        )
    )


    del pixel_probability
    del natural_probability


    selected_z = [
        z_gt,
        z_difference,
    ]


    probability_slices = {}


    for condition in CONDITIONS:

        with np.load(
            PREDICTION_ROOT
            / condition
            / (
                case_id
                + ".npz"
            ),
            allow_pickle=False,
        ) as npz:

            probability = np.asarray(
                npz[
                    "probability_xyz"
                ],
                dtype=np.float32,
            )


            probability_slices[
                condition
            ] = [
                probability[
                    :,
                    :,
                    z_index,
                ].copy()
                for z_index
                in selected_z
            ]


        del probability


    fig, axes = plt.subplots(
        2,
        5,
        figsize=(
            18,
            8.4,
        ),
    )


    for row_index, z_index in enumerate(
        selected_z
    ):

        ct_slice = display_ct[
            :,
            :,
            z_index,
        ].T


        gt_slice = reference[
            :,
            :,
            z_index,
        ].T


        axes[
            row_index,
            0
        ].imshow(
            ct_slice,
            cmap="gray",
            origin="lower",
            vmin=0,
            vmax=1,
        )


        if gt_slice.any():

            axes[
                row_index,
                0
            ].contour(
                gt_slice,
                levels=[
                    0.5
                ],
                colors=[
                    "lime"
                ],
                linewidths=1.4,
            )


        axes[
            row_index,
            0
        ].set_title(
            (
                "Ground Truth"
                if row_index == 0
                else "GT / Reference"
            ),
            fontweight="bold",
        )


        row_label = (
            "Max GT slice"
            if row_index == 0
            else "Max Pixel–Omission Disagreement"
        )


        axes[
            row_index,
            0
        ].set_ylabel(
            row_label
            + "\n"
            + "z="
            + str(
                z_index
            ),
            fontweight="bold",
        )


        for column_index, condition in enumerate(
            CONDITIONS,
            start=1,
        ):

            probability_slice = (
                probability_slices[
                    condition
                ][
                    row_index
                ].T
            )


            axes[
                row_index,
                column_index
            ].imshow(
                ct_slice,
                cmap="gray",
                origin="lower",
                vmin=0,
                vmax=1,
            )


            axes[
                row_index,
                column_index
            ].imshow(
                probability_slice,
                cmap="magma",
                origin="lower",
                vmin=0,
                vmax=1,
                alpha=0.32,
            )


            if gt_slice.any():

                axes[
                    row_index,
                    column_index
                ].contour(
                    gt_slice,
                    levels=[
                        0.5
                    ],
                    colors=[
                        "lime"
                    ],
                    linewidths=1.2,
                )


            prediction_binary = (
                probability_slice
                >= 0.5
            )


            if prediction_binary.any():

                axes[
                    row_index,
                    column_index
                ].contour(
                    prediction_binary,
                    levels=[
                        0.5
                    ],
                    colors=[
                        "red"
                    ],
                    linewidths=0.9,
                )


            axes[
                row_index,
                column_index
            ].set_title(
                condition_display[
                    condition
                ],
                fontweight="bold",
            )


    for axis in axes.reshape(
        -1
    ):

        axis.set_xticks(
            []
        )

        axis.set_yticks(
            []
        )


    fig.suptitle(
        (
            "Gate-B Failure Mechanism Audit — "
            + case_id
            + " ("
            + source_origin
            + ")\n"
            + "Green = Dense Reference | Red = p≥0.5 Boundary | Heat = Probability"
        ),
        fontweight="bold",
        fontsize=14,
    )


    fig.tight_layout(
        rect=[
            0,
            0,
            1,
            0.93,
        ]
    )


    safe_case = case_id.replace(
        "/",
        "_",
    )


    fig.savefig(
        diagnostic_figure_dir
        / (
            "fig_gate_b_failure_overlay_"
            + safe_case
            + ".png"
        ),
        dpi=500,
        bbox_inches="tight",
    )


    fig.savefig(
        diagnostic_figure_dir
        / (
            "fig_gate_b_failure_overlay_"
            + safe_case
            + ".pdf"
        ),
        bbox_inches="tight",
    )


    plt.close(
        fig
    )


    del ct
    del display_ct
    del reference
    del probability_slices

    gc.collect()


# ==========================================================================================
# 13. AGGREGATE DIAGNOSTIC FIGURES
# ==========================================================================================

# ------------------------------------------------------------------------------------------
# Figure 17 — fragmentation versus probability threshold.
# ------------------------------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(
        10.5,
        6.3,
    )
)


for condition in CONDITIONS:

    subset = threshold_summary_df[
        threshold_summary_df[
            "condition"
        ]
        == condition
    ]


    ax.plot(
        subset[
            "threshold"
        ],
        subset[
            "mean_predicted_components"
        ],
        marker="o",
        linewidth=2,
        label=condition,
    )


ax.set_yscale(
    "log"
)


ax.axvline(
    0.5,
    linestyle="--",
    linewidth=1.2,
)


ax.set_xlabel(
    "Voxel Probability Threshold",
    fontweight="bold",
)


ax.set_ylabel(
    "Mean Number of 3-D Prediction Components (log scale)",
    fontweight="bold",
)


ax.set_title(
    "Gate-B Failure Audit: Prediction Fragmentation Across Thresholds\n"
    "No Prediction-Size Filtering",
    fontweight="bold",
)


ax.grid(
    alpha=0.25,
)


ax.legend(
    fontsize=8,
)


fig.tight_layout()


fig.savefig(
    diagnostic_figure_dir
    / "fig17_fragmentation_vs_threshold.png",
    dpi=600,
    bbox_inches="tight",
)


fig.savefig(
    diagnostic_figure_dir
    / "fig17_fragmentation_vs_threshold.pdf",
    bbox_inches="tight",
)


plt.close(
    fig
)


# ------------------------------------------------------------------------------------------
# Figure 18 — Dice versus diagnostic threshold.
# ------------------------------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(
        10.5,
        6.3,
    )
)


for condition in CONDITIONS:

    subset = threshold_summary_df[
        threshold_summary_df[
            "condition"
        ]
        == condition
    ]


    ax.plot(
        subset[
            "threshold"
        ],
        subset[
            "macro_dice"
        ],
        marker="o",
        linewidth=2,
        label=condition,
    )


ax.axvline(
    0.5,
    linestyle="--",
    linewidth=1.2,
)


ax.set_xlabel(
    "Voxel Probability Threshold",
    fontweight="bold",
)


ax.set_ylabel(
    "Mean Development Dice",
    fontweight="bold",
)


ax.set_ylim(
    0,
    max(
        0.4,
        float(
            threshold_summary_df[
                "macro_dice"
            ].max()
        )
        * 1.12,
    ),
)


ax.set_title(
    "Gate-B Failure Audit: Threshold Sensitivity of Segmentation Overlap\n"
    "Diagnostic Only — Locked Gate-B Result Remains NO-GO",
    fontweight="bold",
)


ax.grid(
    alpha=0.25,
)


ax.legend(
    fontsize=8,
)


fig.tight_layout()


fig.savefig(
    diagnostic_figure_dir
    / "fig18_dice_vs_threshold.png",
    dpi=600,
    bbox_inches="tight",
)


fig.savefig(
    diagnostic_figure_dir
    / "fig18_dice_vs_threshold.pdf",
    bbox_inches="tight",
)


plt.close(
    fig
)


# ------------------------------------------------------------------------------------------
# Figure 19 — probability separation.
# ------------------------------------------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(
        10,
        6.2,
    )
)


output_plot = output_summary_df.set_index(
    "condition"
).loc[
    CONDITIONS
]


x = np.arange(
    len(
        CONDITIONS
    )
)


width = 0.34


ax.bar(
    x
    - width
    / 2,
    output_plot[
        "mean_lesion_probability"
    ],
    width,
    label="Lesion voxels",
)


ax.bar(
    x
    + width
    / 2,
    output_plot[
        "mean_background_probability"
    ],
    width,
    label="Background voxels",
)


ax.set_xticks(
    x
)


ax.set_xticklabels(
    [
        condition_display[
            condition
        ]
        for condition in CONDITIONS
    ],
    rotation=15,
    ha="right",
    fontweight="bold",
)


ax.set_ylim(
    0,
    1,
)


ax.set_ylabel(
    "Mean Predicted Probability",
    fontweight="bold",
)


ax.set_title(
    "Gate-B Failure Audit: Lesion–Background Probability Separation",
    fontweight="bold",
)


ax.grid(
    axis="y",
    alpha=0.25,
)


ax.legend()


fig.tight_layout()


fig.savefig(
    diagnostic_figure_dir
    / "fig19_probability_separation.png",
    dpi=600,
    bbox_inches="tight",
)


fig.savefig(
    diagnostic_figure_dir
    / "fig19_probability_separation.pdf",
    bbox_inches="tight",
)


plt.close(
    fig
)


# ------------------------------------------------------------------------------------------
# Figure 20 — actual foreground supervision exposure.
# ------------------------------------------------------------------------------------------

exposure_plot = exposure_summary_df.set_index(
    "condition"
).loc[
    CONDITIONS
]


fig, ax = plt.subplots(
    figsize=(
        10,
        6.2,
    )
)


bars = ax.bar(
    np.arange(
        len(
            CONDITIONS
        )
    ),
    exposure_plot[
        "foreground_voxels_seen"
    ],
)


ax.set_xticks(
    np.arange(
        len(
            CONDITIONS
        )
    )
)


ax.set_xticklabels(
    [
        condition_display[
            condition
        ]
        for condition in CONDITIONS
    ],
    rotation=15,
    ha="right",
    fontweight="bold",
)


ax.set_ylabel(
    "Foreground Sparse-Voxel Exposures Across Training",
    fontweight="bold",
)


ax.set_title(
    "Gate-B Failure Audit: Actual Foreground-Supervision Exposure\n"
    "Exact Replay of the Frozen Training Schedule",
    fontweight="bold",
)


ax.grid(
    axis="y",
    alpha=0.25,
)


fig.tight_layout()


fig.savefig(
    diagnostic_figure_dir
    / "fig20_sparse_supervision_exposure.png",
    dpi=600,
    bbox_inches="tight",
)


fig.savefig(
    diagnostic_figure_dir
    / "fig20_sparse_supervision_exposure.pdf",
    bbox_inches="tight",
)


plt.close(
    fig
)


# ==========================================================================================
# 14. SAVE DIAGNOSTIC TABLES
# ==========================================================================================

heading(
    "STEP 8/9 — FREEZE FAILURE-AUDIT TABLES AND SCIENTIFIC INTERPRETATION"
)


manifest_dir = (
    REPO
    / "data/manifests"
)


threshold_df.to_csv(
    manifest_dir
    / "gate_b_failure_threshold_case_audit.csv",
    index=False,
)


threshold_summary_df.to_csv(
    manifest_dir
    / "gate_b_failure_threshold_summary.csv",
    index=False,
)


output_df.to_csv(
    manifest_dir
    / "gate_b_failure_output_case_audit.csv",
    index=False,
)


output_summary_df.to_csv(
    manifest_dir
    / "gate_b_failure_output_summary.csv",
    index=False,
)


exposure_case_df.to_csv(
    manifest_dir
    / "gate_b_failure_supervision_exposure_case.csv",
    index=False,
)


exposure_summary_df.to_csv(
    manifest_dir
    / "gate_b_failure_supervision_exposure_summary.csv",
    index=False,
)


best_threshold_df.to_csv(
    manifest_dir
    / "gate_b_failure_best_diagnostic_threshold.csv",
    index=False,
)


source_primary_df.to_csv(
    manifest_dir
    / "gate_b_failure_source_primary_r_at_1.csv",
    index=False,
)


source_threshold_05_df.to_csv(
    manifest_dir
    / "gate_b_failure_source_threshold_0p5.csv",
    index=False,
)


# ==========================================================================================
# 15. AUDIT JSON + DOCUMENT
# ==========================================================================================

audit_payload = {
    "project":
        "CORA-Lung",

    "block":
        "08F",

    "status":
        "PASS",

    "generated_at_utc":
        NOW_ISO,

    "purpose":
        "development-only mechanism audit after locked Gate-B NO-GO",

    "gate_b_before_audit":
        "NO_GO",

    "gate_b_after_audit":
        "NO_GO",

    "gate_b_decision_modified":
        False,

    "gate_c":
        "NOT_RUN",

    "optimizer_steps_in_block08f":
        0,

    "models_retrained":
        False,

    "checkpoints_reselected":
        False,

    "predictions_regenerated":
        False,

    "development_dense_cases_used":
        4,

    "final_outer_cv_cases_used":
        0,

    "thresholds_diagnostic_only":
        THRESHOLDS.tolist(),

    "fixed_threshold_reproduction":
        {
            "status":
                "PASS",

            "maximum_absolute_error":
                max(
                    validation_errors.values()
                ),
        },

    "fragmentation_ratio_at_0p5":
        fragmentation_ratios,

    "mechanism_flags":
        mechanism_flags,

    "best_diagnostic_thresholds": {
        str(
            row[
                "condition"
            ]
        ):
            {
                "threshold":
                    float(
                        row[
                            "best_diagnostic_threshold"
                        ]
                    ),

                "macro_dice":
                    float(
                        row[
                            "best_diagnostic_macro_dice"
                        ]
                    ),

                "improvement_vs_0p5":
                    float(
                        row[
                            "dice_improvement_vs_0p5"
                        ]
                    ),
            }
        for _, row
        in best_threshold_df.iterrows()
    },

    "output_separation": {
        str(
            row[
                "condition"
            ]
        ):
            {
                "mean_sampled_auc":
                    float(
                        row[
                            "mean_sampled_voxel_auc"
                        ]
                    ),

                "mean_lesion_probability":
                    float(
                        row[
                            "mean_lesion_probability"
                        ]
                    ),

                "mean_background_probability":
                    float(
                        row[
                            "mean_background_probability"
                        ]
                    ),

                "mean_probability_separation":
                    float(
                        row[
                            "mean_probability_separation"
                        ]
                    ),
            }
        for _, row
        in output_summary_df.iterrows()
    },

    "source_primary_deltas": {
        str(
            row[
                "source_origin"
            ]
        ):
            float(
                row[
                    "mean_delta_R_at_1"
                ]
            )
        for _, row
        in source_primary_df.iterrows()
    },

    "scientific_conclusion":
        (
            "The locked Gate-B NO-GO result is unchanged. "
            "This diagnostic audit may identify baseline pathologies or "
            "regularization-like behavior, but it does not itself establish "
            "an implementation defect or authorize protocol modification."
        ),

    "next_action":
        (
            "Scientific review of Block 08F. "
            "Do not retrain, alter Gate B, implement CORA replay, "
            "or access final outer-CV outcomes until a specific defect or "
            "new protocol hypothesis is justified and frozen."
        ),
}


audit_dir = (
    REPO
    / "experiments/audits"
)


write_json(
    audit_dir
    / "block08f_gate_b_failure_mechanism_audit.json",
    audit_payload,
)


diagnostic_document = """
# Gate-B Failure Mechanism Audit

## Status

The locked Gate-B decision remains **NO-GO**.

Block 08F is diagnostic. It cannot revise Gate B, select a new checkpoint,
change a probability threshold for the locked result, or authorize CORA
replay.

## Questions examined

1. Prediction fragmentation across probability thresholds.
2. Diagnostic threshold sensitivity of Dice and lesion recall.
3. Lesion-versus-background probability separation.
4. Exact sparse-supervision exposure under the original frozen training
   schedule.
5. Source-specific behavior for Coronacases and Radiopaedia.
6. A descriptive regularization-like pattern in the natural component-
   omission condition.

## Interpretation constraint

A better diagnostic threshold does not invalidate the locked Gate-B result.

Likewise, finding that natural component omission has lower fragmentation or
better lesion/background separation does not establish a causal regularization
mechanism from four development cases and one seed.

Block 08F may justify a subsequent protocol-design discussion only if the
observed diagnostics identify a concrete, technically defensible problem.

No final outer-CV result may be inspected during that discussion.
"""


write_text(
    REPO
    / "docs/gate_b_failure_mechanism_audit.md",
    diagnostic_document,
)


# ==========================================================================================
# 16. CAPTURE EXECUTED SOURCE
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
        "CORA-LUNG — CODE BLOCK 08F"
        in cell
    ):

        source_path = (
            REPO
            / "scripts/code_blocks/"
            "block08f_gate_b_failure_mechanism_audit.py"
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
# 17. UPDATE PROJECT STATE — GATE B MUST REMAIN NO_GO
# ==========================================================================================

state.update(
    {
        "last_attempted_block":
            "08F",

        "last_completed_block":
            "08F",

        "last_completed_block_name":
            "gate_b_failure_mechanism_audit",

        "current_stage":
            "gate_b_failure_mechanism_audit_complete",

        "current_gate":
            "B",

        "gate_b":
            "NO_GO",

        "gate_c":
            "NOT_RUN",

        "gate_b_failure_audit":
            "PASS",

        "gate_b_failure_fragmentation_review":
            (
                "FLAGGED"
                if severe_fragmentation_conditions
                else "NOT_FLAGGED"
            ),

        "gate_b_failure_threshold_sensitivity_review":
            (
                "FLAGGED"
                if threshold_sensitive_conditions
                else "NOT_FLAGGED"
            ),

        "gate_b_failure_regularization_like_pattern":
            (
                "DESCRIPTIVE_YES"
                if regularization_like_pattern
                else "DESCRIPTIVE_NO"
            ),

        "gate_b_failure_root_cause":
            "SCIENTIFIC_REVIEW_REQUIRED",

        "training_authorized":
            False,

        "pilot_training_authorized":
            False,

        "optimizer_steps_performed":
            int(
                state[
                    "optimizer_steps_performed"
                ]
            ),

        "valid_gate_b_optimizer_steps":
            6000,

        "gate_b_outcomes_opened":
            True,

        "dense_development_evaluation":
            "PASS",

        "final_outer_cv_failure_audit_access":
            0,

        "next_action":
            (
                "Review Block 08F diagnostics. "
                "Do not retrain or implement CORA until a specific "
                "scientific defect or revised hypothesis is justified."
            ),

        "updated_at_utc":
            NOW_ISO,
    }
)


if (
    state[
        "gate_b"
    ]
    != "NO_GO"
):

    raise RuntimeError(
        "Block 08F illegally changed Gate B."
    )


if (
    state[
        "gate_c"
    ]
    != "NOT_RUN"
):

    raise RuntimeError(
        "Block 08F illegally changed Gate C."
    )


write_json(
    state_path,
    state,
)


# ==========================================================================================
# 18. REPOSITORY MANIFEST
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
            "08F",

        "gate_b":
            "NO_GO",

        "gate_c":
            "NOT_RUN",

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
# 19. COMMIT + PUSH
# ==========================================================================================

heading(
    "STEP 9/9 — COMMIT FAILURE-MECHANISM AUDIT"
)


git_paths = [
    "PROJECT_STATE.json",
    "REPOSITORY_MANIFEST.json",
    "data/manifests/gate_b_failure_threshold_case_audit.csv",
    "data/manifests/gate_b_failure_threshold_summary.csv",
    "data/manifests/gate_b_failure_output_case_audit.csv",
    "data/manifests/gate_b_failure_output_summary.csv",
    "data/manifests/gate_b_failure_supervision_exposure_case.csv",
    "data/manifests/gate_b_failure_supervision_exposure_summary.csv",
    "data/manifests/gate_b_failure_best_diagnostic_threshold.csv",
    "data/manifests/gate_b_failure_source_primary_r_at_1.csv",
    "data/manifests/gate_b_failure_source_threshold_0p5.csv",
    "experiments/audits/block08f_gate_b_failure_mechanism_audit.json",
    "docs/gate_b_failure_mechanism_audit.md",
    "figures/diagnostics/gate_b_failure",
]


source_path = (
    REPO
    / "scripts/code_blocks/"
    "block08f_gate_b_failure_mechanism_audit.py"
)


if source_path.exists():

    git_paths.append(
        "scripts/code_blocks/"
        "block08f_gate_b_failure_mechanism_audit.py"
    )


sh(
    [
        "git",
        "add",
        *git_paths,
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
        "No Block-08F artifacts available to commit."
    )


print(
    status
)


sh(
    [
        "git",
        "commit",
        "-m",
        "audit: diagnose locked Gate-B no-go mechanisms",
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
# 20. FINAL REPORT
# ==========================================================================================

print(
    "\n"
    + "=" * 122
)

print(
    "CORA-LUNG CODE BLOCK 08F — FINAL FAILURE-MECHANISM REPORT"
)

print(
    "=" * 122
)


print(
    "Locked Gate-B decision                : NO_GO"
)

print(
    "Gate-B decision modified              : NO"
)

print(
    "Gate C                                : NOT RUN"
)

print(
    "Final outer-CV cases accessed         : 0"
)

print(
    "Models retrained                      : NO"
)

print(
    "Checkpoints reselected                : NO"
)

print(
    "Predictions regenerated               : NO"
)

print(
    "Optimizer steps in Block 08F          : 0"
)

print(
    "Diagnostic thresholds                 :",
    THRESHOLDS.tolist(),
)

print(
    "Threshold-0.5 metric reproduction     : PASS"
)

print(
    "Maximum Block-08E reproduction error  :",
    "{:.3e}".format(
        max(
            validation_errors.values()
        )
    ),
)

print()
print(
    "FRAGMENTATION @ p=0.5"
)

print(
    "---------------------"
)


for condition in CONDITIONS:

    row = threshold_05_summary.loc[
        condition
    ]

    print(
        "  {:29s} components={:8.2f}  comp/ref={:8.2f}x  "
        "tiny<0.1mL={:.3f}  largest_frac={:.3f}".format(
            condition,
            float(
                row[
                    "mean_predicted_components"
                ]
            ),
            float(
                row[
                    "mean_component_to_reference_ratio"
                ]
            ),
            float(
                row[
                    "mean_fraction_components_lt_0p1ml"
                ]
            ),
            float(
                row[
                    "mean_largest_component_fraction"
                ]
            ),
        )
    )


print()
print(
    "DIAGNOSTIC THRESHOLD SENSITIVITY"
)

print(
    "--------------------------------"
)

print(
    best_threshold_df[
        [
            "condition",
            "dice_at_0p5",
            "best_diagnostic_threshold",
            "best_diagnostic_macro_dice",
            "dice_improvement_vs_0p5",
            "best_diagnostic_any_overlap_recall",
            "threshold_sensitive_heuristic",
        ]
    ].to_string(
        index=False
    )
)


print()
print(
    "MODEL OUTPUT SEPARATION"
)

print(
    "-----------------------"
)

print(
    output_summary_df[
        [
            "condition",
            "mean_sampled_voxel_auc",
            "mean_lesion_probability",
            "mean_background_probability",
            "mean_probability_separation",
            "mean_balanced_brier",
        ]
    ].to_string(
        index=False
    )
)


print()
print(
    "SPARSE-SUPERVISION EXPOSURE"
)

print(
    "---------------------------"
)

print(
    exposure_summary_df[
        [
            "condition",
            "labelled_voxels_seen",
            "foreground_voxels_seen",
            "background_voxels_seen",
            "mean_fg_per_patch",
            "mean_bg_per_patch",
            "available_fg_groups",
            "seen_fg_groups",
            "group_coverage_fraction",
        ]
    ].to_string(
        index=False
    )
)


print()
print(
    "SOURCE-SPECIFIC PRIMARY GATE-B DELTA"
)

print(
    "------------------------------------"
)

print(
    source_primary_df.to_string(
        index=False
    )
)


print()
print(
    "MECHANISM FLAGS"
)

print(
    "---------------"
)

print(
    "Severe fragmentation conditions      :",
    (
        len(
            severe_fragmentation_conditions
        )
    ),
    "/4",
)

print(
    "Threshold-sensitive conditions       :",
    (
        len(
            threshold_sensitive_conditions
        )
    ),
    "/4",
)

print(
    "Natural-omission regularization-like :",
    (
        "YES"
        if regularization_like_pattern
        else "NO"
    ),
)

print(
    "Baseline pathology heuristic          :",
    (
        "FLAGGED"
        if baseline_pathology_flag
        else "NOT FLAGGED"
    ),
)

print(
    "Implementation defect proven          : NO"
)

print(
    "Automated root-cause conclusion       : SCIENTIFIC_REVIEW_REQUIRED"
)

print()
print(
    "Publication-quality overlay figures   :",
    len(
        development_cases
    ),
    "case sets",
)

print(
    "Exact Block-08F source captured       :",
    source_capture,
)

print(
    "Starting commit                       :",
    starting_commit[:12],
)

print(
    "Final audit commit                    :",
    final_commit[:12],
)

print(
    "GitHub synchronization                : PASS"
)

print()
print(
    "NEXT:"
)

print(
    "Send me this COMPLETE report, especially the four sections:"
)

print(
    "FRAGMENTATION, DIAGNOSTIC THRESHOLD SENSITIVITY, "
    "MODEL OUTPUT SEPARATION, and SPARSE-SUPERVISION EXPOSURE."
)

print(
    "Do NOT retrain anything yet."
)

print(
    "=" * 122
)