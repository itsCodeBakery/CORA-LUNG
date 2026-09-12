# ==========================================================================================
# CORA-LUNG — CODE BLOCK 06A
# Post-Transfer Causal-Matching Feasibility Audit
#
# PURPOSE
#   Determine whether the causal controls can be made EXACT on the actual
#   3-D training grid after nearest-voxel mapping and same-class merging.
#
# THIS BLOCK DOES NOT:
#   * access dense masks
#   * modify any cached annotation
#   * regenerate scribbles
#   * alter the frozen split
#   * train a model
#
# IT ONLY PROVES FEASIBILITY.
#
# PROPOSED PREPROCESSING v1.1 POLICY
#
# A. NATURAL COMPONENT OMISSION
#    Keep exactly as transferred.
#
# B. RANDOM-PIXEL CONTROL
#    Candidate pool:
#       transferred component-complete foreground supervision.
#
#    Target:
#       exact unique-FG count of the corresponding transferred
#       natural component-omission condition.
#
#    Constraint:
#       every component-complete FG group must remain represented.
#
# C. FIXED-BUDGET PAIR
#    Candidate pools:
#       transferred component_fixed_X
#       transferred complete_fixed_X
#
#    Shared target:
#       min(unique FG capacity of the two transferred annotations).
#
#    Constraint:
#       every foreground group originally represented on each side
#       must remain represented after thinning.
#
# BACKGROUND
#    Unchanged and already known to be coordinate-identical.
#
# IF ALL 120 TARGETS ARE CONSTRUCTIVELY FEASIBLE:
#    Block 06B will create preprocessing/cache v1.1.
# ==========================================================================================

from __future__ import annotations

from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import subprocess
import hashlib
import json
import os
import sys
import textwrap

import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from kaggle_secrets import UserSecretsClient


# ==========================================================================================
# 0. CONSTANTS
# ==========================================================================================

PROJECT = "CORA-Lung"

BLOCK_ID = "06A"
AUDIT_NAME = "post_transfer_causal_matching_feasibility"

GITHUB_OWNER = "itsCodeBakery"
GITHUB_REPO = "CORA-LUNG"

REMOTE_URL = (
    f"https://github.com/"
    f"{GITHUB_OWNER}/"
    f"{GITHUB_REPO}.git"
)

WORK = Path(
    "/kaggle/working"
)

REPO = (
    WORK
    / GITHUB_REPO
)

CACHE_ROOT = (
    WORK
    / "cora_train_cache_v1"
)

CACHE_MANIFEST = (
    CACHE_ROOT
    / "manifest.csv"
)

COVERAGES = [
    75,
    50,
    25,
]

MATCH_SEED = 20260912

NOW = datetime.now(
    timezone.utc
)

NOW_ISO = NOW.strftime(
    "%Y-%m-%dT%H:%M:%SZ"
)


# ==========================================================================================
# 1. HELPERS
# ==========================================================================================

def heading(text):
    print(
        "\n"
        + "=" * 114
    )

    print(
        text
    )

    print(
        "=" * 114
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
        and result.returncode
        != 0
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


def sha256_file(
    path,
    chunk_size=8 * 1024 * 1024,
):
    h = hashlib.sha256()

    with open(
        path,
        "rb",
    ) as f:

        for chunk in iter(
            lambda:
                f.read(
                    chunk_size
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
    text = "|".join(
        str(x)
        for x in parts
    )

    digest = hashlib.sha256(
        text.encode(
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


def sorted_coordinate_hash(
    coords,
):
    coords = np.asarray(
        coords,
        dtype=np.int32,
    )

    if len(
        coords
    ) == 0:

        return hashlib.sha256(
            b""
        ).hexdigest()

    order = np.lexsort(
        (
            coords[
                :,
                2
            ],
            coords[
                :,
                1
            ],
            coords[
                :,
                0
            ],
        )
    )

    coords = np.ascontiguousarray(
        coords[
            order
        ]
    )

    return hashlib.sha256(
        coords.tobytes()
    ).hexdigest()


# ==========================================================================================
# 2. LOAD TRAINING-GRID ANNOTATION
# ==========================================================================================

def load_training_annotation(
    case_id,
    condition,
):
    path = (
        CACHE_ROOT
        / "annotations"
        / (
            f"{case_id}"
            f"__"
            f"{condition}"
            f".npz"
        )
    )

    if not path.exists():
        raise RuntimeError(
            f"Missing training annotation: "
            f"{path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        supervision_coords = np.asarray(
            data[
                "supervision_voxel_zyx"
            ],
            dtype=np.int32,
        )

        supervision_labels = np.asarray(
            data[
                "supervision_label"
            ],
            dtype=np.int8,
        )

        membership_coords = np.asarray(
            data[
                "fg_membership_voxel_zyx"
            ],
            dtype=np.int32,
        )

        membership_groups = np.asarray(
            data[
                "fg_membership_group_id"
            ],
            dtype=np.int32,
        )

    fg_direct = (
        supervision_coords[
            supervision_labels
            == 1
        ]
    )

    bg_direct = (
        supervision_coords[
            supervision_labels
            == 0
        ]
    )

    return {
        "path":
            path,

        "fg_direct":
            fg_direct,

        "bg_direct":
            bg_direct,

        "membership_coords":
            membership_coords,

        "membership_groups":
            membership_groups,
    }


# ==========================================================================================
# 3. BUILD UNIQUE TRAINING-GRID CANDIDATE POOL
# ==========================================================================================

def build_candidate_pool(
    annotation,
):
    """
    Returns:
      coord_to_groups:
          unique training-grid FG voxel -> all FG groups represented there.

      groups:
          required source group IDs.

    Every selected coordinate is a UNIQUE actual training voxel.
    """

    coord_to_groups = defaultdict(
        set
    )

    for coord, group_id in zip(
        annotation[
            "membership_coords"
        ],
        annotation[
            "membership_groups"
        ],
    ):
        coord_to_groups[
            tuple(
                int(x)
                for x in coord
            )
        ].add(
            int(
                group_id
            )
        )

    groups = set(
        int(x)
        for x in np.unique(
            annotation[
                "membership_groups"
            ]
        )
    )

    fg_direct_set = {
        tuple(
            int(x)
            for x in coord
        )
        for coord in annotation[
            "fg_direct"
        ]
    }

    membership_coord_set = set(
        coord_to_groups.keys()
    )

    if (
        fg_direct_set
        != membership_coord_set
    ):
        raise RuntimeError(
            "FG direct-supervision coordinates "
            "do not match FG membership-coordinate support."
        )

    return (
        dict(
            coord_to_groups
        ),
        groups,
    )


# ==========================================================================================
# 4. CONSTRUCT EXACT UNIQUE-VOXEL SUBSET WITH GROUP COVERAGE
# ==========================================================================================

def construct_exact_subset(
    coord_to_groups,
    required_groups,
    target_unique_voxels,
    seed,
):
    """
    Constructively proves feasibility.

    Phase 1:
      deterministic greedy set cover of required groups.

    Phase 2:
      deterministically add unused unique candidate voxels
      until EXACT target count is reached.

    No dense information is involved.
    """

    required_groups = set(
        int(x)
        for x in required_groups
    )

    target_unique_voxels = int(
        target_unique_voxels
    )

    candidate_coords = sorted(
        coord_to_groups.keys()
    )

    candidate_count = len(
        candidate_coords
    )

    if target_unique_voxels < 0:
        return {
            "feasible":
                False,

            "reason":
                "negative_target",
        }

    if (
        target_unique_voxels
        > candidate_count
    ):
        return {
            "feasible":
                False,

            "reason":
                (
                    f"target_above_candidate_capacity:"
                    f"{target_unique_voxels}>"
                    f"{candidate_count}"
                ),
        }

    available_groups = set()

    for group_set in (
        coord_to_groups.values()
    ):
        available_groups.update(
            group_set
        )

    missing_groups = (
        required_groups
        - available_groups
    )

    if missing_groups:
        return {
            "feasible":
                False,

            "reason":
                (
                    "candidate_pool_missing_groups:"
                    + ",".join(
                        map(
                            str,
                            sorted(
                                missing_groups
                            ),
                        )
                    )
                ),
        }

    rng = np.random.default_rng(
        seed
    )

    priorities = {
        coord:
            float(
                rng.random()
            )

        for coord in candidate_coords
    }

    selected = set()

    uncovered = set(
        required_groups
    )

    # ------------------------------------------------------------------
    # Greedy group-cover selection.
    # ------------------------------------------------------------------

    while uncovered:

        useful = []

        for coord in candidate_coords:

            newly_covered = (
                coord_to_groups[
                    coord
                ]
                & uncovered
            )

            if not newly_covered:
                continue

            useful.append(
                (
                    -len(
                        newly_covered
                    ),
                    priorities[
                        coord
                    ],
                    coord,
                )
            )

        if not useful:
            return {
                "feasible":
                    False,

                "reason":
                    "group_cover_stalled",
            }

        useful.sort()

        best_coord = useful[
            0
        ][
            2
        ]

        selected.add(
            best_coord
        )

        uncovered -= (
            coord_to_groups[
                best_coord
            ]
        )

    minimum_constructive_cover = len(
        selected
    )

    if (
        minimum_constructive_cover
        > target_unique_voxels
    ):
        return {
            "feasible":
                False,

            "reason":
                (
                    "group_cover_exceeds_target:"
                    f"{minimum_constructive_cover}>"
                    f"{target_unique_voxels}"
                ),

            "constructive_cover":
                minimum_constructive_cover,
        }

    # ------------------------------------------------------------------
    # Fill to exact target using remaining unique training voxels.
    # ------------------------------------------------------------------

    remaining_coords = [
        coord
        for coord in candidate_coords
        if coord not in selected
    ]

    if remaining_coords:
        permutation = rng.permutation(
            len(
                remaining_coords
            )
        )

        needed = (
            target_unique_voxels
            - len(
                selected
            )
        )

        for index in permutation[
            :needed
        ]:
            selected.add(
                remaining_coords[
                    int(
                        index
                    )
                ]
            )

    if (
        len(
            selected
        )
        != target_unique_voxels
    ):
        return {
            "feasible":
                False,

            "reason":
                (
                    "exact_fill_failed:"
                    f"{len(selected)}!="
                    f"{target_unique_voxels}"
                ),
        }

    represented_groups = set()

    for coord in selected:
        represented_groups.update(
            coord_to_groups[
                coord
            ]
        )

    if not required_groups.issubset(
        represented_groups
    ):
        return {
            "feasible":
                False,

            "reason":
                "group_loss_after_selection",
        }

    selected_array = np.asarray(
        sorted(
            selected
        ),
        dtype=np.int32,
    )

    return {
        "feasible":
            True,

        "reason":
            "PASS",

        "candidate_unique_voxels":
            candidate_count,

        "required_groups":
            len(
                required_groups
            ),

        "constructive_cover":
            minimum_constructive_cover,

        "target_unique_voxels":
            target_unique_voxels,

        "selected":
            selected_array,
    }


# ==========================================================================================
# 5. VERIFY CURRENT BLOCK-06 FAILURE STATE
# ==========================================================================================

heading(
    "CORA-LUNG BLOCK 06A — POST-TRANSFER MATCHING FEASIBILITY"
)

if not (
    REPO
    / ".git"
).exists():
    raise RuntimeError(
        "CORA-LUNG repository missing."
    )

if not CACHE_MANIFEST.exists():
    raise RuntimeError(
        "Block-06 training cache is missing from this Kaggle session."
    )

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
        "training_grid_cache"
    )
    != "REVIEW_REQUIRED_PRIMARY_CAUSAL_BUDGET"
):
    raise RuntimeError(
        "Repository is not in the expected Block-06 review state."
    )

manifest = pd.read_csv(
    CACHE_MANIFEST
)

if len(
    manifest
) != 260:
    raise RuntimeError(
        f"Expected 260 cache rows; found {len(manifest)}."
    )

cases = sorted(
    manifest[
        "case_id"
    ].astype(
        str
    ).unique()
)

if len(
    cases
) != 20:
    raise RuntimeError(
        "Expected exactly 20 CT volumes."
    )

print(
    f"✓ Cache rows          : {len(manifest)}"
)

print(
    f"✓ Cases               : {len(cases)}"
)

print(
    f"✓ Dense masks accessed: NO"
)


# ==========================================================================================
# 6. VERIFY CURRENT BACKGROUND IDENTITY
# ==========================================================================================

heading(
    "STEP 1/6 — VERIFY BACKGROUND REMAINS FROZEN"
)

background_rows = []

for case_id in tqdm(
    cases,
    desc="Checking background coordinates",
):

    hashes = set()

    for condition in manifest.loc[
        manifest[
            "case_id"
        ].astype(
            str
        )
        == case_id,
        "condition",
    ]:

        ann = load_training_annotation(
            case_id,
            str(
                condition
            ),
        )

        hashes.add(
            sorted_coordinate_hash(
                ann[
                    "bg_direct"
                ]
            )
        )

    background_rows.append(
        {
            "case_id":
                case_id,

            "unique_background_hashes":
                len(
                    hashes
                ),
        }
    )


background_df = pd.DataFrame(
    background_rows
)

background_exact = bool(
    (
        background_df[
            "unique_background_hashes"
        ]
        == 1
    ).all()
)

print(
    "Exact same transferred BG coordinates per case: "
    + (
        "PASS"
        if background_exact
        else "FAIL"
    )
)

if not background_exact:
    raise RuntimeError(
        "Background identity failed. "
        "Do not attempt causal equalization."
    )


# ==========================================================================================
# 7. NATURAL-vs-PIXEL FEASIBILITY
# ==========================================================================================

heading(
    "STEP 2/6 — PROVE NATURAL-vs-PIXEL EXACT MATCHING FEASIBILITY"
)

pixel_rows = []

pixel_feasible_count = 0

primary50_pixel_feasible = 0

for case_id in tqdm(
    cases,
    desc="Natural/pixel feasibility",
):

    complete = load_training_annotation(
        case_id,
        "complete",
    )

    (
        complete_pool,
        complete_groups,
    ) = build_candidate_pool(
        complete
    )

    complete_unique_fg = len(
        complete_pool
    )

    for cov in COVERAGES:

        natural = load_training_annotation(
            case_id,
            f"component_natural_{cov}",
        )

        raw_pixel = load_training_annotation(
            case_id,
            f"pixel_dropout_matched_{cov}",
        )

        natural_target = len(
            natural[
                "fg_direct"
            ]
        )

        raw_pixel_unique = len(
            raw_pixel[
                "fg_direct"
            ]
        )

        result = construct_exact_subset(
            complete_pool,
            complete_groups,
            natural_target,
            stable_seed(
                MATCH_SEED,
                case_id,
                "training_grid_pixel_match",
                cov,
            ),
        )

        feasible = bool(
            result[
                "feasible"
            ]
        )

        if feasible:
            pixel_feasible_count += 1

            if cov == 50:
                primary50_pixel_feasible += 1

        pixel_rows.append(
            {
                "case_id":
                    case_id,

                "coverage":
                    cov,

                "complete_unique_fg_capacity":
                    complete_unique_fg,

                "required_complete_groups":
                    len(
                        complete_groups
                    ),

                "natural_target_unique_fg":
                    natural_target,

                "raw_pixel_unique_fg":
                    raw_pixel_unique,

                "raw_pair_exact":
                    natural_target
                    == raw_pixel_unique,

                "constructive_group_cover":
                    result.get(
                        "constructive_cover",
                        np.nan,
                    ),

                "prospective_exact_match_feasible":
                    feasible,

                "reason":
                    result[
                        "reason"
                    ],

                "pixel_voxel_change_if_corrected":
                    (
                        natural_target
                        - raw_pixel_unique
                    ),
            }
        )


pixel_df = pd.DataFrame(
    pixel_rows
)

print(
    f"All-coverage feasible : "
    f"{pixel_feasible_count}/60"
)

print(
    f"PRIMARY 50% feasible  : "
    f"{primary50_pixel_feasible}/20"
)

print(
    f"Currently exact       : "
    f"{int(pixel_df['raw_pair_exact'].sum())}/60"
)

print(
    f"Controls requiring post-transfer reconstruction: "
    f"{int((~pixel_df['raw_pair_exact']).sum())}/60"
)


# ==========================================================================================
# 8. FIXED-PAIR FEASIBILITY
# ==========================================================================================

heading(
    "STEP 3/6 — PROVE FIXED-BUDGET EXACT MATCHING FEASIBILITY"
)

fixed_rows = []

fixed_feasible_count = 0

primary50_fixed_feasible = 0

for case_id in tqdm(
    cases,
    desc="Fixed-pair feasibility",
):

    for cov in COVERAGES:

        incomplete = load_training_annotation(
            case_id,
            f"component_fixed_{cov}",
        )

        complete_control = load_training_annotation(
            case_id,
            f"complete_fixed_{cov}",
        )

        (
            incomplete_pool,
            incomplete_groups,
        ) = build_candidate_pool(
            incomplete
        )

        (
            complete_pool,
            complete_groups,
        ) = build_candidate_pool(
            complete_control
        )

        incomplete_capacity = len(
            incomplete_pool
        )

        complete_capacity = len(
            complete_pool
        )

        shared_target = min(
            incomplete_capacity,
            complete_capacity,
        )

        incomplete_result = construct_exact_subset(
            incomplete_pool,
            incomplete_groups,
            shared_target,
            stable_seed(
                MATCH_SEED,
                case_id,
                "training_grid_component_fixed",
                cov,
            ),
        )

        complete_result = construct_exact_subset(
            complete_pool,
            complete_groups,
            shared_target,
            stable_seed(
                MATCH_SEED,
                case_id,
                "training_grid_complete_fixed",
                cov,
            ),
        )

        pair_feasible = bool(
            incomplete_result[
                "feasible"
            ]
            and complete_result[
                "feasible"
            ]
        )

        if pair_feasible:
            fixed_feasible_count += 1

            if cov == 50:
                primary50_fixed_feasible += 1

        fixed_rows.append(
            {
                "case_id":
                    case_id,

                "coverage":
                    cov,

                "component_fixed_unique_fg":
                    incomplete_capacity,

                "complete_fixed_unique_fg":
                    complete_capacity,

                "raw_pair_exact":
                    incomplete_capacity
                    == complete_capacity,

                "shared_training_grid_target":
                    shared_target,

                "component_fixed_groups":
                    len(
                        incomplete_groups
                    ),

                "complete_fixed_groups":
                    len(
                        complete_groups
                    ),

                "component_constructive_cover":
                    incomplete_result.get(
                        "constructive_cover",
                        np.nan,
                    ),

                "complete_constructive_cover":
                    complete_result.get(
                        "constructive_cover",
                        np.nan,
                    ),

                "component_side_feasible":
                    bool(
                        incomplete_result[
                            "feasible"
                        ]
                    ),

                "complete_side_feasible":
                    bool(
                        complete_result[
                            "feasible"
                        ]
                    ),

                "prospective_exact_pair_feasible":
                    pair_feasible,

                "component_reason":
                    incomplete_result[
                        "reason"
                    ],

                "complete_reason":
                    complete_result[
                        "reason"
                    ],

                "component_voxel_reduction":
                    (
                        incomplete_capacity
                        - shared_target
                    ),

                "complete_voxel_reduction":
                    (
                        complete_capacity
                        - shared_target
                    ),
            }
        )


fixed_df = pd.DataFrame(
    fixed_rows
)

print(
    f"All-coverage feasible : "
    f"{fixed_feasible_count}/60"
)

print(
    f"PRIMARY 50% feasible  : "
    f"{primary50_fixed_feasible}/20"
)

print(
    f"Currently exact       : "
    f"{int(fixed_df['raw_pair_exact'].sum())}/60"
)

print(
    f"Pairs requiring post-transfer equalization: "
    f"{int((~fixed_df['raw_pair_exact']).sum())}/60"
)


# ==========================================================================================
# 9. CHECK PROSPECTIVE GROUP-PRESERVING TARGET MARGINS
# ==========================================================================================

heading(
    "STEP 4/6 — AUDIT TARGET MARGINS"
)

pixel_df[
    "target_minus_constructive_cover"
] = (
    pixel_df[
        "natural_target_unique_fg"
    ]
    - pixel_df[
        "constructive_group_cover"
    ]
)

fixed_df[
    "component_target_margin"
] = (
    fixed_df[
        "shared_training_grid_target"
    ]
    - fixed_df[
        "component_constructive_cover"
    ]
)

fixed_df[
    "complete_target_margin"
] = (
    fixed_df[
        "shared_training_grid_target"
    ]
    - fixed_df[
        "complete_constructive_cover"
    ]
)


all_pixel_feasible = bool(
    pixel_feasible_count
    == 60
)

all_fixed_feasible = bool(
    fixed_feasible_count
    == 60
)

primary50_feasible = bool(
    primary50_pixel_feasible
    == 20
    and primary50_fixed_feasible
    == 20
)

overall_feasible = bool(
    all_pixel_feasible
    and all_fixed_feasible
    and primary50_feasible
)

print(
    f"Minimum pixel target margin            : "
    f"{pixel_df['target_minus_constructive_cover'].min():.0f} voxels"
)

print(
    f"Minimum component-fixed target margin  : "
    f"{fixed_df['component_target_margin'].min():.0f} voxels"
)

print(
    f"Minimum complete-fixed target margin   : "
    f"{fixed_df['complete_target_margin'].min():.0f} voxels"
)

print()
print(
    f"ALL 60 natural/pixel controls feasible : "
    f"{'PASS' if all_pixel_feasible else 'FAIL'}"
)

print(
    f"ALL 60 fixed-budget pairs feasible      : "
    f"{'PASS' if all_fixed_feasible else 'FAIL'}"
)

print(
    f"PRIMARY 50% correction feasibility      : "
    f"{'PASS' if primary50_feasible else 'FAIL'}"
)


# ==========================================================================================
# 10. SAVE REPRODUCIBILITY AUDIT
# ==========================================================================================

heading(
    "STEP 5/6 — FREEZE FEASIBILITY AUDIT"
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

pixel_df.to_csv(
    manifest_dir
    / "training_grid_pixel_equalization_feasibility.csv",
    index=False,
)

fixed_df.to_csv(
    manifest_dir
    / "training_grid_fixed_equalization_feasibility.csv",
    index=False,
)


audit_payload = {
    "project":
        PROJECT,

    "block":
        BLOCK_ID,

    "audit_name":
        AUDIT_NAME,

    "generated_at_utc":
        NOW_ISO,

    "dense_masks_accessed":
        False,

    "cache_modified":
        False,

    "model_training_performed":
        False,

    "background_coordinate_identity":
        bool(
            background_exact
        ),

    "natural_pixel": {
        "existing_exact_pairs":
            int(
                pixel_df[
                    "raw_pair_exact"
                ].sum()
            ),

        "total_pairs":
            60,

        "constructively_feasible_pairs":
            int(
                pixel_feasible_count
            ),

        "primary_50_feasible":
            int(
                primary50_pixel_feasible
            )
            == 20,

        "proposed_candidate_pool":
            (
                "transferred component-complete "
                "unique foreground voxels"
            ),

        "proposed_target":
            (
                "transferred natural-component-omission "
                "unique foreground count"
            ),
    },

    "fixed_budget": {
        "existing_exact_pairs":
            int(
                fixed_df[
                    "raw_pair_exact"
                ].sum()
            ),

        "total_pairs":
            60,

        "constructively_feasible_pairs":
            int(
                fixed_feasible_count
            ),

        "primary_50_feasible":
            int(
                primary50_fixed_feasible
            )
            == 20,

        "proposed_target":
            (
                "minimum transferred unique-FG capacity "
                "of component-fixed and complete-fixed pair"
            ),
    },

    "group_preservation": {
        "algorithm":
            (
                "deterministic greedy group cover followed "
                "by exact unique-voxel fill"
            ),

        "minimum_pixel_target_margin":
            float(
                pixel_df[
                    "target_minus_constructive_cover"
                ].min()
            ),

        "minimum_component_target_margin":
            float(
                fixed_df[
                    "component_target_margin"
                ].min()
            ),

        "minimum_complete_target_margin":
            float(
                fixed_df[
                    "complete_target_margin"
                ].min()
            ),
    },

    "overall_feasible":
        bool(
            overall_feasible
        ),

    "scientific_decision":
        (
            "POST_TRANSFER_EQUALIZATION_FEASIBLE"
            if overall_feasible
            else
            "POST_TRANSFER_EQUALIZATION_NOT_PROVEN"
        ),
}

write_json(
    audit_dir
    / "block06a_post_transfer_equalization_feasibility.json",
    audit_payload,
)


# ==========================================================================================
# 11. UPDATE PROJECT STATE WITHOUT ACCEPTING BLOCK 06 YET
# ==========================================================================================

state.update(
    {
        "last_attempted_block":
            "06A",

        "current_stage":
            (
                "post_transfer_causal_equalization_feasibility_verified"
                if overall_feasible
                else
                "post_transfer_causal_equalization_not_feasible"
            ),

        "training_grid_cache":
            "REVIEW_REQUIRED_PRIMARY_CAUSAL_BUDGET",

        "post_transfer_equalization_feasibility":
            (
                "PASS"
                if overall_feasible
                else "FAIL"
            ),

        "primary_50_equalization_feasibility":
            (
                "PASS"
                if primary50_feasible
                else "FAIL"
            ),

        "training_authorized":
            False,

        "next_action":
            (
                "Code Block 06B: implement deterministic "
                "post-transfer unique-voxel causal equalization "
                "and freeze preprocessing/cache v1.1."
                if overall_feasible
                else
                "STOP: redesign causal-control representation "
                "before model training."
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
# 12. CAPTURE EXACT EXECUTED SOURCE
# ==========================================================================================

source_capture = "NOT_AVAILABLE"

try:

    ip = get_ipython()

    raw_cell = (
        ip.history_manager
        .input_hist_raw[
            -1
        ]
    )

    if (
        "CORA-LUNG — CODE BLOCK 06A"
        in raw_cell
    ):
        code_dir = (
            REPO
            / "scripts/code_blocks"
        )

        code_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        (
            code_dir
            / "block06a_post_transfer_equalization_feasibility.py"
        ).write_text(
            raw_cell,
            encoding="utf-8",
        )

        source_capture = "PASS"

except Exception:
    pass


# ==========================================================================================
# 13. REFRESH REPOSITORY MANIFEST
# ==========================================================================================

repo_manifest = []

repo_files = sorted(
    [
        p
        for p in REPO.rglob(
            "*"
        )
        if p.is_file()
        and ".git"
        not in p.parts
    ],
    key=lambda p:
        str(
            p.relative_to(
                REPO
            )
        ),
)

for path in tqdm(
    repo_files,
    desc="Refreshing repository manifest",
):
    repo_manifest.append(
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
    )

write_json(
    REPO
    / "REPOSITORY_MANIFEST.json",
    {
        "generated_at_utc":
            NOW_ISO,

        "completed_block":
            "05",

        "last_attempted_block":
            "06A",

        "files":
            repo_manifest,
    },
)


# ==========================================================================================
# 14. COMMIT AND PUSH DIAGNOSTIC
# ==========================================================================================

heading(
    "STEP 6/6 — COMMIT FEASIBILITY AUDIT"
)

secrets = UserSecretsClient()

token = secrets.get_secret(
    "pushCora"
)

if not token:
    raise RuntimeError(
        "Kaggle secret 'pushCora' unavailable."
    )

token = token.strip()

askpass = Path(
    "/tmp/cora_git_askpass_block06a.sh"
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
] = token

git_env[
    "GIT_ASKPASS"
] = str(
    askpass
)

git_env[
    "GIT_TERMINAL_PROMPT"
] = "0"

starting_commit = sh(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    cwd=REPO,
).stdout.strip()

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

if status:

    changes = status.splitlines()

    print(
        f"Tracked changes: {len(changes)}"
    )

    for line in changes[
        :35
    ]:
        print(
            " ",
            line,
        )

    commit_message = (
        "audit: verify post-transfer causal matching feasibility"
    )

    sh(
        [
            "git",
            "commit",
            "-m",
            commit_message,
        ],
        cwd=REPO,
    )

    print(
        f"✓ Commit created: "
        f"{commit_message}"
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

if (
    push.returncode
    != 0
):
    safe_error = (
        push.stderr
        or ""
    ).replace(
        token,
        "***TOKEN_REDACTED***",
    )

    raise RuntimeError(
        "GitHub push failed:\n"
        + safe_error
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

git_env.pop(
    "GITHUB_TOKEN",
    None,
)

token = None

print(
    "✓ GitHub synchronization: PASS"
)


# ==========================================================================================
# 15. FINAL REPORT
# ==========================================================================================

print("\n")
print("=" * 114)
print("CORA-LUNG CODE BLOCK 06A — POST-TRANSFER CAUSAL-MATCHING FEASIBILITY REPORT")
print("=" * 114)

print(f"""
CURRENT BLOCK-06 FAILURE
------------------------
Geometry / physical transfer             : PASS
Dense-label firewall                     : PASS
Foreground-group survival                : PASS
Background coordinate identity           : PASS
Causal budget equality after resampling  : FAILED IN RAW v1.0 CACHE

WHY
---
Native sparse points collapse onto shared voxels on the coarser
3.0 × 1.5 × 1.5 mm training grid.

No dense annotation was accessed in this diagnostic.

NATURAL-vs-PIXEL CONTROL
------------------------
Pairs examined                           : 60
Already exact after transfer             : {int(pixel_df['raw_pair_exact'].sum())}/60
Constructively matchable exactly         : {pixel_feasible_count}/60
Controls requiring reconstruction        : {int((~pixel_df['raw_pair_exact']).sum())}/60
PRIMARY 50% feasible                     : {primary50_pixel_feasible}/20

Proposed final target:
  transferred natural-component-omission UNIQUE FG voxel count

Proposed pixel candidate pool:
  transferred component-complete UNIQUE FG voxels

FIXED-BUDGET CONTROL
--------------------
Pairs examined                           : 60
Already exact after transfer             : {int(fixed_df['raw_pair_exact'].sum())}/60
Constructively matchable exactly         : {fixed_feasible_count}/60
Pairs requiring equalization             : {int((~fixed_df['raw_pair_exact']).sum())}/60
PRIMARY 50% feasible                     : {primary50_fixed_feasible}/20

Proposed shared target:
  min(component-fixed unique FG capacity,
      complete-fixed unique FG capacity)

GROUP-PRESERVATION MARGINS
--------------------------
Minimum pixel target minus group cover   : {pixel_df['target_minus_constructive_cover'].min():.0f}
Minimum component-fixed target margin    : {fixed_df['component_target_margin'].min():.0f}
Minimum complete-fixed target margin     : {fixed_df['complete_target_margin'].min():.0f}

FINAL FEASIBILITY DECISION
--------------------------
All natural/pixel pairs feasible         : {'PASS' if all_pixel_feasible else 'FAIL'}
All fixed-budget pairs feasible          : {'PASS' if all_fixed_feasible else 'FAIL'}
Primary 50% causal controls feasible     : {'PASS' if primary50_feasible else 'FAIL'}

OVERALL                                : {'PASS' if overall_feasible else 'FAIL'}

STATE
-----
Weak-label source artifacts modified     : NO
Training-grid annotations modified       : NO
Dense masks accessed                     : NO
Model training                           : NO
Training authorized                      : NO
Exact Block-06A source captured           : {source_capture}

GITHUB
------
Starting commit                          : {starting_commit[:12]}
Current commit                           : {current_commit[:12]}
Synchronization                          : PASS

NEXT
----
If OVERALL = PASS:

Code Block 06B will create preprocessing/cache v1.1 and deterministically
equalize the ACTUAL unique foreground voxel budgets seen by the network.

Natural component-omission annotations will remain unchanged.

Background coordinates will remain unchanged.

No dense labels will be used.

Do not train yet.

Send me this COMPLETE report.
""")

print("=" * 114)


if not overall_feasible:
    raise RuntimeError(
        "POST-TRANSFER EXACT CAUSAL MATCHING WAS NOT PROVEN FEASIBLE. "
        "Do not modify the cache or begin training."
    )