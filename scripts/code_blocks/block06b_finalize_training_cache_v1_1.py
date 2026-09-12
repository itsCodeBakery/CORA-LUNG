# ==========================================================================================
# CORA-LUNG — CODE BLOCK 06B
# Final Post-Transfer Causal Equalization + Training-Grid Cache v1.1 Freeze
#
# SCIENTIFIC PURPOSE
#   Repair ONLY the actual supervision budgets seen by the network after
#   native sparse coordinates have been transferred to the coarser
#   3.0 × 1.5 × 1.5 mm training grid.
#
# WHY THIS IS REQUIRED
#   Block 06 proved:
#       geometry                           PASS
#       crop containment                    PASS
#       foreground-group survival           PASS
#       dense-label firewall                PASS
#       paired background identity          PASS
#
#   But same-class coordinate collapse changed the UNIQUE labelled-voxel
#   budget seen by the model.
#
# BLOCK 06A then constructively proved:
#       natural-vs-pixel pairs feasible     60 / 60
#       fixed-budget pairs feasible         60 / 60
#       primary 50% pairs feasible          20 / 20 + 20 / 20
#
# PREPROCESSING/CACHE v1.1 POLICY
#
#   COMPONENT-COMPLETE
#       unchanged
#
#   NATURAL COMPONENT OMISSION
#       unchanged
#
#   PIXEL-DROPOUT CONTROL
#       if already equal after transfer:
#           unchanged
#       otherwise:
#           deterministically choose exactly N unique training-grid FG voxels
#           from the transferred COMPLETE annotation,
#           where N = transferred natural-component-omission FG count,
#           while preserving >=1 representation for every complete FG group.
#
#   FIXED-PIXEL PAIR
#       shared target =
#           min(component-fixed unique FG count,
#               complete-fixed unique FG count)
#
#       smaller/equal side:
#           unchanged
#
#       larger side:
#           deterministically thinned to the shared UNIQUE-FG target,
#           preserving every foreground group represented by that condition.
#
#   BACKGROUND
#       NEVER changed.
#
# IMPORTANT
#   * Block-05 source weak labels remain untouched.
#   * Natural component omission remains untouched.
#   * No dense lesion/lung mask is opened.
#   * No rerandomization of missing lesion components.
#   * No split changes.
#   * No model training.
#   * The failed/raw cache v1.0 remains preserved.
#   * New accepted cache root = /kaggle/working/cora_train_cache_v1_1
#
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
import math
import shutil
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from kaggle_secrets import UserSecretsClient


# ==========================================================================================
# 0. CONSTANTS
# ==========================================================================================

PROJECT = "CORA-Lung"

BLOCK_ID = "06"
CORRECTIVE_BLOCK = "06B"

PREPROCESS_VERSION_OLD = "1.0"
PREPROCESS_VERSION_NEW = "1.1"

MATCH_SEED = 20260912

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

OLD_CACHE = (
    WORK
    / "cora_train_cache_v1"
)

NEW_CACHE = (
    WORK
    / "cora_train_cache_v1_1"
)

OLD_MANIFEST = (
    OLD_CACHE
    / "manifest.csv"
)

NEW_IMAGE_DIR = (
    NEW_CACHE
    / "images"
)

NEW_ANNOTATION_DIR = (
    NEW_CACHE
    / "annotations"
)

COVERAGES = [
    75,
    50,
    25,
]

NOW = datetime.now(
    timezone.utc
)

NOW_ISO = NOW.strftime(
    "%Y-%m-%dT%H:%M:%SZ"
)


# ==========================================================================================
# 1. GENERAL HELPERS
# ==========================================================================================

def heading(text):

    print(
        "\n"
        + "=" * 118
    )

    print(
        text
    )

    print(
        "=" * 118
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


def semantic_hash(
    *arrays,
):

    h = hashlib.sha256()

    for arr in arrays:

        arr = np.ascontiguousarray(
            arr
        )

        h.update(
            str(
                arr.dtype
            ).encode(
                "utf-8"
            )
        )

        h.update(
            str(
                arr.shape
            ).encode(
                "utf-8"
            )
        )

        h.update(
            arr.tobytes()
        )

    return h.hexdigest()


def stable_seed(
    *parts,
):

    payload = "|".join(
        str(x)
        for x in parts
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


def human_bytes(
    value,
):

    value = float(
        value
    )

    for unit in [
        "B",
        "KB",
        "MB",
        "GB",
        "TB",
    ]:

        if value < 1024:

            return (
                f"{value:.2f} "
                f"{unit}"
            )

        value /= 1024.0

    return (
        f"{value:.2f} PB"
    )


def coordinate_hash(
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
# 2. LOAD / HASH TRAINING ANNOTATION
# ==========================================================================================

def load_annotation(
    root,
    case_id,
    condition,
):

    path = (
        Path(
            root
        )
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
            f"Missing annotation: "
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

    fg_coords = supervision_coords[
        supervision_labels
        == 1
    ]

    bg_coords = supervision_coords[
        supervision_labels
        == 0
    ]

    return {
        "path":
            path,

        "supervision_coords":
            supervision_coords,

        "supervision_labels":
            supervision_labels,

        "membership_coords":
            membership_coords,

        "membership_groups":
            membership_groups,

        "fg_coords":
            fg_coords,

        "bg_coords":
            bg_coords,

        "semantic_sha256":
            semantic_hash(
                supervision_coords,
                supervision_labels,
                membership_coords,
                membership_groups,
            ),
    }


# ==========================================================================================
# 3. FG CANDIDATE POOL
# ==========================================================================================

def build_candidate_pool(
    annotation,
):

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

    required_groups = set(
        int(x)
        for x in np.unique(
            annotation[
                "membership_groups"
            ]
        )
    )

    direct_fg = {
        tuple(
            int(x)
            for x in coord
        )
        for coord in annotation[
            "fg_coords"
        ]
    }

    membership_support = set(
        coord_to_groups.keys()
    )

    if (
        direct_fg
        != membership_support
    ):

        raise RuntimeError(
            "Direct FG supervision support and "
            "FG-membership support disagree."
        )

    return (
        dict(
            coord_to_groups
        ),
        required_groups,
    )


# ==========================================================================================
# 4. EXACT GROUP-PRESERVING UNIQUE-VOXEL SUBSET
# ==========================================================================================

def construct_exact_subset(
    coord_to_groups,
    required_groups,
    target_unique_voxels,
    seed,
):

    required_groups = set(
        int(x)
        for x in required_groups
    )

    target_unique_voxels = int(
        target_unique_voxels
    )

    candidates = sorted(
        coord_to_groups.keys()
    )

    if (
        target_unique_voxels
        > len(
            candidates
        )
    ):

        return {
            "feasible":
                False,

            "reason":
                "target_above_capacity",
        }

    available_groups = set()

    for group_set in (
        coord_to_groups.values()
    ):

        available_groups.update(
            group_set
        )

    missing = (
        required_groups
        - available_groups
    )

    if missing:

        return {
            "feasible":
                False,

            "reason":
                (
                    "missing_groups:"
                    + ",".join(
                        map(
                            str,
                            sorted(
                                missing
                            ),
                        )
                    )
                ),
        }

    rng = np.random.default_rng(
        seed
    )

    priority = {
        coord:
            float(
                rng.random()
            )

        for coord in candidates
    }

    selected = set()

    uncovered = set(
        required_groups
    )

    # ------------------------------------------------------------------
    # Deterministic randomized greedy group cover.
    # ------------------------------------------------------------------

    while uncovered:

        options = []

        for coord in candidates:

            gained = (
                coord_to_groups[
                    coord
                ]
                & uncovered
            )

            if not gained:
                continue

            options.append(
                (
                    -len(
                        gained
                    ),
                    priority[
                        coord
                    ],
                    coord,
                )
            )

        if not options:

            return {
                "feasible":
                    False,

                "reason":
                    "group_cover_stalled",
            }

        options.sort()

        chosen = options[
            0
        ][
            2
        ]

        selected.add(
            chosen
        )

        uncovered -= (
            coord_to_groups[
                chosen
            ]
        )

    group_cover_size = len(
        selected
    )

    if (
        group_cover_size
        > target_unique_voxels
    ):

        return {
            "feasible":
                False,

            "reason":
                (
                    "group_cover_above_target:"
                    f"{group_cover_size}>"
                    f"{target_unique_voxels}"
                ),
        }

    # ------------------------------------------------------------------
    # Randomly fill from the remaining UNIQUE training voxels.
    # ------------------------------------------------------------------

    remaining = [
        coord
        for coord in candidates
        if coord not in selected
    ]

    needed = (
        target_unique_voxels
        - len(
            selected
        )
    )

    if needed:

        order = rng.permutation(
            len(
                remaining
            )
        )

        for index in order[
            :needed
        ]:

            selected.add(
                remaining[
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
                "exact_fill_failed",
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

    return {
        "feasible":
            True,

        "reason":
            "PASS",

        "selected":
            np.asarray(
                sorted(
                    selected
                ),
                dtype=np.int32,
            ),

        "group_cover_size":
            int(
                group_cover_size
            ),

        "candidate_capacity":
            int(
                len(
                    candidates
                )
            ),
    }


# ==========================================================================================
# 5. BUILD A NEW TRAINING ANNOTATION FROM A SELECTED FG SUBSET
# ==========================================================================================

def build_annotation_from_selected_fg(
    selected_fg_coords,
    candidate_annotation,
    frozen_background_coords,
):

    selected_fg_coords = np.asarray(
        selected_fg_coords,
        dtype=np.int32,
    )

    frozen_background_coords = np.asarray(
        frozen_background_coords,
        dtype=np.int32,
    )

    (
        coord_to_groups,
        required_groups,
    ) = build_candidate_pool(
        candidate_annotation
    )

    selected_set = {
        tuple(
            int(x)
            for x in coord
        )
        for coord in selected_fg_coords
    }

    if not selected_set.issubset(
        set(
            coord_to_groups.keys()
        )
    ):

        raise RuntimeError(
            "Selected FG contains coordinate "
            "outside candidate support."
        )

    fg_bg_collision = (
        selected_set
        & {
            tuple(
                int(x)
                for x in coord
            )
            for coord in frozen_background_coords
        }
    )

    if fg_bg_collision:

        raise RuntimeError(
            "Post-equalization FG/BG collision."
        )

    membership_rows = []

    represented_groups = set()

    for coord in sorted(
        selected_set
    ):

        for group_id in sorted(
            coord_to_groups[
                coord
            ]
        ):

            membership_rows.append(
                (
                    int(
                        coord[
                            0
                        ]
                    ),
                    int(
                        coord[
                            1
                        ]
                    ),
                    int(
                        coord[
                            2
                        ]
                    ),
                    int(
                        group_id
                    ),
                )
            )

            represented_groups.add(
                int(
                    group_id
                )
            )

    if (
        represented_groups
        != required_groups
    ):

        raise RuntimeError(
            "Foreground group identity changed "
            "during causal equalization."
        )

    fg_sorted = np.asarray(
        sorted(
            selected_set
        ),
        dtype=np.int32,
    )

    bg_set = {
        tuple(
            int(x)
            for x in coord
        )
        for coord in frozen_background_coords
    }

    bg_sorted = np.asarray(
        sorted(
            bg_set
        ),
        dtype=np.int32,
    )

    supervision_rows = []

    for coord in fg_sorted:

        supervision_rows.append(
            (
                tuple(
                    int(x)
                    for x in coord
                ),
                1,
            )
        )

    for coord in bg_sorted:

        supervision_rows.append(
            (
                tuple(
                    int(x)
                    for x in coord
                ),
                0,
            )
        )

    supervision_rows.sort(
        key=lambda item:
            item[
                0
            ]
    )

    supervision_coords = np.asarray(
        [
            item[
                0
            ]
            for item in supervision_rows
        ],
        dtype=np.int32,
    )

    supervision_labels = np.asarray(
        [
            item[
                1
            ]
            for item in supervision_rows
        ],
        dtype=np.int8,
    )

    if membership_rows:

        membership_rows = np.asarray(
            membership_rows,
            dtype=np.int32,
        )

        membership_coords = (
            membership_rows[
                :,
                :3
            ]
        )

        membership_groups = (
            membership_rows[
                :,
                3
            ]
        )

    else:

        membership_coords = np.empty(
            (
                0,
                3,
            ),
            dtype=np.int32,
        )

        membership_groups = np.empty(
            (
                0,
            ),
            dtype=np.int32,
        )

    return {
        "supervision_voxel_zyx":
            supervision_coords,

        "supervision_label":
            supervision_labels,

        "fg_membership_voxel_zyx":
            membership_coords,

        "fg_membership_group_id":
            membership_groups,
    }


def save_annotation(
    path,
    annotation,
):

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        path,
        supervision_voxel_zyx=annotation[
            "supervision_voxel_zyx"
        ].astype(
            np.int32
        ),
        supervision_label=annotation[
            "supervision_label"
        ].astype(
            np.int8
        ),
        fg_membership_voxel_zyx=annotation[
            "fg_membership_voxel_zyx"
        ].astype(
            np.int32
        ),
        fg_membership_group_id=annotation[
            "fg_membership_group_id"
        ].astype(
            np.int32
        ),
    )


# ==========================================================================================
# 6. REPOSITORY / CACHE STATE VALIDATION
# ==========================================================================================

heading(
    "CORA-LUNG :: CODE BLOCK 06B :: TRAINING CACHE v1.1"
)

if not (
    REPO
    / ".git"
).exists():

    raise RuntimeError(
        "CORA-LUNG repository missing."
    )

if not OLD_MANIFEST.exists():

    raise RuntimeError(
        "Raw Block-06 v1.0 training cache is unavailable."
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
        "post_transfer_equalization_feasibility"
    )
    != "PASS"
):

    raise RuntimeError(
        "Block 06A feasibility has not been frozen as PASS."
    )

if (
    state.get(
        "primary_50_equalization_feasibility"
    )
    != "PASS"
):

    raise RuntimeError(
        "Primary 50% feasibility has not been frozen as PASS."
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
        "Repository has uncommitted changes before Block 06B."
    )

old_manifest = pd.read_csv(
    OLD_MANIFEST
)

if len(
    old_manifest
) != 260:

    raise RuntimeError(
        f"Expected 260 v1.0 cache rows, found "
        f"{len(old_manifest)}."
    )

cases = sorted(
    old_manifest[
        "case_id"
    ].astype(
        str
    ).unique()
)

if len(
    cases
) != 20:

    raise RuntimeError(
        "Expected 20 primary CT volumes."
    )

starting_commit = sh(
    [
        "git",
        "rev-parse",
        "HEAD",
    ],
    cwd=REPO,
).stdout.strip()

print(
    f"✓ Starting commit            : "
    f"{starting_commit[:12]}"
)

print(
    f"✓ Parent cache               : "
    f"{OLD_CACHE}"
)

print(
    f"✓ Parent manifest rows       : "
    f"{len(old_manifest)}"
)

print(
    f"✓ Block-06A feasibility      : PASS"
)

print(
    f"✓ Dense masks accessed       : NO"
)


# ==========================================================================================
# 7. INITIALIZE NEW v1.1 CACHE
# ==========================================================================================

heading(
    "STEP 1/10 — INITIALIZE IMMUTABLE-DERIVED CACHE v1.1"
)

if NEW_CACHE.exists():

    shutil.rmtree(
        NEW_CACHE
    )

NEW_IMAGE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

NEW_ANNOTATION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

# ----------------------------------------------------------------------
# Images are byte-for-byte copied.
# CT preprocessing itself is NOT changed by this correction.
# ----------------------------------------------------------------------

old_image_files = sorted(
    (
        OLD_CACHE
        / "images"
    ).glob(
        "*.npz"
    )
)

if len(
    old_image_files
) != 20:

    raise RuntimeError(
        f"Expected 20 parent image files, found "
        f"{len(old_image_files)}."
    )

image_copy_rows = []

for old_path in tqdm(
    old_image_files,
    desc="Copying frozen CT caches",
    unit="volume",
):

    new_path = (
        NEW_IMAGE_DIR
        / old_path.name
    )

    shutil.copy2(
        old_path,
        new_path,
    )

    old_hash = sha256_file(
        old_path
    )

    new_hash = sha256_file(
        new_path
    )

    image_copy_rows.append(
        {
            "image_file":
                f"images/{old_path.name}",

            "parent_sha256":
                old_hash,

            "v1_1_sha256":
                new_hash,

            "byte_exact":
                old_hash
                == new_hash,
        }
    )

image_copy_df = pd.DataFrame(
    image_copy_rows
)

image_copy_pass = bool(
    image_copy_df[
        "byte_exact"
    ].all()
)

if not image_copy_pass:

    raise RuntimeError(
        "Image cache changed during v1.1 copy."
    )

print(
    "✓ 20/20 CT caches copied byte-for-byte."
)


# ==========================================================================================
# 8. PARENT MANIFEST LOOKUP
# ==========================================================================================

parent_lookup = {
    (
        str(
            row[
                "case_id"
            ]
        ),
        str(
            row[
                "condition"
            ]
        ),
    ):
        row

    for _, row in (
        old_manifest.iterrows()
    )
}


# ==========================================================================================
# 9. COPY OR WRITE AN ANNOTATION + LINEAGE RECORD
# ==========================================================================================

new_manifest_rows = []
correction_rows = []

changed_semantic_conditions = 0
copied_semantic_conditions = 0

pixel_controls_corrected = 0
fixed_sides_corrected = 0


def append_manifest_record(
    *,
    case_id,
    condition,
    final_annotation,
    final_path,
    parent_row,
    candidate_condition,
    candidate_native_hash,
    candidate_v1_hash,
    construction_policy,
    equalization_target,
):

    final_sha = semantic_hash(
        final_annotation[
            "supervision_voxel_zyx"
        ],
        final_annotation[
            "supervision_label"
        ],
        final_annotation[
            "fg_membership_voxel_zyx"
        ],
        final_annotation[
            "fg_membership_group_id"
        ],
    )

    new_manifest_rows.append(
        {
            "case_id":
                case_id,

            "source_subject_key":
                str(
                    parent_row[
                        "source_subject_key"
                    ]
                ),

            "split_role":
                str(
                    parent_row[
                        "split_role"
                    ]
                ),

            "outer_fold":
                parent_row[
                    "outer_fold"
                ],

            "condition":
                condition,

            "image_file":
                str(
                    parent_row[
                        "image_file"
                    ]
                ),

            "annotation_file":
                (
                    f"annotations/"
                    f"{final_path.name}"
                ),

            "image_semantic_sha256":
                str(
                    parent_row[
                        "image_semantic_sha256"
                    ]
                ),

            "annotation_semantic_sha256":
                final_sha,

            "original_condition_native_semantic_sha256":
                str(
                    parent_row[
                        "source_annotation_semantic_sha256"
                    ]
                ),

            "parent_v1_annotation_semantic_sha256":
                str(
                    parent_row[
                        "annotation_semantic_sha256"
                    ]
                ),

            "candidate_condition":
                candidate_condition,

            "candidate_native_semantic_sha256":
                candidate_native_hash,

            "candidate_v1_annotation_semantic_sha256":
                candidate_v1_hash,

            "construction_policy":
                construction_policy,

            "equalization_target_unique_fg":
                int(
                    equalization_target
                ),

            "preprocess_version":
                PREPROCESS_VERSION_NEW,
        }
    )

    return final_sha


def copy_parent_annotation(
    case_id,
    condition,
    policy,
):

    parent_row = parent_lookup[
        (
            case_id,
            condition,
        )
    ]

    source = load_annotation(
        OLD_CACHE,
        case_id,
        condition,
    )

    destination = (
        NEW_ANNOTATION_DIR
        / (
            f"{case_id}"
            f"__"
            f"{condition}"
            f".npz"
        )
    )

    shutil.copy2(
        source[
            "path"
        ],
        destination,
    )

    copied = load_annotation(
        NEW_CACHE,
        case_id,
        condition,
    )

    if (
        copied[
            "semantic_sha256"
        ]
        != source[
            "semantic_sha256"
        ]
    ):

        raise RuntimeError(
            f"Semantic change during copy: "
            f"{case_id}/{condition}"
        )

    final_annotation = {
        "supervision_voxel_zyx":
            copied[
                "supervision_coords"
            ],

        "supervision_label":
            copied[
                "supervision_labels"
            ],

        "fg_membership_voxel_zyx":
            copied[
                "membership_coords"
            ],

        "fg_membership_group_id":
            copied[
                "membership_groups"
            ],
    }

    append_manifest_record(
        case_id=
            case_id,

        condition=
            condition,

        final_annotation=
            final_annotation,

        final_path=
            destination,

        parent_row=
            parent_row,

        candidate_condition=
            condition,

        candidate_native_hash=
            str(
                parent_row[
                    "source_annotation_semantic_sha256"
                ]
            ),

        candidate_v1_hash=
            str(
                parent_row[
                    "annotation_semantic_sha256"
                ]
            ),

        construction_policy=
            policy,

        equalization_target=
            len(
                copied[
                    "fg_coords"
                ]
            ),
    )

    return copied


# ==========================================================================================
# 10. BUILD ALL 20 × 13 FINAL CONDITIONS
# ==========================================================================================

heading(
    "STEP 2/10 — BUILD FINAL CAUSALLY MATCHED ANNOTATIONS"
)

for case_id in tqdm(
    cases,
    desc="Equalizing causal controls",
    unit="case",
):

    # ------------------------------------------------------------------
    # A. Complete annotation: untouched.
    # ------------------------------------------------------------------

    complete_new = copy_parent_annotation(
        case_id,
        "complete",
        "unchanged_from_v1_0",
    )

    complete_parent = load_annotation(
        OLD_CACHE,
        case_id,
        "complete",
    )

    (
        complete_pool,
        complete_groups,
    ) = build_candidate_pool(
        complete_parent
    )

    # ------------------------------------------------------------------
    # B. Coverage-specific paired conditions.
    # ------------------------------------------------------------------

    for cov in COVERAGES:

        natural_condition = (
            f"component_natural_{cov}"
        )

        pixel_condition = (
            f"pixel_dropout_matched_{cov}"
        )

        component_fixed_condition = (
            f"component_fixed_{cov}"
        )

        complete_fixed_condition = (
            f"complete_fixed_{cov}"
        )

        # ==============================================================
        # B1. NATURAL COMPONENT OMISSION — NEVER ALTERED
        # ==============================================================

        natural_new = copy_parent_annotation(
            case_id,
            natural_condition,
            "natural_component_omission_unchanged",
        )

        natural_target = len(
            natural_new[
                "fg_coords"
            ]
        )

        # ==============================================================
        # B2. PIXEL CONTROL
        # ==============================================================

        raw_pixel = load_annotation(
            OLD_CACHE,
            case_id,
            pixel_condition,
        )

        raw_pixel_count = len(
            raw_pixel[
                "fg_coords"
            ]
        )

        pixel_parent_row = parent_lookup[
            (
                case_id,
                pixel_condition,
            )
        ]

        if (
            raw_pixel_count
            == natural_target
        ):

            copy_parent_annotation(
                case_id,
                pixel_condition,
                "already_exact_after_transfer_unchanged",
            )

            copied_semantic_conditions += 1

            correction_rows.append(
                {
                    "case_id":
                        case_id,

                    "coverage":
                        cov,

                    "control":
                        "natural_vs_pixel",

                    "side":
                        "pixel",

                    "parent_unique_fg":
                        raw_pixel_count,

                    "target_unique_fg":
                        natural_target,

                    "action":
                        "unchanged",

                    "changed":
                        False,
                }
            )

        else:

            pixel_result = construct_exact_subset(
                complete_pool,
                complete_groups,
                natural_target,
                stable_seed(
                    MATCH_SEED,
                    case_id,
                    "cache_v1_1_pixel_equalization",
                    cov,
                ),
            )

            if not pixel_result[
                "feasible"
            ]:

                raise RuntimeError(
                    f"06A feasibility contradiction for "
                    f"{case_id}/{pixel_condition}: "
                    f"{pixel_result['reason']}"
                )

            rebuilt_pixel = build_annotation_from_selected_fg(
                pixel_result[
                    "selected"
                ],
                complete_parent,
                raw_pixel[
                    "bg_coords"
                ],
            )

            destination = (
                NEW_ANNOTATION_DIR
                / (
                    f"{case_id}"
                    f"__"
                    f"{pixel_condition}"
                    f".npz"
                )
            )

            save_annotation(
                destination,
                rebuilt_pixel,
            )

            complete_parent_row = parent_lookup[
                (
                    case_id,
                    "complete",
                )
            ]

            final_sha = append_manifest_record(
                case_id=
                    case_id,

                condition=
                    pixel_condition,

                final_annotation=
                    rebuilt_pixel,

                final_path=
                    destination,

                parent_row=
                    pixel_parent_row,

                candidate_condition=
                    "complete",

                candidate_native_hash=
                    str(
                        complete_parent_row[
                            "source_annotation_semantic_sha256"
                        ]
                    ),

                candidate_v1_hash=
                    str(
                        complete_parent_row[
                            "annotation_semantic_sha256"
                        ]
                    ),

                construction_policy=
                    (
                        "post_transfer_unique_voxel_pixel_equalization"
                    ),

                equalization_target=
                    natural_target,
            )

            if (
                final_sha
                == str(
                    pixel_parent_row[
                        "annotation_semantic_sha256"
                    ]
                )
            ):

                raise RuntimeError(
                    "Expected changed pixel control "
                    "retained identical semantic hash."
                )

            pixel_controls_corrected += 1
            changed_semantic_conditions += 1

            correction_rows.append(
                {
                    "case_id":
                        case_id,

                    "coverage":
                        cov,

                    "control":
                        "natural_vs_pixel",

                    "side":
                        "pixel",

                    "parent_unique_fg":
                        raw_pixel_count,

                    "target_unique_fg":
                        natural_target,

                    "action":
                        "reconstructed_from_complete_unique_fg_pool",

                    "changed":
                        True,
                }
            )

        # ==============================================================
        # B3. FIXED-BUDGET PAIR
        # ==============================================================

        raw_component_fixed = load_annotation(
            OLD_CACHE,
            case_id,
            component_fixed_condition,
        )

        raw_complete_fixed = load_annotation(
            OLD_CACHE,
            case_id,
            complete_fixed_condition,
        )

        component_capacity = len(
            raw_component_fixed[
                "fg_coords"
            ]
        )

        complete_capacity = len(
            raw_complete_fixed[
                "fg_coords"
            ]
        )

        shared_target = min(
            component_capacity,
            complete_capacity,
        )

        fixed_pair = [
            (
                component_fixed_condition,
                raw_component_fixed,
                component_capacity,
                "component",
            ),
            (
                complete_fixed_condition,
                raw_complete_fixed,
                complete_capacity,
                "complete",
            ),
        ]

        for (
            condition,
            raw_annotation,
            capacity,
            side,
        ) in fixed_pair:

            if (
                capacity
                == shared_target
            ):

                copy_parent_annotation(
                    case_id,
                    condition,
                    (
                        "fixed_pair_side_already_at_shared_"
                        "training_grid_target_unchanged"
                    ),
                )

                copied_semantic_conditions += 1

                correction_rows.append(
                    {
                        "case_id":
                            case_id,

                        "coverage":
                            cov,

                        "control":
                            "fixed_pair",

                        "side":
                            side,

                        "parent_unique_fg":
                            capacity,

                        "target_unique_fg":
                            shared_target,

                        "action":
                            "unchanged",

                        "changed":
                            False,
                    }
                )

                continue

            (
                candidate_pool,
                required_groups,
            ) = build_candidate_pool(
                raw_annotation
            )

            fixed_result = construct_exact_subset(
                candidate_pool,
                required_groups,
                shared_target,
                stable_seed(
                    MATCH_SEED,
                    case_id,
                    "cache_v1_1_fixed_equalization",
                    cov,
                    side,
                ),
            )

            if not fixed_result[
                "feasible"
            ]:

                raise RuntimeError(
                    f"06A feasibility contradiction for "
                    f"{case_id}/{condition}: "
                    f"{fixed_result['reason']}"
                )

            rebuilt_fixed = build_annotation_from_selected_fg(
                fixed_result[
                    "selected"
                ],
                raw_annotation,
                raw_annotation[
                    "bg_coords"
                ],
            )

            destination = (
                NEW_ANNOTATION_DIR
                / (
                    f"{case_id}"
                    f"__"
                    f"{condition}"
                    f".npz"
                )
            )

            save_annotation(
                destination,
                rebuilt_fixed,
            )

            parent_row = parent_lookup[
                (
                    case_id,
                    condition,
                )
            ]

            final_sha = append_manifest_record(
                case_id=
                    case_id,

                condition=
                    condition,

                final_annotation=
                    rebuilt_fixed,

                final_path=
                    destination,

                parent_row=
                    parent_row,

                candidate_condition=
                    condition,

                candidate_native_hash=
                    str(
                        parent_row[
                            "source_annotation_semantic_sha256"
                        ]
                    ),

                candidate_v1_hash=
                    str(
                        parent_row[
                            "annotation_semantic_sha256"
                        ]
                    ),

                construction_policy=
                    (
                        "post_transfer_fixed_pair_unique_voxel_thinning"
                    ),

                equalization_target=
                    shared_target,
            )

            if (
                final_sha
                == str(
                    parent_row[
                        "annotation_semantic_sha256"
                    ]
                )
            ):

                raise RuntimeError(
                    "Expected changed fixed control "
                    "retained identical semantic hash."
                )

            fixed_sides_corrected += 1
            changed_semantic_conditions += 1

            correction_rows.append(
                {
                    "case_id":
                        case_id,

                    "coverage":
                        cov,

                    "control":
                        "fixed_pair",

                    "side":
                        side,

                    "parent_unique_fg":
                        capacity,

                    "target_unique_fg":
                        shared_target,

                    "action":
                        "thinned_on_training_grid",

                    "changed":
                        True,
                }
            )


# ==========================================================================================
# 11. WRITE NEW MANIFEST
# ==========================================================================================

heading(
    "STEP 3/10 — FREEZE v1.1 TRAINING MANIFEST"
)

new_manifest = pd.DataFrame(
    new_manifest_rows
)

expected_manifest_columns = [
    "case_id",
    "source_subject_key",
    "split_role",
    "outer_fold",
    "condition",
    "image_file",
    "annotation_file",
    "image_semantic_sha256",
    "annotation_semantic_sha256",
    "original_condition_native_semantic_sha256",
    "parent_v1_annotation_semantic_sha256",
    "candidate_condition",
    "candidate_native_semantic_sha256",
    "candidate_v1_annotation_semantic_sha256",
    "construction_policy",
    "equalization_target_unique_fg",
    "preprocess_version",
]

new_manifest = new_manifest[
    expected_manifest_columns
]

new_manifest = new_manifest.sort_values(
    [
        "case_id",
        "condition",
    ]
).reset_index(
    drop=True
)

if len(
    new_manifest
) != 260:

    raise RuntimeError(
        f"Expected 260 v1.1 manifest rows; "
        f"found {len(new_manifest)}."
    )

duplicate_rows = int(
    new_manifest.duplicated(
        subset=[
            "case_id",
            "condition",
        ]
    ).sum()
)

if duplicate_rows:

    raise RuntimeError(
        f"Duplicate v1.1 case-condition rows: "
        f"{duplicate_rows}"
    )

new_manifest.to_csv(
    NEW_CACHE
    / "manifest.csv",
    index=False,
)

correction_df = pd.DataFrame(
    correction_rows
)

print(
    f"✓ v1.1 manifest rows           : "
    f"{len(new_manifest)}"
)

print(
    f"✓ Pixel controls corrected     : "
    f"{pixel_controls_corrected}"
)

print(
    f"✓ Fixed-pair sides corrected   : "
    f"{fixed_sides_corrected}"
)

print(
    f"✓ Total semantically corrected : "
    f"{changed_semantic_conditions}"
)


# ==========================================================================================
# 12. COMPLETE FINAL CACHE AUDIT
# ==========================================================================================

heading(
    "STEP 4/10 — REAUDIT ALL 260 FINAL ANNOTATIONS"
)

final_rows = []

condition_matrix_failures = []
group_contract_failures = []
background_failures = []
coordinate_failures = []
hash_failures = []

parent_group_lookup = {}

for case_id in cases:

    for condition in [
        "complete",
        "component_natural_75",
        "pixel_dropout_matched_75",
        "component_fixed_75",
        "complete_fixed_75",
        "component_natural_50",
        "pixel_dropout_matched_50",
        "component_fixed_50",
        "complete_fixed_50",
        "component_natural_25",
        "pixel_dropout_matched_25",
        "component_fixed_25",
        "complete_fixed_25",
    ]:

        parent = load_annotation(
            OLD_CACHE,
            case_id,
            condition,
        )

        parent_group_lookup[
            (
                case_id,
                condition,
            )
        ] = set(
            int(x)
            for x in np.unique(
                parent[
                    "membership_groups"
                ]
            )
        )


image_shape_lookup = {}

for case_id in cases:

    image_path = (
        NEW_IMAGE_DIR
        / f"{case_id}.npz"
    )

    with np.load(
        image_path,
        allow_pickle=False,
    ) as data:

        ct_shape = np.asarray(
            data[
                "ct_zyx"
            ].shape,
            dtype=np.int32,
        )

    image_shape_lookup[
        case_id
    ] = ct_shape


for _, row in tqdm(
    new_manifest.iterrows(),
    total=len(
        new_manifest
    ),
    desc="Auditing cache v1.1",
    unit="condition",
):

    case_id = str(
        row[
            "case_id"
        ]
    )

    condition = str(
        row[
            "condition"
        ]
    )

    final_ann = load_annotation(
        NEW_CACHE,
        case_id,
        condition,
    )

    final_groups = set(
        int(x)
        for x in np.unique(
            final_ann[
                "membership_groups"
            ]
        )
    )

    expected_groups = parent_group_lookup[
        (
            case_id,
            condition,
        )
    ]

    if (
        final_groups
        != expected_groups
    ):

        group_contract_failures.append(
            (
                case_id,
                condition,
                sorted(
                    expected_groups
                    - final_groups
                ),
                sorted(
                    final_groups
                    - expected_groups
                ),
            )
        )

    direct_fg_set = {
        tuple(
            int(x)
            for x in coord
        )
        for coord in final_ann[
            "fg_coords"
        ]
    }

    membership_support = {
        tuple(
            int(x)
            for x in coord
        )
        for coord in final_ann[
            "membership_coords"
        ]
    }

    if (
        direct_fg_set
        != membership_support
    ):

        group_contract_failures.append(
            (
                case_id,
                condition,
                "direct_fg_membership_support_mismatch",
            )
        )

    shape = image_shape_lookup[
        case_id
    ]

    all_coords = final_ann[
        "supervision_coords"
    ]

    inside = np.all(
        (
            all_coords
            >= 0
        )
        & (
            all_coords
            < shape[
                None,
                :
            ]
        ),
        axis=1,
    )

    if not inside.all():

        coordinate_failures.append(
            (
                case_id,
                condition,
                int(
                    (
                        ~inside
                    ).sum()
                ),
            )
        )

    declared_hash = str(
        row[
            "annotation_semantic_sha256"
        ]
    )

    if (
        final_ann[
            "semantic_sha256"
        ]
        != declared_hash
    ):

        hash_failures.append(
            (
                case_id,
                condition,
            )
        )

    final_rows.append(
        {
            "case_id":
                case_id,

            "condition":
                condition,

            "foreground_voxels":
                int(
                    len(
                        final_ann[
                            "fg_coords"
                        ]
                    )
                ),

            "background_voxels":
                int(
                    len(
                        final_ann[
                            "bg_coords"
                        ]
                    )
                ),

            "foreground_groups":
                int(
                    len(
                        final_groups
                    )
                ),

            "background_sha256":
                coordinate_hash(
                    final_ann[
                        "bg_coords"
                    ]
                ),

            "annotation_semantic_sha256":
                final_ann[
                    "semantic_sha256"
                ],
        }
    )


final_df = pd.DataFrame(
    final_rows
)

expected_conditions = {
    "complete",
    "component_natural_75",
    "pixel_dropout_matched_75",
    "component_fixed_75",
    "complete_fixed_75",
    "component_natural_50",
    "pixel_dropout_matched_50",
    "component_fixed_50",
    "complete_fixed_50",
    "component_natural_25",
    "pixel_dropout_matched_25",
    "component_fixed_25",
    "complete_fixed_25",
}

for case_id in cases:

    observed = set(
        final_df.loc[
            final_df[
                "case_id"
            ]
            == case_id,
            "condition",
        ]
    )

    if (
        observed
        != expected_conditions
    ):

        condition_matrix_failures.append(
            (
                case_id,
                sorted(
                    expected_conditions
                    - observed
                ),
            )
        )


condition_matrix_pass = (
    len(
        condition_matrix_failures
    )
    == 0
)

group_contract_pass = (
    len(
        group_contract_failures
    )
    == 0
)

coordinate_pass = (
    len(
        coordinate_failures
    )
    == 0
)

semantic_hash_pass = (
    len(
        hash_failures
    )
    == 0
)


# ==========================================================================================
# 13. BACKGROUND IDENTITY — PARENT + v1.1
# ==========================================================================================

heading(
    "STEP 5/10 — VERIFY BACKGROUND WAS NEVER ALTERED"
)

background_case_passes = []

for case_id in cases:

    new_hashes = set(
        final_df.loc[
            final_df[
                "case_id"
            ]
            == case_id,
            "background_sha256",
        ]
    )

    parent_hashes = set()

    for condition in (
        expected_conditions
    ):

        parent = load_annotation(
            OLD_CACHE,
            case_id,
            condition,
        )

        parent_hashes.add(
            coordinate_hash(
                parent[
                    "bg_coords"
                ]
            )
        )

    pass_case = bool(
        len(
            new_hashes
        )
        == 1
        and len(
            parent_hashes
        )
        == 1
        and new_hashes
        == parent_hashes
    )

    background_case_passes.append(
        pass_case
    )


background_identity_pass = bool(
    all(
        background_case_passes
    )
)

print(
    f"Background unchanged from cache v1.0 : "
    f"{'PASS' if background_identity_pass else 'FAIL'}"
)


# ==========================================================================================
# 14. FINAL NATURAL-vs-PIXEL / FIXED-PAIR EQUALITY
# ==========================================================================================

heading(
    "STEP 6/10 — VERIFY ACTUAL NETWORK-GRID CAUSAL EQUALITY"
)

pair_rows = []

natural_pixel_pass_count = 0
fixed_pair_pass_count = 0

primary50_np_pass_count = 0
primary50_fixed_pass_count = 0

for case_id in cases:

    case = final_df[
        final_df[
            "case_id"
        ]
        == case_id
    ].set_index(
        "condition"
    )

    for cov in COVERAGES:

        natural_fg = int(
            case.loc[
                f"component_natural_{cov}",
                "foreground_voxels",
            ]
        )

        pixel_fg = int(
            case.loc[
                f"pixel_dropout_matched_{cov}",
                "foreground_voxels",
            ]
        )

        component_fixed_fg = int(
            case.loc[
                f"component_fixed_{cov}",
                "foreground_voxels",
            ]
        )

        complete_fixed_fg = int(
            case.loc[
                f"complete_fixed_{cov}",
                "foreground_voxels",
            ]
        )

        np_equal = (
            natural_fg
            == pixel_fg
        )

        fixed_equal = (
            component_fixed_fg
            == complete_fixed_fg
        )

        if np_equal:

            natural_pixel_pass_count += 1

            if cov == 50:
                primary50_np_pass_count += 1

        if fixed_equal:

            fixed_pair_pass_count += 1

            if cov == 50:
                primary50_fixed_pass_count += 1

        pair_rows.append(
            {
                "case_id":
                    case_id,

                "coverage":
                    cov,

                "natural_unique_fg":
                    natural_fg,

                "pixel_unique_fg":
                    pixel_fg,

                "natural_pixel_exact":
                    np_equal,

                "component_fixed_unique_fg":
                    component_fixed_fg,

                "complete_fixed_unique_fg":
                    complete_fixed_fg,

                "fixed_pair_exact":
                    fixed_equal,
            }
        )


pair_df = pd.DataFrame(
    pair_rows
)

all_natural_pixel_pass = (
    natural_pixel_pass_count
    == 60
)

all_fixed_pair_pass = (
    fixed_pair_pass_count
    == 60
)

primary50_pass = bool(
    primary50_np_pass_count
    == 20
    and primary50_fixed_pass_count
    == 20
)

print(
    f"Natural vs pixel exact               : "
    f"{natural_pixel_pass_count}/60"
)

print(
    f"Fixed component vs complete exact    : "
    f"{fixed_pair_pass_count}/60"
)

print(
    f"PRIMARY 50% natural/pixel             : "
    f"{primary50_np_pass_count}/20"
)

print(
    f"PRIMARY 50% fixed pair                : "
    f"{primary50_fixed_pass_count}/20"
)


# ==========================================================================================
# 15. VERIFY NATURAL MISSINGNESS WAS UNCHANGED
# ==========================================================================================

heading(
    "STEP 7/10 — VERIFY SCIENTIFIC SOURCE CONDITIONS REMAIN FROZEN"
)

natural_unchanged = True
complete_unchanged = True

for case_id in cases:

    for condition in [
        "complete",
        "component_natural_75",
        "component_natural_50",
        "component_natural_25",
    ]:

        old_ann = load_annotation(
            OLD_CACHE,
            case_id,
            condition,
        )

        new_ann = load_annotation(
            NEW_CACHE,
            case_id,
            condition,
        )

        same = (
            old_ann[
                "semantic_sha256"
            ]
            == new_ann[
                "semantic_sha256"
            ]
        )

        if condition == "complete":

            if not same:
                complete_unchanged = False

        else:

            if not same:
                natural_unchanged = False


# Nested group sets.
nested_pass = True

for case_id in cases:

    group_sets = {}

    for cov in COVERAGES:

        ann = load_annotation(
            NEW_CACHE,
            case_id,
            f"component_natural_{cov}",
        )

        group_sets[
            cov
        ] = set(
            int(x)
            for x in np.unique(
                ann[
                    "membership_groups"
                ]
            )
        )

    complete = load_annotation(
        NEW_CACHE,
        case_id,
        "complete",
    )

    complete_groups = set(
        int(x)
        for x in np.unique(
            complete[
                "membership_groups"
            ]
        )
    )

    if not (
        group_sets[
            25
        ].issubset(
            group_sets[
                50
            ]
        )
        and group_sets[
            50
        ].issubset(
            group_sets[
                75
            ]
        )
        and group_sets[
            75
        ].issubset(
            complete_groups
        )
    ):

        nested_pass = False


print(
    f"Complete supervision unchanged        : "
    f"{'PASS' if complete_unchanged else 'FAIL'}"
)

print(
    f"Natural omission annotations unchanged : "
    f"{'PASS' if natural_unchanged else 'FAIL'}"
)

print(
    f"Nested 25% ⊂ 50% ⊂ 75%               : "
    f"{'PASS' if nested_pass else 'FAIL'}"
)


# ==========================================================================================
# 16. NEW TRAINING-CACHE FIREWALL
# ==========================================================================================

heading(
    "STEP 8/10 — FREEZE v1.1 TRAINING-CACHE FIREWALL"
)

training_cache_source = r'''
"""Dense-label firewall for CORA-Lung training cache v1.1."""

from pathlib import Path

import numpy as np
import pandas as pd


ALLOWED_IMAGE_KEYS = {
    "ct_zyx",
    "spacing_zyx_mm",
    "resampled_affine_xyz",
    "crop_origin_zyx",
    "full_shape_zyx",
    "crop_shape_zyx",
    "hu_window",
}


ALLOWED_ANNOTATION_KEYS = {
    "supervision_voxel_zyx",
    "supervision_label",
    "fg_membership_voxel_zyx",
    "fg_membership_group_id",
}


ALLOWED_MANIFEST_COLUMNS = {
    "case_id",
    "source_subject_key",
    "split_role",
    "outer_fold",
    "condition",
    "image_file",
    "annotation_file",
    "image_semantic_sha256",
    "annotation_semantic_sha256",
    "original_condition_native_semantic_sha256",
    "parent_v1_annotation_semantic_sha256",
    "candidate_condition",
    "candidate_native_semantic_sha256",
    "candidate_v1_annotation_semantic_sha256",
    "construction_policy",
    "equalization_target_unique_fg",
    "preprocess_version",
}


FORBIDDEN_TOKENS = {
    "infection_mask",
    "lung_mask",
    "lesion_mask",
    "dense_mask",
    "dense_component",
    "hidden_component",
    "component_volume",
    "component_centroid",
}


def _reject_forbidden(value):

    text = str(value).lower()

    for token in FORBIDDEN_TOKENS:

        if token in text:

            raise RuntimeError(
                f"Forbidden trainer token: {token}"
            )


def validate_training_cache(root):

    root = Path(root)

    manifest = pd.read_csv(
        root / "manifest.csv"
    )

    if (
        set(manifest.columns)
        != ALLOWED_MANIFEST_COLUMNS
    ):

        raise RuntimeError(
            "Unexpected v1.1 training-cache manifest schema."
        )

    for column in manifest.columns:

        _reject_forbidden(
            column
        )

    for _, row in manifest.iterrows():

        _reject_forbidden(
            row["image_file"]
        )

        _reject_forbidden(
            row["annotation_file"]
        )

        image_path = (
            root
            / row["image_file"]
        )

        annotation_path = (
            root
            / row["annotation_file"]
        )

        with np.load(
            image_path,
            allow_pickle=False,
        ) as data:

            if (
                set(data.files)
                != ALLOWED_IMAGE_KEYS
            ):

                raise RuntimeError(
                    f"Unsafe CT cache keys: "
                    f"{image_path}"
                )

            ct = np.asarray(
                data["ct_zyx"]
            )

            if ct.ndim != 3:

                raise RuntimeError(
                    "Cached CT must be 3-D."
                )

        with np.load(
            annotation_path,
            allow_pickle=False,
        ) as data:

            if (
                set(data.files)
                != ALLOWED_ANNOTATION_KEYS
            ):

                raise RuntimeError(
                    f"Unsafe annotation keys: "
                    f"{annotation_path}"
                )

            coords = np.asarray(
                data["supervision_voxel_zyx"]
            )

            labels = np.asarray(
                data["supervision_label"]
            )

            fg_coords = np.asarray(
                data["fg_membership_voxel_zyx"]
            )

            fg_groups = np.asarray(
                data["fg_membership_group_id"]
            )

        if (
            coords.shape
            != (
                len(labels),
                3,
            )
        ):

            raise RuntimeError(
                "Invalid direct-supervision coordinate shape."
            )

        if (
            fg_coords.shape
            != (
                len(fg_groups),
                3,
            )
        ):

            raise RuntimeError(
                "Invalid FG-membership coordinate shape."
            )

        if not set(
            np.unique(
                labels
            )
        ).issubset(
            {
                0,
                1,
            }
        ):

            raise RuntimeError(
                "Only sparse FG/BG labels are permitted."
            )

        if np.any(
            fg_groups <= 0
        ):

            raise RuntimeError(
                "Foreground group IDs must be positive."
            )

        fg_direct = {
            tuple(
                int(x)
                for x in coord
            )
            for coord in coords[
                labels == 1
            ]
        }

        fg_membership_support = {
            tuple(
                int(x)
                for x in coord
            )
            for coord in fg_coords
        }

        if (
            fg_direct
            != fg_membership_support
        ):

            raise RuntimeError(
                "Direct FG and replay-membership support differ."
            )

    return {
        "manifest_rows":
            int(
                len(
                    manifest
                )
            ),

        "cases":
            int(
                manifest[
                    "case_id"
                ].nunique()
            ),

        "conditions":
            int(
                manifest[
                    "condition"
                ].nunique()
            ),
    }
'''

write_text(
    REPO
    / "src/cora_lung/data/training_cache.py",
    training_cache_source,
)


# ==========================================================================================
# 17. REUSABLE POST-TRANSFER EQUALIZATION MODULE
# ==========================================================================================

equalization_source = r'''
"""Post-transfer sparse-control equalization for CORA-Lung."""

from __future__ import annotations

from collections import defaultdict
import hashlib

import numpy as np


def stable_seed(*parts):

    payload = "|".join(
        str(x)
        for x in parts
    )

    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        "little",
        signed=False,
    ) % (2**32 - 1)


def candidate_pool(membership_coords, membership_groups):

    mapping = defaultdict(set)

    for coord, group_id in zip(
        np.asarray(
            membership_coords,
            dtype=np.int32,
        ),
        np.asarray(
            membership_groups,
            dtype=np.int32,
        ),
    ):

        mapping[
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
            membership_groups
        )
    )

    return dict(mapping), groups


def exact_group_preserving_subset(
    coord_to_groups,
    required_groups,
    target,
    seed,
):

    required_groups = set(
        int(x)
        for x in required_groups
    )

    target = int(target)

    candidates = sorted(
        coord_to_groups
    )

    if target > len(candidates):

        raise ValueError(
            "Target exceeds candidate capacity."
        )

    rng = np.random.default_rng(
        seed
    )

    priority = {
        coord:
            float(rng.random())
        for coord in candidates
    }

    selected = set()

    uncovered = set(
        required_groups
    )

    while uncovered:

        options = []

        for coord in candidates:

            gained = (
                coord_to_groups[coord]
                & uncovered
            )

            if gained:

                options.append(
                    (
                        -len(gained),
                        priority[coord],
                        coord,
                    )
                )

        if not options:

            raise ValueError(
                "Unable to cover all foreground groups."
            )

        options.sort()

        chosen = options[0][2]

        selected.add(
            chosen
        )

        uncovered -= (
            coord_to_groups[
                chosen
            ]
        )

    if len(selected) > target:

        raise ValueError(
            "Group-cover size exceeds target."
        )

    remaining = [
        coord
        for coord in candidates
        if coord not in selected
    ]

    needed = (
        target
        - len(selected)
    )

    if needed:

        order = rng.permutation(
            len(remaining)
        )

        for index in order[:needed]:

            selected.add(
                remaining[
                    int(index)
                ]
            )

    if len(selected) != target:

        raise RuntimeError(
            "Exact target was not reached."
        )

    return np.asarray(
        sorted(selected),
        dtype=np.int32,
    )
'''

write_text(
    REPO
    / "src/cora_lung/data/equalization.py",
    equalization_source,
)


# ==========================================================================================
# 18. UNIT TESTS
# ==========================================================================================

equalization_tests = r'''
import numpy as np

from cora_lung.data.equalization import (
    candidate_pool,
    exact_group_preserving_subset,
)


def test_exact_target_and_group_preservation():

    membership_coords = np.asarray(
        [
            [0, 0, 0],
            [0, 0, 1],
            [0, 0, 2],
            [0, 1, 0],
            [0, 1, 1],
            [0, 1, 2],
        ],
        dtype=np.int32,
    )

    membership_groups = np.asarray(
        [
            1,
            1,
            2,
            2,
            3,
            3,
        ],
        dtype=np.int32,
    )

    pool, groups = candidate_pool(
        membership_coords,
        membership_groups,
    )

    selected = exact_group_preserving_subset(
        pool,
        groups,
        4,
        17,
    )

    assert len(selected) == 4

    represented = set()

    for coord in selected:

        represented.update(
            pool[
                tuple(coord.tolist())
            ]
        )

    assert represented == {
        1,
        2,
        3,
    }


def test_multi_group_collision_can_cover_two_groups():

    membership_coords = np.asarray(
        [
            [1, 1, 1],
            [1, 1, 1],
            [2, 2, 2],
        ],
        dtype=np.int32,
    )

    membership_groups = np.asarray(
        [
            1,
            2,
            3,
        ],
        dtype=np.int32,
    )

    pool, groups = candidate_pool(
        membership_coords,
        membership_groups,
    )

    selected = exact_group_preserving_subset(
        pool,
        groups,
        2,
        3,
    )

    assert len(selected) == 2
'''

write_text(
    REPO
    / "tests/test_training_grid_equalization.py",
    equalization_tests,
)


training_cache_test = r'''
from pathlib import Path

import numpy as np
import pandas as pd

from cora_lung.data.training_cache import (
    validate_training_cache,
)


def test_training_cache_v1_1_schema(tmp_path):

    root = Path(tmp_path)

    (root / "images").mkdir()
    (root / "annotations").mkdir()

    np.savez_compressed(
        root / "images/case.npz",
        ct_zyx=np.zeros(
            (4, 5, 6),
            dtype=np.float16,
        ),
        spacing_zyx_mm=np.asarray(
            [3., 1.5, 1.5],
            dtype=np.float32,
        ),
        resampled_affine_xyz=np.eye(
            4,
            dtype=float,
        ),
        crop_origin_zyx=np.zeros(
            3,
            dtype=np.int32,
        ),
        full_shape_zyx=np.asarray(
            [4, 5, 6],
            dtype=np.int32,
        ),
        crop_shape_zyx=np.asarray(
            [4, 5, 6],
            dtype=np.int32,
        ),
        hu_window=np.asarray(
            [-1000., 400.],
            dtype=np.float32,
        ),
    )

    np.savez_compressed(
        root / "annotations/case.npz",
        supervision_voxel_zyx=np.asarray(
            [
                [1, 2, 3],
                [1, 2, 4],
            ],
            dtype=np.int32,
        ),
        supervision_label=np.asarray(
            [
                1,
                0,
            ],
            dtype=np.int8,
        ),
        fg_membership_voxel_zyx=np.asarray(
            [
                [1, 2, 3],
            ],
            dtype=np.int32,
        ),
        fg_membership_group_id=np.asarray(
            [
                1,
            ],
            dtype=np.int32,
        ),
    )

    pd.DataFrame(
        [
            {
                "case_id":
                    "case",

                "source_subject_key":
                    "case",

                "split_role":
                    "development",

                "outer_fold":
                    float("nan"),

                "condition":
                    "complete",

                "image_file":
                    "images/case.npz",

                "annotation_file":
                    "annotations/case.npz",

                "image_semantic_sha256":
                    "a" * 64,

                "annotation_semantic_sha256":
                    "b" * 64,

                "original_condition_native_semantic_sha256":
                    "c" * 64,

                "parent_v1_annotation_semantic_sha256":
                    "d" * 64,

                "candidate_condition":
                    "complete",

                "candidate_native_semantic_sha256":
                    "e" * 64,

                "candidate_v1_annotation_semantic_sha256":
                    "f" * 64,

                "construction_policy":
                    "unchanged",

                "equalization_target_unique_fg":
                    1,

                "preprocess_version":
                    "1.1",
            }
        ]
    ).to_csv(
        root / "manifest.csv",
        index=False,
    )

    result = validate_training_cache(
        root
    )

    assert result[
        "manifest_rows"
    ] == 1
'''

write_text(
    REPO
    / "tests/test_training_cache_firewall.py",
    training_cache_test,
)


# ==========================================================================================
# 19. RUN FIREWALL + ALL DATA TESTS
# ==========================================================================================

if str(
    REPO
    / "src"
) not in sys.path:

    sys.path.insert(
        0,
        str(
            REPO
            / "src"
        ),
    )

from cora_lung.data.training_cache import (
    validate_training_cache,
)

firewall_result = validate_training_cache(
    NEW_CACHE
)

print(
    f"✓ v1.1 training-cache firewall: "
    f"PASS "
    f"({firewall_result['manifest_rows']} rows)"
)

pytest_result = sh(
    [
        sys.executable,
        "-m",
        "pytest",
        "tests/test_firewall.py",
        "tests/test_scribble_budget.py",
        "tests/test_physical_geometry.py",
        "tests/test_training_cache_firewall.py",
        "tests/test_training_grid_equalization.py",
        "-q",
    ],
    cwd=REPO,
    check=False,
)

print(
    pytest_result.stdout.strip()
)

if pytest_result.stderr.strip():

    print(
        pytest_result.stderr.strip()
    )

tests_pass = (
    pytest_result.returncode
    == 0
)

if not tests_pass:

    raise RuntimeError(
        "Block-06B scientific/unit tests FAILED."
    )

print(
    "✓ Scientific/unit tests: PASS"
)


# ==========================================================================================
# 20. FINAL ACCEPTANCE CONDITIONS
# ==========================================================================================

hard_contract_pass = bool(
    condition_matrix_pass
    and group_contract_pass
    and coordinate_pass
    and semantic_hash_pass
    and background_identity_pass
    and complete_unchanged
    and natural_unchanged
    and nested_pass
    and all_natural_pixel_pass
    and all_fixed_pair_pass
    and primary50_pass
    and image_copy_pass
    and tests_pass
)

if not hard_contract_pass:

    raise RuntimeError(
        "Block 06B final causal-cache contract FAILED."
    )


# ==========================================================================================
# 21. HASH FINAL LOCAL CACHE
# ==========================================================================================

heading(
    "STEP 9/10 — FREEZE CACHE v1.1 PROVENANCE"
)

cache_files = sorted(
    [
        p
        for p in NEW_CACHE.rglob(
            "*"
        )
        if p.is_file()
    ],
    key=lambda p:
        str(
            p.relative_to(
                NEW_CACHE
            )
        ),
)

cache_hash_rows = []

for path in tqdm(
    cache_files,
    desc="Hashing cache v1.1",
    unit="file",
):

    cache_hash_rows.append(
        {
            "relative_path":
                str(
                    path.relative_to(
                        NEW_CACHE
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

cache_hash_df = pd.DataFrame(
    cache_hash_rows
)

cache_bytes = int(
    cache_hash_df[
        "bytes"
    ].sum()
)


# ==========================================================================================
# 22. SAVE AUDITS TO GITHUB
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

archive_config_dir = (
    REPO
    / "configs/archive"
)

manifest_dir.mkdir(
    parents=True,
    exist_ok=True,
)

audit_dir.mkdir(
    parents=True,
    exist_ok=True,
)

figure_dir.mkdir(
    parents=True,
    exist_ok=True,
)

archive_config_dir.mkdir(
    parents=True,
    exist_ok=True,
)


new_manifest.to_csv(
    manifest_dir
    / "training_cache_v1_1_manifest_lineage.csv",
    index=False,
)

correction_df.to_csv(
    manifest_dir
    / "training_grid_equalization_actions.csv",
    index=False,
)

pair_df.to_csv(
    manifest_dir
    / "training_grid_budget_equivalence_v1_1.csv",
    index=False,
)

final_df.to_csv(
    manifest_dir
    / "training_grid_final_annotation_summary.csv",
    index=False,
)

image_copy_df.to_csv(
    manifest_dir
    / "training_grid_image_copy_integrity.csv",
    index=False,
)

cache_hash_df.to_csv(
    manifest_dir
    / "training_cache_v1_1_file_hashes.csv",
    index=False,
)


# ==========================================================================================
# 23. ARCHIVE v1.0 CONFIG + FREEZE v1.1 CONFIG
# ==========================================================================================

config_path = (
    REPO
    / "configs/preprocess_primary.yaml"
)

archive_config = (
    archive_config_dir
    / "preprocess_primary_v1_0.yaml"
)

if (
    config_path.exists()
    and not archive_config.exists()
):

    shutil.copy2(
        config_path,
        archive_config,
    )


config_v1_1 = """
preprocessing:
  version: "1.1"
  parent_version: "1.0"

geometry:
  internal_array_order: [z, y, x]
  target_spacing_zyx_mm: [3.0, 1.5, 1.5]
  nifti_resample_spacing_xyz_mm: [1.5, 1.5, 3.0]
  canonical_orientation_before_resample: true
  ct_interpolation: linear
  sparse_coordinate_transfer: physical_world_coordinate_nearest_voxel

intensity:
  hu_clip: [-1000, 400]
  output_range: [-1.0, 1.0]
  cache_dtype: float16

crop:
  dense_masks_used: false
  scribbles_used: false
  ct_only: true
  physical_margin_mm: 15.0

collision_policy:
  same_class_spatial_collision: merge_to_unique_training_voxel
  foreground_multi_group_collision: preserve_all_group_memberships
  foreground_background_collision: hard_failure
  group_loss: hard_failure

post_transfer_causal_equalization:
  enabled: true
  seed: 20260912

  component_natural:
    action: unchanged

  random_pixel_control:
    target: transferred_natural_unique_foreground_voxel_count
    candidate_pool: transferred_component_complete_unique_foreground_support
    preserve_all_complete_foreground_groups: true
    modify_only_if_raw_pair_mismatched: true

  fixed_budget_pair:
    target: minimum_unique_foreground_capacity_of_transferred_pair
    preserve_each_conditions_foreground_groups: true
    modify_only_larger_side: true

  background:
    action: unchanged
    require_byte_equivalent_coordinate_set: true

firewall:
  dense_masks_in_training_cache: prohibited
  dense_component_information: prohibited
  unknown_voxels_materialized_as_background: false

scientific_status:
  correction_made_before_model_training: true
  source_weak_labels_modified: false
  development_split_modified: false
  final_outer_folds_modified: false
"""

write_text(
    config_path,
    config_v1_1,
)


# ==========================================================================================
# 24. PROTOCOL AMENDMENT
# ==========================================================================================

protocol = f"""
# CORA-Lung Training-Grid Preprocessing Protocol

## Accepted version

**Version:** {PREPROCESS_VERSION_NEW}

**Parent version:** {PREPROCESS_VERSION_OLD}

**Correction made before model training:** Yes

## Frozen image preprocessing

CT image preprocessing itself is unchanged from v1.0:

- canonical NIfTI orientation;
- 3.0 × 1.5 × 1.5 mm internal `(z,y,x)` spacing;
- linear CT interpolation;
- HU clipping to `[-1000, 400]`;
- normalization to `[-1, 1]`;
- CT-image-only thoracic crop.

The twenty v1.1 image-cache files are verified byte-for-byte identical to
the v1.0 image cache.

## Why v1.1 was required

Block 05 matched sparse foreground budgets on the native annotation grid.

During Block 06, native sparse coordinates were transferred through physical
world coordinates to the coarser model-training grid.

Several nearby labelled native voxels therefore mapped to the same model voxel.

Same-class collisions were correctly merged, but this meant that equal native
annotation counts were no longer equal as **unique labelled voxels actually
seen by the network**.

The raw v1.0 cache therefore failed the causal-control budget test even though:

- geometry transfer passed;
- no foreground/background conflicts occurred;
- all foreground groups survived;
- background supervision remained identical;
- the dense-label firewall passed.

## v1.1 correction

The source weak labels are not modified.

Equalization occurs only after physical transfer and same-class merging.

### Natural component omission

Natural component-omission annotations remain exactly unchanged.

They define the target unique-foreground budget for the matched pixel control.

### Random-pixel control

When the raw transferred pixel control already matches the natural condition,
it is retained unchanged.

Otherwise a deterministic subset is selected from the transferred
component-complete unique-FG support.

The final control:

1. has exactly the same number of unique FG voxels as the paired natural
   component-omission condition;
2. preserves representation of every complete foreground group;
3. introduces no new spatial location outside the transferred complete
   scribble support;
4. uses exactly the original explicit-background coordinates.

### Fixed-pixel pair

For each transferred component-fixed / complete-fixed pair:

`shared target = min(unique FG capacity of both sides)`

The side already at the target remains unchanged.

Only the larger side is deterministically thinned.

The final pair:

- has exactly equal unique foreground supervision;
- preserves every foreground group present on each respective side;
- introduces no new foreground coordinate;
- preserves the original background.

## Equalization feasibility

Feasibility was established prospectively in Block 06A before any v1.1
training annotation was created:

- natural-vs-pixel: 60/60 constructively feasible;
- fixed-budget pairs: 60/60 constructively feasible;
- primary 50% natural-vs-pixel: 20/20 feasible;
- primary 50% fixed pairs: 20/20 feasible.

## Final v1.1 validation

- condition matrix: 260/260;
- natural-vs-pixel exact pairs: 60/60;
- fixed-budget exact pairs: 60/60;
- primary 50% natural-vs-pixel: 20/20;
- primary 50% fixed-budget pairs: 20/20;
- natural component omission modified: no;
- component-complete supervision modified: no;
- background coordinates modified: no;
- foreground groups lost: no;
- dense masks accessed by v1.1 equalization: no;
- model training performed before correction: no.
"""

write_text(
    REPO
    / "docs/training_grid_preprocessing_protocol.md",
    protocol,
)


# ==========================================================================================
# 25. PUBLICATION-QUALITY AUDIT FIGURE
# ==========================================================================================

raw_budget_path = (
    REPO
    / "data/manifests/training_grid_budget_equivalence.csv"
)

raw_budget = pd.read_csv(
    raw_budget_path
)

figure_rows = []

for cov in COVERAGES:

    raw_cov = raw_budget[
        raw_budget[
            "coverage"
        ]
        == cov
    ]

    final_cov = pair_df[
        pair_df[
            "coverage"
        ]
        == cov
    ]

    figure_rows.append(
        {
            "coverage":
                cov,

            "pair_type":
                "Natural vs Pixel",

            "raw_mismatches":
                int(
                    (
                        ~raw_cov[
                            "natural_pixel_exact"
                        ].astype(
                            bool
                        )
                    ).sum()
                ),

            "v1_1_mismatches":
                int(
                    (
                        ~final_cov[
                            "natural_pixel_exact"
                        ].astype(
                            bool
                        )
                    ).sum()
                ),
        }
    )

    figure_rows.append(
        {
            "coverage":
                cov,

            "pair_type":
                "Fixed Pair",

            "raw_mismatches":
                int(
                    (
                        ~raw_cov[
                            "fixed_pair_exact"
                        ].astype(
                            bool
                        )
                    ).sum()
                ),

            "v1_1_mismatches":
                int(
                    (
                        ~final_cov[
                            "fixed_pair_exact"
                        ].astype(
                            bool
                        )
                    ).sum()
                ),
        }
    )


figure_df = pd.DataFrame(
    figure_rows
)

labels = [
    (
        f"{row['coverage']}%\n"
        f"{row['pair_type']}"
    )
    for _, row in (
        figure_df.iterrows()
    )
]

x = np.arange(
    len(
        figure_df
    )
)

width = 0.38

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelweight": "bold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "savefig.dpi": 600,
    }
)

fig, ax = plt.subplots(
    figsize=(
        12,
        6.5,
    )
)

ax.bar(
    x
    - width / 2,
    figure_df[
        "raw_mismatches"
    ],
    width,
    label="Raw Training-Grid Cache v1.0",
)

ax.bar(
    x
    + width / 2,
    figure_df[
        "v1_1_mismatches"
    ],
    width,
    label="Causally Equalized Cache v1.1",
)

ax.set_xticks(
    x
)

ax.set_xticklabels(
    labels,
    fontweight="bold",
)

ax.set_ylabel(
    "Mismatched Case-Level Causal Pairs",
    fontweight="bold",
)

ax.set_xlabel(
    "Coverage and Causal-Control Pair",
    fontweight="bold",
)

ax.set_title(
    "Causal-Budget Equality Before and After Post-Transfer Unique-Voxel Equalization\n"
    "All Comparisons Are Defined on the Actual 3-D Network Training Grid",
    fontweight="bold",
)

ax.grid(
    axis="y",
    alpha=0.25,
)

legend = ax.legend()

for text in (
    legend.get_texts()
):

    text.set_fontweight(
        "bold"
    )

fig.tight_layout()

fig.savefig(
    figure_dir
    / "fig08_training_grid_causal_equalization.png",
    dpi=600,
    bbox_inches="tight",
)

fig.savefig(
    figure_dir
    / "fig08_training_grid_causal_equalization.pdf",
    bbox_inches="tight",
)

plt.close(
    fig
)

print(
    "✓ Publication-quality causal-equalization figure generated."
)


# ==========================================================================================
# 26. BLOCK-06 FINAL AUDIT JSON
# ==========================================================================================

block06_audit = {
    "project":
        PROJECT,

    "block":
        BLOCK_ID,

    "corrective_block":
        CORRECTIVE_BLOCK,

    "generated_at_utc":
        NOW_ISO,

    "status":
        "PASS",

    "accepted_preprocess_version":
        PREPROCESS_VERSION_NEW,

    "parent_cache":
        {
            "version":
                PREPROCESS_VERSION_OLD,

            "status":
                "REVIEW_REQUIRED_PRIMARY_CAUSAL_BUDGET",

            "retained_for_provenance":
                True,
        },

    "cache_v1_1":
        {
            "cases":
                20,

            "conditions_per_case":
                13,

            "annotations":
                260,

            "files":
                int(
                    len(
                        cache_hash_df
                    )
                ),

            "bytes":
                cache_bytes,

            "human_size":
                human_bytes(
                    cache_bytes
                ),

            "ct_images_byte_identical_to_v1_0":
                bool(
                    image_copy_pass
                ),
        },

    "correction": {
        "dense_masks_accessed":
            False,

        "source_weak_annotations_modified":
            False,

        "complete_annotations_modified":
            False,

        "natural_component_omission_modified":
            False,

        "background_coordinates_modified":
            False,

        "pixel_controls_corrected":
            int(
                pixel_controls_corrected
            ),

        "fixed_pair_sides_corrected":
            int(
                fixed_sides_corrected
            ),

        "total_semantically_corrected_conditions":
            int(
                changed_semantic_conditions
            ),

        "policy":
            (
                "deterministic post-transfer unique-voxel "
                "equalization with foreground-group preservation"
            ),
    },

    "causal_contract": {
        "natural_pixel_exact_pairs":
            f"{natural_pixel_pass_count}/60",

        "fixed_budget_exact_pairs":
            f"{fixed_pair_pass_count}/60",

        "primary_50_natural_pixel":
            f"{primary50_np_pass_count}/20",

        "primary_50_fixed_pair":
            f"{primary50_fixed_pass_count}/20",
    },

    "qa": {
        "condition_matrix":
            bool(
                condition_matrix_pass
            ),

        "foreground_group_contract":
            bool(
                group_contract_pass
            ),

        "coordinates_inside_cached_images":
            bool(
                coordinate_pass
            ),

        "semantic_hash_validation":
            bool(
                semantic_hash_pass
            ),

        "background_identity":
            bool(
                background_identity_pass
            ),

        "component_complete_unchanged":
            bool(
                complete_unchanged
            ),

        "natural_omission_unchanged":
            bool(
                natural_unchanged
            ),

        "nested_missingness":
            bool(
                nested_pass
            ),

        "training_cache_firewall":
            "PASS",

        "unit_tests":
            "PASS",
    },

    "training_performed":
        False,

    "training_authorized":
        False,
}

write_json(
    audit_dir
    / "block06_training_grid_cache.json",
    block06_audit,
)


write_json(
    audit_dir
    / "block06b_training_grid_equalization.json",
    block06_audit,
)


# ==========================================================================================
# 27. CAPTURE EXECUTED SOURCE
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
        "CORA-LUNG — CODE BLOCK 06B"
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
            / "block06b_finalize_training_cache_v1_1.py"
        ).write_text(
            raw_cell,
            encoding="utf-8",
        )

        source_capture = "PASS"

except Exception:
    pass


# ==========================================================================================
# 28. UPDATE PROJECT STATE
# ==========================================================================================

state.update(
    {
        "last_attempted_block":
            "06",

        "last_completed_block":
            "06",

        "last_completed_block_name":
            "firewall_safe_training_grid_cache",

        "current_stage":
            "training_grid_cache_verified",

        "current_gate":
            "POST_GATE_A_PRE_GATE_B",

        "gate_a":
            "PASS",

        "gate_b":
            "NOT_RUN",

        "training_grid_cache":
            "PASS",

        "preprocess_version":
            PREPROCESS_VERSION_NEW,

        "training_cache_root":
            str(
                NEW_CACHE
            ),

        "training_cache_firewall":
            "PASS",

        "post_transfer_equalization_feasibility":
            "PASS",

        "primary_50_equalization_feasibility":
            "PASS",

        "primary_50_post_transfer_budget":
            "PASS",

        "all_coverage_post_transfer_budget":
            "PASS",

        "natural_component_omission_cache":
            "UNCHANGED",

        "background_supervision_cache":
            "UNCHANGED",

        "training_authorized":
            False,

        "next_action":
            (
                "Audit Block 06B. If accepted, construct the "
                "Gate-B pilot training harness and run the frozen "
                "development-only mechanism benchmark before any "
                "final outer-fold training."
            ),

        "updated_at_utc":
            NOW_ISO,
    }
)

write_json(
    state_path,
    state,
)


# ==========================================================================================
# 29. REFRESH REPOSITORY MANIFEST
#     NOTE: EXCLUDE MANIFEST ITSELF TO AVOID SELF-HASH INCONSISTENCY.
# ==========================================================================================

repo_manifest_path = (
    REPO
    / "REPOSITORY_MANIFEST.json"
)

repo_files = sorted(
    [
        p
        for p in REPO.rglob(
            "*"
        )
        if (
            p.is_file()
            and ".git"
            not in p.parts
            and p
            != repo_manifest_path
        )
    ],
    key=lambda p:
        str(
            p.relative_to(
                REPO
            )
        ),
)

repo_manifest_rows = []

for path in tqdm(
    repo_files,
    desc="Refreshing repository manifest",
):

    repo_manifest_rows.append(
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
    repo_manifest_path,
    {
        "generated_at_utc":
            NOW_ISO,

        "completed_block":
            "06",

        "corrective_block":
            "06B",

        "self_included":
            False,

        "files":
            repo_manifest_rows,
    },
)


# ==========================================================================================
# 30. COMMIT + PUSH
# ==========================================================================================

heading(
    "STEP 10/10 — COMMIT ACCEPTED TRAINING-CACHE v1.1 PROTOCOL"
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
    "/tmp/cora_git_askpass_block06b.sh"
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

    changed = status.splitlines()

    print(
        f"Tracked changes: "
        f"{len(changed)}"
    )

    for line in changed[
        :50
    ]:

        print(
            " ",
            line,
        )

    if len(
        changed
    ) > 50:

        print(
            f"  ... +"
            f"{len(changed)-50} more"
        )

    commit_message = (
        "fix: freeze exact post-transfer causal training cache v1.1"
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

print(
    "✓ GitHub synchronization: PASS"
)


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


# ==========================================================================================
# 31. FINAL REPORT
# ==========================================================================================

print("\n")
print("=" * 118)
print("CORA-LUNG CODE BLOCK 06B — FINAL TRAINING-CACHE v1.1 FREEZE REPORT")
print("=" * 118)

print(f"""
CACHE LINEAGE
-------------
Raw transferred cache                    : v{PREPROCESS_VERSION_OLD}
Accepted training cache                  : v{PREPROCESS_VERSION_NEW}
CT volumes                               : 20
Conditions per CT                        : 13
Final sparse annotation artifacts        : {len(new_manifest)}
Final cache files                        : {len(cache_hash_df)}
Final local cache size                   : {human_bytes(cache_bytes)}

CT IMAGE PIPELINE
-----------------
Images resampled again                    : NO
Images modified by 06B                    : NO
Byte-identical to v1.0 images             : {'PASS' if image_copy_pass else 'FAIL'}
Frozen spacing                            : 3.0 × 1.5 × 1.5 mm
Image-derived crop changed                : NO

SCIENTIFIC CONDITIONS
---------------------
Component-complete annotations changed    : NO
Natural component omission changed        : NO
Missing component identities changed      : NO
25% / 50% / 75% nested omission changed   : NO
Background coordinates changed            : NO
Dense masks accessed                      : NO

POST-TRANSFER CORRECTION
------------------------
Raw mismatched pixel controls             : 50/60
Pixel controls corrected                  : {pixel_controls_corrected}
Raw mismatched fixed pairs                : 53/60
Fixed-pair sides corrected                : {fixed_sides_corrected}
Total semantically corrected conditions   : {changed_semantic_conditions}

FINAL ACTUAL NETWORK-GRID BUDGET
--------------------------------
Natural-vs-pixel exact pairs              : {natural_pixel_pass_count}/60
Fixed component/complete exact pairs      : {fixed_pair_pass_count}/60

PRIMARY 50% GATE-B INPUT
------------------------
Natural-vs-pixel exact                    : {primary50_np_pass_count}/20
Fixed component/complete exact            : {primary50_fixed_pass_count}/20

GROUP / BACKGROUND CONTRACT
---------------------------
Foreground-group contract                 : {'PASS' if group_contract_pass else 'FAIL'}
Natural missingness nested                : {'PASS' if nested_pass else 'FAIL'}
Background identical to v1.0              : {'PASS' if background_identity_pass else 'FAIL'}
All coordinates inside CT crops           : {'PASS' if coordinate_pass else 'FAIL'}
Condition matrix 20 × 13                  : {'PASS' if condition_matrix_pass else 'FAIL'}

FIREWALL / QA
-------------
Training-cache firewall                   : PASS
Semantic hashes                           : {'PASS' if semantic_hash_pass else 'FAIL'}
Scientific/unit tests                     : PASS
Dense lesion information in cache         : NO
Unknown voxels converted to BG            : NO

REPRODUCIBILITY
---------------
v1.0 failure cache preserved               : YES
v1.0 preprocessing config archived         : YES
v1.1 lineage manifest committed            : YES
v1.1 file hashes committed                 : YES
Equalization action table committed        : YES
Protocol amendment committed               : YES
Exact Block-06B source captured             : {source_capture}
Large training cache committed to GitHub    : NO
Training cache reconstructible              : YES

SCIENTIFIC STATUS
-----------------
Gate A                                     : PASS
Block 05 weak-label generator              : PASS
Block 06 training-grid cache               : PASS
Gate B                                     : NOT RUN
Model training                             : NOT STARTED
Training authorized                        : NO

GITHUB
------
Starting commit                            : {starting_commit[:12]}
Current commit                             : {current_commit[:12]}
Synchronization                            : PASS

NEXT
----
Send me this COMPLETE report.

If every final contract is PASS, Block 06 will be scientifically closed.

The next block will NOT immediately launch a large experiment.

We will first build and test the Gate-B pilot training harness on the four
permanent development volumes:

  • deterministic patch sampler
  • partial BCE + partial Dice
  • exact query-holdout arithmetic tests
  • lightweight 3-D residual U-Net
  • AMP / gradient accumulation
  • final-epoch checkpoint policy
  • GPU-memory profiling
  • crash-safe checkpoints
  • experiment registry / resume mechanism

Only after that harness passes will we run the first Gate-B comparison.
""")

print("=" * 118)