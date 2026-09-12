# ==========================================================================================
# CORA-LUNG — CODE BLOCK 06B-A
# Resume Block 06B After Stale Python Module Import
#
# ROOT CAUSE
#   src/cora_lung/data/training_cache.py was overwritten with the v1.1 validator,
#   but Python had already imported the old v1.0 module earlier in the notebook.
#
#   Therefore:
#       source file on disk  = v1.1
#       module in sys.modules = stale v1.0
#
# THIS BLOCK:
#   * DOES NOT rebuild CT images
#   * DOES NOT regenerate weak labels
#   * DOES NOT redo equalization
#   * DOES NOT access dense masks
#   * DOES NOT train
#
# IT:
#   1. verifies the already-created v1.1 cache;
#   2. forces a fresh import of the v1.1 firewall;
#   3. independently audits all 260 annotations;
#   4. verifies 60/60 + 60/60 causal equality;
#   5. verifies 20/20 + 20/20 primary 50% equality;
#   6. verifies complete/natural/background conditions were unchanged;
#   7. freezes cache hashes, protocol, figures, project state;
#   8. commits and pushes Block 06 as PASS.
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
CORRECTIVE_BLOCK = "06B-A"

PREPROCESS_VERSION = "1.1"

WORK = Path("/kaggle/working")

REPO = (
    WORK
    / "CORA-LUNG"
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

NEW_MANIFEST = (
    NEW_CACHE
    / "manifest.csv"
)

NOW = datetime.now(
    timezone.utc
)

NOW_ISO = NOW.strftime(
    "%Y-%m-%dT%H:%M:%SZ"
)

EXPECTED_CONDITIONS = [
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
]

COVERAGES = [
    75,
    50,
    25,
]


# ==========================================================================================
# 1. HELPERS
# ==========================================================================================

def heading(text):

    print(
        "\n"
        + "=" * 118
    )

    print(text)

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


def write_json(
    path,
    obj,
):

    path = Path(path)

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

    path = Path(path)

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


def coordinate_hash(
    coords,
):

    coords = np.asarray(
        coords,
        dtype=np.int32,
    )

    if len(coords) == 0:

        return hashlib.sha256(
            b""
        ).hexdigest()

    order = np.lexsort(
        (
            coords[:, 2],
            coords[:, 1],
            coords[:, 0],
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


def human_bytes(value):

    value = float(value)

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

        value /= 1024

    return (
        f"{value:.2f} PB"
    )


def load_annotation(
    cache_root,
    case_id,
    condition,
):

    path = (
        Path(cache_root)
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
            f"Missing annotation: {path}"
        )

    with np.load(
        path,
        allow_pickle=False,
    ) as data:

        coords = np.asarray(
            data[
                "supervision_voxel_zyx"
            ],
            dtype=np.int32,
        )

        labels = np.asarray(
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

    return {
        "path":
            path,

        "coords":
            coords,

        "labels":
            labels,

        "membership_coords":
            membership_coords,

        "membership_groups":
            membership_groups,

        "fg":
            coords[
                labels == 1
            ],

        "bg":
            coords[
                labels == 0
            ],

        "hash":
            semantic_hash(
                coords,
                labels,
                membership_coords,
                membership_groups,
            ),
    }


# ==========================================================================================
# 2. VERIFY PRESERVED 06B STATE
# ==========================================================================================

heading(
    "CORA-LUNG BLOCK 06B-A — RESUME v1.1 CACHE FREEZE"
)

if not (
    REPO
    / ".git"
).exists():

    raise RuntimeError(
        "CORA-LUNG repository is missing."
    )

if not OLD_MANIFEST.exists():

    raise RuntimeError(
        "Parent training cache v1.0 is missing."
    )

if not NEW_MANIFEST.exists():

    raise RuntimeError(
        "The partially completed v1.1 cache does not exist. "
        "Do not run this resume block."
    )


old_manifest = pd.read_csv(
    OLD_MANIFEST
)

new_manifest = pd.read_csv(
    NEW_MANIFEST
)


print(
    f"Parent cache rows        : "
    f"{len(old_manifest)}"
)

print(
    f"v1.1 cache rows          : "
    f"{len(new_manifest)}"
)

print(
    f"v1.1 annotation files    : "
    f"{len(list((NEW_CACHE / 'annotations').glob('*.npz')))}"
)

print(
    f"v1.1 image files         : "
    f"{len(list((NEW_CACHE / 'images').glob('*.npz')))}"
)


if len(old_manifest) != 260:

    raise RuntimeError(
        "Parent manifest must contain 260 rows."
    )

if len(new_manifest) != 260:

    raise RuntimeError(
        "v1.1 manifest must contain 260 rows."
    )

if len(
    list(
        (
            NEW_CACHE
            / "annotations"
        ).glob(
            "*.npz"
        )
    )
) != 260:

    raise RuntimeError(
        "v1.1 cache must contain 260 sparse annotations."
    )

if len(
    list(
        (
            NEW_CACHE
            / "images"
        ).glob(
            "*.npz"
        )
    )
) != 20:

    raise RuntimeError(
        "v1.1 cache must contain exactly 20 CT images."
    )

print(
    "✓ Partially completed 06B cache is preserved."
)


# ==========================================================================================
# 3. CONFIRM NEW VALIDATOR EXISTS ON DISK
# ==========================================================================================

heading(
    "STEP 1/10 — VERIFY v1.1 FIREWALL SOURCE ON DISK"
)

validator_path = (
    REPO
    / "src/cora_lung/data/training_cache.py"
)

if not validator_path.exists():

    raise RuntimeError(
        "training_cache.py is missing."
    )


validator_text = validator_path.read_text(
    encoding="utf-8"
)


if (
    "Unexpected v1.1 training-cache manifest schema."
    not in validator_text
):

    raise RuntimeError(
        "The v1.1 training-cache firewall source "
        "was not written successfully."
    )


if (
    "original_condition_native_semantic_sha256"
    not in validator_text
):

    raise RuntimeError(
        "The validator on disk does not contain "
        "the v1.1 lineage schema."
    )


print(
    "✓ v1.1 firewall source exists on disk."
)

print(
    "✓ v1.1 lineage manifest schema detected."
)


# ==========================================================================================
# 4. FORCE FRESH PYTHON IMPORT
# ==========================================================================================

heading(
    "STEP 2/10 — PURGE STALE PYTHON MODULE CACHE"
)

SRC = (
    REPO
    / "src"
)

if str(SRC) not in sys.path:

    sys.path.insert(
        0,
        str(SRC),
    )


modules_to_remove = [
    name
    for name in list(
        sys.modules
    )
    if (
        name
        == "cora_lung.data.training_cache"
        or name.startswith(
            "cora_lung.data.training_cache."
        )
    )
]


for module_name in modules_to_remove:

    del sys.modules[
        module_name
    ]


importlib.invalidate_caches()


training_cache_module = importlib.import_module(
    "cora_lung.data.training_cache"
)


module_file = Path(
    training_cache_module.__file__
).resolve()


print(
    f"Fresh module loaded from : "
    f"{module_file}"
)


if (
    module_file
    != validator_path.resolve()
):

    raise RuntimeError(
        "Python imported training_cache from an unexpected location."
    )


print(
    "✓ Stale module cache purged."
)


# ==========================================================================================
# 5. RUN FRESH FIREWALL
# ==========================================================================================

heading(
    "STEP 3/10 — VALIDATE TRAINING CACHE v1.1"
)


firewall_result = (
    training_cache_module
    .validate_training_cache(
        NEW_CACHE
    )
)


print(
    "✓ Runtime training-cache firewall: PASS"
)

print(
    f"  Manifest rows : "
    f"{firewall_result['manifest_rows']}"
)

print(
    f"  Cases         : "
    f"{firewall_result['cases']}"
)

print(
    f"  Conditions    : "
    f"{firewall_result['conditions']}"
)


# ==========================================================================================
# 6. RUN TEST SUITE IN FRESH SUBPROCESS
# ==========================================================================================

heading(
    "STEP 4/10 — RUN FIREWALL / GEOMETRY / EQUALIZATION TESTS"
)


pytest_env = os.environ.copy()

existing_pythonpath = pytest_env.get(
    "PYTHONPATH",
    "",
)

pytest_env[
    "PYTHONPATH"
] = (
    str(SRC)
    if not existing_pythonpath
    else (
        str(SRC)
        + os.pathsep
        + existing_pythonpath
    )
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
        "Post-reload scientific/unit tests FAILED."
    )


print(
    "✓ Fresh-subprocess test suite: PASS"
)


# ==========================================================================================
# 7. VERIFY 20 × 13 CONDITION MATRIX
# ==========================================================================================

heading(
    "STEP 5/10 — REAUDIT ALL 260 FINAL CONDITIONS"
)


cases = sorted(
    new_manifest[
        "case_id"
    ].astype(str)
    .unique()
)


if len(cases) != 20:

    raise RuntimeError(
        "Expected exactly 20 CT volumes."
    )


expected_condition_set = set(
    EXPECTED_CONDITIONS
)


condition_failures = []

for case_id in cases:

    observed = set(
        new_manifest.loc[
            new_manifest[
                "case_id"
            ].astype(str)
            == case_id,
            "condition",
        ].astype(str)
    )

    if observed != expected_condition_set:

        condition_failures.append(
            {
                "case_id":
                    case_id,

                "missing":
                    sorted(
                        expected_condition_set
                        - observed
                    ),

                "extra":
                    sorted(
                        observed
                        - expected_condition_set
                    ),
            }
        )


if condition_failures:

    raise RuntimeError(
        "v1.1 condition matrix failure:\n"
        + json.dumps(
            condition_failures,
            indent=2,
        )
    )


print(
    "✓ Condition matrix: 20 × 13 = 260 PASS"
)


# ==========================================================================================
# 8. LOAD IMAGE SHAPES / VERIFY IMAGE BYTE IDENTITY
# ==========================================================================================

image_shape = {}

image_integrity_rows = []


for case_id in tqdm(
    cases,
    desc="Auditing frozen CT images",
    unit="volume",
):

    old_image = (
        OLD_CACHE
        / "images"
        / f"{case_id}.npz"
    )

    new_image = (
        NEW_CACHE
        / "images"
        / f"{case_id}.npz"
    )


    old_sha = sha256_file(
        old_image
    )

    new_sha = sha256_file(
        new_image
    )


    if old_sha != new_sha:

        raise RuntimeError(
            f"CT image cache changed: "
            f"{case_id}"
        )


    with np.load(
        new_image,
        allow_pickle=False,
    ) as data:

        shape = np.asarray(
            data[
                "ct_zyx"
            ].shape,
            dtype=np.int32,
        )


    image_shape[
        case_id
    ] = shape


    image_integrity_rows.append(
        {
            "case_id":
                case_id,

            "parent_sha256":
                old_sha,

            "v1_1_sha256":
                new_sha,

            "byte_identical":
                True,
        }
    )


image_integrity_df = pd.DataFrame(
    image_integrity_rows
)


print(
    "✓ CT images byte-identical to v1.0: 20/20"
)


# ==========================================================================================
# 9. REAUDIT ANNOTATION SEMANTICS / GROUPS / COORDINATES
# ==========================================================================================

final_rows = []

semantic_failures = []
group_failures = []
coordinate_failures = []


manifest_lookup = {
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
        new_manifest.iterrows()
    )
}


for _, row in tqdm(
    new_manifest.iterrows(),
    total=len(new_manifest),
    desc="Revalidating final annotations",
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


    ann = load_annotation(
        NEW_CACHE,
        case_id,
        condition,
    )


    declared_hash = str(
        row[
            "annotation_semantic_sha256"
        ]
    )


    if ann[
        "hash"
    ] != declared_hash:

        semantic_failures.append(
            (
                case_id,
                condition,
            )
        )


    direct_fg = {
        tuple(
            int(x)
            for x in coord
        )
        for coord in ann[
            "fg"
        ]
    }


    membership_support = {
        tuple(
            int(x)
            for x in coord
        )
        for coord in ann[
            "membership_coords"
        ]
    }


    if direct_fg != membership_support:

        group_failures.append(
            (
                case_id,
                condition,
                "direct_fg_membership_support_mismatch",
            )
        )


    final_groups = set(
        int(x)
        for x in np.unique(
            ann[
                "membership_groups"
            ]
        )
    )


    # Parent condition defines the required group set.
    parent_ann = load_annotation(
        OLD_CACHE,
        case_id,
        condition,
    )


    parent_groups = set(
        int(x)
        for x in np.unique(
            parent_ann[
                "membership_groups"
            ]
        )
    )


    if final_groups != parent_groups:

        group_failures.append(
            (
                case_id,
                condition,
                sorted(
                    parent_groups
                    - final_groups
                ),
                sorted(
                    final_groups
                    - parent_groups
                ),
            )
        )


    shape = image_shape[
        case_id
    ]


    inside = np.all(
        (
            ann[
                "coords"
            ]
            >= 0
        )
        & (
            ann[
                "coords"
            ]
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


    final_rows.append(
        {
            "case_id":
                case_id,

            "condition":
                condition,

            "foreground_voxels":
                int(
                    len(
                        ann[
                            "fg"
                        ]
                    )
                ),

            "background_voxels":
                int(
                    len(
                        ann[
                            "bg"
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
                    ann[
                        "bg"
                    ]
                ),

            "semantic_sha256":
                ann[
                    "hash"
                ],

            "construction_policy":
                str(
                    row[
                        "construction_policy"
                    ]
                ),
        }
    )


final_df = pd.DataFrame(
    final_rows
)


if semantic_failures:

    raise RuntimeError(
        f"Semantic hash failures: "
        f"{semantic_failures[:10]}"
    )


if group_failures:

    raise RuntimeError(
        f"Foreground-group contract failures: "
        f"{group_failures[:10]}"
    )


if coordinate_failures:

    raise RuntimeError(
        f"Out-of-crop coordinates: "
        f"{coordinate_failures[:10]}"
    )


print(
    "✓ Semantic hashes: PASS"
)

print(
    "✓ Foreground-group identity: PASS"
)

print(
    "✓ Every sparse coordinate remains inside cached CT: PASS"
)


# ==========================================================================================
# 10. VERIFY BACKGROUND EXACTLY UNCHANGED
# ==========================================================================================

heading(
    "STEP 6/10 — VERIFY BACKGROUND IMMUTABILITY"
)


background_pass = True


for case_id in cases:

    v1_hashes = set()

    v1_1_hashes = set()


    for condition in EXPECTED_CONDITIONS:

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


        v1_hashes.add(
            coordinate_hash(
                old_ann[
                    "bg"
                ]
            )
        )

        v1_1_hashes.add(
            coordinate_hash(
                new_ann[
                    "bg"
                ]
            )
        )


    if not (
        len(v1_hashes) == 1
        and len(v1_1_hashes) == 1
        and v1_hashes == v1_1_hashes
    ):

        background_pass = False

        break


if not background_pass:

    raise RuntimeError(
        "Background coordinates changed during v1.1 equalization."
    )


print(
    "✓ Explicit-background coordinates unchanged: PASS"
)


# ==========================================================================================
# 11. VERIFY COMPLETE / NATURAL CONDITIONS UNCHANGED
# ==========================================================================================

heading(
    "STEP 7/10 — VERIFY FROZEN SCIENTIFIC CONDITIONS"
)


complete_unchanged = True
natural_unchanged = True
nested_pass = True


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
                "hash"
            ]
            == new_ann[
                "hash"
            ]
        )


        if (
            condition
            == "complete"
            and not same
        ):

            complete_unchanged = False


        if (
            condition.startswith(
                "component_natural"
            )
            and not same
        ):

            natural_unchanged = False


    sets = {}

    for cov in COVERAGES:

        ann = load_annotation(
            NEW_CACHE,
            case_id,
            f"component_natural_{cov}",
        )

        sets[
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
        sets[
            25
        ].issubset(
            sets[
                50
            ]
        )
        and sets[
            50
        ].issubset(
            sets[
                75
            ]
        )
        and sets[
            75
        ].issubset(
            complete_groups
        )
    ):

        nested_pass = False


if not complete_unchanged:

    raise RuntimeError(
        "Component-complete condition changed."
    )


if not natural_unchanged:

    raise RuntimeError(
        "Natural component omission changed."
    )


if not nested_pass:

    raise RuntimeError(
        "Nested missingness contract changed."
    )


print(
    "✓ Component-complete supervision unchanged: PASS"
)

print(
    "✓ Natural component-omission supervision unchanged: PASS"
)

print(
    "✓ Nested 25% ⊂ 50% ⊂ 75% missingness: PASS"
)


# ==========================================================================================
# 12. FINAL CAUSAL EQUALITY
# ==========================================================================================

heading(
    "STEP 8/10 — VERIFY ACTUAL NETWORK-GRID CAUSAL BUDGETS"
)


pair_rows = []


natural_pixel_pass = 0
fixed_pair_pass = 0

primary50_np_pass = 0
primary50_fixed_pass = 0


for case_id in cases:

    case = (
        final_df[
            final_df[
                "case_id"
            ]
            == case_id
        ]
        .set_index(
            "condition"
        )
    )


    for cov in COVERAGES:

        n_fg = int(
            case.loc[
                f"component_natural_{cov}",
                "foreground_voxels",
            ]
        )

        p_fg = int(
            case.loc[
                f"pixel_dropout_matched_{cov}",
                "foreground_voxels",
            ]
        )

        f_fg = int(
            case.loc[
                f"component_fixed_{cov}",
                "foreground_voxels",
            ]
        )

        c_fg = int(
            case.loc[
                f"complete_fixed_{cov}",
                "foreground_voxels",
            ]
        )


        np_equal = (
            n_fg
            == p_fg
        )

        fixed_equal = (
            f_fg
            == c_fg
        )


        natural_pixel_pass += int(
            np_equal
        )

        fixed_pair_pass += int(
            fixed_equal
        )


        if cov == 50:

            primary50_np_pass += int(
                np_equal
            )

            primary50_fixed_pass += int(
                fixed_equal
            )


        pair_rows.append(
            {
                "case_id":
                    case_id,

                "coverage":
                    cov,

                "natural_unique_fg":
                    n_fg,

                "pixel_unique_fg":
                    p_fg,

                "natural_pixel_exact":
                    np_equal,

                "component_fixed_unique_fg":
                    f_fg,

                "complete_fixed_unique_fg":
                    c_fg,

                "fixed_pair_exact":
                    fixed_equal,
            }
        )


pair_df = pd.DataFrame(
    pair_rows
)


print(
    f"Natural-vs-pixel exact pairs         : "
    f"{natural_pixel_pass}/60"
)

print(
    f"Fixed component-vs-complete pairs   : "
    f"{fixed_pair_pass}/60"
)

print(
    f"PRIMARY 50% natural-vs-pixel        : "
    f"{primary50_np_pass}/20"
)

print(
    f"PRIMARY 50% fixed-budget pair       : "
    f"{primary50_fixed_pass}/20"
)


if natural_pixel_pass != 60:

    raise RuntimeError(
        "Final natural/pixel causal equality failed."
    )


if fixed_pair_pass != 60:

    raise RuntimeError(
        "Final fixed-budget causal equality failed."
    )


if not (
    primary50_np_pass == 20
    and primary50_fixed_pass == 20
):

    raise RuntimeError(
        "Primary 50% causal equality failed."
    )


# ==========================================================================================
# 13. RECONSTRUCT CORRECTION AUDIT FROM LINEAGE MANIFEST
# ==========================================================================================

heading(
    "STEP 9/10 — FREEZE v1.1 PROVENANCE / AUDIT"
)


correction_rows = []


for _, row in new_manifest.iterrows():

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


    old_fg = int(
        len(
            old_ann[
                "fg"
            ]
        )
    )

    new_fg = int(
        len(
            new_ann[
                "fg"
            ]
        )
    )


    changed = (
        old_ann[
            "hash"
        ]
        != new_ann[
            "hash"
        ]
    )


    correction_rows.append(
        {
            "case_id":
                case_id,

            "condition":
                condition,

            "construction_policy":
                str(
                    row[
                        "construction_policy"
                    ]
                ),

            "parent_unique_fg":
                old_fg,

            "final_unique_fg":
                new_fg,

            "equalization_target_unique_fg":
                int(
                    row[
                        "equalization_target_unique_fg"
                    ]
                ),

            "semantic_changed":
                changed,
        }
    )


correction_df = pd.DataFrame(
    correction_rows
)


pixel_controls_corrected = int(
    correction_df[
        (
            correction_df[
                "condition"
            ].str.startswith(
                "pixel_dropout_matched_"
            )
        )
        & (
            correction_df[
                "semantic_changed"
            ]
        )
    ].shape[
        0
    ]
)


fixed_sides_corrected = int(
    correction_df[
        (
            (
                correction_df[
                    "condition"
                ].str.startswith(
                    "component_fixed_"
                )
            )
            | (
                correction_df[
                    "condition"
                ].str.startswith(
                    "complete_fixed_"
                )
            )
        )
        & (
            correction_df[
                "semantic_changed"
            ]
        )
    ].shape[
        0
    ]
)


total_changed = int(
    correction_df[
        "semantic_changed"
    ].sum()
)


print(
    f"Pixel controls semantically corrected : "
    f"{pixel_controls_corrected}"
)

print(
    f"Fixed-pair sides corrected             : "
    f"{fixed_sides_corrected}"
)

print(
    f"Total corrected conditions             : "
    f"{total_changed}"
)


# ==========================================================================================
# 14. HASH FINAL CACHE
# ==========================================================================================

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
    desc="Hashing final v1.1 cache",
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
# 15. SAVE REPOSITORY AUDITS
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


for directory in [
    manifest_dir,
    audit_dir,
    figure_dir,
    archive_config_dir,
]:

    directory.mkdir(
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


image_integrity_df.to_csv(
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
# 16. ARCHIVE v1.0 PREPROCESS CONFIG / FREEZE v1.1 CONFIG
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
    require_coordinate_identity_with_parent_cache: true

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
# 17. FINAL PROTOCOL DOCUMENT
# ==========================================================================================

protocol = f"""
# CORA-Lung Training-Grid Preprocessing Protocol

## Accepted version

**Version:** 1.1

**Parent version:** 1.0

**Correction made before model training:** Yes

## Image preprocessing

The image pipeline is unchanged from preprocessing v1.0:

- canonical NIfTI orientation;
- target spacing 3.0 × 1.5 × 1.5 mm in `(z,y,x)`;
- linear CT interpolation;
- HU clipping to `[-1000, 400]`;
- normalization to `[-1, 1]`;
- CT-image-only thoracic crop.

All twenty v1.1 cached CT images are byte-for-byte identical to their
v1.0 counterparts.

## Reason for the v1.1 amendment

Sparse labels were originally matched on the native annotation grid.

After physical-coordinate transfer to the coarser model grid, nearby
same-class labelled voxels can collapse to the same unique model voxel.

This changed the effective causal annotation budget received by the network.

The raw v1.0 training cache therefore failed causal-budget equivalence despite
passing geometry, foreground-group survival, background identity and firewall QA.

## Feasibility audit

Block 06A was performed before any v1.1 equalization.

It established constructive feasibility for:

- 60/60 natural-component vs random-pixel pairs;
- 60/60 fixed-budget pairs;
- 20/20 primary 50% natural/pixel comparisons;
- 20/20 primary 50% fixed comparisons.

No dense masks were accessed.

## Natural component omission

Natural component-omission conditions are unchanged from v1.0.

Missing-component identities are unchanged.

The nested 25%, 50%, and 75% coverage hierarchy is unchanged.

## Random-pixel control

The target is the exact unique-FG count of the paired natural component-omission
annotation after transfer.

If the raw transferred control already satisfies that target it is unchanged.

Otherwise a deterministic group-preserving subset is selected from the
transferred component-complete unique foreground support.

## Fixed-budget pair

For every component-fixed / complete-fixed pair:

`target = min(unique FG capacity of the two transferred conditions)`

The side already at the target is unchanged.

Only an oversized side is thinned.

All foreground groups represented by that condition are preserved.

## Background

Explicit-background coordinates are never changed.

The v1.1 background coordinate set is verified identical to the corresponding
v1.0 background set for every case and condition.

## Final validation

- condition matrix: 260/260;
- natural/pixel exact causal pairs: {natural_pixel_pass}/60;
- fixed-budget exact causal pairs: {fixed_pair_pass}/60;
- primary 50% natural/pixel: {primary50_np_pass}/20;
- primary 50% fixed pair: {primary50_fixed_pass}/20;
- component-complete annotations unchanged: yes;
- natural component-omission annotations unchanged: yes;
- background coordinates unchanged: yes;
- foreground-group loss: none;
- dense masks accessed during v1.1 equalization: no;
- model training performed before correction: no.
"""


write_text(
    REPO
    / "docs/training_grid_preprocessing_protocol.md",
    protocol,
)


# ==========================================================================================
# 18. PUBLICATION FIGURE
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
            "label":
                (
                    f"{cov}%\n"
                    f"Natural vs Pixel"
                ),

            "raw":
                int(
                    (
                        ~raw_cov[
                            "natural_pixel_exact"
                        ].astype(bool)
                    ).sum()
                ),

            "final":
                int(
                    (
                        ~final_cov[
                            "natural_pixel_exact"
                        ].astype(bool)
                    ).sum()
                ),
        }
    )


    figure_rows.append(
        {
            "label":
                (
                    f"{cov}%\n"
                    f"Fixed Pair"
                ),

            "raw":
                int(
                    (
                        ~raw_cov[
                            "fixed_pair_exact"
                        ].astype(bool)
                    ).sum()
                ),

            "final":
                int(
                    (
                        ~final_cov[
                            "fixed_pair_exact"
                        ].astype(bool)
                    ).sum()
                ),
        }
    )


figure_df = pd.DataFrame(
    figure_rows
)


x = np.arange(
    len(
        figure_df
    )
)

width = 0.38


plt.rcParams.update(
    {
        "font.size":
            11,

        "axes.titlesize":
            14,

        "axes.titleweight":
            "bold",

        "axes.labelsize":
            12,

        "axes.labelweight":
            "bold",

        "xtick.labelsize":
            9,

        "ytick.labelsize":
            10,

        "legend.fontsize":
            10,

        "savefig.dpi":
            600,
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
        "raw"
    ],
    width,
    label=(
        "Raw Training-Grid "
        "Cache v1.0"
    ),
)


ax.bar(
    x
    + width / 2,
    figure_df[
        "final"
    ],
    width,
    label=(
        "Accepted Causally "
        "Equalized Cache v1.1"
    ),
)


ax.set_xticks(
    x
)


ax.set_xticklabels(
    figure_df[
        "label"
    ],
    fontweight="bold",
)


ax.set_ylabel(
    "Case-Level Pairs with Unequal Unique-FG Budgets",
    fontweight="bold",
)


ax.set_xlabel(
    "Coverage and Causal-Control Comparison",
    fontweight="bold",
)


ax.set_title(
    "Restoring Exact Causal Annotation-Budget Equality on the Actual CORA-Lung Training Grid\n"
    "Post-Transfer Equalization Is Performed Before Any Model Training",
    fontweight="bold",
)


ax.grid(
    axis="y",
    alpha=0.25,
)


legend = ax.legend()


for text in legend.get_texts():

    text.set_fontweight(
        "bold"
    )


fig.tight_layout()


fig.savefig(
    REPO
    / "figures/audit/fig08_training_grid_causal_equalization.png",
    dpi=600,
    bbox_inches="tight",
)


fig.savefig(
    REPO
    / "figures/audit/fig08_training_grid_causal_equalization.pdf",
    bbox_inches="tight",
)


plt.close(
    fig
)


print(
    "✓ Publication-quality equalization figure generated."
)


# ==========================================================================================
# 19. FINAL BLOCK-06 AUDIT
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
        PREPROCESS_VERSION,

    "parent_cache_status":
        "REVIEW_REQUIRED_PRIMARY_CAUSAL_BUDGET",

    "cache_v1_1": {
        "cases":
            20,

        "conditions_per_case":
            13,

        "annotations":
            260,

        "cache_files":
            int(
                len(
                    cache_hash_df
                )
            ),

        "cache_bytes":
            cache_bytes,

        "cache_human_size":
            human_bytes(
                cache_bytes
            ),

        "ct_images_byte_identical_to_parent":
            True,
    },

    "correction": {
        "pixel_controls_corrected":
            pixel_controls_corrected,

        "fixed_pair_sides_corrected":
            fixed_sides_corrected,

        "total_semantically_changed_conditions":
            total_changed,

        "source_weak_labels_modified":
            False,

        "component_complete_modified":
            False,

        "natural_component_omission_modified":
            False,

        "background_modified":
            False,

        "dense_masks_accessed":
            False,

        "model_training_before_correction":
            False,
    },

    "causal_contract": {
        "natural_pixel_exact":
            f"{natural_pixel_pass}/60",

        "fixed_pair_exact":
            f"{fixed_pair_pass}/60",

        "primary_50_natural_pixel":
            f"{primary50_np_pass}/20",

        "primary_50_fixed":
            f"{primary50_fixed_pass}/20",
    },

    "qa": {
        "condition_matrix":
            "PASS",

        "semantic_hashes":
            "PASS",

        "foreground_groups":
            "PASS",

        "coordinate_bounds":
            "PASS",

        "background_identity":
            "PASS",

        "complete_unchanged":
            "PASS",

        "natural_omission_unchanged":
            "PASS",

        "nested_missingness":
            "PASS",

        "training_cache_firewall":
            "PASS",

        "scientific_unit_tests":
            "PASS",
    },

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
# 20. CAPTURE ORIGINAL 06B + REPAIR SOURCE
# ==========================================================================================

source_06b_capture = "NOT_AVAILABLE"
source_repair_capture = "NOT_AVAILABLE"


try:

    ip = get_ipython()

    history = (
        ip.history_manager
        .input_hist_raw
    )


    for cell in reversed(
        history
    ):

        if (
            "CORA-LUNG — CODE BLOCK 06B"
            in cell
            and "CODE BLOCK 06B-A"
            not in cell
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
                cell,
                encoding="utf-8",
            )

            source_06b_capture = "PASS"

            break


    current_cell = (
        history[
            -1
        ]
    )


    if (
        "CORA-LUNG — CODE BLOCK 06B-A"
        in current_cell
    ):

        (
            REPO
            / "scripts/code_blocks/block06b_a_resume_after_module_cache.py"
        ).write_text(
            current_cell,
            encoding="utf-8",
        )

        source_repair_capture = "PASS"


except Exception:
    pass


# ==========================================================================================
# 21. UPDATE PROJECT STATE
# ==========================================================================================

state_path = (
    REPO
    / "PROJECT_STATE.json"
)


state = json.loads(
    state_path.read_text(
        encoding="utf-8"
    )
)


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
            "1.1",

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
                "Audit Block 06 final report. "
                "Then build the Gate-B pilot training harness "
                "on the four permanent development volumes."
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
# 22. REFRESH REPOSITORY MANIFEST — EXCLUDE SELF
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
            and p != repo_manifest_path
        )
    ],
    key=lambda p:
        str(
            p.relative_to(
                REPO
            )
        ),
)


repo_rows = []


for path in tqdm(
    repo_files,
    desc="Refreshing repository manifest",
):

    repo_rows.append(
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
            "06B-A",

        "self_included":
            False,

        "files":
            repo_rows,
    },
)


# ==========================================================================================
# 23. COMMIT + PUSH
# ==========================================================================================

heading(
    "STEP 10/10 — COMMIT ACCEPTED TRAINING-CACHE v1.1"
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
    "/tmp/cora_git_askpass_block06ba.sh"
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
        f"Tracked changes: "
        f"{len(changes)}"
    )

    for line in changes[
        :50
    ]:

        print(
            " ",
            line,
        )

    if len(changes) > 50:

        print(
            f"  ... +"
            f"{len(changes)-50} more"
        )


    commit_message = (
        "fix: finalize causally matched training cache v1.1"
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


if push.returncode != 0:

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
# 24. FINAL REPORT
# ==========================================================================================

print("\n")

print(
    "=" * 118
)

print(
    "CORA-LUNG CODE BLOCK 06B-A — FINAL TRAINING-CACHE v1.1 REPORT"
)

print(
    "=" * 118
)


print(f"""
ROOT CAUSE OF 06B INTERRUPT
---------------------------
v1.1 firewall source written to disk     : YES
Python reused previously imported v1.0   : YES
Fresh module reload performed            : YES
Cache regenerated                         : NO
Equalization rerun                        : NO

CACHE
-----
Accepted preprocessing version            : 1.1
CT volumes                                : 20
Conditions per CT                         : 13
Final sparse annotations                  : {len(new_manifest)}
Final cache files                         : {len(cache_hash_df)}
Final cache size                          : {human_bytes(cache_bytes)}

CT IMAGE INTEGRITY
------------------
Images modified by correction             : NO
Images byte-identical to v1.0             : PASS
Image-derived crop changed                : NO

FROZEN SCIENTIFIC CONDITIONS
----------------------------
Component-complete changed                : NO
Natural component omission changed        : NO
Missing component identities changed      : NO
Nested 25% / 50% / 75% missingness        : PASS
Explicit background changed               : NO
Dense masks accessed                      : NO

POST-TRANSFER EQUALIZATION
--------------------------
Pixel controls semantically corrected     : {pixel_controls_corrected}
Fixed-pair sides semantically corrected   : {fixed_sides_corrected}
Total semantically changed conditions     : {total_changed}

ACTUAL NETWORK-GRID CAUSAL BUDGET
---------------------------------
Natural-vs-pixel exact                    : {natural_pixel_pass}/60
Fixed component-vs-complete exact         : {fixed_pair_pass}/60

PRIMARY 50% GATE-B INPUT
------------------------
Natural-vs-pixel exact                    : {primary50_np_pass}/20
Fixed component-vs-complete exact         : {primary50_fixed_pass}/20

GROUP / GEOMETRY / BACKGROUND
-----------------------------
Foreground group preservation             : PASS
All sparse coordinates inside CT crop     : PASS
Background identical to parent cache      : PASS
Condition matrix                          : PASS (20 × 13)

FIREWALL / QA
-------------
Fresh runtime v1.1 firewall               : PASS
Fresh-subprocess unit tests               : PASS
Semantic hashes                           : PASS
Dense lesion information in cache         : NO
Unknown voxels converted to background    : NO

REPRODUCIBILITY
---------------
v1.0 review cache preserved locally       : YES
v1.0 preprocess config archived           : YES
v1.1 lineage manifest committed           : YES
v1.1 cache hashes committed               : YES
Equalization action audit committed       : YES
Protocol amendment committed              : YES
Original 06B source captured              : {source_06b_capture}
06B-A repair source captured              : {source_repair_capture}
Large training cache committed to GitHub  : NO
Cache reconstructible                     : YES

SCIENTIFIC STATUS
-----------------
Gate A                                    : PASS
Block 05                                  : PASS
Block 06                                  : PASS
Gate B                                    : NOT RUN
Model training                            : NOT STARTED
Training authorized                       : NO

GITHUB
------
Starting commit                           : {starting_commit[:12]}
Current commit                            : {current_commit[:12]}
Synchronization                           : PASS

NEXT
----
Send me this COMPLETE report.

If the two causal families both read 60/60 and the primary 50% comparisons
both read 20/20, Block 06 is scientifically closed.

The next block will build the Gate-B PILOT HARNESS only.

It will not yet run the full experiment.
""")


print(
    "=" * 118
)