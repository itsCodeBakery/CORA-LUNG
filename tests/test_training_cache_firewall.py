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
