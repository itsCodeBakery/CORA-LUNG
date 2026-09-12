import json
import numpy as np
import pandas as pd

from cora_lung.data.firewall import (
    validate_annotation_file,
    validate_geometry_file,
)


def test_safe_sparse_annotation(tmp_path):
    path = tmp_path / "safe.npz"

    np.savez_compressed(
        path,
        voxel_ijk=np.asarray(
            [[1, 2, 3], [4, 5, 6]],
            dtype=np.int32,
        ),
        world_xyz_mm=np.asarray(
            [[1., 2., 3.], [4., 5., 6.]],
            dtype=np.float32,
        ),
        label=np.asarray(
            [1, 0],
            dtype=np.int8,
        ),
        group_id=np.asarray(
            [1, -1],
            dtype=np.int32,
        ),
    )

    assert validate_annotation_file(path)


def test_unknown_voxels_are_not_materialized(tmp_path):
    path = tmp_path / "safe.npz"

    np.savez_compressed(
        path,
        voxel_ijk=np.asarray(
            [[2, 2, 2]],
            dtype=np.int32,
        ),
        world_xyz_mm=np.asarray(
            [[2., 2., 2.]],
            dtype=np.float32,
        ),
        label=np.asarray(
            [1],
            dtype=np.int8,
        ),
        group_id=np.asarray(
            [3],
            dtype=np.int32,
        ),
    )

    with np.load(path, allow_pickle=False) as data:
        assert "unknown_mask" not in data.files
        assert "dense_mask" not in data.files


def test_geometry_rejects_dense_mask_field(tmp_path):
    path = tmp_path / "geometry.json"

    payload = {
        "case_id": "synthetic",
        "infection_mask": "forbidden.nii",
    }

    path.write_text(
        json.dumps(payload),
        encoding="utf-8",
    )

    failed = False

    try:
        validate_geometry_file(path)
    except RuntimeError:
        failed = True

    assert failed
