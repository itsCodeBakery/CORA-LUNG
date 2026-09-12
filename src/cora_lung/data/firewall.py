"""Dense-label firewall for CORA-Lung trainer-safe weak supervision."""

from pathlib import Path
import json
import numpy as np
import pandas as pd


ALLOWED_NPZ_KEYS = {
    "voxel_ijk",
    "world_xyz_mm",
    "label",
    "group_id",
}

ALLOWED_MANIFEST_COLUMNS = {
    "case_id",
    "source_subject_key",
    "split_role",
    "outer_fold",
    "condition",
    "requested_coverage",
    "ct_scan",
    "geometry_file",
    "annotation_file",
    "foreground_labelled_voxels",
    "background_labelled_voxels",
    "generator_version",
    "generator_seed",
    "semantic_sha256",
}

FORBIDDEN_TOKENS = {
    "infection_mask",
    "lung_mask",
    "lesion_mask",
    "dense_mask",
    "dense_component",
    "hidden_component",
    "component_centroid",
    "component_size",
    "component_volume",
    "full_mask_distance",
}


def _assert_no_forbidden_text(value):
    text = str(value).lower()

    for token in FORBIDDEN_TOKENS:
        if token in text:
            raise RuntimeError(
                f"Forbidden dense-supervision token detected: {token}"
            )


def validate_manifest(manifest_path):
    manifest_path = Path(manifest_path)

    df = pd.read_csv(manifest_path)

    actual = set(df.columns)

    if actual != ALLOWED_MANIFEST_COLUMNS:
        extra = actual - ALLOWED_MANIFEST_COLUMNS
        missing = ALLOWED_MANIFEST_COLUMNS - actual

        raise RuntimeError(
            f"Unsafe manifest schema. Extra={extra}, Missing={missing}"
        )

    for column in df.columns:
        _assert_no_forbidden_text(column)

    for column in ["ct_scan", "geometry_file", "annotation_file"]:
        for value in df[column].dropna():
            _assert_no_forbidden_text(value)

    return df


def validate_annotation_file(path):
    path = Path(path)

    with np.load(path, allow_pickle=False) as data:
        keys = set(data.files)

        if keys != ALLOWED_NPZ_KEYS:
            raise RuntimeError(
                f"Unsafe annotation keys in {path.name}: {keys}"
            )

        voxel_ijk = np.asarray(data["voxel_ijk"])
        world_xyz_mm = np.asarray(data["world_xyz_mm"])
        label = np.asarray(data["label"])
        group_id = np.asarray(data["group_id"])

    n = len(label)

    if voxel_ijk.shape != (n, 3):
        raise RuntimeError("Invalid voxel_ijk shape.")

    if world_xyz_mm.shape != (n, 3):
        raise RuntimeError("Invalid world_xyz_mm shape.")

    if group_id.shape != (n,):
        raise RuntimeError("Invalid group_id shape.")

    if not set(np.unique(label)).issubset({0, 1}):
        raise RuntimeError("Sparse labels must be only 0 or 1.")

    fg = label == 1
    bg = label == 0

    if np.any(group_id[fg] <= 0):
        raise RuntimeError(
            "Every foreground labelled voxel must have positive group ID."
        )

    if np.any(group_id[bg] != -1):
        raise RuntimeError(
            "Every background labelled voxel must use group_id=-1."
        )

    coord_label = {}

    for coord, y in zip(voxel_ijk, label):
        key = tuple(int(x) for x in coord)

        if key in coord_label and coord_label[key] != int(y):
            raise RuntimeError(
                "Conflicting foreground/background sparse coordinate."
            )

        coord_label[key] = int(y)

    return True


def validate_geometry_file(path):
    path = Path(path)

    payload = json.loads(
        path.read_text(encoding="utf-8")
    )

    serialized = json.dumps(payload).lower()

    for token in FORBIDDEN_TOKENS:
        if token in serialized:
            raise RuntimeError(
                f"Forbidden dense field in geometry file: {token}"
            )

    required = {
        "case_id",
        "source_subject_key",
        "split_role",
        "outer_fold",
        "ct_scan",
        "native_shape",
        "native_spacing_mm",
        "native_affine",
        "coordinate_convention",
        "generator_version",
        "generator_seed",
    }

    if set(payload.keys()) != required:
        raise RuntimeError(
            "Unexpected trainer geometry schema."
        )

    return True


def validate_weak_export(root):
    root = Path(root)

    manifest = validate_manifest(
        root / "manifest.csv"
    )

    for _, row in manifest.iterrows():
        validate_annotation_file(
            root / row["annotation_file"]
        )

        validate_geometry_file(
            root / row["geometry_file"]
        )

    return {
        "manifest_rows": int(len(manifest)),
        "annotation_files": int(
            manifest["annotation_file"].nunique()
        ),
        "case_count": int(
            manifest["case_id"].nunique()
        ),
    }
