"""Firewall for CORA-Lung model-fitting cache."""

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
    "source_annotation_semantic_sha256",
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


def _forbidden(text):
    text = str(text).lower()

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

    if set(manifest.columns) != ALLOWED_MANIFEST_COLUMNS:
        raise RuntimeError(
            "Unexpected model-fitting manifest schema."
        )

    for col in manifest.columns:
        _forbidden(col)

    for _, row in manifest.iterrows():
        image_path = (
            root
            / row["image_file"]
        )

        ann_path = (
            root
            / row["annotation_file"]
        )

        _forbidden(
            row["image_file"]
        )

        _forbidden(
            row["annotation_file"]
        )

        with np.load(
            image_path,
            allow_pickle=False,
        ) as data:
            if set(
                data.files
            ) != ALLOWED_IMAGE_KEYS:
                raise RuntimeError(
                    f"Unsafe image cache keys: {image_path}"
                )

            ct = np.asarray(
                data["ct_zyx"]
            )

            if ct.ndim != 3:
                raise RuntimeError(
                    "Training CT must be 3-D."
                )

        with np.load(
            ann_path,
            allow_pickle=False,
        ) as data:
            if set(
                data.files
            ) != ALLOWED_ANNOTATION_KEYS:
                raise RuntimeError(
                    f"Unsafe annotation cache keys: {ann_path}"
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

            if coords.shape != (
                len(labels),
                3,
            ):
                raise RuntimeError(
                    "Invalid supervision coordinate shape."
                )

            if fg_coords.shape != (
                len(fg_groups),
                3,
            ):
                raise RuntimeError(
                    "Invalid FG membership coordinate shape."
                )

            if not set(
                np.unique(
                    labels
                )
            ).issubset(
                {0, 1}
            ):
                raise RuntimeError(
                    "Training labels must be sparse binary labels."
                )

            if np.any(
                fg_groups
                <= 0
            ):
                raise RuntimeError(
                    "Replay FG group IDs must be positive."
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
