from pathlib import Path

import numpy as np
import nibabel as nib

from cora_lung.eval.cova3d_final import (
    VOXEL_THRESHOLDS,
    FIXED_SECONDARY_THRESHOLD,
    label_components,
    reference_partition,
    prediction_partition,
    maximum_cardinality_then_iou,
    evaluate_case_threshold,
    select_operating_point,
    fixed_threshold_metrics,
    pooled_volume_tertiles,
    resample_reference_to_training_crop,
)


def test_threshold_grid_is_exact():

    expected = np.asarray(
        [
            0.05,
            0.10,
            0.15,
            0.20,
            0.25,
            0.30,
            0.35,
            0.40,
            0.45,
            0.50,
            0.55,
            0.60,
            0.65,
            0.70,
            0.75,
            0.80,
            0.85,
            0.90,
            0.95,
            0.975,
            0.99,
            0.995,
            1.000001,
        ],
        dtype=np.float64,
    )

    assert np.array_equal(
        VOXEL_THRESHOLDS,
        expected,
    )

    assert FIXED_SECONDARY_THRESHOLD == 0.50


def test_reference_volume_eligibility_boundary():

    reference = np.zeros(
        (1, 1, 20),
        dtype=bool,
    )

    # Exactly 0.10 mL:
    # 5 voxels × 0.02 mL = 0.10 mL.
    reference[0, 0, 0:5] = True

    # Small reference:
    # 3 voxels × 0.02 = 0.06 mL.
    reference[0, 0, 10:13] = True

    part = reference_partition(
        reference,
        voxel_volume_ml=0.02,
        minimum_volume_ml=0.10,
    )

    assert part["eligible_count"] == 1
    assert part["small_count"] == 1


def test_matching_is_maximum_cardinality_before_total_iou():

    matrix = np.asarray(
        [
            [0.90, 0.20],
            [0.80, 0.00],
        ],
        dtype=np.float64,
    )

    matches = maximum_cardinality_then_iou(
        matrix,
        minimum_iou=0.10,
    )

    # Greedy highest-IoU matching would incorrectly select only 0.90.
    # Correct lexicographic matching obtains two matches:
    # R0-P1 and R1-P0.
    assert len(matches) == 2

    pairs = {
        (row, col)
        for row, col, _
        in matches
    }

    assert pairs == {
        (0, 1),
        (1, 0),
    }


def test_small_reference_prediction_is_neutral_not_fp():

    reference = np.zeros(
        (1, 1, 30),
        dtype=bool,
    )

    # Eligible lesion: 5 × 0.02 = 0.10 mL
    reference[0, 0, 0:5] = True

    # Small lesion: 3 × 0.02 = 0.06 mL
    reference[0, 0, 15:18] = True

    probability = np.zeros(
        reference.shape,
        dtype=np.float32,
    )

    probability[0, 0, 0:5] = 0.90
    probability[0, 0, 15:18] = 0.90

    result = evaluate_case_threshold(
        probability,
        reference,
        voxel_volume_ml=0.02,
        threshold=0.50,
        minimum_iou=0.10,
    )

    assert result["eligible_references"] == 1
    assert result["small_references"] == 1
    assert result["true_positives"] == 1
    assert result["neutral_small_predictions"] == 1
    assert result["false_positives"] == 0
    assert result["recall"] == 1.0


def test_prediction_components_are_recomputed_at_every_threshold():

    probability = np.zeros(
        (1, 3, 9),
        dtype=np.float32,
    )

    # Two high-confidence regions.
    probability[0, 1, 1:3] = 0.90
    probability[0, 1, 6:8] = 0.90

    # Medium-confidence bridge.
    probability[0, 1, 3:6] = 0.60

    low = prediction_partition(
        probability,
        threshold=0.50,
    )

    high = prediction_partition(
        probability,
        threshold=0.70,
    )

    # Bridge exists at 0.50 => one component.
    assert low["count"] == 1

    # Bridge disappears at 0.70 => two independently relabelled components.
    assert high["count"] == 2


def test_froc_operating_point_selection_tiebreak():

    curve = [
        {
            "voxel_threshold": 0.40,
            "fp_per_patient": 1.0,
            "macro_recall": 0.80,
            "pooled_recall": 0.80,
        },
        {
            "voxel_threshold": 0.60,
            "fp_per_patient": 0.5,
            "macro_recall": 0.80,
            "pooled_recall": 0.80,
        },
        {
            "voxel_threshold": 0.70,
            "fp_per_patient": 0.5,
            "macro_recall": 0.80,
            "pooled_recall": 0.80,
        },
        {
            "voxel_threshold": 0.30,
            "fp_per_patient": 1.5,
            "macro_recall": 0.95,
            "pooled_recall": 0.95,
        },
    ]

    selected = select_operating_point(
        curve,
        fp_budget=1.0,
    )

    # 0.30 is infeasible.
    # 0.60 beats 0.40 because of fewer FPs.
    # 0.70 beats 0.60 by the final higher-threshold tiebreak.
    assert selected["voxel_threshold"] == 0.70


def test_fixed_threshold_metrics_perfect_prediction():

    reference = np.zeros(
        (1, 4, 8),
        dtype=bool,
    )

    reference[0, 1:3, 2:7] = True

    probability = np.where(
        reference,
        0.90,
        0.10,
    ).astype(np.float32)

    metrics = fixed_threshold_metrics(
        probability,
        reference,
        voxel_volume_ml=0.02,
    )

    assert metrics["dice"] == 1.0
    assert metrics["iou"] == 1.0
    assert metrics["any_overlap_lesion_recall"] == 1.0
    assert metrics["iou_0p25_lesion_recall"] == 1.0
    assert metrics["false_positive_volume_ml"] == 0.0
    assert metrics["prediction_component_count"] == 1
    assert metrics["component_to_reference_fragmentation_ratio"] == 1.0


def test_fragmentation_ratio_detects_split_prediction():

    reference = np.zeros(
        (1, 3, 12),
        dtype=bool,
    )

    reference[0, 1, 1:11] = True

    probability = np.zeros(
        reference.shape,
        dtype=np.float32,
    )

    probability[0, 1, 1:4] = 0.90
    probability[0, 1, 8:11] = 0.90

    metrics = fixed_threshold_metrics(
        probability,
        reference,
        voxel_volume_ml=0.02,
    )

    assert metrics["prediction_component_count"] == 2

    assert (
        metrics[
            "component_to_reference_fragmentation_ratio"
        ]
        == 2.0
    )


def test_volume_tertiles_are_prediction_independent_numeric_rule():

    values = np.asarray(
        [0.10, 0.20, 0.30, 0.40, 0.50, 0.60],
        dtype=np.float64,
    )

    q1, q2 = pooled_volume_tertiles(
        values
    )

    assert np.isfinite(q1)
    assert np.isfinite(q2)
    assert q1 < q2


def test_reference_reconstruction_xyz_to_zyx_and_crop(tmp_path):

    # Native/synthetic file already uses the frozen target spacing:
    # X=1.5 mm, Y=1.5 mm, Z=3.0 mm.
    xyz = np.zeros(
        (6, 7, 8),
        dtype=np.uint8,
    )

    xyz[
        2:5,
        2:6,
        2:7,
    ] = 1

    affine = np.diag(
        [
            1.5,
            1.5,
            3.0,
            1.0,
        ]
    )

    path = (
        Path(tmp_path)
        / "synthetic_reference.nii.gz"
    )

    nib.save(
        nib.Nifti1Image(
            xyz,
            affine,
        ),
        str(path),
    )

    observed = resample_reference_to_training_crop(
        path,
        crop_origin_zyx=(1, 1, 1),
        crop_shape_zyx=(6, 5, 4),
        expected_full_shape_zyx=(8, 7, 6),
    )

    expected_full = np.transpose(
        xyz > 0,
        (2, 1, 0),
    )

    expected = expected_full[
        1:7,
        1:6,
        1:5,
    ]

    assert observed.shape == (6, 5, 4)

    assert np.array_equal(
        observed,
        expected,
    )
