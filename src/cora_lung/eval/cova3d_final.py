from __future__ import annotations

from pathlib import Path

import numpy as np
import nibabel as nib

from nibabel.processing import resample_to_output
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment


CONNECTIVITY_26 = ndi.generate_binary_structure(3, 3)

VOXEL_THRESHOLDS = np.asarray(
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

REFERENCE_MINIMUM_VOLUME_ML = 0.10
PRIMARY_MATCH_IOU = 0.10
SECONDARY_MATCH_IOU = 0.25
FIXED_SECONDARY_THRESHOLD = 0.50
FP_BUDGETS = (0.5, 1.0, 2.0, 4.0)


def label_components(mask):

    labels, count = ndi.label(
        np.asarray(mask, dtype=bool),
        structure=CONNECTIVITY_26,
    )

    return (
        labels.astype(np.int32, copy=False),
        int(count),
    )


def dice_score(prediction, reference):

    p = np.asarray(prediction, dtype=bool)
    r = np.asarray(reference, dtype=bool)

    inter = int(np.logical_and(p, r).sum())
    denom = int(p.sum()) + int(r.sum())

    if denom == 0:
        return 1.0

    return float(2.0 * inter / denom)


def iou_score(prediction, reference):

    p = np.asarray(prediction, dtype=bool)
    r = np.asarray(reference, dtype=bool)

    inter = int(np.logical_and(p, r).sum())
    union = int(np.logical_or(p, r).sum())

    if union == 0:
        return 1.0

    return float(inter / union)


def reference_partition(
    reference,
    *,
    voxel_volume_ml,
    minimum_volume_ml=REFERENCE_MINIMUM_VOLUME_ML,
):

    labels, count = label_components(reference)

    sizes = np.bincount(
        labels.reshape(-1),
        minlength=count + 1,
    ).astype(np.int64)

    ids = np.arange(
        1,
        count + 1,
        dtype=np.int32,
    )

    volumes = (
        sizes[1:].astype(np.float64)
        * float(voxel_volume_ml)
    )

    eligible_mask = volumes >= float(minimum_volume_ml)

    eligible_ids = ids[eligible_mask]
    small_ids = ids[~eligible_mask]

    return {
        "labels": labels,
        "sizes": sizes,
        "component_ids": ids,
        "volumes_ml": volumes,
        "eligible_ids": eligible_ids,
        "small_ids": small_ids,
        "eligible_count": int(len(eligible_ids)),
        "small_count": int(len(small_ids)),
    }


def prediction_partition(
    probability,
    *,
    threshold,
):

    probability = np.asarray(
        probability,
        dtype=np.float32,
    )

    binary = probability >= float(threshold)

    labels, count = label_components(binary)

    sizes = np.bincount(
        labels.reshape(-1),
        minlength=count + 1,
    ).astype(np.int64)

    return {
        "binary": binary,
        "labels": labels,
        "sizes": sizes,
        "count": int(count),
    }


def component_iou_matrix(
    reference_labels,
    reference_sizes,
    reference_ids,
    prediction_labels,
    prediction_sizes,
):

    reference_ids = np.asarray(
        reference_ids,
        dtype=np.int32,
    )

    n_prediction = int(
        len(prediction_sizes) - 1
    )

    matrix = np.zeros(
        (len(reference_ids), n_prediction),
        dtype=np.float64,
    )

    if len(reference_ids) == 0 or n_prediction == 0:
        return matrix

    lookup = {
        int(component_id): row
        for row, component_id
        in enumerate(reference_ids.tolist())
    }

    overlap = (
        (reference_labels > 0)
        &
        (prediction_labels > 0)
    )

    if not overlap.any():
        return matrix

    ref_values = reference_labels[overlap].astype(
        np.int64,
        copy=False,
    )

    pred_values = prediction_labels[overlap].astype(
        np.int64,
        copy=False,
    )

    code = (
        ref_values
        * (n_prediction + 1)
        + pred_values
    )

    unique_code, intersections = np.unique(
        code,
        return_counts=True,
    )

    for item, intersection in zip(
        unique_code.tolist(),
        intersections.tolist(),
    ):

        ref_id = int(
            item // (n_prediction + 1)
        )

        pred_id = int(
            item % (n_prediction + 1)
        )

        row = lookup.get(ref_id)

        if row is None or pred_id <= 0:
            continue

        union = (
            int(reference_sizes[ref_id])
            + int(prediction_sizes[pred_id])
            - int(intersection)
        )

        if union <= 0:
            continue

        matrix[row, pred_id - 1] = (
            float(intersection)
            / float(union)
        )

    return matrix


def maximum_cardinality_then_iou(
    iou_matrix,
    *,
    minimum_iou,
):

    matrix = np.asarray(
        iou_matrix,
        dtype=np.float64,
    )

    n_ref, n_pred = matrix.shape

    if n_ref == 0 or n_pred == 0:
        return []

    if float(minimum_iou) <= 0.0:
        admissible = matrix > 0.0
    else:
        admissible = matrix >= float(minimum_iou)

    if not admissible.any():
        return []

    # Larger than maximum possible total IoU contribution.
    bonus = min(n_ref, n_pred) + 1.0

    reward = np.where(
        admissible,
        bonus + matrix,
        0.0,
    )

    rows, cols = linear_sum_assignment(
        -reward
    )

    output = []

    for row, col in zip(
        rows.tolist(),
        cols.tolist(),
    ):

        if not admissible[row, col]:
            continue

        output.append(
            (
                int(row),
                int(col),
                float(matrix[row, col]),
            )
        )

    return output


def evaluate_case_threshold(
    probability,
    reference,
    *,
    voxel_volume_ml,
    threshold,
    minimum_reference_volume_ml=REFERENCE_MINIMUM_VOLUME_ML,
    minimum_iou=PRIMARY_MATCH_IOU,
):

    probability = np.asarray(
        probability,
        dtype=np.float32,
    )

    reference = np.asarray(
        reference,
        dtype=bool,
    )

    if probability.shape != reference.shape:
        raise ValueError(
            "Prediction/reference shape mismatch."
        )

    ref = reference_partition(
        reference,
        voxel_volume_ml=voxel_volume_ml,
        minimum_volume_ml=minimum_reference_volume_ml,
    )

    pred = prediction_partition(
        probability,
        threshold=threshold,
    )

    eligible_iou = component_iou_matrix(
        ref["labels"],
        ref["sizes"],
        ref["eligible_ids"],
        pred["labels"],
        pred["sizes"],
    )

    small_iou = component_iou_matrix(
        ref["labels"],
        ref["sizes"],
        ref["small_ids"],
        pred["labels"],
        pred["sizes"],
    )

    matches = maximum_cardinality_then_iou(
        eligible_iou,
        minimum_iou=minimum_iou,
    )

    matched_prediction_ids = {
        int(item[1])
        for item in matches
    }

    unmatched = [
        idx
        for idx in range(pred["count"])
        if idx not in matched_prediction_ids
    ]

    neutral_small = 0

    for pred_idx in unmatched:

        if small_iou.shape[0] == 0:
            continue

        if float(
            np.max(
                small_iou[:, pred_idx]
            )
        ) >= float(minimum_iou):

            neutral_small += 1

    false_positives = (
        len(unmatched)
        - neutral_small
    )

    eligible_count = int(
        ref["eligible_count"]
    )

    tp = int(
        len(matches)
    )

    recall = (
        float(tp / eligible_count)
        if eligible_count
        else 0.0
    )

    return {
        "voxel_threshold": float(threshold),
        "prediction_components": int(pred["count"]),
        "eligible_references": eligible_count,
        "small_references": int(ref["small_count"]),
        "true_positives": tp,
        "false_positives": int(false_positives),
        "neutral_small_predictions": int(neutral_small),
        "recall": recall,
    }


def froc_curve(
    case_records,
    *,
    voxel_thresholds=VOXEL_THRESHOLDS,
    minimum_reference_volume_ml=REFERENCE_MINIMUM_VOLUME_ML,
    minimum_iou=PRIMARY_MATCH_IOU,
):

    if len(case_records) == 0:
        raise ValueError(
            "case_records must not be empty."
        )

    rows = []

    for threshold in np.asarray(
        voxel_thresholds,
        dtype=np.float64,
    ).tolist():

        case_results = {}

        recalls = []
        fp_total = 0
        tp_total = 0
        eligible_total = 0

        for record in case_records:

            case_id = str(
                record["case_id"]
            )

            result = evaluate_case_threshold(
                record["probability"],
                record["reference"],
                voxel_volume_ml=float(
                    record["voxel_volume_ml"]
                ),
                threshold=float(threshold),
                minimum_reference_volume_ml=minimum_reference_volume_ml,
                minimum_iou=minimum_iou,
            )

            case_results[case_id] = result

            recalls.append(
                float(result["recall"])
            )

            fp_total += int(
                result["false_positives"]
            )

            tp_total += int(
                result["true_positives"]
            )

            eligible_total += int(
                result["eligible_references"]
            )

        macro_recall = float(
            np.mean(recalls)
        )

        pooled_recall = (
            float(tp_total / eligible_total)
            if eligible_total
            else 0.0
        )

        fp_per_patient = float(
            fp_total / len(case_records)
        )

        rows.append(
            {
                "voxel_threshold": float(threshold),
                "fp_total": int(fp_total),
                "fp_per_patient": fp_per_patient,
                "true_positives": int(tp_total),
                "eligible_references": int(eligible_total),
                "macro_recall": macro_recall,
                "pooled_recall": pooled_recall,
                "case_results": case_results,
            }
        )

    return rows


def select_operating_point(
    curve,
    *,
    fp_budget,
):

    feasible = [
        row
        for row in curve
        if float(row["fp_per_patient"])
        <= float(fp_budget) + 1e-12
    ]

    if not feasible:
        raise RuntimeError(
            "No feasible FROC operating point."
        )

    def key(row):

        return (
            float(row["macro_recall"]),
            float(row["pooled_recall"]),
            -float(row["fp_per_patient"]),
            float(row["voxel_threshold"]),
        )

    return max(
        feasible,
        key=key,
    )


def fixed_threshold_metrics(
    probability,
    reference,
    *,
    voxel_volume_ml,
    probability_threshold=FIXED_SECONDARY_THRESHOLD,
    minimum_reference_volume_ml=REFERENCE_MINIMUM_VOLUME_ML,
):

    probability = np.asarray(
        probability,
        dtype=np.float32,
    )

    reference = np.asarray(
        reference,
        dtype=bool,
    )

    if probability.shape != reference.shape:
        raise ValueError(
            "Prediction/reference shape mismatch."
        )

    ref = reference_partition(
        reference,
        voxel_volume_ml=voxel_volume_ml,
        minimum_volume_ml=minimum_reference_volume_ml,
    )

    pred = prediction_partition(
        probability,
        threshold=probability_threshold,
    )

    eligible_iou = component_iou_matrix(
        ref["labels"],
        ref["sizes"],
        ref["eligible_ids"],
        pred["labels"],
        pred["sizes"],
    )

    any_matches = maximum_cardinality_then_iou(
        eligible_iou,
        minimum_iou=0.0,
    )

    iou025_matches = maximum_cardinality_then_iou(
        eligible_iou,
        minimum_iou=SECONDARY_MATCH_IOU,
    )

    eligible_count = int(
        ref["eligible_count"]
    )

    any_overlap_recall = (
        float(len(any_matches) / eligible_count)
        if eligible_count
        else 0.0
    )

    iou025_recall = (
        float(len(iou025_matches) / eligible_count)
        if eligible_count
        else 0.0
    )

    binary = pred["binary"]

    false_positive_voxels = int(
        np.logical_and(
            binary,
            np.logical_not(reference),
        ).sum()
    )

    if eligible_count:

        overlapping_components_per_reference = (
            eligible_iou > 0.0
        ).sum(axis=1)

        fragmentation_ratio = float(
            np.mean(
                overlapping_components_per_reference
            )
        )

    else:

        fragmentation_ratio = float("nan")

    return {
        "probability_threshold":
            float(probability_threshold),

        "dice":
            dice_score(
                binary,
                reference,
            ),

        "iou":
            iou_score(
                binary,
                reference,
            ),

        "any_overlap_lesion_recall":
            any_overlap_recall,

        "iou_0p25_lesion_recall":
            iou025_recall,

        "false_positive_volume_ml":
            float(
                false_positive_voxels
                * float(voxel_volume_ml)
            ),

        "prediction_component_count":
            int(pred["count"]),

        "component_to_reference_fragmentation_ratio":
            fragmentation_ratio,

        "eligible_reference_count":
            eligible_count,
    }


def reference_lesion_volumes_ml(
    reference,
    *,
    voxel_volume_ml,
    minimum_reference_volume_ml=REFERENCE_MINIMUM_VOLUME_ML,
):

    partition = reference_partition(
        reference,
        voxel_volume_ml=voxel_volume_ml,
        minimum_volume_ml=minimum_reference_volume_ml,
    )

    component_ids = partition[
        "component_ids"
    ]

    volumes = partition[
        "volumes_ml"
    ]

    lookup = {
        int(component_id): float(volume)
        for component_id, volume
        in zip(
            component_ids.tolist(),
            volumes.tolist(),
        )
    }

    return np.asarray(
        [
            lookup[int(component_id)]
            for component_id
            in partition["eligible_ids"].tolist()
        ],
        dtype=np.float64,
    )


def pooled_volume_tertiles(
    eligible_volumes_ml,
):

    values = np.asarray(
        eligible_volumes_ml,
        dtype=np.float64,
    )

    if values.ndim != 1 or len(values) < 3:
        raise ValueError(
            "At least three eligible lesions are required."
        )

    q1, q2 = np.quantile(
        values,
        [1.0 / 3.0, 2.0 / 3.0],
    )

    return (
        float(q1),
        float(q2),
    )


def resample_reference_to_training_crop(
    mask_path,
    *,
    crop_origin_zyx,
    crop_shape_zyx,
    expected_full_shape_zyx=None,
):

    image = nib.load(
        str(Path(mask_path)),
        mmap=True,
    )

    canonical = nib.as_closest_canonical(
        image
    )

    resampled = resample_to_output(
        canonical,
        voxel_sizes=(
            1.5,
            1.5,
            3.0,
        ),
        order=0,
        cval=0,
    )

    xyz = (
        np.asarray(
            resampled.dataobj
        )
        > 0
    )

    zyx = np.transpose(
        xyz,
        (2, 1, 0),
    )

    if expected_full_shape_zyx is not None:

        expected = tuple(
            int(v)
            for v in expected_full_shape_zyx
        )

        if tuple(zyx.shape) != expected:

            raise RuntimeError(
                "Resampled reference full-grid shape mismatch. "
                f"Expected {expected}, observed {tuple(zyx.shape)}."
            )

    oz, oy, ox = [
        int(v)
        for v in crop_origin_zyx
    ]

    sz, sy, sx = [
        int(v)
        for v in crop_shape_zyx
    ]

    cropped = zyx[
        oz:oz + sz,
        oy:oy + sy,
        ox:ox + sx,
    ]

    expected_crop = (
        sz,
        sy,
        sx,
    )

    if tuple(cropped.shape) != expected_crop:

        raise RuntimeError(
            "Reference crop shape mismatch."
        )

    return cropped.astype(
        bool,
        copy=False,
    )
