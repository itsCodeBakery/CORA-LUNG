"""Frozen Gate-B component and FROC evaluator for CORA-Lung."""

from __future__ import annotations

import numpy as np

from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment


CONNECTIVITY_26 = ndi.generate_binary_structure(
    3,
    3,
)


def dice_score(
    prediction,
    reference,
):
    prediction = np.asarray(
        prediction,
        dtype=bool,
    )

    reference = np.asarray(
        reference,
        dtype=bool,
    )

    intersection = int(
        np.logical_and(
            prediction,
            reference,
        ).sum()
    )

    denominator = (
        int(
            prediction.sum()
        )
        + int(
            reference.sum()
        )
    )

    if denominator == 0:
        return 1.0

    return (
        2.0
        * intersection
        / denominator
    )


def iou_score(
    prediction,
    reference,
):
    prediction = np.asarray(
        prediction,
        dtype=bool,
    )

    reference = np.asarray(
        reference,
        dtype=bool,
    )

    intersection = int(
        np.logical_and(
            prediction,
            reference,
        ).sum()
    )

    union = int(
        np.logical_or(
            prediction,
            reference,
        ).sum()
    )

    if union == 0:
        return 1.0

    return intersection / union


def label_binary_components(
    mask,
):
    return ndi.label(
        np.asarray(
            mask,
            dtype=bool,
        ),
        structure=CONNECTIVITY_26,
    )


def reference_component_partition(
    reference,
    *,
    voxel_volume_ml,
    minimum_volume_ml,
):
    labels, count = label_binary_components(
        reference
    )

    sizes = np.bincount(
        labels.reshape(
            -1
        ),
        minlength=count + 1,
    ).astype(
        np.int64
    )

    ids = np.arange(
        1,
        count + 1,
        dtype=np.int32,
    )

    volumes = (
        sizes[
            1:
        ].astype(
            np.float64
        )
        * float(
            voxel_volume_ml
        )
    )

    eligible = ids[
        volumes
        >= float(
            minimum_volume_ml
        )
    ]

    small = ids[
        volumes
        < float(
            minimum_volume_ml
        )
    ]

    return {
        "labels":
            labels.astype(
                np.int32,
                copy=False,
            ),

        "sizes":
            sizes,

        "eligible_ids":
            eligible.astype(
                np.int32
            ),

        "small_ids":
            small.astype(
                np.int32
            ),

        "eligible_count":
            int(
                len(
                    eligible
                )
            ),

        "small_count":
            int(
                len(
                    small
                )
            ),
    }


def prediction_component_partition(
    probability,
    *,
    base_threshold,
):
    probability = np.asarray(
        probability,
        dtype=np.float32,
    )

    binary = (
        probability
        >= float(
            base_threshold
        )
    )

    labels, count = label_binary_components(
        binary
    )

    labels = labels.astype(
        np.int32,
        copy=False,
    )

    sizes = np.bincount(
        labels.reshape(
            -1
        ),
        minlength=count + 1,
    ).astype(
        np.int64
    )

    if count == 0:

        scores = np.empty(
            0,
            dtype=np.float32,
        )

    else:

        scores = np.asarray(
            ndi.mean(
                probability,
                labels=labels,
                index=np.arange(
                    1,
                    count + 1,
                ),
            ),
            dtype=np.float32,
        )

    return {
        "binary":
            binary,

        "labels":
            labels,

        "sizes":
            sizes,

        "scores":
            scores,

        "count":
            int(
                count
            ),
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
        len(
            prediction_sizes
        )
        - 1
    )

    matrix = np.zeros(
        (
            len(
                reference_ids
            ),
            n_prediction,
        ),
        dtype=np.float32,
    )

    if (
        len(
            reference_ids
        )
        == 0
        or n_prediction
        == 0
    ):

        return matrix

    row_lookup = {
        int(
            component_id
        ):
            row_index
        for row_index, component_id
        in enumerate(
            reference_ids.tolist()
        )
    }

    overlap_mask = (
        (
            reference_labels
            > 0
        )
        & (
            prediction_labels
            > 0
        )
    )

    if not bool(
        overlap_mask.any()
    ):

        return matrix

    reference_overlap = (
        reference_labels[
            overlap_mask
        ].astype(
            np.int64,
            copy=False,
        )
    )

    prediction_overlap = (
        prediction_labels[
            overlap_mask
        ].astype(
            np.int64,
            copy=False,
        )
    )

    pair_code = (
        reference_overlap
        * (
            n_prediction
            + 1
        )
        + prediction_overlap
    )

    unique_codes, intersections = np.unique(
        pair_code,
        return_counts=True,
    )

    for code, intersection in zip(
        unique_codes.tolist(),
        intersections.tolist(),
    ):

        reference_id = int(
            code
            // (
                n_prediction
                + 1
            )
        )

        prediction_id = int(
            code
            % (
                n_prediction
                + 1
            )
        )

        row = row_lookup.get(
            reference_id
        )

        if row is None:
            continue

        if prediction_id <= 0:
            continue

        union = (
            int(
                reference_sizes[
                    reference_id
                ]
            )
            + int(
                prediction_sizes[
                    prediction_id
                ]
            )
            - int(
                intersection
            )
        )

        if union <= 0:
            continue

        matrix[
            row,
            prediction_id - 1,
        ] = (
            float(
                intersection
            )
            / float(
                union
            )
        )

    return matrix


def maximum_cardinality_iou_matching(
    iou_matrix,
    *,
    minimum_iou,
):
    iou_matrix = np.asarray(
        iou_matrix,
        dtype=np.float64,
    )

    n_reference, n_prediction = (
        iou_matrix.shape
    )

    if (
        n_reference == 0
        or n_prediction == 0
    ):

        return []


    if float(
        minimum_iou
    ) <= 0.0:

        admissible = (
            iou_matrix
            > 0.0
        )

    else:

        admissible = (
            iou_matrix
            >= float(
                minimum_iou
            )
        )


    if not bool(
        admissible.any()
    ):

        return []


    # A cardinality bonus larger than the maximum possible total IoU
    # ensures lexicographic optimization:
    #
    #   1. maximum number of admissible matches
    #   2. maximum total IoU among those matchings
    cardinality_bonus = (
        min(
            n_reference,
            n_prediction,
        )
        + 1.0
    )


    reward = np.where(
        admissible,
        cardinality_bonus
        + iou_matrix,
        0.0,
    )


    rows, columns = linear_sum_assignment(
        -reward
    )


    output = []


    for row, column in zip(
        rows.tolist(),
        columns.tolist(),
    ):

        if not admissible[
            row,
            column
        ]:
            continue

        output.append(
            (
                int(
                    row
                ),
                int(
                    column
                ),
                float(
                    iou_matrix[
                        row,
                        column
                    ]
                ),
            )
        )


    return output


def evaluate_selected_candidates(
    *,
    candidate_scores,
    eligible_iou,
    small_iou,
    score_threshold,
    minimum_iou,
):
    candidate_scores = np.asarray(
        candidate_scores,
        dtype=np.float32,
    )

    selected = np.flatnonzero(
        candidate_scores
        >= float(
            score_threshold
        )
    )


    selected_eligible_iou = (
        eligible_iou[
            :,
            selected,
        ]
        if len(
            selected
        )
        else eligible_iou[
            :,
            :0,
        ]
    )


    matches = maximum_cardinality_iou_matching(
        selected_eligible_iou,
        minimum_iou=minimum_iou,
    )


    matched_local_predictions = {
        int(
            item[
                1
            ]
        )
        for item
        in matches
    }


    matched_global_predictions = {
        int(
            selected[
                local_index
            ]
        )
        for local_index
        in matched_local_predictions
    }


    unmatched_global_predictions = [
        int(
            prediction_index
        )
        for prediction_index
        in selected.tolist()
        if int(
            prediction_index
        )
        not in matched_global_predictions
    ]


    neutral_small = 0


    for prediction_index in unmatched_global_predictions:

        if small_iou.shape[
            0
        ] == 0:

            continue


        max_small_iou = float(
            np.max(
                small_iou[
                    :,
                    prediction_index
                ]
            )
        )


        if max_small_iou >= float(
            minimum_iou
        ):

            neutral_small += 1


    false_positives = (
        len(
            unmatched_global_predictions
        )
        - neutral_small
    )


    eligible_count = int(
        eligible_iou.shape[
            0
        ]
    )


    true_positives = int(
        len(
            matches
        )
    )


    recall = (
        true_positives
        / eligible_count
        if eligible_count
        else 0.0
    )


    return {
        "selected_candidates":
            int(
                len(
                    selected
                )
            ),

        "true_positives":
            true_positives,

        "false_positives":
            int(
                false_positives
            ),

        "neutral_small_reference_predictions":
            int(
                neutral_small
            ),

        "eligible_references":
            eligible_count,

        "recall":
            float(
                recall
            ),
    }


def cohort_froc_curve(
    case_data,
    *,
    minimum_iou,
    maximum_fp_per_patient,
):
    case_ids = sorted(
        case_data.keys()
    )


    all_scores = []


    for case_id in case_ids:

        scores = np.asarray(
            case_data[
                case_id
            ][
                "candidate_scores"
            ],
            dtype=np.float32,
        )

        if len(
            scores
        ):

            all_scores.append(
                scores
            )


    if all_scores:

        unique_scores = np.unique(
            np.concatenate(
                all_scores
            )
        )

        unique_scores = np.sort(
            unique_scores
        )[
            ::-1
        ]

        thresholds = np.concatenate(
            [
                np.asarray(
                    [
                        np.inf
                    ],
                    dtype=np.float64,
                ),
                unique_scores.astype(
                    np.float64
                ),
            ]
        )

    else:

        thresholds = np.asarray(
            [
                np.inf
            ],
            dtype=np.float64,
        )


    rows = []

    previous_fp_total = -1


    for threshold in thresholds.tolist():

        case_results = {}

        fp_total = 0

        tp_total = 0

        eligible_total = 0

        recalls = []


        for case_id in case_ids:

            data = case_data[
                case_id
            ]


            result = evaluate_selected_candidates(
                candidate_scores=data[
                    "candidate_scores"
                ],
                eligible_iou=data[
                    "eligible_iou"
                ],
                small_iou=data[
                    "small_iou"
                ],
                score_threshold=threshold,
                minimum_iou=minimum_iou,
            )


            case_results[
                case_id
            ] = result


            fp_total += int(
                result[
                    "false_positives"
                ]
            )

            tp_total += int(
                result[
                    "true_positives"
                ]
            )

            eligible_total += int(
                result[
                    "eligible_references"
                ]
            )

            recalls.append(
                float(
                    result[
                        "recall"
                    ]
                )
            )


        fp_per_patient = (
            fp_total
            / len(
                case_ids
            )
        )


        macro_recall = float(
            np.mean(
                recalls
            )
        )


        pooled_recall = (
            tp_total
            / eligible_total
            if eligible_total
            else 0.0
        )


        row = {
            "score_threshold":
                float(
                    threshold
                ),

            "fp_total":
                int(
                    fp_total
                ),

            "fp_per_patient":
                float(
                    fp_per_patient
                ),

            "true_positives":
                int(
                    tp_total
                ),

            "eligible_references":
                int(
                    eligible_total
                ),

            "macro_recall":
                macro_recall,

            "pooled_recall":
                float(
                    pooled_recall
                ),

            "case_results":
                case_results,
        }


        rows.append(
            row
        )


        # FP must not decrease as the candidate-score threshold falls.
        # If monotonicity holds and we have moved beyond the largest
        # prespecified FP budget, no lower threshold can become feasible again.
        if (
            previous_fp_total >= 0
            and fp_total
            < previous_fp_total
        ):

            raise RuntimeError(
                "FROC false-positive count became non-monotonic."
            )


        previous_fp_total = (
            fp_total
        )


        if fp_per_patient > float(
            maximum_fp_per_patient
        ):

            break


    return rows


def select_froc_operating_point(
    curve,
    *,
    fp_budget,
):
    feasible = [
        row
        for row in curve
        if float(
            row[
                "fp_per_patient"
            ]
        )
        <= float(
            fp_budget
        )
        + 1e-12
    ]


    if not feasible:

        raise RuntimeError(
            "No feasible FROC operating point."
        )


    def key(
        row,
    ):

        threshold = float(
            row[
                "score_threshold"
            ]
        )

        threshold_tiebreak = (
            threshold
            if np.isfinite(
                threshold
            )
            else float(
                "inf"
            )
        )

        return (
            float(
                row[
                    "macro_recall"
                ]
            ),
            float(
                row[
                    "pooled_recall"
                ]
            ),
            -float(
                row[
                    "fp_per_patient"
                ]
            ),
            threshold_tiebreak,
        )


    return max(
        feasible,
        key=key,
    )


def fixed_threshold_metrics(
    *,
    probability,
    reference,
    voxel_volume_ml,
    minimum_reference_volume_ml,
    base_probability_threshold,
    primary_match_iou,
    secondary_match_iou,
):
    reference = np.asarray(
        reference,
        dtype=bool,
    )

    probability = np.asarray(
        probability,
        dtype=np.float32,
    )


    reference_partition = reference_component_partition(
        reference,
        voxel_volume_ml=voxel_volume_ml,
        minimum_volume_ml=minimum_reference_volume_ml,
    )


    prediction_partition = prediction_component_partition(
        probability,
        base_threshold=base_probability_threshold,
    )


    eligible_iou = component_iou_matrix(
        reference_partition[
            "labels"
        ],
        reference_partition[
            "sizes"
        ],
        reference_partition[
            "eligible_ids"
        ],
        prediction_partition[
            "labels"
        ],
        prediction_partition[
            "sizes"
        ],
    )


    small_iou = component_iou_matrix(
        reference_partition[
            "labels"
        ],
        reference_partition[
            "sizes"
        ],
        reference_partition[
            "small_ids"
        ],
        prediction_partition[
            "labels"
        ],
        prediction_partition[
            "sizes"
        ],
    )


    all_selected_primary = evaluate_selected_candidates(
        candidate_scores=prediction_partition[
            "scores"
        ],
        eligible_iou=eligible_iou,
        small_iou=small_iou,
        score_threshold=-np.inf,
        minimum_iou=primary_match_iou,
    )


    all_selected_secondary = evaluate_selected_candidates(
        candidate_scores=prediction_partition[
            "scores"
        ],
        eligible_iou=eligible_iou,
        small_iou=small_iou,
        score_threshold=-np.inf,
        minimum_iou=secondary_match_iou,
    )


    # "Any overlap" means one-to-one matching with strictly positive IoU.
    any_overlap_matches = maximum_cardinality_iou_matching(
        eligible_iou,
        minimum_iou=0.0,
    )


    eligible_count = int(
        reference_partition[
            "eligible_count"
        ]
    )


    any_overlap_recall = (
        len(
            any_overlap_matches
        )
        / eligible_count
        if eligible_count
        else 0.0
    )


    binary_prediction = prediction_partition[
        "binary"
    ]


    false_positive_voxels = int(
        np.logical_and(
            binary_prediction,
            np.logical_not(
                reference
            ),
        ).sum()
    )


    return {
        "dice":
            float(
                dice_score(
                    binary_prediction,
                    reference,
                )
            ),

        "iou":
            float(
                iou_score(
                    binary_prediction,
                    reference,
                )
            ),

        "predicted_components":
            int(
                prediction_partition[
                    "count"
                ]
            ),

        "eligible_reference_components":
            eligible_count,

        "small_reference_components":
            int(
                reference_partition[
                    "small_count"
                ]
            ),

        "fp_components_at_0p5":
            int(
                all_selected_primary[
                    "false_positives"
                ]
            ),

        "fp_volume_ml_at_0p5":
            float(
                false_positive_voxels
                * float(
                    voxel_volume_ml
                )
            ),

        "primary_iou_component_recall_at_0p5":
            float(
                all_selected_primary[
                    "recall"
                ]
            ),

        "iou_0p25_component_recall":
            float(
                all_selected_secondary[
                    "recall"
                ]
            ),

        "any_overlap_component_recall":
            float(
                any_overlap_recall
            ),

        "candidate_scores":
            prediction_partition[
                "scores"
            ],

        "prediction_sizes":
            prediction_partition[
                "sizes"
            ][
                1:
            ],

        "eligible_iou":
            eligible_iou,

        "small_iou":
            small_iou,
    }
