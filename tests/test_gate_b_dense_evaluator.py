import numpy as np

from cora_lung.eval.gate_b import (
    maximum_cardinality_iou_matching,
    evaluate_selected_candidates,
    select_froc_operating_point,
)

from cora_lung.eval.inference import (
    sliding_window_starts,
)


def test_matching_prioritizes_cardinality_before_iou():

    # Greedy IoU would take 0.90 and leave only one match.
    # Correct lexicographic matching takes:
    #   ref0 -> pred1 (0.11)
    #   ref1 -> pred0 (0.10)
    # giving cardinality 2.
    matrix = np.asarray(
        [
            [0.90, 0.11],
            [0.10, 0.00],
        ],
        dtype=np.float32,
    )

    matches = maximum_cardinality_iou_matching(
        matrix,
        minimum_iou=0.10,
    )

    assert len(matches) == 2


def test_small_reference_prediction_is_not_false_positive():

    result = evaluate_selected_candidates(
        candidate_scores=np.asarray(
            [0.8],
            dtype=np.float32,
        ),
        eligible_iou=np.asarray(
            [
                [0.0],
            ],
            dtype=np.float32,
        ),
        small_iou=np.asarray(
            [
                [0.30],
            ],
            dtype=np.float32,
        ),
        score_threshold=0.5,
        minimum_iou=0.10,
    )

    assert result[
        "true_positives"
    ] == 0

    assert result[
        "false_positives"
    ] == 0

    assert result[
        "neutral_small_reference_predictions"
    ] == 1


def test_operating_point_respects_fp_budget_and_tie_break():

    curve = [
        {
            "score_threshold": 0.9,
            "fp_per_patient": 0.5,
            "macro_recall": 0.4,
            "pooled_recall": 0.4,
        },
        {
            "score_threshold": 0.8,
            "fp_per_patient": 1.0,
            "macro_recall": 0.6,
            "pooled_recall": 0.6,
        },
        {
            "score_threshold": 0.7,
            "fp_per_patient": 2.0,
            "macro_recall": 0.8,
            "pooled_recall": 0.8,
        },
    ]

    result = select_froc_operating_point(
        curve,
        fp_budget=1.0,
    )

    assert result[
        "score_threshold"
    ] == 0.8


def test_sliding_window_starts_cover_endpoint():

    starts = sliding_window_starts(
        101,
        48,
        0.5,
    )

    assert starts[
        0
    ] == 0

    assert starts[
        -1
    ] == (
        101
        - 48
    )
