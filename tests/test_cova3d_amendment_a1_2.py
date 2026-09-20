from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_A1_2_is_frozen():

    config = yaml.safe_load(
        (
            ROOT
            / "configs/"
            "cova3d_resolution_aware_amendment_A1_2.yaml"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        config[
            "status"
        ]
        == "PROSPECTIVELY_FROZEN_BEFORE_COVA_TRAINING"
    )

    assert (
        config[
            "GLOBAL_RESOLUTION_LIMITED"
        ][
            "minimum_quota"
        ]
        == 1
    )

    assert (
        config[
            "GLOBAL_RESOLUTION_LIMITED"
        ][
            "geometry_manipulation"
        ]
        is False
    )

    assert (
        config[
            "GLOBAL_RESOLUTION_LIMITED"
        ][
            "cross_coverage_prefix_property"
        ]
        is True
    )

    assert (
        config[
            "factorial_training_authorized"
        ]
        is False
    )


def test_neutral_bank_capacity_manifest():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_resolution_limited_neutral_bank_capacity_v1_0.csv"
    )

    assert len(
        frame
    ) == 28

    assert frame[
        "case_id"
    ].nunique() == 13

    assert (
        frame[
            "mapped_physical_support_capacity"
        ]
        >= 2
    ).all()


def test_neutral_bank_case_diagnostic():

    frame = pd.read_csv(
        ROOT
        / "experiments/audits/"
        "block09d_b_neutral_bank_diagnostic/"
        "case_neutral_bank_feasibility.csv"
    )

    assert len(
        frame
    ) == 20

    assert frame[
        "neutral_bank_aggregate_feasible"
    ].astype(
        bool
    ).all()

    special = frame[
        frame[
            "case_id"
        ]
        == "radiopaedia_29_86490_1"
    ].iloc[
        0
    ]

    assert int(
        special[
            "common_minimum"
        ]
    ) == 2

    assert int(
        special[
            "common_maximum"
        ]
    ) == 20
