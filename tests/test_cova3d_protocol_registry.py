import json
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_cova3d_factorial_registry_is_complete():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_factorial_register_v1_0.csv"
    )

    assert len(frame) == 6

    assert set(
        frame[
            "coverage_percent"
        ].astype(int)
    ) == {
        50,
        100,
    }

    assert set(
        frame[
            "geometry"
        ]
    ) == {
        "coherent",
        "dispersed",
        "fragmented",
    }

    assert not frame[
        "training_authorized"
    ].astype(bool).any()


def test_cova3d_equal_budget_contract_is_frozen():

    config = yaml.safe_load(
        (
            ROOT
            / "configs/"
            "cova3d_protocol_v1_0.yaml"
        ).read_text(
            encoding="utf-8"
        )
    )

    matching = config[
        "causal_matching"
    ]

    assert (
        matching[
            "within_case_total_positive_budget"
        ]
        == "EXACTLY_EQUAL_ACROSS_ALL_SIX_CELLS"
    )

    assert (
        matching[
            "same_selected_component_set_within_coverage_across_geometries"
        ]
        is True
    )

    assert (
        matching[
            "same_per_component_positive_quota_within_coverage_across_geometries"
        ]
        is True
    )

    assert (
        matching[
            "same_background_coordinates_across_all_six_cells"
        ]
        is True
    )


def test_cora_no_go_is_preserved():

    state = json.loads(
        (
            ROOT
            / "PROJECT_STATE.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert state[
        "gate_b"
    ] == "NO_GO"

    assert state[
        "gate_c"
    ] == "NOT_RUN"
