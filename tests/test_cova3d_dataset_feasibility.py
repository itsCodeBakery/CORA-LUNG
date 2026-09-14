import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_primary_structural_factorial_feasibility():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_primary_factorial_structural_feasibility_v1_0.csv"
    )

    assert len(frame) == 20

    assert (
        frame[
            "eligible_components"
        ]
        >= 2
    ).all()

    assert (
        frame[
            "selected_components_50"
        ]
        >= 1
    ).all()

    assert (
        frame[
            "selected_components_50"
        ]
        <= frame[
            "selected_components_100"
        ]
    ).all()

    assert frame[
        "primary_factorial_structural_eligible"
    ].astype(bool).all()


def test_primary_roles_remain_frozen():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_primary_factorial_structural_feasibility_v1_0.csv"
    )

    counts = frame[
        "role"
    ].value_counts().to_dict()

    assert counts[
        "permanent_development"
    ] == 4

    assert counts[
        "final_outer_cv"
    ] == 16

    role_count = (
        frame.groupby(
            "source_subject_key"
        )[
            "role"
        ]
        .nunique()
    )

    assert int(
        role_count.max()
    ) == 1


def test_secondary_dataset_is_candidate_only():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_dataset_register_v1_0.csv"
    )

    secondary = frame[
        frame[
            "dataset_id"
        ]
        == "MSD_Task03_Liver"
    ]

    assert len(
        secondary
    ) == 1

    row = secondary.iloc[
        0
    ]

    assert row[
        "structural_factorial_status"
    ] == "NOT_YET_AUDITED"

    assert not bool(
        row[
            "factorial_training_authorized"
        ]
    )


def test_training_remains_unauthorized():

    state = json.loads(
        (
            ROOT
            / "COVA3D_STATE.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert state[
        "factorial_training_authorized"
    ] is False

    assert state[
        "method_development_authorized"
    ] is False

    assert state[
        "final_outer_cv_outcomes_authorized"
    ] is False

    assert state[
        "optimizer_steps_in_cova3d"
    ] == 0
