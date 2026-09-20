from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


EXPECTED_TRAINER_KEYS = {
    "supervision_voxel_zyx",
    "supervision_label",
    "fg_membership_voxel_zyx",
    "fg_membership_group_id",
}


def test_final_coordinate_level_budget():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_training_grid_budget_v1_2.csv"
    )

    assert len(frame) == 20

    assert frame[
        "coordinate_level_six_cell_feasible"
    ].astype(bool).all()

    assert not frame[
        "factorial_training_authorized"
    ].astype(bool).any()

    assert (
        frame[
            "frozen_train_budget_B_i"
        ]
        >= frame[
            "aggregate_lower_bound"
        ]
    ).all()

    assert (
        frame[
            "frozen_train_budget_B_i"
        ]
        <= frame[
            "aggregate_upper_bound"
        ]
    ).all()


def test_final_six_cell_matching():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_training_grid_condition_manifest_v1_2.csv"
    )

    assert len(frame) == 120

    for _, group in frame.groupby(
        "case_id"
    ):

        assert len(group) == 6

        assert (
            group[
                "train_positive_budget_B_i"
            ].nunique()
            == 1
        )

        assert (
            group[
                "foreground_unique_voxels"
            ].nunique()
            == 1
        )

        assert (
            group[
                "background_semantic_sha256"
            ].nunique()
            == 1
        )

        for coverage in [
            0.5,
            1.0,
        ]:

            subset = group[
                np.isclose(
                    group[
                        "coverage_fraction"
                    ],
                    coverage,
                )
            ]

            assert (
                subset[
                    "selected_component_ids"
                ].nunique()
                == 1
            )

            assert (
                subset[
                    "selected_component_quotas"
                ].nunique()
                == 1
            )


def test_neutral_bank_manifest():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_neutral_anchor_bank_manifest_v1_0.csv"
    )

    assert len(frame) == 28

    assert frame[
        "case_id"
    ].nunique() == 13

    assert (
        frame[
            "capacity"
        ]
        >= 2
    ).all()

    assert frame[
        "shared_across_geometry"
    ].astype(bool).all()

    assert frame[
        "shared_across_coverage"
    ].astype(bool).all()

    assert not frame[
        "geometry_claim_allowed"
    ].astype(bool).any()


def test_resolution_class_rules():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_training_grid_component_audit_v1_2.csv"
    )

    limited = frame[
        frame[
            "global_resolution_class"
        ]
        == "GLOBAL_RESOLUTION_LIMITED"
    ]

    assert (
        limited[
            "quota"
        ]
        >= 1
    ).all()

    assert not limited[
        "geometry_claim_allowed"
    ].astype(bool).any()

    assert (
        limited[
            "representation"
        ]
        == "neutral_physical_bank_prefix"
    ).all()

    resolvable = frame[
        frame[
            "global_resolution_class"
        ]
        == "GLOBAL_GEOMETRY_RESOLVABLE"
    ]

    assert (
        resolvable[
            "quota"
        ]
        >= 2
    ).all()

    coherent = resolvable[
        resolvable[
            "geometry_condition"
        ]
        == "coherent"
    ]

    dispersed = resolvable[
        resolvable[
            "geometry_condition"
        ]
        == "dispersed"
    ]

    fragmented = resolvable[
        resolvable[
            "geometry_condition"
        ]
        == "fragmented"
    ]

    assert (
        coherent[
            "connected_sets"
        ]
        == 1
    ).all()

    assert (
        dispersed[
            "connected_sets"
        ]
        >= 2
    ).all()

    assert (
        fragmented[
            "connected_sets"
        ]
        >= 2
    ).all()


def test_sparse_trainer_artifacts():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_training_grid_candidate_artifact_manifest_v1_2.csv"
    )

    assert len(frame) == 120

    for _, row in frame.iterrows():

        path = ROOT / row[
            "artifact_file"
        ]

        with np.load(
            path,
            allow_pickle=False,
        ) as data:

            assert set(
                data.files
            ) == EXPECTED_TRAINER_KEYS

            labels = np.asarray(
                data[
                    "supervision_label"
                ]
            )

            coords = np.asarray(
                data[
                    "supervision_voxel_zyx"
                ]
            )

            assert set(
                np.unique(
                    labels
                ).tolist()
            ).issubset(
                {
                    0,
                    1,
                }
            )

            assert len(
                np.unique(
                    coords,
                    axis=0,
                )
            ) == len(
                coords
            )
