from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_six_cells_per_case_and_equal_native_budget():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_native_condition_manifest_v1_0.csv"
    )

    assert frame[
        "case_id"
    ].nunique() == 20

    assert len(frame) == 120

    for _, group in frame.groupby(
        "case_id"
    ):

        assert len(group) == 6

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

        assert not group[
            "trainer_eligible"
        ].astype(bool).any()


def test_geometry_component_sets_and_quotas_are_paired():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_native_condition_manifest_v1_0.csv"
    )

    for _, case_group in frame.groupby(
        "case_id"
    ):

        for coverage in [
            0.5,
            1.0,
        ]:

            subset = case_group[
                np.isclose(
                    case_group[
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


def test_native_artifacts_have_no_dense_arrays():

    manifest = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_native_candidate_artifact_manifest_v1_0.csv"
    )

    forbidden_tokens = {
        "infection",
        "lesion_mask",
        "lung_mask",
        "dense_mask",
        "reference_mask",
    }

    expected_keys = {
        "case_id",
        "condition_id",
        "simulator_version",
        "coverage_fraction",
        "geometry",
        "trainer_eligible",
        "native_affine_xyz",
        "supervision_voxel_xyz",
        "supervision_world_xyz",
        "supervision_label",
        "foreground_voxel_xyz",
        "foreground_component_id",
        "background_voxel_xyz",
        "selected_component_ids",
        "selected_component_quotas",
    }

    for _, row in manifest.iterrows():

        path = ROOT / row[
            "artifact_file"
        ]

        assert path.exists()

        with np.load(
            path,
            allow_pickle=False,
        ) as data:

            assert set(
                data.files
            ) == expected_keys

            assert not bool(
                data[
                    "trainer_eligible"
                ]
            )

            joined = " ".join(
                data.files
            ).lower()

            for token in forbidden_tokens:

                assert token not in joined


def test_geometry_structural_distinction():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_native_geometry_component_audit_v1_0.csv"
    )

    coherent = frame[
        frame[
            "geometry"
        ]
        == "coherent"
    ]

    dispersed = frame[
        frame[
            "geometry"
        ]
        == "dispersed"
    ]

    fragmented = frame[
        frame[
            "geometry"
        ]
        == "fragmented"
    ]

    assert (
        coherent[
            "selected_point_components"
        ]
        == 1
    ).all()

    assert (
        dispersed[
            "selected_point_components"
        ]
        >= 2
    ).all()

    assert (
        fragmented[
            "selected_point_components"
        ]
        >= 2
    ).all()


def test_native_budget_table_is_fully_feasible():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_native_budget_v1_0.csv"
    )

    assert len(frame) == 20

    assert frame[
        "native_six_cell_feasible"
    ].astype(bool).all()

    assert not frame[
        "trainer_eligible"
    ].astype(bool).any()

    assert set(
        frame[
            "trainer_grid_feasibility"
        ]
    ) == {
        "PENDING_09D"
    }
