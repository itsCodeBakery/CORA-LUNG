from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_sanity_gate_case_table():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_sanity_gate_case_metrics_v1_0.csv"
    )

    assert len(frame) == 4

    assert (
        frame["role"]
        == "permanent_development"
    ).all()

    assert (
        frame["lesion_voxels_in_crop"]
        > 0
    ).all()

    assert (
        frame["outside_voxels_in_crop"]
        > 0
    ).all()


def test_sanity_gate_audit():

    audit = json.loads(
        (
            ROOT
            / "experiments/audits/"
            "block09e_sanity_gate_close.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert audit[
        "dense_development_separation"
    ][
        "cases_evaluated"
    ] == 4

    assert audit[
        "firewall"
    ][
        "development_infection_masks_accessed"
    ] == 4

    assert audit[
        "firewall"
    ][
        "development_lung_masks_accessed"
    ] == 0

    assert audit[
        "firewall"
    ][
        "final_outer_cv_dense_masks_accessed"
    ] == 0

    assert audit[
        "firewall"
    ][
        "optimizer_steps"
    ] == 0

    assert audit[
        "firewall"
    ][
        "threshold_search"
    ] is False

    assert audit[
        "firewall"
    ][
        "Dice_computed"
    ] is False


def test_state_matches_gate():

    audit = json.loads(
        (
            ROOT
            / "experiments/audits/"
            "block09e_sanity_gate_close.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    state = json.loads(
        (
            ROOT
            / "COVA3D_STATE.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert state[
        "last_completed_block"
    ] == "09E-SANITY-EVAL"

    assert state[
        "sanity_gate_status"
    ] == (
        "PASS"
        if audit[
            "overall_gate_pass"
        ]
        else "FAIL"
    )

    assert state[
        "factorial_training_authorized"
    ] is audit[
        "overall_gate_pass"
    ]

    assert state[
        "final_outer_cv_access_in_cova3d"
    ] == 0
