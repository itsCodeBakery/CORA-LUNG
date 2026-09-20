from pathlib import Path
import json

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_factorial_execution_registry():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_final_factorial_execution_registry_v1_0.csv"
    )

    assert len(frame) == 72
    assert frame["run_id"].nunique() == 72
    assert frame["training_authorized"].all()
    assert (~frame["final_dense_outcome_access_during_fit"]).all()

    assert set(frame["condition"]) == {
        "C50_COH",
        "C50_DIS",
        "C50_FRG",
        "C100_COH",
        "C100_DIS",
        "C100_FRG",
    }

    assert set(frame["outer_fold"]) == {0, 1, 2, 3}
    assert set(frame["seed"]) == {17, 29, 43}

    assert frame["paired_batch_id"].nunique() == 12


def test_factorial_batch_registry():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_final_factorial_batch_registry_v1_0.csv"
    )

    assert len(frame) == 12
    assert (frame["run_count"] == 6).all()
    assert (frame["training_case_count"] == 12).all()
    assert (frame["heldout_case_count"] == 4).all()
    assert (frame["execution_status"] == "PENDING").all()


def test_factorial_execution_lock():

    cfg = yaml.safe_load(
        (
            ROOT
            / "configs/"
            "cova3d_factorial_execution_lock_v1_0.yaml"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert cfg["status"] == "FROZEN"
    assert cfg["sanity_gate"] == "PASS"
    assert cfg["primary_factorial"]["runs"] == 72
    assert cfg["primary_factorial"]["paired_batches"] == 12
    assert cfg["training"]["optimizer_steps_per_run"] == 2400

    assert (
        cfg[
            "outcome_firewall"
        ][
            "final_outer_cv_dense_outcomes_authorized_during_fit"
        ]
        is False
    )


def test_factorial_preflight_state():

    state = json.loads(
        (
            ROOT
            / "COVA3D_STATE.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert state["last_completed_block"] == "10A-FACTORIAL-PREFLIGHT"
    assert state["factorial_preflight_status"] == "PASS"
    assert state["factorial_training_authorized"] is True
    assert state["factorial_batches_total"] == 12
    assert state["factorial_batches_completed"] == 0
    assert state["factorial_runs_completed"] == 0
    assert state["final_outer_cv_outcomes_authorized"] is False
    assert state["final_outer_cv_access_in_cova3d"] == 0
    assert state["next_block"] == "10B-FACTORIAL-BATCH-01"
