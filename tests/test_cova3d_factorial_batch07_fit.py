from pathlib import Path
import json

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_batch07_fit_summary():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_factorial_batch07_fit_summary_v1_0.csv"
    )

    assert len(frame) == 6

    assert set(frame["condition"]) == {
        "C50_COH",
        "C50_DIS",
        "C50_FRG",
        "C100_COH",
        "C100_DIS",
        "C100_FRG",
    }

    assert (frame["outer_fold"] == 2).all()
    assert (frame["seed"] == 17).all()
    assert (frame["successful_optimizer_steps"] == 2400).all()

    assert frame["shared_initialization_sha256"].nunique() == 1
    assert frame["paired_plan_sha256"].nunique() == 1

    assert (frame["dense_training_masks_accessed"] == 0).all()
    assert (frame["heldout_dense_outcomes_accessed"] == 0).all()


def test_batch07_epoch_log():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_factorial_batch07_epoch_log_v1_0.csv"
    )

    assert len(frame) == 240

    assert (
        frame.groupby("condition")["epoch"].count()
        == 40
    ).all()

    assert (
        frame.groupby("condition")[
            "successful_optimizer_steps_total"
        ].max()
        == 2400
    ).all()


def test_batch07_audit():

    audit = json.loads(
        (
            ROOT
            / "experiments/audits/"
            "block10b_factorial_batch07_fit.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert audit["status"] == "FIT_COMPLETE"
    assert audit["batch_id"] == "fold2_seed17"
    assert audit["fits_completed"] == 6
    assert audit["optimizer_steps_batch"] == 14400

    assert audit["pairing"]["model_initialization"] == "PASS"
    assert audit["pairing"]["patient_schedule"] == "PASS"
    assert audit["pairing"]["augmentation_randomization"] == "PASS"

    assert audit["firewall"]["dense_training_masks_accessed"] == 0
    assert audit["firewall"]["heldout_dense_outcomes_accessed"] == 0
    assert audit["firewall"]["final_outer_cv_outcome_access"] == 0


def test_batch07_state():

    state = json.loads(
        (
            ROOT
            / "COVA3D_STATE.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert state["last_completed_block"] == "10B-FACTORIAL-BATCH-07"
    assert state["factorial_runs_completed"] == 24
    assert state["factorial_batches_completed"] == 4
    assert state["factorial_batch07_fit_status"] == "COMPLETE"
    assert state["final_outer_cv_outcomes_authorized"] is False
    assert state["final_outer_cv_access_in_cova3d"] == 0

    assert (
        state["next_block"]
        == "10C-FACTORIAL-BATCH-07-PRED-FREEZE"
    )
