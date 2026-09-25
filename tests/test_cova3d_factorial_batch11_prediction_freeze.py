from pathlib import Path
import json

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_batch11_prediction_manifest():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_factorial_batch11_prediction_freeze_v1_0.csv"
    )

    assert len(frame) == 24

    assert frame["condition"].nunique() == 6
    assert frame["case_id"].nunique() == 4

    assert (
        frame.groupby("condition")["case_id"].count()
        == 4
    ).all()

    assert (frame["outer_fold"] == 3).all()
    assert (frame["seed"] == 29).all()

    assert (frame["storage_dtype"] == "float32").all()
    assert (~frame["dense_mask_accessed"]).all()
    assert (~frame["final_outcome_access"]).all()

    assert frame["prediction_sha256"].notna().all()


def test_factorial_inference_lock():

    cfg = yaml.safe_load(
        (
            ROOT
            / "configs/"
            "cova3d_factorial_inference_implementation_lock_v1_0.yaml"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        cfg["status"]
        == "FROZEN_BEFORE_FIRST_FINAL_FACTORIAL_DENSE_OUTCOME"
    )

    assert cfg["inference"]["patch_zyx"] == [48, 128, 128]
    assert cfg["inference"]["overlap"] == 0.5
    assert cfg["inference"]["gaussian_sigma_scale"] == 0.125
    assert cfg["inference"]["blend_domain"] == "logits"
    assert cfg["inference"]["TTA"] is False
    assert cfg["inference"]["output_storage_dtype"] == "float32"


def test_batch11_prediction_audit():

    audit = json.loads(
        (
            ROOT
            / "experiments/audits/"
            "block10c_factorial_batch11_prediction_freeze.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert audit["status"] == "PREDICTIONS_FROZEN"
    assert audit["batch_id"] == "fold3_seed29"
    assert audit["models"] == 6
    assert audit["prediction_maps"] == 24
    assert audit["prediction_SHA_verified"] == 24

    assert audit["firewall"]["infection_masks_accessed"] == 0
    assert audit["firewall"]["lung_masks_accessed"] == 0
    assert audit["firewall"]["final_dense_outcomes_accessed"] == 0
    assert audit["firewall"]["optimizer_steps"] == 0
    assert audit["firewall"]["training"] is False


def test_batch11_prediction_state():

    state = json.loads(
        (
            ROOT
            / "COVA3D_STATE.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        state["last_completed_block"]
        == "10C-FACTORIAL-BATCH-11-PRED-FREEZE"
    )

    assert state["factorial_runs_completed"] == 36
    assert state["factorial_prediction_runs_frozen"] == 36
    assert state["factorial_prediction_maps_frozen"] == 144

    assert state["final_outer_cv_outcomes_authorized"] is False
    assert state["final_outer_cv_access_in_cova3d"] == 0

    assert state["next_factorial_batch"] == "fold3_seed43"
    assert state["next_block"] == "10B-FACTORIAL-BATCH-12"


def test_registry_advanced():

    execution = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_final_factorial_execution_registry_v1_0.csv"
    )

    rows = execution[
        execution["paired_batch_id"]
        == "fold3_seed29"
    ]

    assert len(rows) == 6

    assert (
        rows["execution_status"]
        == "PREDICTIONS_FROZEN"
    ).all()


    batch = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_final_factorial_batch_registry_v1_0.csv"
    )

    row = batch[
        batch["batch_id"]
        == "fold3_seed29"
    ]

    assert len(row) == 1

    assert (
        row.iloc[0]["execution_status"]
        == "PREDICTIONS_FROZEN"
    )
