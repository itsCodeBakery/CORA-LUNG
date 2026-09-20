from pathlib import Path
import json

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_R2_epoch_contract():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_sanity_fit_R2_epoch_log_v1_0.csv"
    )

    assert len(frame) == 20

    assert (
        frame[
            "logical_microbatches"
        ]
        == 100
    ).all()

    assert (
        frame[
            "logical_optimizer_steps"
        ]
        == 50
    ).all()

    assert int(
        frame.iloc[-1][
            "successful_optimizer_steps_total"
        ]
    ) == 1000

    assert (
        frame[
            [
                "coronacases_004_samples",
                "coronacases_008_samples",
                "radiopaedia_14_85914_0_samples",
                "radiopaedia_27_86410_0_samples",
            ]
        ]
        == 25
    ).all().all()

    assert (
        frame[
            "foreground_centered"
        ]
        + frame[
            "background_centered"
        ]
        + frame[
            "uniform_crop"
        ]
        == 100
    ).all()

    assert (
        frame[
            "amp_scale_min"
        ]
        >= 1.0
    ).all()

    assert (
        frame[
            "amp_scale_max"
        ]
        <= 4096.0
    ).all()

    assert np.isfinite(
        frame[
            [
                "mean_loss",
                "mean_partial_bce",
                "mean_partial_dice_fp32",
                "mean_preclip_grad_norm",
                "max_preclip_grad_norm",
            ]
        ].to_numpy()
    ).all()


def test_R2_firewall():

    audit = json.loads(
        (
            ROOT
            / "experiments/audits/"
            "block09e_sanity_fit_R2.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert audit[
        "status"
    ] == "FIT_COMPLETE_EVALUATION_PENDING"

    assert audit[
        "R1_updates_retained"
    ] == 0

    assert audit[
        "successful_optimizer_steps"
    ] == 1000

    assert audit[
        "firewall"
    ][
        "dense_lesion_masks_accessed"
    ] == 0

    assert audit[
        "firewall"
    ][
        "dense_lung_masks_accessed"
    ] == 0

    assert audit[
        "firewall"
    ][
        "final_outer_cv_access"
    ] == 0

    assert audit[
        "sanity_gate_applied"
    ] is False

    assert audit[
        "factorial_training_authorized"
    ] is False


def test_R2_state():

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
    ] == "09E-SANITY-FIT-R2"

    assert state[
        "sanity_fit_R2_status"
    ] == "COMPLETE"

    assert state[
        "sanity_gate_status"
    ] == "EVALUATION_PENDING"

    assert state[
        "optimizer_steps_retained_for_current_frozen_run"
    ] == 1000

    assert state[
        "factorial_training_authorized"
    ] is False

    assert state[
        "dense_outcomes_opened_in_cova3d"
    ] is False

    assert state[
        "final_outer_cv_access_in_cova3d"
    ] == 0
