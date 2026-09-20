from pathlib import Path
import json

import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_n1_is_frozen():

    cfg = yaml.safe_load(
        (
            ROOT
            / "configs/"
            "cova3d_numerical_amendment_N1_amp.yaml"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        cfg[
            "status"
        ]
        == "PROSPECTIVELY_FROZEN_BEFORE_FIRST_SUCCESSFUL_COVA_OPTIMIZER_STEP"
    )

    assert (
        cfg[
            "amp_policy"
        ][
            "initial_scale"
        ]
        == 4096.0
    )

    assert (
        cfg[
            "amp_policy"
        ][
            "growth_interval_successful_steps"
        ]
        == 1000000
    )

    assert (
        cfg[
            "amp_policy"
        ][
            "retry_same_logical_pair"
        ]
        is True
    )

    assert (
        cfg[
            "scientifically_unchanged"
        ][
            "successful_sanity_optimizer_step_target"
        ]
        == 1000
    )

    assert (
        cfg[
            "factorial_training_authorized"
        ]
        is False
    )


def test_state_after_n1():

    state = json.loads(
        (
            ROOT
            / "COVA3D_STATE.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        state[
            "last_completed_block"
        ]
        == "09E-N1-LOCK"
    )

    assert (
        state[
            "numerical_amendment"
        ]
        == "N1"
    )

    assert (
        state[
            "optimizer_steps_in_cova3d"
        ]
        == 0
    )

    assert (
        state[
            "sanity_training_authorized"
        ]
        is True
    )

    assert (
        state[
            "factorial_training_authorized"
        ]
        is False
    )

    assert (
        state[
            "dense_outcomes_opened_in_cova3d"
        ]
        is False
    )
