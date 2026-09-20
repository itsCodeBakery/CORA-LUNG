from pathlib import Path
import json

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_partial_dice_fp32_matches_reference():

    from cora_lung.losses.partial import (
        partial_dice,
        partial_dice_fp32,
    )

    logits = torch.tensor(
        [
            -16.0,
            -15.5,
            2.0,
            -1.0,
        ],
        dtype=torch.float16,
        requires_grad=True,
    )

    target = torch.tensor(
        [
            0,
            0,
            1,
            -1,
        ],
        dtype=torch.int8,
    )

    observed = partial_dice_fp32(
        logits,
        target,
    )

    reference = partial_dice(
        logits.float(),
        target,
    )

    assert observed.dtype == torch.float32

    assert torch.allclose(
        observed,
        reference,
        atol=1e-7,
        rtol=1e-7,
    )


def test_partial_dice_fp32_unknown_only_zero_gradient():

    from cora_lung.losses.partial import (
        partial_dice_fp32,
    )

    logits = torch.randn(
        12,
        dtype=torch.float16,
        requires_grad=True,
    )

    target = torch.full(
        (
            12,
        ),
        -1,
        dtype=torch.int8,
    )

    loss = partial_dice_fp32(
        logits,
        target,
    )

    assert float(
        loss.detach()
    ) == 0.0

    loss.backward()

    assert torch.all(
        logits.grad
        == 0
    )


def test_n2_protocol_is_frozen():

    cfg = yaml.safe_load(
        (
            ROOT
            / "configs/"
            "cova3d_numerical_amendment_N2_fp32_dice.yaml"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        cfg[
            "status"
        ]
        == "PROSPECTIVELY_FROZEN_BEFORE_CLEAN_SANITY_RESTART_R2"
    )

    assert (
        cfg[
            "N2_policy"
        ][
            "network_forward_dtype"
        ]
        == "float16"
    )

    assert (
        cfg[
            "N2_policy"
        ][
            "partial_Dice_arithmetic_dtype"
        ]
        == "float32"
    )

    assert (
        cfg[
            "N2_policy"
        ][
            "partial_Dice_epsilon"
        ]
        == 1e-6
    )

    assert (
        cfg[
            "N2_policy"
        ][
            "partial_BCE"
        ]
        == "UNCHANGED"
    )

    assert (
        cfg[
            "restart_policy"
        ][
            "R1_updates_retained"
        ]
        == 0
    )

    assert (
        cfg[
            "factorial_training_authorized"
        ]
        is False
    )


def test_n2_state():

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
        == "09E-N2-LOCK"
    )

    assert (
        state[
            "effective_protocol"
        ]
        == "COVA3D_1.0+A1+A1.1+A1.2+N1+N2"
    )

    assert (
        state[
            "sanity_fit_R1_successful_updates_executed"
        ]
        == 233
    )

    assert (
        state[
            "sanity_fit_R1_updates_retained"
        ]
        == 0
    )

    assert (
        state[
            "optimizer_steps_retained_for_current_frozen_run"
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
