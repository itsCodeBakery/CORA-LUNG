from pathlib import Path
import json

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_n2():

    return yaml.safe_load(
        (
            ROOT
            / "configs/"
            "cova3d_numerical_amendment_N2_fp32_dice.yaml"
        ).read_text(
            encoding="utf-8"
        )
    )


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

    cfg = load_n2()

    assert (
        cfg[
            "status"
        ]
        == "PROSPECTIVELY_FROZEN_BEFORE_CLEAN_SANITY_RESTART_R2"
    )

    assert (
        cfg[
            "amendment_id"
        ]
        == "N2_FP32_SPARSE_DICE_ARITHMETIC"
    )

    assert (
        cfg[
            "protocol_before"
        ]
        == "COVA3D_1.0+A1+A1.1+A1.2+N1"
    )

    assert (
        cfg[
            "protocol_after"
        ]
        == "COVA3D_1.0+A1+A1.1+A1.2+N1+N2"
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


def test_n2_historical_audit_is_immutable():

    audit = json.loads(
        (
            ROOT
            / "experiments/audits/"
            "block09e_n2_lock_fp32_dice.json"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        audit[
            "status"
        ]
        == "PASS"
    )

    assert (
        audit[
            "amendment"
        ]
        == "N2_FP32_SPARSE_DICE_ARITHMETIC"
    )

    assert (
        audit[
            "R1_successful_updates_executed"
        ]
        == 233
    )

    assert (
        audit[
            "R1_successful_updates_retained"
        ]
        == 0
    )

    assert (
        audit[
            "legacy_Dice_scale1_finite"
        ]
        is False
    )

    assert (
        audit[
            "N2_Dice_scale1_finite"
        ]
        is True
    )

    assert (
        audit[
            "Dice_arithmetic_dtype"
        ]
        == "FP32"
    )

    assert (
        audit[
            "Dice_formula_changed"
        ]
        is False
    )

    assert (
        audit[
            "Dice_epsilon_changed"
        ]
        is False
    )

    assert (
        audit[
            "BCE_changed"
        ]
        is False
    )

    assert (
        audit[
            "factorial_training_authorized"
        ]
        is False
    )
