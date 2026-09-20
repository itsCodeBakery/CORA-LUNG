from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def load_n1():

    return yaml.safe_load(
        (
            ROOT
            / "configs/"
            "cova3d_numerical_amendment_N1_amp.yaml"
        ).read_text(
            encoding="utf-8"
        )
    )


def test_n1_is_frozen():

    cfg = load_n1()

    assert (
        cfg[
            "status"
        ]
        == "PROSPECTIVELY_FROZEN_BEFORE_FIRST_SUCCESSFUL_COVA_OPTIMIZER_STEP"
    )

    assert (
        cfg[
            "amendment_id"
        ]
        == "N1_FP16_STABLE_LOSS_SCALING"
    )

    assert (
        cfg[
            "protocol_before"
        ]
        == "COVA3D_1.0+A1+A1.1+A1.2"
    )

    assert (
        cfg[
            "protocol_after"
        ]
        == "COVA3D_1.0+A1+A1.1+A1.2+N1"
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
            "amp_policy"
        ][
            "gradient_clip_only_after_finite_unscale"
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


def test_n1_historical_provenance_is_immutable():

    """Validate the historical N1 artifact, not mutable CURRENT project state."""

    cfg = load_n1()

    assert (
        cfg[
            "diagnostic"
        ][
            "diagnosis"
        ]
        == "TRANSIENT_FP16_LOSS_SCALE_OVERFLOW"
    )

    assert (
        cfg[
            "diagnostic"
        ][
            "largest_observed_finite_scale"
        ]
        == 4096.0
    )

    assert (
        cfg[
            "diagnostic"
        ][
            "scale_1_finite"
        ]
        is True
    )

    assert (
        cfg[
            "scientifically_unchanged"
        ][
            "architecture"
        ]
        is True
    )

    assert (
        cfg[
            "scientifically_unchanged"
        ][
            "loss"
        ]
        is True
    )

    assert (
        cfg[
            "scientifically_unchanged"
        ][
            "annotation_protocol"
        ]
        is True
    )
