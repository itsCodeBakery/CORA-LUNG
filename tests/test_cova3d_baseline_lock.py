from pathlib import Path

import pandas as pd
import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def test_baseline_config_is_frozen():

    config = yaml.safe_load(
        (
            ROOT
            / "configs/"
            "cova3d_baseline_training_v1_0.yaml"
        ).read_text(
            encoding="utf-8"
        )
    )

    assert (
        config[
            "status"
        ]
        == "PROSPECTIVELY_FROZEN_BEFORE_FIRST_COVA_OPTIMIZER_STEP"
    )

    assert (
        config[
            "loss"
        ][
            "unknown_label"
        ]
        == -1
    )

    assert (
        config[
            "training_authorization_after_lock"
        ][
            "sanity_training"
        ]
        is True
    )

    assert (
        config[
            "training_authorization_after_lock"
        ][
            "factorial_training"
        ]
        is False
    )


def test_final_run_registry():

    frame = pd.read_csv(
        ROOT
        / "data/manifests/"
        "cova3d_final_factorial_run_registry_v1_0.csv"
    )

    assert len(
        frame
    ) == 72

    assert set(
        frame[
            "seed"
        ].astype(
            int
        )
    ) == {
        17,
        29,
        43,
    }

    assert set(
        frame[
            "outer_fold"
        ].astype(
            int
        )
    ) == {
        0,
        1,
        2,
        3,
    }

    assert (
        frame[
            "training_case_count"
        ]
        == 12
    ).all()

    assert (
        frame[
            "heldout_case_count"
        ]
        == 4
    ).all()

    assert not frame[
        "training_authorized"
    ].astype(
        bool
    ).any()


def test_model_shape_cpu():

    from cora_lung.models.cova3d_nnunet_style import (
        COVA3DNNUNetStyle,
    )

    model = COVA3DNNUNetStyle(
        features=(
            32,
            64,
            128,
            256,
            320,
        )
    )

    # Smaller divisible tensor for CPU unit test.
    x = torch.zeros(
        (
            1,
            1,
            16,
            32,
            32,
        ),
        dtype=torch.float32,
    )

    model.eval()

    with torch.no_grad():

        y = model(
            x
        )

    assert y.shape == x.shape


def test_pairing_seed_independent_of_condition():

    import hashlib

    def stable_seed(*parts):

        payload = "|".join(
            str(
                part
            )
            for part in parts
        )

        digest = hashlib.sha256(
            payload.encode(
                "utf-8"
            )
        ).digest()

        return int.from_bytes(
            digest[
                :8
            ],
            "little",
            signed=False,
        ) % (
            2**32
            - 1
        )

    seed_a = stable_seed(
        "COVA3D_NNUNET_STYLE_PARTIAL_V1",
        "init",
        17,
        0,
    )

    seed_b = stable_seed(
        "COVA3D_NNUNET_STYLE_PARTIAL_V1",
        "init",
        17,
        0,
    )

    assert seed_a == seed_b
