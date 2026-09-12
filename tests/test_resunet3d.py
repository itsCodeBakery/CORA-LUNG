import torch

from cora_lung.models.resunet3d import (
    CORALungResidualUNet,
)


def test_model_output_shapes():
    model = CORALungResidualUNet(
        base_channels=4,
        embedding_dim=8,
    )

    x = torch.randn(
        1,
        1,
        16,
        32,
        32,
    )

    with torch.no_grad():
        out = model(
            x
        )

    assert out[
        "logits"
    ].shape == (
        1,
        1,
        16,
        32,
        32,
    )

    assert out[
        "embedding"
    ].shape == (
        1,
        8,
        16,
        32,
        32,
    )


def test_embedding_is_l2_normalized():
    model = CORALungResidualUNet(
        base_channels=4,
        embedding_dim=8,
    )

    x = torch.randn(
        1,
        1,
        8,
        16,
        16,
    )

    with torch.no_grad():
        emb = model(
            x
        )[
            "embedding"
        ]

    norm = torch.linalg.vector_norm(
        emb,
        dim=1,
    )

    assert torch.allclose(
        norm,
        torch.ones_like(
            norm
        ),
        atol=1e-4,
        rtol=1e-4,
    )
