import torch

from cora_lung.losses.partial import (
    partial_bce,
    partial_dice,
    query_holdout_bce_exact,
)


def test_unknown_voxels_have_zero_bce_gradient():
    logits = torch.tensor(
        [
            [
                [
                    [
                        [0.2, -0.3, 0.7]
                    ]
                ]
            ]
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    target = torch.tensor(
        [
            [
                [
                    [
                        [1, -1, 0]
                    ]
                ]
            ]
        ],
        dtype=torch.int8,
    )

    loss = partial_bce(
        logits,
        target,
    )

    grad = torch.autograd.grad(
        loss,
        logits,
    )[0]

    assert grad.flatten()[1].item() == 0.0


def test_unknown_voxels_have_zero_dice_gradient():
    logits = torch.tensor(
        [
            [
                [
                    [
                        [0.2, -0.3, 0.7]
                    ]
                ]
            ]
        ],
        dtype=torch.float64,
        requires_grad=True,
    )

    target = torch.tensor(
        [
            [
                [
                    [
                        [1, -1, 0]
                    ]
                ]
            ]
        ],
        dtype=torch.int8,
    )

    loss = partial_dice(
        logits,
        target,
    )

    grad = torch.autograd.grad(
        loss,
        logits,
    )[0]

    assert grad.flatten()[1].item() == 0.0


def test_query_holdout_value_exact():
    torch.manual_seed(7)

    logits = torch.randn(
        1,
        1,
        3,
        4,
        5,
        dtype=torch.float64,
        requires_grad=True,
    )

    target = torch.full(
        (
            1,
            1,
            3,
            4,
            5,
        ),
        -1,
        dtype=torch.int8,
    )

    labelled = [
        (0, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 1, 1),
        (0, 0, 1, 1, 1, 0),
        (0, 0, 2, 3, 4, 0),
        (0, 0, 1, 2, 3, 1),
    ]

    for b, c, z, y, x, value in labelled:
        target[
            b,
            c,
            z,
            y,
            x,
        ] = value

    query_mask = torch.zeros_like(
        target,
        dtype=torch.bool,
    )

    query_mask[
        0,
        0,
        0,
        0,
        0,
    ] = True

    query_mask[
        0,
        0,
        1,
        2,
        3,
    ] = True

    split = query_holdout_bce_exact(
        logits,
        target,
        query_mask,
        query_lambda=1.0,
    )

    assert torch.allclose(
        split[
            "ordinary_bce"
        ],
        split[
            "decomposed_bce"
        ],
        atol=1e-12,
        rtol=1e-12,
    )


def test_query_holdout_gradient_exact():
    torch.manual_seed(11)

    base_logits = torch.randn(
        1,
        1,
        2,
        3,
        4,
        dtype=torch.float64,
    )

    target = torch.full(
        (
            1,
            1,
            2,
            3,
            4,
        ),
        -1,
        dtype=torch.int8,
    )

    target[
        0,
        0,
        0,
        0,
        0,
    ] = 1

    target[
        0,
        0,
        0,
        0,
        1,
    ] = 1

    target[
        0,
        0,
        1,
        1,
        1,
    ] = 0

    target[
        0,
        0,
        1,
        2,
        3,
    ] = 0

    query_mask = torch.zeros_like(
        target,
        dtype=torch.bool,
    )

    query_mask[
        0,
        0,
        0,
        0,
        0,
    ] = True

    logits_a = base_logits.clone().requires_grad_()

    ordinary = partial_bce(
        logits_a,
        target,
    )

    grad_a = torch.autograd.grad(
        ordinary,
        logits_a,
    )[0]

    logits_b = base_logits.clone().requires_grad_()

    split = query_holdout_bce_exact(
        logits_b,
        target,
        query_mask,
        query_lambda=1.0,
    )

    grad_b = torch.autograd.grad(
        split[
            "decomposed_bce"
        ],
        logits_b,
    )[0]

    assert torch.allclose(
        grad_a,
        grad_b,
        atol=1e-12,
        rtol=1e-12,
    )


def test_partial_bce_preserves_float64_precision():
    logits = torch.tensor(
        [[[[[0.13, -0.71, 1.19, -0.44]]]]],
        dtype=torch.float64,
        requires_grad=True,
    )

    target = torch.tensor(
        [[[[[1, 0, 1, 0]]]]],
        dtype=torch.int8,
    )

    loss = partial_bce(
        logits,
        target,
    )

    assert loss.dtype == torch.float64


def test_query_decomposition_float64_precision_regression():
    logits = torch.tensor(
        [[[[[0.13, -0.71, 1.19, -0.44]]]]],
        dtype=torch.float64,
        requires_grad=True,
    )

    target = torch.tensor(
        [[[[[1, 0, 1, 0]]]]],
        dtype=torch.int8,
    )

    query_mask = torch.zeros_like(
        target,
        dtype=torch.bool,
    )

    query_mask[
        0,
        0,
        0,
        0,
        0,
    ] = True

    split = query_holdout_bce_exact(
        logits,
        target,
        query_mask,
        query_lambda=1.0,
    )

    assert split["ordinary_bce"].dtype == torch.float64

    assert split["decomposed_bce"].dtype == torch.float64

    assert torch.allclose(
        split["ordinary_bce"],
        split["decomposed_bce"],
        atol=1e-12,
        rtol=1e-12,
    )
