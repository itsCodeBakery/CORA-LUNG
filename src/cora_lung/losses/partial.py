"""Sparse partial losses and exact query-holdout arithmetic for CORA-Lung."""

from __future__ import annotations

import torch
import torch.nn.functional as F


UNKNOWN_LABEL = -1


def labelled_mask(target):
    return target != UNKNOWN_LABEL


def sparse_bce_sum(logits, target, mask):
    if not torch.any(mask):
        return logits.sum() * 0.0

    return F.binary_cross_entropy_with_logits(
        logits[mask],
        target[mask].to(dtype=logits.dtype),
        reduction="sum",
    )


def partial_bce(logits, target):
    mask = labelled_mask(target)

    n = mask.sum().clamp_min(1)

    return (
        sparse_bce_sum(
            logits,
            target,
            mask,
        )
        / n
    )


def partial_dice(
    logits,
    target,
    eps=1e-6,
):
    mask = labelled_mask(
        target
    )

    if not torch.any(mask):
        return logits.sum() * 0.0

    p = torch.sigmoid(
        logits[mask]
    )

    y = target[
        mask
    ].float()

    numerator = (
        2.0
        * torch.sum(
            p * y
        )
        + eps
    )

    denominator = (
        torch.sum(p)
        + torch.sum(y)
        + eps
    )

    return (
        1.0
        - numerator
        / denominator
    )


def partial_segmentation_loss(
    logits,
    target,
):
    return (
        partial_bce(
            logits,
            target,
        )
        + partial_dice(
            logits,
            target,
        )
    )


def query_group_mask(
    target,
    membership_voxel_zyx,
    membership_group_id,
    query_group_id,
):
    query_group_id = int(
        query_group_id
    )

    dense = torch.zeros_like(
        target,
        dtype=torch.bool,
    )

    if membership_group_id.numel() == 0:
        return dense

    q_rows = (
        membership_group_id
        == query_group_id
    )

    coords = membership_voxel_zyx[
        q_rows
    ]

    for coord in coords:
        z, y, x = (
            int(coord[0]),
            int(coord[1]),
            int(coord[2]),
        )

        dense[
            ...,
            z,
            y,
            x,
        ] = True

    dense &= (
        target == 1
    )

    return dense


def query_holdout_bce_exact(
    logits,
    target,
    query_mask,
    query_lambda=1.0,
):
    labelled = labelled_mask(
        target
    )

    query_mask = (
        query_mask
        & labelled
    )

    support_mask = (
        labelled
        & (~query_mask)
    )

    total_n = (
        labelled.sum()
        .clamp_min(1)
    )

    support = (
        sparse_bce_sum(
            logits,
            target,
            support_mask,
        )
        / total_n
    )

    query = (
        sparse_bce_sum(
            logits,
            target,
            query_mask,
        )
        / total_n
    )

    decomposed = (
        support
        + float(
            query_lambda
        )
        * query
    )

    return {
        "support_bce":
            support,

        "query_bce":
            query,

        "decomposed_bce":
            decomposed,

        "ordinary_bce":
            partial_bce(
                logits,
                target,
            ),

        "labelled_count":
            int(
                labelled.sum()
            ),

        "query_count":
            int(
                query_mask.sum()
            ),
    }


def query_holdout_segmentation_loss(
    logits,
    target,
    query_mask,
    query_lambda=1.0,
):
    split = query_holdout_bce_exact(
        logits,
        target,
        query_mask,
        query_lambda=query_lambda,
    )

    dice = partial_dice(
        logits,
        target,
    )

    total = (
        split[
            "decomposed_bce"
        ]
        + dice
    )

    return {
        **split,

        "partial_dice":
            dice,

        "total":
            total,
    }
