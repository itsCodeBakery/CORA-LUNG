"""Pilot training contract for CORA-Lung.

This module defines infrastructure only.
Scientific training is launched by a later block.
"""

from __future__ import annotations

import torch


def primary_checkpoint_epoch(
    total_epochs,
):
    total_epochs = int(
        total_epochs
    )

    if total_epochs < 1:
        raise ValueError(
            "total_epochs must be positive."
        )

    return (
        total_epochs
        - 1
    )


def should_activate_replay(
    epoch,
    warmup_epochs,
):
    return int(
        epoch
    ) >= int(
        warmup_epochs
    )


def scale_loss_for_accumulation(
    loss,
    accumulation_steps,
):
    accumulation_steps = int(
        accumulation_steps
    )

    if accumulation_steps < 1:
        raise ValueError(
            "accumulation_steps must be >=1"
        )

    return (
        loss
        / accumulation_steps
    )


def optimizer_step_due(
    microbatch_index,
    accumulation_steps,
):
    return (
        (
            int(
                microbatch_index
            )
            + 1
        )
        % int(
            accumulation_steps
        )
        == 0
    )
