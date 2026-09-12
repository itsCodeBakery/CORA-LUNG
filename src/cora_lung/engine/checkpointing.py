"""Crash-safe checkpoint utilities for CORA-Lung."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json
import os
import tempfile

import torch


def sha256_file(
    path,
    chunk_size=8 * 1024 * 1024,
):
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(
            lambda:
                f.read(chunk_size),
            b"",
        ):
            h.update(chunk)

    return h.hexdigest()


def atomic_torch_save(
    payload,
    destination,
):
    destination = Path(
        destination
    )

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fd, tmp_name = tempfile.mkstemp(
        prefix=(
            destination.name
            + ".tmp."
        ),
        dir=str(
            destination.parent
        ),
    )

    os.close(
        fd
    )

    tmp_path = Path(
        tmp_name
    )

    try:
        torch.save(
            payload,
            tmp_path,
        )

        os.replace(
            tmp_path,
            destination,
        )

    finally:
        if tmp_path.exists():
            tmp_path.unlink()

    return sha256_file(
        destination
    )


def save_training_checkpoint(
    destination,
    *,
    model,
    optimizer,
    scaler,
    epoch,
    global_step,
    config_hash,
    cache_manifest_hash,
    run_id,
):
    payload = {
        "run_id":
            run_id,

        "epoch":
            int(epoch),

        "global_step":
            int(global_step),

        "model":
            model.state_dict(),

        "optimizer":
            (
                optimizer.state_dict()
                if optimizer is not None
                else None
            ),

        "scaler":
            (
                scaler.state_dict()
                if scaler is not None
                else None
            ),

        "config_hash":
            str(
                config_hash
            ),

        "cache_manifest_hash":
            str(
                cache_manifest_hash
            ),
    }

    return atomic_torch_save(
        payload,
        destination,
    )


def load_training_checkpoint(
    path,
    *,
    expected_config_hash,
    expected_cache_manifest_hash,
):
    payload = torch.load(
        path,
        map_location="cpu",
        weights_only=False,
    )

    if (
        payload[
            "config_hash"
        ]
        != expected_config_hash
    ):
        raise RuntimeError(
            "Checkpoint config lineage mismatch."
        )

    if (
        payload[
            "cache_manifest_hash"
        ]
        != expected_cache_manifest_hash
    ):
        raise RuntimeError(
            "Checkpoint cache-manifest lineage mismatch."
        )

    return payload
