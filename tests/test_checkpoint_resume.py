from pathlib import Path

import torch

from cora_lung.engine.checkpointing import (
    save_training_checkpoint,
    load_training_checkpoint,
)


def test_checkpoint_lineage_roundtrip(tmp_path):
    model = torch.nn.Linear(
        3,
        2,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    path = (
        Path(tmp_path)
        / "checkpoint.pt"
    )

    save_training_checkpoint(
        path,
        model=model,
        optimizer=optimizer,
        scaler=None,
        epoch=3,
        global_step=12,
        config_hash="a" * 64,
        cache_manifest_hash="b" * 64,
        run_id="test",
    )

    payload = load_training_checkpoint(
        path,
        expected_config_hash="a" * 64,
        expected_cache_manifest_hash="b" * 64,
    )

    assert payload[
        "epoch"
    ] == 3

    assert payload[
        "global_step"
    ] == 12
