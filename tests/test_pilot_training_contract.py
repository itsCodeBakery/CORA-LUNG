from cora_lung.engine.pilot_contract import (
    primary_checkpoint_epoch,
    should_activate_replay,
    optimizer_step_due,
)


def test_final_epoch_is_primary_checkpoint():
    assert primary_checkpoint_epoch(
        30
    ) == 29


def test_replay_starts_after_warmup():
    assert not should_activate_replay(
        9,
        10,
    )

    assert should_activate_replay(
        10,
        10,
    )


def test_gradient_accumulation_contract():
    assert not optimizer_step_due(
        0,
        2,
    )

    assert optimizer_step_due(
        1,
        2,
    )
