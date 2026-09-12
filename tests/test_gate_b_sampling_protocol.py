import numpy as np

from cora_lung.data.pilot_dataset import (
    deterministic_patch_origin,
)


def test_gate_b_patch_source_probabilities():

    shape = (
        64,
        160,
        160,
    )

    patch = (
        48,
        128,
        128,
    )

    coords = np.asarray(
        [
            [20, 80, 80],
            [40, 100, 100],
        ],
        dtype=np.int32,
    )

    labels = np.asarray(
        [
            1,
            0,
        ],
        dtype=np.int8,
    )

    counts = {
        "foreground": 0,
        "background": 0,
        "random": 0,
    }

    n = 10000

    for i in range(n):

        _, source = deterministic_patch_origin(
            shape,
            patch,
            coords,
            labels,
            seed=17,
            sample_index=i,
        )

        counts[source] += 1

    proportions = {
        key:
            value / n
        for key, value
        in counts.items()
    }

    assert abs(
        proportions["foreground"]
        - 0.50
    ) < 0.025

    assert abs(
        proportions["background"]
        - 0.25
    ) < 0.025

    assert abs(
        proportions["random"]
        - 0.25
    ) < 0.025
