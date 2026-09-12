"""Post-transfer sparse-control equalization for CORA-Lung."""

from __future__ import annotations

from collections import defaultdict
import hashlib

import numpy as np


def stable_seed(*parts):

    payload = "|".join(
        str(x)
        for x in parts
    )

    digest = hashlib.sha256(
        payload.encode("utf-8")
    ).digest()

    return int.from_bytes(
        digest[:8],
        "little",
        signed=False,
    ) % (2**32 - 1)


def candidate_pool(membership_coords, membership_groups):

    mapping = defaultdict(set)

    for coord, group_id in zip(
        np.asarray(
            membership_coords,
            dtype=np.int32,
        ),
        np.asarray(
            membership_groups,
            dtype=np.int32,
        ),
    ):

        mapping[
            tuple(
                int(x)
                for x in coord
            )
        ].add(
            int(
                group_id
            )
        )

    groups = set(
        int(x)
        for x in np.unique(
            membership_groups
        )
    )

    return dict(mapping), groups


def exact_group_preserving_subset(
    coord_to_groups,
    required_groups,
    target,
    seed,
):

    required_groups = set(
        int(x)
        for x in required_groups
    )

    target = int(target)

    candidates = sorted(
        coord_to_groups
    )

    if target > len(candidates):

        raise ValueError(
            "Target exceeds candidate capacity."
        )

    rng = np.random.default_rng(
        seed
    )

    priority = {
        coord:
            float(rng.random())
        for coord in candidates
    }

    selected = set()

    uncovered = set(
        required_groups
    )

    while uncovered:

        options = []

        for coord in candidates:

            gained = (
                coord_to_groups[coord]
                & uncovered
            )

            if gained:

                options.append(
                    (
                        -len(gained),
                        priority[coord],
                        coord,
                    )
                )

        if not options:

            raise ValueError(
                "Unable to cover all foreground groups."
            )

        options.sort()

        chosen = options[0][2]

        selected.add(
            chosen
        )

        uncovered -= (
            coord_to_groups[
                chosen
            ]
        )

    if len(selected) > target:

        raise ValueError(
            "Group-cover size exceeds target."
        )

    remaining = [
        coord
        for coord in candidates
        if coord not in selected
    ]

    needed = (
        target
        - len(selected)
    )

    if needed:

        order = rng.permutation(
            len(remaining)
        )

        for index in order[:needed]:

            selected.add(
                remaining[
                    int(index)
                ]
            )

    if len(selected) != target:

        raise RuntimeError(
            "Exact target was not reached."
        )

    return np.asarray(
        sorted(selected),
        dtype=np.int32,
    )
