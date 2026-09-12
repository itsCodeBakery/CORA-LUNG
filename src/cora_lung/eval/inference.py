"""Frozen sliding-window inference used for CORA-Lung Gate B."""

from __future__ import annotations

import numpy as np
import torch


def sliding_window_starts(
    length,
    patch,
    overlap,
):
    length = int(
        length
    )

    patch = int(
        patch
    )

    if length <= patch:

        return [
            0
        ]


    stride = max(
        1,
        int(
            round(
                patch
                * (
                    1.0
                    - float(
                        overlap
                    )
                )
            )
        ),
    )


    starts = list(
        range(
            0,
            length
            - patch
            + 1,
            stride,
        )
    )


    last = (
        length
        - patch
    )


    if starts[
        -1
    ] != last:

        starts.append(
            last
        )


    return starts


def gaussian_importance_map(
    patch_shape,
    *,
    sigma_scale,
):
    vectors = []


    for size in patch_shape:

        size = int(
            size
        )

        coordinate = np.arange(
            size,
            dtype=np.float32,
        )


        center = (
            size
            - 1
        ) / 2.0


        sigma = max(
            1.0,
            float(
                sigma_scale
            )
            * size,
        )


        vector = np.exp(
            -0.5
            * (
                (
                    coordinate
                    - center
                )
                / sigma
            )
            ** 2
        )


        vectors.append(
            vector.astype(
                np.float32
            )
        )


    weight = (
        vectors[
            0
        ][
            :,
            None,
            None,
        ]
        * vectors[
            1
        ][
            None,
            :,
            None,
        ]
        * vectors[
            2
        ][
            None,
            None,
            :,
        ]
    )


    weight = np.maximum(
        weight,
        1e-3,
    )


    return weight.astype(
        np.float32
    )


@torch.no_grad()
def sliding_window_probability(
    model,
    image_zyx,
    *,
    patch_shape,
    overlap,
    sigma_scale,
    device,
    amp=True,
):
    image_zyx = np.asarray(
        image_zyx,
        dtype=np.float32,
    )

    patch_shape = tuple(
        int(
            x
        )
        for x in patch_shape
    )


    original_shape = tuple(
        int(
            x
        )
        for x in image_zyx.shape
    )


    padded_shape = tuple(
        max(
            original_shape[
                axis
            ],
            patch_shape[
                axis
            ],
        )
        for axis in range(
            3
        )
    )


    pad_width = [
        (
            0,
            padded_shape[
                axis
            ]
            - original_shape[
                axis
            ],
        )
        for axis in range(
            3
        )
    ]


    padded = np.pad(
        image_zyx,
        pad_width,
        mode="constant",
        constant_values=-1.0,
    )


    starts = [
        sliding_window_starts(
            padded_shape[
                axis
            ],
            patch_shape[
                axis
            ],
            overlap,
        )
        for axis in range(
            3
        )
    ]


    importance = gaussian_importance_map(
        patch_shape,
        sigma_scale=sigma_scale,
    )


    logit_sum = np.zeros(
        padded_shape,
        dtype=np.float32,
    )


    weight_sum = np.zeros(
        padded_shape,
        dtype=np.float32,
    )


    patch_count = (
        len(
            starts[
                0
            ]
        )
        * len(
            starts[
                1
            ]
        )
        * len(
            starts[
                2
            ]
        )
    )


    model.eval()


    for z in starts[
        0
    ]:

        for y in starts[
            1
        ]:

            for x in starts[
                2
            ]:

                patch = padded[
                    z:
                    z
                    + patch_shape[
                        0
                    ],

                    y:
                    y
                    + patch_shape[
                        1
                    ],

                    x:
                    x
                    + patch_shape[
                        2
                    ],
                ]


                tensor = torch.from_numpy(
                    patch[
                        None,
                        None,
                        ...
                    ]
                ).to(
                    device=device,
                    dtype=torch.float32,
                )


                with torch.autocast(
                    device_type=device.type,
                    dtype=(
                        torch.float16
                        if device.type
                        == "cuda"
                        else torch.bfloat16
                    ),
                    enabled=bool(
                        amp
                        and device.type
                        == "cuda"
                    ),
                ):

                    output = model(
                        tensor
                    )


                    logits = output[
                        "logits"
                    ]


                patch_logits = (
                    logits[
                        0,
                        0
                    ]
                    .float()
                    .cpu()
                    .numpy()
                )


                logit_sum[
                    z:
                    z
                    + patch_shape[
                        0
                    ],

                    y:
                    y
                    + patch_shape[
                        1
                    ],

                    x:
                    x
                    + patch_shape[
                        2
                    ],
                ] += (
                    patch_logits
                    * importance
                )


                weight_sum[
                    z:
                    z
                    + patch_shape[
                        0
                    ],

                    y:
                    y
                    + patch_shape[
                        1
                    ],

                    x:
                    x
                    + patch_shape[
                        2
                    ],
                ] += importance


                del tensor
                del output
                del logits


    if np.any(
        weight_sum
        <= 0
    ):

        raise RuntimeError(
            "Sliding-window coverage contains zero-weight voxels."
        )


    merged_logits = (
        logit_sum
        / weight_sum
    )


    merged_logits = merged_logits[
        :
        original_shape[
            0
        ],

        :
        original_shape[
            1
        ],

        :
        original_shape[
            2
        ],
    ]


    # Stable sigmoid.
    probability = np.empty_like(
        merged_logits,
        dtype=np.float32,
    )


    positive = (
        merged_logits
        >= 0
    )


    probability[
        positive
    ] = (
        1.0
        / (
            1.0
            + np.exp(
                -merged_logits[
                    positive
                ]
            )
        )
    )


    exp_x = np.exp(
        merged_logits[
            ~positive
        ]
    )


    probability[
        ~positive
    ] = (
        exp_x
        / (
            1.0
            + exp_x
        )
    )


    return (
        probability.astype(
            np.float32
        ),
        int(
            patch_count
        ),
    )
