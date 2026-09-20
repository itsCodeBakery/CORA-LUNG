"""COVA-3D strong nnU-Net-style sparse-supervision baseline.

This module intentionally contains no COVA-specific replay, teacher/student,
pseudo-label, graph, prototype, VLM or confidence mechanism.

The scientific intervention is the annotation condition, not the network.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvNormAct3D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
    ):
        super().__init__()

        if isinstance(
            kernel_size,
            int,
        ):
            kernel_size = (
                kernel_size,
                kernel_size,
                kernel_size,
            )

        padding = tuple(
            int(
                k
            )
            // 2
            for k in kernel_size
        )

        self.block = nn.Sequential(
            nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                bias=False,
            ),
            nn.InstanceNorm3d(
                out_channels,
                affine=True,
            ),
            nn.LeakyReLU(
                negative_slope=0.01,
                inplace=True,
            ),
        )

    def forward(
        self,
        x,
    ):
        return self.block(
            x
        )


class PlainConvStage3D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        convolutions=2,
    ):
        super().__init__()

        layers = []

        current = int(
            in_channels
        )

        for _ in range(
            int(
                convolutions
            )
        ):

            layers.append(
                ConvNormAct3D(
                    current,
                    out_channels,
                    kernel_size,
                )
            )

            current = int(
                out_channels
            )

        self.block = nn.Sequential(
            *layers
        )

    def forward(
        self,
        x,
    ):
        return self.block(
            x
        )


class Downsample3D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        stride,
    ):
        super().__init__()

        self.down = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=stride,
            stride=stride,
            bias=False,
        )

    def forward(
        self,
        x,
    ):
        return self.down(
            x
        )


class DecoderStage3D(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        stride,
        kernel_size,
    ):
        super().__init__()

        self.up = nn.ConvTranspose3d(
            in_channels,
            out_channels,
            kernel_size=stride,
            stride=stride,
            bias=False,
        )

        self.stage = PlainConvStage3D(
            out_channels
            + skip_channels,
            out_channels,
            kernel_size=kernel_size,
            convolutions=2,
        )

    def forward(
        self,
        x,
        skip,
    ):
        x = self.up(
            x
        )

        if x.shape[
            2:
        ] != skip.shape[
            2:
        ]:
            raise RuntimeError(
                "Decoder/skip shape mismatch: "
                + str(
                    tuple(
                        x.shape
                    )
                )
                + " versus "
                + str(
                    tuple(
                        skip.shape
                    )
                )
            )

        x = torch.cat(
            [
                x,
                skip,
            ],
            dim=1,
        )

        return self.stage(
            x
        )


class COVA3DNNUNetStyle(nn.Module):

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        features=(
            32,
            64,
            128,
            256,
            320,
        ),
    ):
        super().__init__()

        features = tuple(
            int(
                value
            )
            for value in features
        )

        if len(
            features
        ) != 5:
            raise ValueError(
                "COVA3DNNUNetStyle requires exactly five feature stages."
            )

        kernels = (
            (
                1,
                3,
                3,
            ),
            (
                3,
                3,
                3,
            ),
            (
                3,
                3,
                3,
            ),
            (
                3,
                3,
                3,
            ),
            (
                3,
                3,
                3,
            ),
        )

        strides = (
            (
                1,
                2,
                2,
            ),
            (
                2,
                2,
                2,
            ),
            (
                2,
                2,
                2,
            ),
            (
                2,
                2,
                2,
            ),
        )

        self.encoder0 = PlainConvStage3D(
            in_channels,
            features[
                0
            ],
            kernels[
                0
            ],
        )

        self.down1 = Downsample3D(
            features[
                0
            ],
            features[
                1
            ],
            strides[
                0
            ],
        )

        self.encoder1 = PlainConvStage3D(
            features[
                1
            ],
            features[
                1
            ],
            kernels[
                1
            ],
        )

        self.down2 = Downsample3D(
            features[
                1
            ],
            features[
                2
            ],
            strides[
                1
            ],
        )

        self.encoder2 = PlainConvStage3D(
            features[
                2
            ],
            features[
                2
            ],
            kernels[
                2
            ],
        )

        self.down3 = Downsample3D(
            features[
                2
            ],
            features[
                3
            ],
            strides[
                2
            ],
        )

        self.encoder3 = PlainConvStage3D(
            features[
                3
            ],
            features[
                3
            ],
            kernels[
                3
            ],
        )

        self.down4 = Downsample3D(
            features[
                3
            ],
            features[
                4
            ],
            strides[
                3
            ],
        )

        self.bottleneck = PlainConvStage3D(
            features[
                4
            ],
            features[
                4
            ],
            kernels[
                4
            ],
        )

        self.decoder3 = DecoderStage3D(
            features[
                4
            ],
            features[
                3
            ],
            features[
                3
            ],
            strides[
                3
            ],
            kernels[
                3
            ],
        )

        self.decoder2 = DecoderStage3D(
            features[
                3
            ],
            features[
                2
            ],
            features[
                2
            ],
            strides[
                2
            ],
            kernels[
                2
            ],
        )

        self.decoder1 = DecoderStage3D(
            features[
                2
            ],
            features[
                1
            ],
            features[
                1
            ],
            strides[
                1
            ],
            kernels[
                1
            ],
        )

        self.decoder0 = DecoderStage3D(
            features[
                1
            ],
            features[
                0
            ],
            features[
                0
            ],
            strides[
                0
            ],
            kernels[
                0
            ],
        )

        self.output_head = nn.Conv3d(
            features[
                0
            ],
            out_channels,
            kernel_size=1,
            bias=True,
        )

        self.apply(
            self._initialize
        )

    @staticmethod
    def _initialize(module):

        if isinstance(
            module,
            (
                nn.Conv3d,
                nn.ConvTranspose3d,
            ),
        ):
            nn.init.kaiming_normal_(
                module.weight,
                a=0.01,
                mode="fan_out",
                nonlinearity="leaky_relu",
            )

            if getattr(
                module,
                "bias",
                None,
            ) is not None:
                nn.init.zeros_(
                    module.bias
                )

    def forward(
        self,
        x,
    ):
        e0 = self.encoder0(
            x
        )

        e1 = self.encoder1(
            self.down1(
                e0
            )
        )

        e2 = self.encoder2(
            self.down2(
                e1
            )
        )

        e3 = self.encoder3(
            self.down3(
                e2
            )
        )

        bottleneck = self.bottleneck(
            self.down4(
                e3
            )
        )

        d3 = self.decoder3(
            bottleneck,
            e3,
        )

        d2 = self.decoder2(
            d3,
            e2,
        )

        d1 = self.decoder1(
            d2,
            e1,
        )

        d0 = self.decoder0(
            d1,
            e0,
        )

        return self.output_head(
            d0
        )
