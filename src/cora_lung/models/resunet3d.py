"""Lightweight residual 3-D U-Net used by the CORA-Lung pilot."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNormAct(nn.Module):

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
            padding = (
                kernel_size
                // 2
            )

        else:
            padding = tuple(
                int(k) // 2
                for k in kernel_size
            )

        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.norm = nn.InstanceNorm3d(
            out_channels,
            affine=True,
        )

        self.act = nn.LeakyReLU(
            negative_slope=0.01,
            inplace=True,
        )

    def forward(
        self,
        x,
    ):
        return self.act(
            self.norm(
                self.conv(
                    x
                )
            )
        )


class ResidualBlock3D(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
    ):
        super().__init__()

        self.conv1 = ConvNormAct(
            in_channels,
            out_channels,
            kernel_size,
        )

        if isinstance(
            kernel_size,
            int,
        ):
            padding = (
                kernel_size
                // 2
            )

        else:
            padding = tuple(
                int(k) // 2
                for k in kernel_size
            )

        self.conv2 = nn.Conv3d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        self.norm2 = nn.InstanceNorm3d(
            out_channels,
            affine=True,
        )

        if (
            in_channels
            != out_channels
        ):
            self.skip = nn.Conv3d(
                in_channels,
                out_channels,
                kernel_size=1,
                bias=False,
            )

        else:
            self.skip = nn.Identity()

        self.act = nn.LeakyReLU(
            negative_slope=0.01,
            inplace=True,
        )

    def forward(
        self,
        x,
    ):
        residual = self.skip(
            x
        )

        x = self.conv1(
            x
        )

        x = self.norm2(
            self.conv2(
                x
            )
        )

        return self.act(
            x
            + residual
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


class DecoderBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
    ):
        super().__init__()

        self.reduce = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )

        self.block = ResidualBlock3D(
            out_channels
            + skip_channels,
            out_channels,
            kernel_size=3,
        )

    def forward(
        self,
        x,
        skip,
    ):
        x = F.interpolate(
            x,
            size=skip.shape[
                2:
            ],
            mode="trilinear",
            align_corners=False,
        )

        x = self.reduce(
            x
        )

        x = torch.cat(
            [
                x,
                skip,
            ],
            dim=1,
        )

        return self.block(
            x
        )


class CORALungResidualUNet(nn.Module):

    def __init__(
        self,
        in_channels=1,
        base_channels=16,
        embedding_dim=64,
    ):
        super().__init__()

        c1 = int(
            base_channels
        )

        c2 = (
            c1 * 2
        )

        c3 = (
            c1 * 4
        )

        c4 = (
            c1 * 8
        )

        self.enc1 = ResidualBlock3D(
            in_channels,
            c1,
            kernel_size=(
                1,
                3,
                3,
            ),
        )

        self.down1 = Downsample3D(
            c1,
            c2,
            stride=(
                1,
                2,
                2,
            ),
        )

        self.enc2 = ResidualBlock3D(
            c2,
            c2,
            kernel_size=3,
        )

        self.down2 = Downsample3D(
            c2,
            c3,
            stride=(
                2,
                2,
                2,
            ),
        )

        self.enc3 = ResidualBlock3D(
            c3,
            c3,
            kernel_size=3,
        )

        self.down3 = Downsample3D(
            c3,
            c4,
            stride=(
                2,
                2,
                2,
            ),
        )

        self.bottleneck = ResidualBlock3D(
            c4,
            c4,
            kernel_size=3,
        )

        self.dec3 = DecoderBlock(
            c4,
            c3,
            c3,
        )

        self.dec2 = DecoderBlock(
            c3,
            c2,
            c2,
        )

        self.dec1 = DecoderBlock(
            c2,
            c1,
            c1,
        )

        self.embedding_head = nn.Conv3d(
            c1,
            embedding_dim,
            kernel_size=1,
        )

        self.logit_head = nn.Conv3d(
            c1,
            1,
            kernel_size=1,
        )

    def forward(
        self,
        x,
    ):
        e1 = self.enc1(
            x
        )

        e2 = self.enc2(
            self.down1(
                e1
            )
        )

        e3 = self.enc3(
            self.down2(
                e2
            )
        )

        b = self.bottleneck(
            self.down3(
                e3
            )
        )

        d3 = self.dec3(
            b,
            e3,
        )

        d2 = self.dec2(
            d3,
            e2,
        )

        d1 = self.dec1(
            d2,
            e1,
        )

        embedding = self.embedding_head(
            d1
        )

        embedding = F.normalize(
            embedding,
            dim=1,
            eps=1e-6,
        )

        logits = self.logit_head(
            d1
        )

        return {
            "logits":
                logits,

            "embedding":
                embedding,
        }
