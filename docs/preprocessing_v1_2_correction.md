# CORA-Lung preprocessing correction v1.2

A CT-only audit was performed before opening any Gate-B dense development
outcome.

The audit confirmed two documented intensity encodings.

## Coronacases

These scans use raw HU-like intensities.

The v1.2 rule is:

1. clip to [-1250,250] HU;
2. resample on the frozen physical grid;
3. linearly map to [-1,1].

## Radiopaedia

These scans were already windowed to the same represented HU interval and
stored in [0,255].

The v1.2 rule therefore does not apply HU clipping or claim to reconstruct
original HU.

After resampling, values are mapped with:

    2*x/255 - 1

## Crop

The primary crop remains image-only.

Intensity and morphology thresholds now operate in the common normalized
space rather than assuming raw HU input.

A failed normalized-space lung extraction falls back to the image-derived
body bounding box.

No released lung mask or lesion mask repairs the crop.

## Sparse supervision

The physical/resampled image grid did not change.

Previously equalized sparse annotations therefore retain the same full-grid
coordinates. Only crop-relative coordinates are rebased when the corrected
image-derived crop changes.

No sparse label is regenerated from a dense mask.

## Invalidated pilot

The Block-08A checkpoints were produced with the superseded v1.1 intensity
rule and the superseded 0.60/0.20/0.20 patch-source proportions.

They were invalidated before dense Gate-B outcomes were opened and cannot be
used in Gate-B tables, figures or decisions.
