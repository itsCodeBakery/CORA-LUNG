# COVA-3D Baseline and Training Protocol v1.0

## Purpose

The sparse annotation protocol was completed before model development.

The primary experiment tests annotation coverage and annotation geometry.
The segmentation network is therefore intentionally kept as a strong,
conventional 3-D baseline rather than introducing a new architecture.

## Baseline

The model is a five-stage anisotropic nnU-Net-style 3-D U-Net.

Feature widths are 32, 64, 128, 256 and 320.

The first encoder kernel is 1x3x3. Later kernels are 3x3x3.

The first downsampling operation is 1x2x2. Later downsampling operations
are 2x2x2. This respects the frozen 3.0x1.5x1.5 mm grid.

Each stage contains two convolutions with affine instance normalization
and LeakyReLU.

Deep supervision is disabled because downsampling extremely sparse labels
would add a second annotation-remapping operation to the factorial
experiment.

## Sparse loss

Unknown voxels have label -1 and are ignored.

The objective is partial BCE plus partial Dice with equal weights.

No dense mask, pseudo-label, teacher/student loss, consistency loss,
component replay or other auxiliary objective is used.

## Patch protocol

Patch size is 48x128x128 voxels.

Patients are sampled uniformly.

Patch-source probabilities are:

- foreground-supervision centered: 0.50
- background-supervision centered: 0.25
- uniform crop: 0.25

## Optimization

AdamW is used with learning rate 3e-4 and weight decay 1e-4.

Gradient accumulation is two microbatches.

AMP float16 is enabled.

Learning rate uses 100 optimizer-step linear warmup followed by cosine
decay to 3e-6.

No early stopping is allowed.

Only the final epoch checkpoint is eligible.

## Paired randomization

Within each fold and seed, all six annotation conditions receive identical
model initialization.

Patient-order randomness is paired across conditions.

Augmentation randomness is paired across conditions.

The annotation coordinates themselves remain condition-specific because
they are the scientific intervention.

## Development sanity gate

The sanity model uses C100_COH, seed 17 and only the four permanent
development volumes.

It runs for 20 epochs and 1000 optimizer
steps.

Dense masks are not accessed during fitting.

The final checkpoint is evaluated once against the dense permanent-
development masks.

The sanity gate is intended only to detect a broken trainer or degenerate
inference pipeline. It is not a hyperparameter-selection stage.

If the gate fails, factorial training remains blocked. Only implementation
defects may be corrected. Model or hyperparameter changes require a new
prospective protocol amendment.

The sanity checkpoint is never reused in the final experiment.

## Final factorial experiment

Final evaluation uses four frozen outer folds.

Each model trains on 12 volumes and predicts four held-out volumes.

There are six annotation conditions and three seeds: 17, 29 and 43.

This yields 72 independently initialized final fits.

Each fit uses 40 epochs, 120 microbatches per epoch and
2400 optimizer updates.

All final models and predictions must be frozen before dense final
outcomes are opened.

## Current authorization

Sanity training is authorized after this lock.

Factorial training is not authorized.

Final-outcome evaluation is not authorized.
