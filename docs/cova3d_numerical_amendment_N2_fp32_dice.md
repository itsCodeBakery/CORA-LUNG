# COVA-3D Numerical Amendment N2

## Trigger

The N1-corrected sanity run reached epoch 5 and 233 successful optimizer
updates before gradients became non-finite at AMP scale 1.

The exact failed model state was preserved and diagnosed before any dense
outcome was opened.

The failed logical pair contained two microbatches. One was an
all-background sparse patch with only two labelled background voxels.

## Exact-state diagnosis

BCE alone produced finite gradients.

Sparse Dice alone produced non-finite gradients in the original FP16
arithmetic.

The combined original objective therefore produced non-finite gradients.

Recomputing the sparse Dice arithmetic in FP32 while retaining the AMP
FP16 network forward produced finite gradients.

Full FP32 training also produced finite gradients.

The failure was therefore localized to FP16 sparse-Dice arithmetic.

## Numerical mechanism

For the all-background sparse patch, the learned probabilities at the two
labelled voxels were extremely small.

The FP16 Dice denominator approached the epsilon term of 1e-6.

Its reciprocal exceeded the representable finite FP16 range and became
infinity.

The corresponding FP32 reciprocal remained finite.

## Frozen N2 policy

The network forward remains under FP16 autocast.

Partial BCE is unchanged.

Sparse Dice uses the exact same equation and epsilon, but its arithmetic
is explicitly performed in FP32 with autocast disabled.

The total objective remains:

    partial BCE + partial Dice

with equal weights.

N1 remains active for ordinary mixed-precision loss-scale overflow.

## Restart

The R1 model is retired.

Its 233 executed optimizer updates are recorded for provenance but none
are retained in the clean scientific sanity run.

R2 will restart from the original frozen initialization.

No dense development outcome has yet been opened.

Final factorial training remains locked.
