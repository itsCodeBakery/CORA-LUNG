# COVA-3D Numerical Amendment N1

## Trigger

The first COVA-3D sanity-fit attempt stopped on its first accumulated
gradient pair before the first optimizer update.

Forward losses were finite, but FP16 scaled gradients were non-finite at
the default loss scale of 65536.

A pre-outcome numerical diagnostic found non-finite gradients at scales
65536, 32768, 16384 and 8192. Gradients were finite from scale 4096 down
to scale 1.

Therefore the failure was classified as loss-scale overflow rather than
intrinsic model or loss instability.

No dense development outcome was opened. No final-CV outcome was opened.
No successful COVA optimizer update preceded this amendment.

## Frozen N1 policy

FP16 autocast is retained.

GradScaler starts at 4096.

The growth factor is 2 and the backoff factor is 0.5.

The growth interval is 1,000,000 successful optimizer steps, preventing
upward loss-scale growth during the frozen sanity and final runs.

After gradients are unscaled, their finiteness is checked before
gradient clipping.

If overflow occurs, GradScaler skips the update and backs off the scale.
The same logical two-microbatch accumulation pair is then recomputed.

A logical optimizer step is counted only after a finite, successful
weight update.

If gradients remain non-finite at scale 1, training stops.

## Scientific protocol

N1 does not alter the COVA-3D architecture, sparse loss, annotation
conditions, optimizer, learning rate, scheduler, sampling, augmentation,
patch size, accumulation factor, epoch count, checkpoint rule, or final
factorial experiment.

Factorial training remains locked.
