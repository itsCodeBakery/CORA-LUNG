# COVA-3D Protocol Amendment A1

## Resolution-aware sparse annotation geometry

Block 09D-A was performed before any COVA-3D optimizer step.

The diagnostic showed that the original training-grid rule was feasible for
only 7/20 cases.

No lesion component disappeared completely after grid transfer.

However, 18/308 C100 lesion components
could not retain a meaningful two-or-more-voxel geometry representation at
the frozen 3.0 x 1.5 x 1.5 mm training resolution. These components occurred
in 11/20 cases. Their median physical lesion volume was
0.1409 mL.

The original rule would therefore remove many otherwise valid cases because
of the representation limit of the training grid rather than because the
lesions themselves were absent.

## Amended rule

A selected lesion is classified as **geometry-resolvable** when its common
mapped geometry capacity is at least two voxels.

These lesions retain the original coherent, dispersed and fragmented
annotation manipulation.

A selected lesion is classified as **resolution-limited** when it retains at
least one mapped foreground voxel but cannot represent all three geometry
conditions with at least two distinct voxels.

A resolution-limited lesion remains part of the coverage factor. It receives
one identical geometry-neutral anchor in all three geometry conditions.

Thus the amendment preserves lesion-instance coverage while refusing to claim
a geometry manipulation that cannot physically exist at the frozen training
resolution.

No selected lesion is discarded.

## Primary analysis

The primary endpoint remains macro patient lesion recall at <=1 false-positive
component per patient.

The coverage factor remains unchanged.

The geometry factor is interpreted as manipulation of annotation geometry
where that geometry is physically resolvable.

Resolution-limited lesions remain in the overall lesion-recovery endpoint.

Secondary results will also stratify lesion recovery by resolution class and
lesion volume.

## Next required test

Block 09D-B must demonstrate that every resolution-limited component has a
deterministic common anchor shared across all three geometry conditions.

It must then prove exact six-cell foreground/background matching for all
20 cases and freeze the final B_i_train.

Training remains prohibited until that audit passes.
