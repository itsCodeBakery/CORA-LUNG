# CORA-Lung Pretraining Safety Closure

Block 07D is the final no-training safeguard before Gate B.

## Native voxel-volume audit

All 20 primary CT NIfTI headers and all 60 associated mask headers were
inspected without loading dense voxel arrays.

For every file, native voxel volume computed from the product of the first
three header zooms was compared with the absolute determinant of the spatial
3x3 affine matrix.

The relative tolerance is 1e-6.

Paired CT/mask native voxel volume must also agree within the same tolerance.

## Optimizer-boundary firewall

The future scientific trainer must call:

`guard_optimizer_batch(batch, dense_sha256_denylist)`

before consuming every optimizer-bound batch.

The firewall requires an exact sparse-batch schema and rejects:

- unexpected keys;
- dense-label terminology;
- known dense source-file SHA-256 checksums;
- invalid target values;
- inconsistent foreground replay membership;
- invalid membership coordinates.

The dense SHA-256 denylist is derived from the already committed frozen source
checksum manifest. Dense source files are not reopened to construct it.

## Scientific status

No model fitting and no optimizer update occur in Block 07D.

Gate B remains not run until this audit is manually accepted.
