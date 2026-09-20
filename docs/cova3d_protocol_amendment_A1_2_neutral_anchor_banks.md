# COVA-3D Protocol Amendment A1.2

## Geometry-neutral physical anchor banks

Amendment A1.1 used exactly one foreground anchor for every globally
resolution-limited lesion.

This created a mathematical contradiction in
`radiopaedia_29_86490_1`.

The 50% condition selects one lesion while the 100% condition selects two.
With one fixed anchor per lesion, C50 could contain at most one positive
supervision voxel while C100 required at least two. An equal foreground
annotation budget was therefore impossible.

This failure occurred before any COVA-3D training.

A pre-outcome diagnostic then measured the physical training-grid support
of all 28 globally resolution-limited lesions.

Every limited lesion retained at least 17 unique grid voxels.
The median support was 39.0 voxels and the maximum was
101 voxels.

Allowing these lesions to use a geometry-neutral ordered physical anchor
bank made the aggregate budget interval feasible in all 20
primary cases.

## Frozen A1.2 rule

Globally resolution-limited lesions still have no coherent, dispersed or
fragmented geometry claim.

They instead have a deterministic physical anchor bank.

The bank consists of unique training-grid voxels occupied by the physical
lesion component. Voxels are ordered by increasing world-space distance
from the physical native-component centroid. Ties are resolved
lexicographically by crop-grid z, y and x.

The minimum quota is one voxel.

The maximum quota is the mapped physical support capacity.

For quota q, the first q voxels of the ordered bank are used.

The same ordered bank is used for all geometry conditions and all coverage
conditions in which the lesion appears.

Different coverage conditions may use different prefix lengths because
fixed total annotation budget is redistributed across different numbers of
lesion instances. The prefixes remain nested.

Geometry-resolvable lesions retain the original geometry manipulation and
require at least two annotation voxels.

## Budget rule

For each patient, the final B_i_train is the largest common foreground
budget that can be constructed in all six factorial cells.

Component allocation begins at the frozen minimum quotas. Remaining quota
is assigned deterministically in selected-component order, one voxel at a
time while capacity remains.

The final construction must also pass coordinate-level collision tests.
Therefore the 20/20 result obtained before A1.2 is an aggregate capacity
result, not yet the final annotation-feasibility result.

Block 09D-B2 must verify the complete coordinate-level construction.

Training remains unauthorized.
