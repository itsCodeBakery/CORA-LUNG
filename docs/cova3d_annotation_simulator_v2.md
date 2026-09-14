# COVA-3D Block 09C — Annotation Simulator v2

## Status

Native-space six-cell annotation feasibility: **PASS**

Trainer-grid feasibility: **PENDING BLOCK 09D**

No model training has occurred.

## Factorial construction

For each case with `K` eligible lesion components:

- 50% coverage selects `ceil(0.5 K)` components;
- 100% coverage selects all `K`;
- the 50% set is a seeded nested subset of the 100% set.

Three within-instance geometries are constructed:

1. coherent;
2. dispersed;
3. fragmented.

Within a coverage level, all three geometries use the same selected
components and the same per-component positive quotas.

Across all six cells, the total foreground count equals a common
case-specific native budget `B_i_native`.

All six cells also reuse the identical background coordinates.

## Geometry definitions

**Coherent** annotations are deterministic 26-connected compact interior
traces.

**Dispersed** annotations use physical farthest-point sampling.

**Fragmented** annotations use two or three separated compact fragments.

All geometries are generated from the same component-specific candidate pool.

## Native annotation budget

A maximum of 20 positive voxels per
selected component is permitted at the native-candidate stage.

At least 3 voxels per selected
component are required so all three geometries remain meaningful.

The largest feasible exact common native budget is frozen independently for
each case.

## Critical training-grid safeguard

These artifacts are **not trainer eligible**.

Sparse points that are unique in native space may map to the same voxel after
resampling to the COVA training grid. Coherent annotations are particularly
susceptible because adjacent native voxels may collapse.

Therefore Block 09D must:

1. reconstruct the frozen corrected preprocessing/training grid;
2. map every candidate point by physical coordinates;
3. measure unique FG/BG counts after transfer;
4. verify that geometry conditions remain budget matched;
5. if necessary, lower the shared budget using a deterministic
   transfer-feasibility rule;
6. freeze `B_i_train`;
7. only then build trainer-eligible sparse caches.

No model outcome may influence that reduction.

## Firewall

Dense infection/lung masks were accessed only as offline annotation sources.

CT voxel arrays were not read.

Model predictions/checkpoints were not read.

Final outer-CV model outcomes remain unopened.

## Next

Block 09D — exact training-grid causal-matching and collision audit.
