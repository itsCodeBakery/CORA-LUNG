# CORA-Lung Weak-Label Generation Protocol

## Status

**Generator version:** 1.1

**Generator seed:** 20260912

**Frozen before Gate B training:** Yes

## Dense-supervision boundary

Dense infection masks and released lung masks are accessible only to the isolated
weak-label generator.

The model trainer receives no:

- dense lesion mask;
- dense lesion-component map;
- omitted-component coordinates;
- lesion centroid;
- lesion volume;
- full-mask-derived distance map.

Unspecified image voxels remain unknown.

They are never silently converted to background.

## Native component definition

- 26-connected infection-mask components.
- Minimum physical component volume: 0.1 mL.
- Scribbles originate on the native annotation grid.

## Foreground scribbles

Each eligible observed lesion component contributes one foreground stroke/group.

The generator uses:

- 1.0-mm physical interior erosion;
- 3-D skeletonization;
- connected skeleton-path extraction;
- deterministic randomized contiguous path selection;
- persistent positive group identifier.

The frozen base engineering budget is **20 native voxels per observed component**,
subject to available connected path length.

## Background scribbles

Background tracks are generated from released lung tissue outside a **3-mm physical
safety band** around annotated lesions.

Released lung masks are simulator-only inputs and are not trainer inputs.

Exactly the same background coordinate realization is reused for every paired condition
of a given volume.

## Component coverage

Coverage levels:

- 100%;
- 75%;
- 50%;
- 25%.

The retained component count is:

`max(1, ceil(rho * K))`

A deterministic component ordering is generated once per case, producing nested coverage:

25% subset of 50%, 50% subset of 75%, and 75% subset of the complete annotation.

## Natural component omission

Omitted lesion groups lose their foreground stroke completely.

Removed foreground-labelled voxels are not redistributed.

## Random-pixel causal control

The random-pixel control attempts to exactly match the foreground-labelled voxel count of
the corresponding natural component-omission condition while retaining at least one
foreground-labelled voxel for every eligible lesion component.

When the reduced foreground budget is smaller than the total number of eligible
components, exact matching is mathematically impossible without either duplicating voxels
or dropping an entire component.

Such secondary-coverage instances are recorded explicitly and excluded from that paired
pixel-control comparison.

The **primary 50% Gate-B comparison requires complete exact matching for all study volumes.**

## Fixed-positive-pixel component omission

Only retained lesion groups may receive foreground annotation.

Foreground budget is redistributed over connected paths of retained components.

When retained-path capacity is insufficient for the complete foreground budget, a lower
shared pixel budget is used for both:

1. the incomplete component-omission annotation; and
2. its component-complete paired control.

No voxel is duplicated to manufacture a matched budget.

## Trainer-safe sparse representation

Each sparse annotation artifact contains only:

- `voxel_ijk`;
- `world_xyz_mm`;
- sparse binary `label`;
- foreground `group_id`.

Explicit background uses `group_id = -1`.

Unknown voxels are absent from the sparse coordinate table.

## Reproducibility

Patient sparse-coordinate arrays are intentionally not committed to ordinary Git history.

They are reproducible using:

- frozen source dataset hashes;
- frozen primary split;
- generator configuration;
- generator seed;
- executable generator source;
- committed semantic SHA-256 annotation hashes.

## Block-05 outcome

Generator audit status:

**PASS**

Primary 50% component/pixel budget match:

**PASS**

Primary 50% fixed component/complete budget match:

**PASS**

Identical paired background coordinates:

**PASS**


## Generator v1.1 allocator correction

Before any model training, the Block-05 audit detected an implementation-only failure in
`coronacases_003` at the secondary 25% fixed-pixel condition.

The retained ten groups had exactly **376 voxels** of allowed connected-path capacity, and
the requested shared budget was also **376 voxels**. The comparison was therefore
mathematically feasible.

Generator v1.0 used a round-robin allocator with an arbitrary iteration safety threshold.
For this case it stopped at iteration 2505 when the safety threshold was 2504, with five
valid allocations still remaining.

Generator v1.1 removes the arbitrary iteration cutoff. It preserves the same deterministic
shuffled cyclic group visitation order and terminates only when:

1. the exact requested budget is allocated; or
2. a complete pass makes no progress, indicating true infeasibility.

The original executed v1.0 generator is preserved under
`scripts/code_blocks/archive/`.

No model had been trained when this correction was made.

After correction:

- all **260/260** expected weak-supervision artifacts are present;
- all **60/60** natural-component vs random-pixel budget pairs match exactly;
- all **60/60** component-fixed vs complete-fixed budget pairs match exactly;
- all paired background coordinate realizations are identical;
- 25%, 50%, and 75% component missingness remains nested;
- the dense-label firewall passes.
