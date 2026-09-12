# CORA-Lung Training-Grid Preprocessing Protocol

## Accepted version

**Version:** 1.1

**Parent version:** 1.0

**Correction made before model training:** Yes

## Image preprocessing

The image pipeline is unchanged from preprocessing v1.0:

- canonical NIfTI orientation;
- target spacing 3.0 × 1.5 × 1.5 mm in `(z,y,x)`;
- linear CT interpolation;
- HU clipping to `[-1000, 400]`;
- normalization to `[-1, 1]`;
- CT-image-only thoracic crop.

All twenty v1.1 cached CT images are byte-for-byte identical to their
v1.0 counterparts.

## Reason for the v1.1 amendment

Sparse labels were originally matched on the native annotation grid.

After physical-coordinate transfer to the coarser model grid, nearby
same-class labelled voxels can collapse to the same unique model voxel.

This changed the effective causal annotation budget received by the network.

The raw v1.0 training cache therefore failed causal-budget equivalence despite
passing geometry, foreground-group survival, background identity and firewall QA.

## Feasibility audit

Block 06A was performed before any v1.1 equalization.

It established constructive feasibility for:

- 60/60 natural-component vs random-pixel pairs;
- 60/60 fixed-budget pairs;
- 20/20 primary 50% natural/pixel comparisons;
- 20/20 primary 50% fixed comparisons.

No dense masks were accessed.

## Natural component omission

Natural component-omission conditions are unchanged from v1.0.

Missing-component identities are unchanged.

The nested 25%, 50%, and 75% coverage hierarchy is unchanged.

## Random-pixel control

The target is the exact unique-FG count of the paired natural component-omission
annotation after transfer.

If the raw transferred control already satisfies that target it is unchanged.

Otherwise a deterministic group-preserving subset is selected from the
transferred component-complete unique foreground support.

## Fixed-budget pair

For every component-fixed / complete-fixed pair:

`target = min(unique FG capacity of the two transferred conditions)`

The side already at the target is unchanged.

Only an oversized side is thinned.

All foreground groups represented by that condition are preserved.

## Background

Explicit-background coordinates are never changed.

The v1.1 background coordinate set is verified identical to the corresponding
v1.0 background set for every case and condition.

## Final validation

- condition matrix: 260/260;
- natural/pixel exact causal pairs: 60/60;
- fixed-budget exact causal pairs: 60/60;
- primary 50% natural/pixel: 20/20;
- primary 50% fixed pair: 20/20;
- component-complete annotations unchanged: yes;
- natural component-omission annotations unchanged: yes;
- background coordinates unchanged: yes;
- foreground-group loss: none;
- dense masks accessed during v1.1 equalization: no;
- model training performed before correction: no.
