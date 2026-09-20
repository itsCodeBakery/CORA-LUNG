# COVA-3D Block 09D-B2

## Final coordinate-level training-grid construction

Effective protocol: `COVA3D_1.0+A1+A1.1+A1.2`.

Block 09D-B2 is the final structural annotation-feasibility audit before
baseline training is specified.

Globally geometry-resolvable lesion components retain the coherent,
dispersed and fragmented supervision manipulations.

Globally resolution-limited lesion components use deterministic
geometry-neutral physical anchor banks.

Each bank is formed from unique training-grid voxels occupied by the
physical lesion component.

Bank voxels are ordered by world-space distance from the native physical
lesion centroid. Lexicographic z-y-x order resolves exact ties.

For an assigned quota q, the first q bank voxels are used.

The same bank is used across geometry and coverage conditions.
Coverage-specific quota differences therefore produce nested prefixes
rather than different annotation locations.

The final patient-specific B_i_train is the largest foreground budget
that passes actual coordinate-level construction in all six factorial
cells.

The final construction requires equal FG counts, identical BG
coordinates, matched component sets and matched per-component quotas
across geometry.

Cross-component foreground collisions and foreground/background
collisions are prohibited.

Dense lesion masks are used only offline to derive neutral anchor banks.
They are never included in trainer artifacts.

Block 09D-B2 passed all 20 primary cases.

This establishes structural readiness of the COVA-3D annotation
methodology on the primary dataset.

Training is still not authorized.

The next block is 09E-LOCK. It must prospectively freeze the strong
sparse-supervision baseline, sparse loss, optimizer, schedule,
development-only sanity gate and authorization rules before the first
COVA-3D optimizer step.
