# COVA-3D Protocol Amendment A1.1

## Cross-coverage-stable resolution classification

Amendment A1 introduced a resolution-aware rule before any COVA-3D
training.

A subsequent audit of the already archived feasibility table showed
10 lesion components whose resolution class differed
between their 50% and 100% coverage realizations.

Allowing that difference would make the availability of the geometry
manipulation partly dependent on the coverage factor itself.

Therefore A1.1 freezes a case-component resolution class that is invariant
across coverage.

A lesion is globally geometry-resolvable only when every available
coverage realization is geometry-resolvable.

A lesion is globally resolution-limited when any available realization is
resolution-limited.

This conservative rule produces 28/308
globally resolution-limited lesions across 13/20 cases.

No lesion is coverage-lost.

Globally resolution-limited lesions receive a single geometry-neutral
anchor in every factorial cell in which they are selected. The exact same
anchor will be reused across geometry and coverage.

Block 09D-B will derive this anchor from the physical lesion component on
the frozen training grid rather than from a condition-specific candidate
annotation. This prevents the anchor location itself from becoming a
coverage-dependent intervention.

Geometry-resolvable lesions retain the coherent, dispersed and fragmented
manipulation.

No model outcomes, predictions or optimizer steps were used to define this
clarification.

Training remains prohibited until Block 09D-B proves exact six-cell
feasibility.
