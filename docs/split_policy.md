# CORA-Lung Primary Split Policy

## Frozen decision

**Split seed:** `1701`

**Independence unit:** `source_subject_key`

The primary data release contains **20 CT volumes** but only **18 conservative
source groups** because two filename-derived Radiopaedia source groups contain
two series each.

Until external provenance proves otherwise, series belonging to the same
source group are treated as potentially non-independent.

## Leakage rule

All volumes belonging to one `source_subject_key` must remain in exactly one:

- permanent development set; or
- final outer fold.

They may never cross development/training/test roles.

## Development set

Four singleton source groups are selected before model development:

- two Coronacases source groups;
- two Radiopaedia source groups.

Selection uses only provenance category, singleton/repeated status and the
fixed random seed.

No lesion mask, component count, lesion volume, contrast, model prediction or
evaluation outcome is used.

## Final outer cross-validation

The remaining sixteen image volumes are assigned to four outer folds containing
exactly four volumes each.

Whole source groups are allocated together. Because two final source groups
contain two image series, the number of independent conservative source groups
per outer fold may be three or four.

Fold balancing uses source-group **volume count only**, never dense lesion
information.

## Statistical unit

For confirmatory patient/source-level analyses:

- `source_subject_key` is the conservative clustering unit;
- repeated image series are not counted as independent patients;
- repeated-volume measurements are aggregated within source group before
  macro source/patient statistics or bootstrap resampling.

## Preregistration amendment

The original planning document described 20 cases as if they were 20 independent
patients. The provenance audit found 18 conservative source groups.

This amendment is made **before scribble generation, model fitting or test
outcome inspection**.

It is therefore a leakage-control correction, not a result-driven protocol change.

## Redraw policy

The split is now frozen.

Component counts may be summarized after freezing for descriptive feasibility,
but they cannot be used to redraw the development set or outer folds.

Model performance can never trigger a split redraw.
