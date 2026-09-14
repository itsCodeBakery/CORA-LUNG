# COVA-3D Block 09B — Dataset Suitability and Structural Factorial Feasibility

## Primary cohort

The existing lung CT cohort contains 20 volumes.

Using the already frozen lesion definition:

- 26-connectivity;
- lesion component volume >= 0.1 mL;

all 20/20 volumes contain at
least two eligible lesion components and therefore satisfy the structural
minimum for comparing nominal 50% versus 100% lesion-instance coverage.

The current cohort contains 308 eligible components at the
per-volume level.

Under the prospectively frozen count rule:

`K_50(i) = ceil(0.5 K_i)`

the cohort contributes:

- 159 selected lesion components at nominal 50% coverage;
- 308 selected lesion components at 100% coverage.

The four permanent-development cases contain:

- 82 eligible components at 100% coverage;
- 42 selected components under the nominal 50% rule.

No source-subject group crosses development and final roles.

## Coverage discreteness

Nominal 50% coverage is a component-count factor.

Because lesion counts are discrete, `ceil(0.5 K)` is not exactly 0.50 for
every odd-valued K.

The primary protocol is not changed after this audit.

A pre-outcome sensitivity analysis restricted to cases with `K >= 4` is now
registered to examine whether extremely low component counts influence the
interpretation of the nominal coverage factor.

This sensitivity analysis cannot replace the primary analysis.

## Coordinate-level annotation feasibility

Block 09B establishes structural feasibility only.

It does **not** establish that coherent, dispersed and fragmented geometry can
all realize an identical unique-foreground budget `B_i`.

That question is reserved for Block 09C, where the three annotation algorithms
will be implemented using dense masks offline and the largest common feasible
budget will be determined prospectively.

No COVA training is authorized by this block.

## Second dataset

The priority cross-organ candidate is:

**Medical Segmentation Decathlon Task03 Liver — Liver Tumours**

Official metadata:

- modality: Portal venous phase CT;
- target: Liver and tumour;
- total volumes: 201;
- labelled training volumes: 131;
- test volumes: 70;
- license: CC-BY-SA 4.0.

It is only a candidate.

The published dataset description does not establish that enough training
cases contain multiple tumour components for the COVA factorial experiment.
A local NIfTI component audit is mandatory before the dataset can be admitted
to the final study.

The original NIfTI/physical-geometry representation is preferred over
pre-normalized convenience conversions.

## Status

Primary structural feasibility: **PASS**

Primary coordinate-level six-cell feasibility: **PENDING 09C**

Secondary dataset inclusion: **PENDING LOCAL AUDIT**

COVA training authorization: **NO**
