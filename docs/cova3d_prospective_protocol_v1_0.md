# COVA-3D: Coverage–Coherence Trade-offs in Fixed-Budget Sparse Supervision for 3-D Multi-Lesion Segmentation

**Track:** COVA3D  
**Protocol version:** 1.0  
**Frozen:** 2026-09-14T08:33:40Z  
**Predecessor:** CORA-Lung  
**Predecessor outcome:** Gate B = NO-GO; Gate C was never run.

---

## 1. Scientific lineage

COVA-3D is a new prospective research track motivated by, but analytically
separate from, the completed CORA-Lung pilot.

The predecessor experiment tested whether omission of whole lesion components
caused a larger lesion-recovery deficit than matched random-pixel thinning.
That directional premise was not supported. The locked Gate-B result was
NO-GO, and a subsequent dynamic-topology diagnostic also failed to reveal the
hypothesized deficit.

Those results are preserved as historical pilot evidence. They are not reused
as confirmatory evidence for COVA-3D.

No CORA checkpoint is automatically eligible as a COVA-3D model.

---

## 2. Research question

> **Under an equal sparse-annotation budget, how do lesion-instance coverage and within-instance annotation coherence independently and jointly affect 3-D multi-lesion segmentation performance and recovery of unannotated lesions?**

Only one primary research question is registered.

---

## 3. Scientific factors

### Factor A — lesion-instance coverage

Two levels are registered:

1. **50% coverage:** annotate `ceil(0.5 K_i)` eligible lesion components,
   with a minimum of one.
2. **100% coverage:** annotate all `K_i` eligible lesion components.

The 50% component set must be a deterministic nested subset of the 100% set.

Selection is uniform without replacement from a seeded permutation and must
not use lesion difficulty, model prediction, model confidence, or future
outcomes.

### Factor B — within-instance annotation geometry

Three levels are registered:

1. **Coherent:** connected interior stroke/path.
2. **Dispersed:** deterministic spatially separated points.
3. **Fragmented:** multiple separated pieces derived from a coherent interior
   path.

The exact coordinate algorithms will be implemented and frozen in Block 09C
before training.

---

## 4. Primary factorial cells

| Condition | Coverage | Geometry |
|---|---:|---|
| C50_COH | 50% | Coherent |
| C50_DIS | 50% | Dispersed |
| C50_FRG | 50% | Fragmented |
| C100_COH | 100% | Coherent |
| C100_DIS | 100% | Dispersed |
| C100_FRG | 100% | Fragmented |

These six cells are the primary experiment.

---

## 5. Equal-budget causal matching

For patient/case `i`, define `B_i` as the number of unique foreground
annotation voxels.

The primary requirement is:

`N_FG(i, condition) = B_i`

for all six conditions.

`B_i` will be selected before any COVA-3D training as the largest common
feasible unique-positive budget that can be instantiated under every primary
cell for that case.

If no common feasible budget exists, the case is not repaired using outcomes.
Its eligibility is handled prospectively in the feasibility audit.

Within a fixed coverage level:

- selected lesion components are identical across geometries;
- per-component positive-voxel quotas are identical across geometries;
- only annotation spatial organization changes.

Across all six cells:

- background supervision coordinates are identical;
- CT image and preprocessing are identical;
- crop is identical;
- model architecture is identical;
- optimizer is identical;
- model-seed-specific patient and augmentation schedules are paired.

This design is intended to isolate coverage and geometry rather than annotation
quantity.

---

## 6. Component eligibility

Primary lesion references use:

- 26-connectivity;
- physical lesion volume >= 0.10 mL.

A case requires at least two eligible lesion components to enter the primary
coverage factorial experiment.

---

## 7. Primary endpoint

The primary endpoint is:

**macro patient lesion recall at <= 1 false-positive
component per patient (R@1).**

Prediction components are recomputed at each frozen voxel-probability
threshold. This is deliberately different from the historical CORA Gate-B
fixed-p=0.5 topology.

Primary matching uses:

- 26-connected prediction components;
- no prediction-size removal;
- one-to-one maximum-cardinality matching;
- maximum-total-IoU tie break;
- admissible match IoU >= 0.10;
- references < 0.10 mL excluded from the
  primary reference set;
- predictions matching only such small excluded references are neutral rather
  than counted as false positives.

Secondary FROC budgets are 0.5, 2 and 4 FP/patient.

---

## 8. Primary estimands

Let `R(c,g)` denote macro R@1 for coverage `c` and geometry `g`.

### Coverage effects

`R(100%, g) - R(50%, g)` for every geometry.

### Geometry effects

At each coverage level:

`R(c, dispersed) - R(c, coherent)`

and

`R(c, fragmented) - R(c, coherent)`.

### Interaction contrasts

Dispersed interaction:

`[R(100%,dispersed)-R(100%,coherent)] -
 [R(50%,dispersed)-R(50%,coherent)]`

Fragmented interaction:

`[R(100%,fragmented)-R(100%,coherent)] -
 [R(50%,fragmented)-R(50%,coherent)]`

No interaction direction is hypothesized prospectively.

---

## 9. Primary hypothesis

### Null

Coverage and within-instance annotation geometry do not interact with respect
to lesion-recovery performance.

### Alternative

At least one geometry contrast changes as lesion-instance coverage changes.

The hypothesis is two-sided.

The previous CORA pilot observed an unexpected ordering and therefore cannot be
used to justify a directional COVA-3D hypothesis.

---

## 10. Development stage

The existing four permanent development volumes remain development-only.

Initially:

- one model seed: 17;
- no final outer-CV outcome access;
- no method invention;
- no architecture comparison inside the factorial experiment.

A strong 3-D sparse-supervision baseline must first pass a separate sanity
gate.

The current CORA custom U-Net is not automatically accepted as the primary
COVA-3D baseline.

---

## 11. Baseline requirement

A strong 3-D nnU-Net-family sparse-supervision baseline will be frozen before
factorial fitting.

The baseline must demonstrate non-pathological dense segmentation on the
development cohort before the six-condition causal experiment is interpreted.

The exact baseline implementation and numerical sanity criteria are frozen in
Block 09E before factorial outcomes are generated.

---

## 12. Development signal rule

For method-development purposes only, a prospective interaction signal is:

- absolute interaction contrast >= 0.05 in macro
  R@1; and
- same-sign case-level interaction in at least
  3/4 development cases.

This is not final confirmatory evidence.

If no interaction exists but a reproducible main effect exists, a later method
may target that main effect but may not claim an interaction mechanism.

If neither meaningful interaction nor meaningful main effects emerge, method
development stops.

---

## 13. Final-stage requirements

The final stage will require:

- model seeds 17, 29 and 43;
- the currently sealed outer-CV cohort;
- source-subject-level independence;
- cluster bootstrap confidence intervals;
- 10,000 bootstrap replicates;
- bootstrap seed 20260914;
- Holm correction for the two primary interaction contrasts;
- at least one additional 3-D multifocal lesion dataset before a broad
  cross-domain claim.

The current 20-volume lung cohort alone is not sufficient for a broad general
claim about sparse annotation design.

---

## 14. Annotation simulator firewall

Dense masks may be used offline to simulate annotations.

Dense masks must never enter:

- training cache;
- optimizer-bound batches;
- model inputs;
- model-selection logic on the final set.

The existing dense-checksum denylist/firewall remains applicable.

---

## 15. Frozen execution sequence

1. **09A — prospective protocol freeze** — this block.
2. **09B — dataset suitability and factorial feasibility audit.**
3. **09C — annotation simulator v2 implementation.**
4. **09D — exact causal-matching QA.**
5. **09E — strong sparse baseline + baseline sanity gate.**
6. **Block 10 — six-condition development factorial experiment.**
7. Review main effects and interactions.
8. Only then decide whether any adaptive method should be invented.
9. Final outer-CV and second-dataset confirmation only after the method and
   final protocol are frozen.

---

## 16. Prohibited actions at protocol freeze

Until explicitly authorized:

- no COVA-3D model training;
- no six-cell factorial fitting;
- no COVA-3D method development;
- no final outer-CV dense outcome access;
- no budget selection from model outcomes;
- no factor-level changes after seeing COVA-3D results;
- no reinterpretation of the historical CORA Gate-B result.

---

## 17. Status after Block 09A

**CORA-Lung:** closed as a NO-GO pilot.  
**COVA-3D protocol:** frozen prospectively.  
**COVA-3D training:** not authorized.  
**Next action:** Block 09B.
