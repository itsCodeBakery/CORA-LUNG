# CORA-Lung

## Component-Omission Replay for Incomplete Scribble-Supervised Lung Lesion Segmentation

CORA-Lung investigates lung-lesion segmentation when entire lesion components may receive
no foreground scribble during weakly supervised training.

The central experimental question is whether omission of complete lesion components creates
a failure mode that is different from simply reducing the number of labelled pixels, and
whether component-omission replay improves lesion recovery at a controlled false-positive
burden.

---

## Core Experimental Comparison

The project will explicitly compare:

1. component-complete scribbles;
2. random-pixel dropout;
3. whole-component dropout;
4. partial-loss segmentation;
5. ordinary contrastive learning;
6. component-omission replay.

The random-pixel and whole-component conditions will be matched by foreground-labelled
voxel budget whenever feasible.

---

## Dense-Label Firewall

Dense lesion masks are permitted only for:

- isolated weak-label generation;
- dataset auditing;
- final evaluation.

Dense lesion masks are forbidden from gradient-based model fitting.

Training data exports will contain only permitted weak-supervision information.

---

## Repository Structure

    CORA-LUNG/
    ├── app/                 Future practical application
    ├── assets/              Repository assets
    ├── configs/             Version-controlled experiment configs
    ├── data/
    │   ├── manifests/       Dataset manifests/checksums only
    │   └── splits/          Frozen patient splits
    ├── docs/                Scientific and engineering documentation
    ├── environment/         Runtime and package provenance
    ├── experiments/         Experiment registry
    ├── figures/             Publication-ready aggregate figures
    ├── notebooks/           Kaggle notebook stages
    ├── outputs/             Runtime outputs
    ├── scripts/             Recovery and utility scripts
    ├── src/cora_lung/       Reusable source package
    └── tests/               Scientific correctness tests

---

## Notebook Sequence

| Notebook | Purpose |
|---|---|
| 00 | Environment and provenance |
| 01 | Dataset viability and preprocessing |
| 02 | Weak-label generation and dense-label firewall |
| 03 | Partial-loss and missingness baselines |
| 04 | CORA-Lung training |
| 05 | Inference and 3-D component matching |
| 06 | Statistics, tables and publication figures |
| 07 | Locked MosMed external validation |

---

## Scientific Gates

### Gate A — Data Viability

Confirm that enough genuine 3-D lesion components and multifocal cases exist.

### Gate B — Problem Validity

Demonstrate that whole-component omission creates a reproducible deficit beyond
matched random-pixel dropout.

### Gate C — Mechanism

Demonstrate that component-omission replay provides signal beyond partial loss and
ordinary contrastive learning.

### Gate D — Compute

Verify that the accepted implementation fits the declared Kaggle resource envelope.

### Gate E — Primary Evaluation

Evaluate held-out lesion-component recovery at standardized false-positive burden.

### Gate F — External Validation

Evaluate the frozen method on MosMed without target-domain retuning.

---

## Reproducibility

Every important run will record:

- Git commit;
- configuration hash;
- dataset manifest hash;
- patient split;
- fold;
- random seed;
- GPU;
- package versions;
- checkpoint rule;
- run status;
- result-file hashes.

Large medical images and model checkpoints are not stored in ordinary Git history.

---

## Figure Policy

Publication figures will use:

- meaningful technical titles;
- bold titles;
- bold axis labels;
- clear legends;
- explicit physical units;
- publication-quality resolution;
- vector PDF/SVG where appropriate;
- high-resolution PNG where raster output is required.

Figures must answer scientific questions rather than serve decorative purposes.

---

## Practical Application

A practical CORA-Lung application will be developed only after the research inference
pipeline is frozen and validated.

Application code is isolated under `app/` so deployment work cannot alter reported
research experiments.

---

## Author

Syed Shayan Ali Shah  
GitHub: itsCodeBakery
