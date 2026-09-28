COVA-3D FINAL PUBLICATION OUTPUTS — VERSION 1.0
================================================

PRIMARY RESULT
--------------
The prospectively frozen primary endpoint, macro-patient lesion recall
at <=1 FP/patient (R@1), was 0 for all six factorial conditions.

The same floor effect persisted at the predeclared R@0.5, R@2, and R@4
operating points.

Both primary coverage-by-geometry interaction estimates were exactly 0.
The exact two-sided source-subject cluster sign-flip p-values were 1.0,
and Holm-adjusted p-values were also 1.0.

INTERPRETATION RULE
-------------------
The primary endpoint must remain the primary result.

Secondary fixed-threshold metrics are descriptive/mechanistic results.
They must not be presented as replacements for the failed primary endpoint.

The secondary analyses show that probability maps contain lesion-related
signal, but predictions exhibit severe component fragmentation and false-
positive burden, preventing useful operation within the frozen low-FP regime.

FIGURE PURPOSE
--------------
Figure 1:
    Demonstrates how lesion recall becomes available only at very large
    false-positive burden.

Figure 2:
    Shows low absolute volumetric Dice and IoU.

Figure 3:
    Shows the gap between merely touching lesions and achieving IoU>=0.25.

Figure 4:
    Relates component proliferation to false-positive volume.

Figure 5:
    Shows seed-level variability in fragmentation.

Figure 6:
    Descriptive coverage-by-geometry profile for any-overlap recall only.
    This figure is NOT a primary inferential interaction analysis.

STATISTICAL DESIGN
------------------
Independent inferential cluster:
    source_subject_key

Clusters:
    14

Final volumes:
    16

Bootstrap:
    10,000 paired source-subject cluster replicates

Bootstrap seed:
    20260914

Primary interaction tests:
    exact two-sided cluster sign-flip

Multiplicity:
    Holm correction across two primary interaction contrasts

IMPORTANT
---------
No model retraining, threshold alteration, endpoint substitution, or
condition-specific post-processing was performed after final outcome access.
