# COVA-3D Manuscript Integration Notes

## Frozen primary conclusion

The prospective primary endpoint failed.

All six conditions produced:

- R@1 = 0.000
- 95% cluster-bootstrap CI = [0.000, 0.000]

The same floor was observed at:

- R@0.5
- R@2
- R@4

Both primary coverage-by-geometry interactions were:

- estimate = 0.000
- raw exact p = 1.000
- Holm-adjusted p = 1.000

## Descriptive secondary observations

Highest mean Dice:
C100_FRG = 0.058624

Highest mean IoU:
C100_FRG = 0.031312

Highest any-overlap lesion recall:
C50_COH = 0.663554

Highest IoU>=0.25 lesion recall:
C50_COH = 0.212366

Lowest mean FP volume:
C100_FRG = 11610.771 mL

Lowest mean fragmentation ratio:
C100_COH = 3.239262

## Manuscript rule

The manuscript must lead with the failed primary endpoint.

Secondary metrics may be used to explain the failure mechanism but must not
replace the primary endpoint or be described as statistically significant
unless a separately justified prospective analysis is conducted.

## Recommended scientific framing

The strongest defensible contribution is currently methodological/diagnostic:

Under a controlled fixed-budget sparse-supervision design, neither increased
lesion-instance coverage nor altered within-lesion scribble geometry was sufficient
to achieve reliable low-false-positive multi-lesion recovery with the tested
baseline. Secondary analyses indicate that lesion-contact information persists,
but predictions are dominated by severe fragmentation and false-positive burden.

## What must NOT be claimed

- state-of-the-art performance
- superiority over fully supervised methods
- statistically significant secondary differences
- successful primary coverage effect
- successful primary geometry effect
- successful coverage-by-geometry interaction
