# CORA-Lung Gate-B Pilot Harness Protocol

Scientific training in Block 07: **No**
Optimizer updates in Block 07: **0**

## Numerical precision correction

The initial query-BCE verification used `target.float()`.
This forced the BCE result to float32 even when the audit logits were float64.

Observed initial scalar discrepancy:
`5.9604644775390625e-08`

Observed initial gradient discrepancy:
`0.0`

Before any model fitting, the cast was corrected to:
`target.to(dtype=logits.dtype)`

The verification tolerance was not relaxed.

Final maximum BCE value error:
`2.220446049250313081e-16`

Final maximum BCE gradient error:
`0.000000000000000000e+00`

## GPU dry run

Peak allocated memory: **0.879 GB**

Peak reserved memory: **1.217 GB**

Optimizer step performed: **No**
