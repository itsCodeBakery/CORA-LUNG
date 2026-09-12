# Gate-B Failure Mechanism Audit

## Status

The locked Gate-B decision remains **NO-GO**.

Block 08F is diagnostic. It cannot revise Gate B, select a new checkpoint,
change a probability threshold for the locked result, or authorize CORA
replay.

## Questions examined

1. Prediction fragmentation across probability thresholds.
2. Diagnostic threshold sensitivity of Dice and lesion recall.
3. Lesion-versus-background probability separation.
4. Exact sparse-supervision exposure under the original frozen training
   schedule.
5. Source-specific behavior for Coronacases and Radiopaedia.
6. A descriptive regularization-like pattern in the natural component-
   omission condition.

## Interpretation constraint

A better diagnostic threshold does not invalidate the locked Gate-B result.

Likewise, finding that natural component omission has lower fragmentation or
better lesion/background separation does not establish a causal regularization
mechanism from four development cases and one seed.

Block 08F may justify a subsequent protocol-design discussion only if the
observed diagnostics identify a concrete, technically defensible problem.

No final outer-CV result may be inspected during that discussion.
