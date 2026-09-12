# Block 08G — Dynamic-Topology FROC Diagnostic

The official Gate-B result remains NO-GO.

This analysis recomputes connected components independently at each voxel
probability threshold. It therefore asks whether the fixed p=0.5 candidate
topology used by the locked evaluator materially affected the scientific
ordering between matched random-pixel sparsity and natural component omission.

The analysis is post-hoc and diagnostic. It cannot replace Gate B.

The same block also quantifies the class-prior dependence of the current
partial BCE + partial Dice loss using the sparse supervision actually observed
during the frozen training schedule.

No model is retrained and no final outer-CV case is accessed.
