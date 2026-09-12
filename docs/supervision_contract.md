# CORA-Lung Supervision Contract

## Permitted During Training

- CT image;
- physical geometry;
- foreground scribbles;
- background scribbles;
- foreground scribble-group IDs;
- image-derived region maps;
- patient split identifier;
- generation seed.

## Forbidden During Training

- dense lesion mask;
- dense lesion-component IDs;
- hidden-component locations;
- full-mask-derived distances;
- test outcomes.

## Inference

Inference is CT-image-only.

No scribbles, prompts, reports, VLM, SAM, dense teacher, or target labels are permitted.
