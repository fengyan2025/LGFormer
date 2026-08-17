# MRI-LGFormer-T1 model card

## Intended use

Research on deterministic same-grid enhancement of normalized T1-weighted
brain MRI slices. The model is not a medical device and must not be used for
clinical diagnosis or treatment decisions without independent validation.

## Inputs and outputs

- Input: one normalized T1w slice, `B x 1 x H x W`.
- Output: one enhanced slice with the same shape.
- Spatial dimensions must be divisible by eight.

## Training data

The reported model was trained on authorized HCP-derived paired T1w data. Data
are not distributed with this repository.

## Known limitations

- Internal HCP repartition only; no external clinical validation.
- Subject-disjoint but not proven family-disjoint.
- Same-grid 2D T1w setting only.
- No guarantee that enhanced structures are diagnostically faithful.
- Pretrained weights are not released.
