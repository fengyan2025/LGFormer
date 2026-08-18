# Model specification

MRI-LGFormer-T1 maps one normalized T1w slice of shape `B x 1 x H x W` to an
enhanced slice with the same shape. `H` and `W` must be divisible by eight.

| Stage | Channels | Default size | Blocks | Role |
|---|---:|---:|---:|---|
| Encoder L1 | 48 | 304 x 256 | 4 MLDB | local anatomical detail |
| Encoder L2 | 96 | 152 x 128 | 3 MLDB + 3 LTAB | local-global transition |
| Encoder L3 | 192 | 76 x 64 | 6 LTAB | global channel interaction |
| Latent L4 | 384 | 38 x 32 | 8 LTAB | global representation |
| Decoder L3 | 192 | 76 x 64 | 6 LTAB | global channel interaction |
| Decoder L2 | 96 | 152 x 128 | 3 MLDB + 3 LTAB | local-global transition |
| Decoder L1 | 48 | 304 x 256 | 4 MLDB | local detail |
| Refinement | 48 | 304 x 256 | 4 MLDB | local refinement |

The default model has 24,897,808 trainable parameters. The output follows
`enhanced = input + predicted_residual`.

## MRI Local Detail Block

MLDB applies LayerNorm and `1 x 1` expansion, then splits features into an
isotropic depthwise `3 x 3` branch and parallel depthwise `1 x 5` / `5 x 1`
directional branches. The GELU-activated local feature is multiplied by a
sigmoid directional gate, projected to the original width, scaled by a
per-channel zero-initialized coefficient, and added to the input.

## Local-QKV Transposed Attention Block

LTAB uses pre-normalized transposed channel attention. A `1 x 1` QKV projection
and depthwise `3 x 3` encoding precede L2 normalization of Q and K. Attention
is computed as a per-head `C_head x C_head` matrix rather than a spatial
`HW x HW` matrix. 
