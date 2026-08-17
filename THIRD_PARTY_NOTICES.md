# Third-party notices

MRI-LGFormer-T1 is inspired by and adapts implementation concepts from:

- Syed Waqas Zamir et al., **Restormer: Efficient Transformer for
  High-Resolution Image Restoration**, CVPR 2022.
  Official repository: <https://github.com/swz30/Restormer>

The hierarchical encoder-decoder, transposed channel attention, gated
depthwise-convolution feed-forward network, pixel shuffle/unshuffle resizing,
and image-level residual prediction derive from Restormer concepts.

The upstream Restormer MIT license is reproduced at
`third_party_licenses/Restormer_LICENSE.md`.

The MRI Local Detail Block (MLDB), the resolution-specialized allocation
(high-resolution local, intermediate mixed, low-resolution global), and the
T1w same-grid experimental protocol are project-specific contributions.

The aggregate baseline results in this repository refer to separately adapted
research implementations. Third-party baseline source code is not distributed
in this public repository.
