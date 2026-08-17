# Training and model selection

The main model was trained from random initialization for 140,000 steps using
seed `20260730`, batch size 2, paired `256 x 256` crops, AdamW, cosine learning
rate decay from `2e-4` to `2e-6`, weight decay `1e-2`, and Charbonnier plus
`0.1 x` gradient loss.

Checkpoints from 50,000 to 140,000 steps were evaluated every 5,000 steps on
Validation. Mean Global PSNR was the primary metric. The earliest checkpoint
within `0.01 dB` of the observed maximum was selected. The raw maximum was at
140,000 steps (`29.020616 dB`); step 120,000 (`29.010798 dB`) was selected by
the frozen plateau rule.

Test was evaluated once after checkpoint selection and was not used for
training, early stopping, or model selection.
