# Training and model selection

LGFormer was trained from random initialization for 140,000 steps with seeds
`20260730`, `20261117`, and `20261118`, batch size 2, paired `256 x 256` crops,
AdamW, cosine learning rate decay from `2e-4` to `2e-6`, weight decay `1e-2`,
and Charbonnier plus `0.1 x` gradient loss.

Checkpoints from 50,000 to 140,000 steps were evaluated every 5,000 steps on
Validation. The earliest checkpoint within `0.01 dB` of the maximum mean
Global PSNR was selected: 120k, 125k, and 125k for the three seeds, respectively.

For seed `20260730`, the raw maximum was at 140k (`29.020616 dB`);
120k (`29.010798 dB`) was selected by the plateau rule.

Each selected checkpoint was evaluated once on the same Test split.
The two additional seed runs were conducted after the original Test results
were known; checkpoint selection used Validation only.