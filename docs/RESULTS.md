# Aggregate experimental results

## Test quality

| Method | Global PSNR | FG PSNR | Global SSIM | FG SSIM | HFEN |
|---|---:|---:|---:|---:|---:|
| MRI-LGFormer-T1 | **29.1362** | **28.1362** | **0.880736** | **0.861070** | **0.239608** |
| NAFNet | 28.7304 | 27.7301 | 0.873805 | 0.852784 | 0.252859 |
| RCAN-1x | 28.9753 | 27.9747 | 0.876367 | 0.855827 | 0.244655 |
| ResUNet | 28.7267 | 27.7263 | 0.873490 | 0.852401 | 0.252537 |
| LMLT-Base-1x | 27.8509 | 27.0215 | 0.859937 | 0.836316 | 0.283605 |
| SPAN-1x | 27.8784 | 26.8798 | 0.857647 | 0.833296 | 0.283579 |
| SAFMN-1x | 27.7602 | 26.7639 | 0.856202 | 0.831583 | 0.287958 |
| SMFANet-1x | 26.9395 | 25.9530 | 0.841675 | 0.814170 | 0.321276 |

Training budgets were mixed, and each model used one training seed. Restormer
was not rerun on this current split and is not part of this comparison.

## Formal Validation ablations

| Variant | Selected step | Global PSNR | Delta vs. proposed |
|---|---:|---:|---:|
| Proposed | 120k | 29.010798 | 0 |
| All-local matched | 115k | 28.939208 | -0.071590 |
| Reversed matched | 115k | 28.957607 | -0.053191 |
| No directional | 120k | 28.998137 | -0.012661 |
| No local QKV | 115k | 29.001516 | -0.009282 |
| No ReZero | 115k | 29.013210 | +0.002412 |

Ablations were not evaluated on Test.
