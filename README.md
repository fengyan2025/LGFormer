# MRI-LGFormer-T1: Stage-Specialized Local-Global Transformer for Same-Grid T1w MRI Enhancement

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4%2B-ee4c2c.svg)](https://pytorch.org/)
[![Task](https://img.shields.io/badge/Task-T1w%20MRI%20Enhancement-6f42c1.svg)](#model-architecture)
[![License](https://img.shields.io/badge/License-All%20Rights%20Reserved-lightgrey.svg)](LICENSE)

> 🧠 A deterministic Transformer framework for transforming 2.0-mm-quality T1-weighted brain MRI into a 0.7-mm-reference-quality estimate on the same spatial grid.

![MRI-LGFormer-T1 architecture](figures/MRI_LGFormer_T1_architecture.png)

---

## 📖 Table of Contents

- [Abstract](#abstract)
- [Highlights](#highlights)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Ablation Study](#ablation-study)
- [Repository Structure](#repository-structure)
- [License](#license)

<a id="abstract"></a>

## 🧠 Abstract

MRI-LGFormer-T1 is a deterministic image-to-image model for **same-grid T1w brain MRI enhancement**. It maps a normalized `1 × 304 × 256` 2.0-mm-quality input slice to a `1 × 304 × 256` estimate of the paired 0.7-mm-reference-quality image. The task improves image quality without enlarging the spatial matrix and should not be interpreted as conventional ×2 or ×4 super-resolution.

The model follows a multi-scale encoder-decoder design inspired by Restormer, but reorganizes feature modeling according to spatial resolution. High-resolution stages use the proposed **MRI Local Detail Block (MLDB)**, the intermediate stage combines MLDB with a **Local-QKV Transposed Attention Block (LTAB)**, and low-resolution stages use LTAB for efficient global channel interaction. A residual head predicts a correction that is added to the input image. On the frozen 107-subject Test split, the Validation-selected step-120,000 checkpoint achieved `29.1362 dB` Global PSNR and `0.880736` Global SSIM.

<a id="highlights"></a>

## ✨ Highlights

- **Stage-specialized local-global modeling:** local anatomical detail is emphasized at high spatial resolution, while efficient global channel dependencies are modeled at lower resolutions.
- **MRI Local Detail Block (MLDB):** an isotropic depthwise `3 × 3` branch is modulated by parallel depthwise `1 × 5` and `5 × 1` directional context.
- **Local-QKV Transposed Attention Block (LTAB):** local depthwise QKV encoding precedes transposed channel attention and a gated depthwise-convolution feed-forward network.
- **Same-grid residual enhancement:** the model predicts an image correction rather than increasing the input matrix size.
- **Reproducible selection protocol:** checkpoints are selected using mean Global PSNR on Validation, with the earliest point chosen within a pre-specified `0.01 dB` plateau.

<a id="installation"></a>

## ⚙️ Installation

Python 3.10 or later and a CUDA-capable GPU are recommended for training.

```bash
git clone https://github.com/fengyan2025/LGFormer.git
cd LGFormer

python -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -e .
pytest -q
```

The frozen experiments used PyTorch 2.4.1 with CUDA acceleration. On Windows PowerShell, activate the virtual environment with `.\.venv\Scripts\Activate.ps1`.

<a id="dataset-preparation"></a>

## 📂 Dataset Preparation

This repository does **not** distribute HCP images, subject identifiers, split CSV files, or other restricted data. Obtain authorized T1w data from the official HCP data provider and comply with the applicable data-use terms.

Each paired normalized NPZ file must contain a two-dimensional array under the key `image`. Prepare the manifest and subject-disjoint splits with:

```bash
python scripts/build_t1_manifest.py --help
python scripts/make_splits.py --help
python scripts/audit_splits.py --help
```

The expected CSV schema is documented in [`data/README.md`](data/README.md). Before training, update `paths.data_root` and the relevant manifest paths in [`configs/train_t1_main.yaml`](configs/train_t1_main.yaml).

The main experiment used the following subject-level partition:

| Split | Subjects | T1w pairs | Purpose |
|---|---:|---:|---|
| Train | 857 | 17,140 | Parameter optimization |
| Validation | 107 | 2,140 | Checkpoint selection |
| Test | 107 | 2,140 | One-time final evaluation |

<a id="usage"></a>

## 🚀 Usage

### Train from scratch

Pretrained weights are not released. Train MRI-LGFormer-T1 from random initialization:

```bash
python scripts/train.py --config configs/train_t1_main.yaml
```

Frozen main-run settings:

| Setting | Value |
|---|---:|
| Seed | 20260730 |
| Maximum steps | 140,000 |
| Batch size | 2 |
| Crop size | 256 × 256 |
| Optimizer | AdamW |
| Learning rate | `2e-4 → 2e-6` cosine decay |
| Loss | Charbonnier + `0.1 ×` gradient loss |

### Select the checkpoint

Checkpoint selection uses **mean Global PSNR** on Validation. Among checkpoints within `0.01 dB` of the observed maximum, the earliest checkpoint is selected. Test data must not be used for model selection.

```bash
bash scripts/run_validation_selection.sh
```

The frozen rule selected step 120,000; the raw Validation maximum occurred at step 140,000, only `0.009818 dB` higher.

### Evaluate a trained checkpoint

```bash
python scripts/evaluate.py \
  --config configs/train_t1_main.yaml \
  --checkpoint runs/t1_main_seed20260730/checkpoint_step_120000.pth \
  --split validation \
  --output-dir results/local_validation \
  --expected-subjects 107 \
  --expected-pairs 2140
```

### Run inference

```bash
python scripts/infer.py \
  --checkpoint runs/t1_main_seed20260730/checkpoint_step_120000.pth \
  --input path/to/input.npz \
  --output path/to/enhanced.npz
```

The input NPZ must contain `image` with shape `H × W`; both dimensions must be divisible by eight. The enhanced image is stored under the same key in the output NPZ.

<a id="model-architecture"></a>

## 🏗️ Model Architecture

MRI-LGFormer-T1 contains approximately **24.90 million parameters** and follows a four-level U-shaped encoder-decoder:

```text
Low-quality T1w slice
        │
        ▼
3 × 3 overlapping embedding
        │
        ├─ L1: MLDB × 4                     (high-resolution local stage)
        ├─ L2: MLDB × 3 → LTAB × 3          (local-to-global transition)
        ├─ L3: LTAB × 6                      (low-resolution global stage)
        └─ L4: LTAB × 8                      (latent global stage)
        │
        ▼
Symmetric decoder + plain skip fusion
        │
        ▼
MLDB refinement × 4 → residual head
        │
        ▼
Enhanced image = input + predicted residual
```

### MRI Local Detail Block (MLDB)

For an input feature `X`, MLDB expands the channels and creates two complementary paths:

1. a depthwise `3 × 3` local branch followed by GELU;
2. parallel depthwise `1 × 5` and `5 × 1` directional branches followed by sigmoid gating.

Their gated interaction is projected back to the original channel dimension and added through a channel-wise ReZero residual scale initialized to zero. This design targets local tissue boundaries and directional anatomical structures while preserving stable optimization.

### Local-QKV Transposed Attention Block (LTAB)

LTAB uses a `1 × 1` QKV projection followed by depthwise `3 × 3` local QKV encoding. Q and K are L2-normalized along the spatial dimension, and attention is computed across channels rather than through an `HW × HW` spatial attention matrix. A GDFN sub-block then performs gated local feature transformation.


<a id="results"></a>

## 📊 Results

The step-120,000 model was selected on Validation and evaluated once on the frozen 107-subject Test split.

| Global PSNR ↑ | Foreground PSNR ↑ | Global SSIM ↑ | Foreground SSIM ↑ | NMSE ↓ | HFEN ↓ |
|---:|---:|---:|---:|---:|---:|
| **29.1362** | **28.1362** | **0.880736** | **0.861070** | **0.011475** | **0.239608** |

Additional Test metrics include Global MAE `0.044018`, Foreground MAE `0.053454`, gradient MAE `0.091928`, Foreground P95 `0.159106`, and Foreground P99 `0.278512`. All 107 Test subjects improved over their corresponding 2.0-mm-quality input in subject-mean Global and Foreground PSNR and MAE.

These values characterize performance on the frozen internal HCP-derived split. They do not establish superiority over every published method or external clinical generalization. See [`docs/RESULTS.md`](docs/RESULTS.md) and the aggregate files in [`results/`](results/) for the full report.

<a id="ablation-study"></a>

## 🧪 Ablation Study

All ablations below were selected by the same mean Global-PSNR plateau rule.

| Configuration | Selected Global PSNR | Δ vs. proposed |
|---|---:|---:|
| **Proposed stage-specialized model** | **29.1362** | — |
| All-local allocation | 28.939208 | -0.071590 dB |
| Reversed allocation | 28.957607 | -0.053191 dB |
| Without directional filtering | 28.998137 | -0.012661 dB |
| Without local-QKV encoding | 29.001516 | -0.009282 dB |
| Without ReZero | 29.013210 | +0.002412 dB |

The clearest evidence supports the proposed high-resolution-local/intermediate-mixed/low-resolution-global allocation over equal-budget all-local and reversed controls. Directional filtering showed a small positive effect. The isolated local-QKV and ReZero differences fall within approximately `0.01 dB`; they are therefore described as design and stability choices rather than independent performance breakthroughs.

<a id="repository-structure"></a>

## 📁 Repository Structure

```text
configs/              Model, training, and ablation configurations
src/mri_lgformer/     MRI-LGFormer-T1 implementation
scripts/              Training, evaluation, inference, and data utilities
scripts/ablations/    Formal ablation utilities
ablation_models/      Isolated ablation variants
tests/                Shape, metric, dataset, and checkpoint tests
results/              Aggregate, non-subject-level experimental results
figures/              Model architecture figure
docs/                 Data, model, training, and reproducibility notes
```


<a id="license"></a>

## 📜 License

Copyright © 2026 MRI-LGFormer-T1 authors. All rights reserved.

The repository is publicly source-visible for academic inspection and reproducibility, but no permission to use, copy, modify, or redistribute the project is granted unless explicitly stated. See [`LICENSE`](LICENSE) and [`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md) for the complete terms and upstream acknowledgements.
