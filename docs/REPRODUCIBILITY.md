# Reproducibility checklist

- [x] Standalone T1-only model implementation.
- [x] No runtime dependency on historical UPSR, G2, RCR, or Restormer trees.
- [x] Subject-disjoint split construction and audit scripts.
- [x] Configuration, data hash, source commit, and step stored in checkpoints.
- [x] Global-PSNR plateau checkpoint-selection script.
- [x] Aggregate Test, efficiency, and ablation results.
- [x] Unit tests for shapes, metrics, datasets, and checkpoint compatibility.
- [ ] HCP family metadata and family-disjoint split.
- [ ] External clinical or cross-dataset validation.
- [ ] Multi-seed training uncertainty for the final comparison.
- [ ] Public pretrained weights (not planned for this release).
