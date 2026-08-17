# Data preparation

No HCP images, manifests, split assignments, or subject identifiers are
distributed in this repository.

The training code expects a CSV with these columns:

```text
pair_id,subject_id,modality,slice_id,low_path,high_path
```

Requirements:

- `modality` must be `T1w`;
- `low_path` and `high_path` are relative to `paths.data_root` whenever
  possible;
- each NPZ file contains a finite two-dimensional float array named `image`;
- low-quality and reference images have identical shapes;
- subject IDs must not cross Train, Validation, and Test splits.

Use `scripts/build_t1_manifest.py`, `scripts/make_splits.py`, and
`scripts/audit_splits.py` to construct and audit local CSV files. These CSVs
are ignored by Git because they may contain HCP subject identifiers.
