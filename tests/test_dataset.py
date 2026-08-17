import csv

import numpy as np

from mri_lgformer.dataset import PairedT1Dataset


def test_relative_paths(tmp_path):
    data_root = tmp_path / "raw"
    data_root.mkdir()
    for name in ("low.npz", "high.npz"):
        np.savez_compressed(data_root / name, image=np.zeros((8, 8), np.float32))
    split = tmp_path / "split.csv"
    with split.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "pair_id",
                "subject_id",
                "modality",
                "slice_id",
                "low_path",
                "high_path",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "pair_id": "s_T1w_1",
                "subject_id": "s",
                "modality": "T1w",
                "slice_id": "1",
                "low_path": "low.npz",
                "high_path": "high.npz",
            }
        )
    dataset = PairedT1Dataset(split, data_root, validate_paths=True)
    item = dataset[0]
    assert item["low"].shape == (1, 8, 8)
    assert item["high"].shape == (1, 8, 8)
