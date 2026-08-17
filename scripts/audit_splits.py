#!/usr/bin/env python3
"""Audit T1-only split integrity and optional image-file validity."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument(
        "--check-images",
        choices=("none", "sample", "all"),
        default="sample",
    )
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def validate_npz(path: Path) -> tuple[tuple[int, ...], float, float]:
    with np.load(path, allow_pickle=False) as payload:
        image = np.asarray(payload["image"], dtype=np.float32)
    if image.ndim != 2 or not np.all(np.isfinite(image)):
        raise ValueError(f"Invalid image {path}: shape={image.shape}")
    return image.shape, float(image.min()), float(image.max())


def main() -> None:
    args = parse_args()
    split_dir = args.split_dir.resolve()
    data_root = args.data_root.resolve()
    split_rows = {
        split: load_rows(split_dir / f"{split}.csv")
        for split in ("train", "validation", "test")
    }
    subjects = {
        split: {row["subject_id"] for row in rows}
        for split, rows in split_rows.items()
    }
    overlaps = {
        "train_validation": sorted(subjects["train"] & subjects["validation"]),
        "train_test": sorted(subjects["train"] & subjects["test"]),
        "validation_test": sorted(subjects["validation"] & subjects["test"]),
    }
    pair_ids = [
        row["pair_id"]
        for rows in split_rows.values()
        for row in rows
    ]
    errors: list[str] = []
    if any(overlaps.values()):
        errors.append("subject_overlap")
    if len(pair_ids) != len(set(pair_ids)):
        errors.append("duplicate_pair_id")
    if any(
        row["modality"] != "T1w"
        for rows in split_rows.values()
        for row in rows
    ):
        errors.append("non_t1_modality")

    checked_images = 0
    image_ranges: list[tuple[float, float]] = []
    if args.check_images != "none":
        all_rows = [row for rows in split_rows.values() for row in rows]
        if args.check_images == "sample":
            count = min(args.sample_count, len(all_rows))
            indices = np.linspace(0, len(all_rows) - 1, count, dtype=int)
            rows_to_check = [all_rows[int(index)] for index in indices]
        else:
            rows_to_check = all_rows
        for row in rows_to_check:
            low_path = data_root / row["low_path"]
            high_path = data_root / row["high_path"]
            if not low_path.is_file() or not high_path.is_file():
                errors.append(f"missing:{row['pair_id']}")
                continue
            low_shape, low_min, low_max = validate_npz(low_path)
            high_shape, high_min, high_max = validate_npz(high_path)
            if low_shape != high_shape:
                errors.append(f"shape_mismatch:{row['pair_id']}")
            image_ranges.extend([(low_min, low_max), (high_min, high_max)])
            checked_images += 2

    result = {
        "status": "pass" if not errors else "fail",
        "inventories": {
            split: {
                "subjects": len(subjects[split]),
                "pairs": len(rows),
            }
            for split, rows in split_rows.items()
        },
        "subject_overlaps": {key: len(value) for key, value in overlaps.items()},
        "unique_pair_ids": len(set(pair_ids)),
        "total_pairs": len(pair_ids),
        "checked_images": checked_images,
        "observed_min": min((value[0] for value in image_ranges), default=None),
        "observed_max": max((value[1] for value in image_ranges), default=None),
        "errors": errors[:100],
    }
    text = json.dumps(result, indent=2)
    print(text)
    output = args.output or split_dir / "split_audit.json"
    output.write_text(text, encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
