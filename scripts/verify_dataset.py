#!/usr/bin/env python3
"""Verify the portable T1 manifest against the raw NPZ root."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--mode", choices=("paths", "full"), default="paths")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    data_root = args.data_root.resolve()
    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    errors: list[str] = []
    shapes: set[tuple[int, ...]] = set()
    for row in rows:
        for key in ("low_path", "high_path"):
            path = data_root / row[key]
            if not path.is_file():
                errors.append(f"missing:{path}")
                continue
            if args.mode == "full":
                with np.load(path, allow_pickle=False) as payload:
                    image = np.asarray(payload["image"], dtype=np.float32)
                if image.ndim != 2 or not np.all(np.isfinite(image)):
                    errors.append(f"invalid:{path}")
                shapes.add(tuple(image.shape))
    result = {
        "status": "pass" if not errors else "fail",
        "rows": len(rows),
        "subjects": len({row["subject_id"] for row in rows}),
        "modalities": sorted({row["modality"] for row in rows}),
        "unique_pair_ids": len({row["pair_id"] for row in rows}),
        "mode": args.mode,
        "shapes": sorted([list(shape) for shape in shapes]),
        "errors": errors[:100],
    }
    text = json.dumps(result, indent=2)
    print(text)
    output = args.output or manifest.with_name("dataset_audit.json")
    output.write_text(text, encoding="utf-8")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
