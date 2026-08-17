#!/usr/bin/env python3
"""Create a portable T1-only manifest from the frozen HCP pair manifest."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


FIELDS = [
    "pair_id",
    "subject_id",
    "modality",
    "slice_id",
    "low_path",
    "high_path",
    "low_resolution_mm",
    "high_resolution_mm",
    "field_strength_low",
    "field_strength_high",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def portable_path(path_value: str, data_root: Path) -> str:
    path = Path(path_value).resolve()
    try:
        return path.relative_to(data_root).as_posix()
    except ValueError as exc:
        raise ValueError(f"{path} is not under data root {data_root}") from exc


def main() -> None:
    args = parse_args()
    source = args.source_manifest.resolve()
    data_root = args.data_root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    with source.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row["modality"] != "T1w":
                continue
            item = {field: row.get(field, "") for field in FIELDS}
            item["low_path"] = portable_path(row["low_path"], data_root)
            item["high_path"] = portable_path(row["high_path"], data_root)
            rows.append(item)
    rows.sort(key=lambda row: (row["subject_id"], int(row["slice_id"])))
    if not rows:
        raise RuntimeError("Source manifest contains no T1w rows")
    if len({row["pair_id"] for row in rows}) != len(rows):
        raise RuntimeError("Duplicate pair_id values in T1 manifest")

    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(
        {
            "output": str(output),
            "rows": len(rows),
            "subjects": len({row["subject_id"] for row in rows}),
            "modality": "T1w",
            "paths": "relative_to_data_root",
        }
    )


if __name__ == "__main__":
    main()
