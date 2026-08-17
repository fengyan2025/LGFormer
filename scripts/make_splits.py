#!/usr/bin/env python3
"""Create deterministic subject- or family-disjoint T1-only splits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument("--train-subjects", type=int, default=857)
    parser.add_argument("--validation-subjects", type=int, default=107)
    parser.add_argument("--test-subjects", type=int, default=107)
    parser.add_argument(
        "--family-map",
        type=Path,
        help="Optional CSV with subject_id,family_id. Groups never cross splits.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_family_map(path: Path | None) -> dict[str, str]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"subject_id", "family_id"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"{path} must contain {sorted(required)}")
    return {row["subject_id"]: row["family_id"] for row in rows}


def assign_subjects(
    subjects: list[str],
    family_map: dict[str, str],
    counts: dict[str, int],
    seed: int,
) -> dict[str, str]:
    if not family_map:
        shuffled = sorted(subjects)
        random.Random(seed).shuffle(shuffled)
        boundaries = (
            counts["train"],
            counts["train"] + counts["validation"],
        )
        return {
            subject: (
                "train"
                if index < boundaries[0]
                else "validation"
                if index < boundaries[1]
                else "test"
            )
            for index, subject in enumerate(shuffled)
        }

    missing = sorted(set(subjects).difference(family_map))
    if missing:
        raise ValueError(
            f"Family map lacks {len(missing)} subjects; first={missing[0]}"
        )
    groups: dict[str, list[str]] = defaultdict(list)
    for subject in subjects:
        groups[family_map[subject]].append(subject)
    families = sorted(groups)
    random.Random(seed).shuffle(families)
    assignment: dict[str, str] = {}
    current = {name: 0 for name in counts}
    for family in families:
        members = sorted(groups[family])
        # Greedy normalized-deficit assignment keeps families intact.
        split = max(
            counts,
            key=lambda name: (counts[name] - current[name]) / max(counts[name], 1),
        )
        for subject in members:
            assignment[subject] = split
        current[split] += len(members)
    return assignment


def main() -> None:
    args = parse_args()
    manifest = args.manifest.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    with manifest.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    if {row["modality"] for row in rows} != {"T1w"}:
        raise ValueError("Split manifest must be T1w-only")
    subjects = sorted({row["subject_id"] for row in rows})
    expected_total = (
        args.train_subjects + args.validation_subjects + args.test_subjects
    )
    if len(subjects) != expected_total:
        raise ValueError(
            f"Found {len(subjects)} subjects, requested counts sum to {expected_total}"
        )
    family_map = load_family_map(args.family_map.resolve() if args.family_map else None)
    counts = {
        "train": args.train_subjects,
        "validation": args.validation_subjects,
        "test": args.test_subjects,
    }
    assignment = assign_subjects(subjects, family_map, counts, args.seed)

    assignments_path = output_dir / "subject_assignments.csv"
    with assignments_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["subject_id", "family_id", "split"],
        )
        writer.writeheader()
        for subject in subjects:
            writer.writerow(
                {
                    "subject_id": subject,
                    "family_id": family_map.get(subject, ""),
                    "split": assignment[subject],
                }
            )

    split_paths: dict[str, Path] = {}
    inventories: dict[str, dict[str, int]] = {}
    for split in ("train", "validation", "test"):
        split_rows = [
            row for row in rows if assignment[row["subject_id"]] == split
        ]
        split_rows.sort(key=lambda row: (row["subject_id"], int(row["slice_id"])))
        path = output_dir / f"{split}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(split_rows)
        split_paths[split] = path
        inventories[split] = {
            "subjects": len({row["subject_id"] for row in split_rows}),
            "pairs": len(split_rows),
        }

    split_manifest = {
        "source_manifest": str(manifest),
        "source_manifest_sha256": sha256(manifest),
        "seed": args.seed,
        "family_disjoint": bool(family_map),
        "family_map": str(args.family_map.resolve()) if args.family_map else None,
        "warning": (
            None
            if family_map
            else "HCP family metadata was unavailable; splits are subject-disjoint "
            "but not proven family-disjoint."
        ),
        "inventories": inventories,
        "sha256": {
            path.name: sha256(path)
            for path in [assignments_path, *split_paths.values()]
        },
        "historical_data_use": (
            "All 1,071 HCP subjects appeared in prior project development. "
            "This is a clean re-partition for a new protocol, not an untouched "
            "external test cohort."
        ),
    }
    (output_dir / "split_manifest.json").write_text(
        json.dumps(split_manifest, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(split_manifest, indent=2))


if __name__ == "__main__":
    main()
