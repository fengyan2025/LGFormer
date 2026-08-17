#!/usr/bin/env python3
"""Select the earliest checkpoint within a frozen validation PSNR plateau."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--evaluation-root",
        type=Path,
        required=True,
        help="Directory containing one subdirectory per evaluated checkpoint.",
    )
    parser.add_argument("--plateau-db", type=float, default=0.01)
    parser.add_argument(
        "--primary-metric",
        choices=("psnr", "foreground_psnr"),
        default="psnr",
        help="Validation summary metric used for checkpoint selection.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summaries = []
    for path in sorted(args.evaluation_root.glob("*/summary.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["_summary_path"] = str(path)
        summaries.append(payload)
    if not summaries:
        raise RuntimeError(f"No summary.json files under {args.evaluation_root}")
    best_psnr = max(
        float(item["output"][args.primary_metric]["mean"])
        for item in summaries
    )
    plateau = [
        item
        for item in summaries
        if best_psnr - float(item["output"][args.primary_metric]["mean"])
        <= args.plateau_db
    ]
    selected = min(plateau, key=lambda item: int(item["checkpoint_step"]))
    result = {
        "selection_split": "validation",
        "primary_metric": f"mean global {args.primary_metric.upper()}",
        "primary_metric_key": args.primary_metric,
        "plateau_tolerance_db": args.plateau_db,
        "raw_best_primary_metric": best_psnr,
        "selected_step": int(selected["checkpoint_step"]),
        "selected_checkpoint": selected["checkpoint"],
        "selected_checkpoint_sha256": selected["checkpoint_sha256"],
        "selected_primary_metric": float(
            selected["output"][args.primary_metric]["mean"]
        ),
        "selected_global_psnr": float(selected["output"]["psnr"]["mean"]),
        "selected_foreground_psnr": float(
            selected["output"]["foreground_psnr"]["mean"]
        ),
        "evaluated_steps": sorted(
            int(item["checkpoint_step"]) for item in summaries
        ),
        "test_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
