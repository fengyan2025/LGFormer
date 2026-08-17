#!/usr/bin/env python3
"""Select ablation checkpoints from existing Validation trajectories only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_STEPS = list(range(50_000, 120_001, 5_000))
EXPECTED_VARIANTS = {
    "all_local_matched",
    "reversed_matched",
    "no_directional",
    "no_rezero",
    "no_local_qkv",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tolerance-db", type=float, default=0.01)
    args = parser.parse_args()

    suite_root = args.suite_root.resolve()
    output = args.output.resolve()
    tolerance = float(args.tolerance_db)
    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    if output.exists():
        raise FileExistsError(output)

    variant_dirs = {
        path.name: path
        for path in suite_root.iterdir()
        if path.is_dir() and path.name in EXPECTED_VARIANTS
    }
    if set(variant_dirs) != EXPECTED_VARIANTS:
        raise RuntimeError(
            f"Expected variants {sorted(EXPECTED_VARIANTS)}, found {sorted(variant_dirs)}"
        )

    selections = []
    for variant in sorted(variant_dirs):
        rows = []
        trajectory_root = variant_dirs[variant] / "validation_trajectory"
        for summary_path in sorted(trajectory_root.glob("step_*/summary.json")):
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            step = int(summary["checkpoint_step"])
            if summary["split"] != "validation":
                raise RuntimeError(f"Non-Validation result encountered: {summary_path}")
            if summary.get("test_accessed") is not False:
                raise RuntimeError(f"Test access flag is not false: {summary_path}")
            if summary["variant"] != variant:
                raise RuntimeError(f"Variant mismatch: {summary_path}")
            rows.append(
                {
                    "step": step,
                    "global_psnr_db": float(summary["output"]["psnr"]["mean"]),
                    "checkpoint": summary["checkpoint"],
                    "checkpoint_sha256": summary["checkpoint_sha256"],
                    "summary": str(summary_path),
                    "metrics": summary["output"],
                }
            )

        steps = [row["step"] for row in rows]
        if steps != EXPECTED_STEPS:
            raise RuntimeError(f"Incomplete or unordered trajectory for {variant}: {steps}")

        peak_row = max(rows, key=lambda row: row["global_psnr_db"])
        eligible = [
            row
            for row in rows
            if peak_row["global_psnr_db"] - row["global_psnr_db"]
            <= tolerance + 1e-12
        ]
        selected = min(eligible, key=lambda row: row["step"])
        checkpoint = Path(selected["checkpoint"])
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        actual_hash = sha256(checkpoint)
        if actual_hash != selected["checkpoint_sha256"]:
            raise RuntimeError(f"Checkpoint hash mismatch: {checkpoint}")

        selections.append(
            {
                "variant": variant,
                "peak_step": peak_row["step"],
                "peak_global_psnr_db": peak_row["global_psnr_db"],
                "plateau_steps": [row["step"] for row in eligible],
                "selected_step": selected["step"],
                "selected_global_psnr_db": selected["global_psnr_db"],
                "gap_to_peak_db": peak_row["global_psnr_db"]
                - selected["global_psnr_db"],
                "selected_checkpoint": selected["checkpoint"],
                "selected_checkpoint_sha256": actual_hash,
                "selected_summary": selected["summary"],
                "selected_metrics": selected["metrics"],
                "trajectory": [
                    {
                        "step": row["step"],
                        "global_psnr_db": row["global_psnr_db"],
                        "checkpoint_sha256": row["checkpoint_sha256"],
                    }
                    for row in rows
                ],
            }
        )

    payload = {
        "comparison_split": "validation",
        "primary_metric": "mean_global_psnr_db",
        "selection_rule": "earliest_checkpoint_within_0.01_db_of_observed_maximum",
        "plateau_tolerance_db": tolerance,
        "trajectory_steps": EXPECTED_STEPS,
        "training_rerun": False,
        "evaluation_rerun": False,
        "test_accessed": False,
        "selections": selections,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
