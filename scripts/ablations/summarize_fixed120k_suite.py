#!/usr/bin/env python3
"""Summarize fixed-step Validation results without accessing Test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


METRIC_SOURCE_KEYS = {
    "foreground_p95": "foreground_abs_error_p95",
    "foreground_p99": "foreground_abs_error_p99",
}


def metric_mean(summary: dict, key: str) -> float:
    source_key = METRIC_SOURCE_KEYS.get(key, key)
    return float(summary["output"][source_key]["mean"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposed-summary", type=Path, required=True)
    parser.add_argument("--suite-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    proposed = json.loads(args.proposed_summary.resolve().read_text(encoding="utf-8"))
    if proposed["split"] != "validation" or int(proposed["checkpoint_step"]) != 120000:
        raise RuntimeError("Proposed reference must be Validation step 120k")
    metrics = ["psnr", "foreground_psnr", "ssim", "foreground_ssim", "mae", "foreground_mae", "nmse", "hfen", "gradient_mae", "foreground_p95", "foreground_p99"]
    variants = []
    for path in sorted(args.suite_root.resolve().glob("*/validation_trajectory/step_120000/summary.json")):
        summary = json.loads(path.read_text(encoding="utf-8"))
        if summary["split"] != "validation" or summary.get("test_accessed") is not False:
            raise RuntimeError(f"Invalid ablation summary: {path}")
        variants.append({
            "variant": summary["variant"],
            "checkpoint_step": summary["checkpoint_step"],
            "checkpoint_sha256": summary["checkpoint_sha256"],
            "metrics": {key: metric_mean(summary, key) for key in metrics},
            "delta_variant_minus_proposed": {key: metric_mean(summary, key) - metric_mean(proposed, key) for key in metrics},
        })
    if len(variants) != 5:
        raise RuntimeError(f"Expected five completed variants, found {len(variants)}")
    payload = {
        "comparison_split": "validation",
        "formal_step": 120000,
        "selection": "fixed_step_no_per_variant_selection",
        "proposed_checkpoint_sha256": proposed["checkpoint_sha256"],
        "proposed_metrics": {key: metric_mean(proposed, key) for key in metrics},
        "variants": variants,
        "test_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
