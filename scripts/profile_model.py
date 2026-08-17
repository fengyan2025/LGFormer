#!/usr/bin/env python3
"""Report parameter count and optional CUDA latency/VRAM."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mri_lgformer import build_model
from mri_lgformer.reproducibility import git_commit, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = build_model(config["model"])
    result = {
        "model_name": config["model"].get("name", "mri_lgformer_t1"),
        "config": str(config_path),
        "config_sha256": sha256(config_path),
        "git_commit": git_commit(),
        "amp": bool(args.amp),
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters()
            if parameter.requires_grad
        ),
    }
    if torch.cuda.is_available():
        model = model.cuda().eval()
        sample = torch.zeros(1, 1, 304, 256, device="cuda")
        timings = []
        with torch.no_grad():
            for _ in range(args.warmup):
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=args.amp,
                ):
                    model(sample)
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            for _ in range(args.runs):
                started = time.perf_counter()
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=args.amp,
                ):
                    output = model(sample)
                torch.cuda.synchronize()
                timings.append((time.perf_counter() - started) * 1000.0)
        result["output_shape"] = list(output.shape)
        result["mean_latency_ms"] = statistics.mean(timings)
        result["median_latency_ms"] = statistics.median(timings)
        result["p95_latency_ms"] = sorted(timings)[
            max(0, int(0.95 * len(timings)) - 1)
        ]
        result["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 1024**3
    payload = json.dumps(result, indent=2)
    if args.output is not None:
        output_path = args.output.resolve()
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite {output_path}")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
    print(payload)


if __name__ == "__main__":
    main()
