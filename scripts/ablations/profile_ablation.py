#!/usr/bin/env python3
"""Profile an independent ablation architecture on the frozen full image."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ablation_models import build_ablation_model
from mri_lgformer.reproducibility import git_commit, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=200)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model = build_ablation_model(config["model"]).cuda().eval()
    sample = torch.zeros(1, 1, 304, 256, device="cuda")
    timings = []
    with torch.no_grad():
        for _ in range(args.warmup):
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                model(sample)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.runs):
            start = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                result = model(sample)
            torch.cuda.synchronize()
            timings.append((time.perf_counter() - start) * 1000.0)
    payload = {
        "variant": config["model"]["variant"],
        "config_sha256": sha256(config_path),
        "git_commit": git_commit(),
        "parameters": sum(p.numel() for p in model.parameters()),
        "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        "output_shape": list(result.shape),
        "mean_latency_ms": statistics.mean(timings),
        "median_latency_ms": statistics.median(timings),
        "p95_latency_ms": sorted(timings)[max(0, int(0.95 * len(timings)) - 1)],
        "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
        "test_accessed": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
