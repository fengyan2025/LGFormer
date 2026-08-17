#!/usr/bin/env python3
"""Effect-free synthetic engineering gate for all five ablation models."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ablation_models import build_ablation_model
from ablation_models.mri_lgformer_t1_variants import (
    NoDirectionalMLDB,
    NoLocalQKVAttention,
    NoReZeroMLDB,
)
from mri_lgformer import build_model
from mri_lgformer.losses import CharbonnierLoss, gradient_loss
from mri_lgformer.reproducibility import git_commit, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(output)
    if len(args.config) != 5:
        raise RuntimeError("Exactly five ablation configs are required")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    proposed_config = {
        "inp_channels": 1,
        "out_channels": 1,
        "dim": 48,
        "num_blocks": [4, 6, 6, 8],
        "num_refinement_blocks": 4,
        "heads": [1, 2, 4, 8],
        "ffn_expansion_factor": 2.66,
        "bias": False,
        "LayerNorm_type": "WithBias",
        "dual_pixel_task": False,
        "allocation": "proposed",
    }
    proposed = build_model(proposed_config)
    proposed_parameters = sum(p.numel() for p in proposed.parameters())
    if proposed_parameters != 24897808:
        raise RuntimeError("Frozen proposed parameter count mismatch")
    del proposed

    expected_variants = {
        "all_local_matched",
        "reversed_matched",
        "no_directional",
        "no_rezero",
        "no_local_qkv",
    }
    results = []
    seen = set()
    for path_arg in args.config:
        config_path = path_arg.resolve()
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        variant = config["model"]["variant"]
        seen.add(variant)
        train_cfg = config["training"]
        if int(train_cfg["max_steps"]) != 120000 or int(train_cfg["scheduler_t_max"]) != 140000:
            raise RuntimeError(f"Fixed-step protocol mismatch for {variant}")
        expected_lr = float(config["freeze"]["proposed_step120000_learning_rate"])
        calculated_lr = float(train_cfg["eta_min"]) + (
            float(train_cfg["learning_rate"]) - float(train_cfg["eta_min"])
        ) * (1.0 + math.cos(math.pi * 120000 / 140000)) / 2.0
        if abs(calculated_lr - expected_lr) > 1e-15:
            raise RuntimeError(f"Step-120k LR path mismatch for {variant}")

        torch.manual_seed(int(train_cfg["seed"]))
        model = build_ablation_model(config["model"]).cuda().train()
        parameters = sum(p.numel() for p in model.parameters())
        if variant in {"all_local_matched", "reversed_matched"}:
            parameter_difference = abs(parameters - proposed_parameters) / proposed_parameters
            if parameter_difference > 0.05:
                raise RuntimeError(f"Parameter-match gate failed for {variant}")
        module_counts = {
            "no_directional_mldb": sum(isinstance(m, NoDirectionalMLDB) for m in model.modules()),
            "no_rezero_mldb": sum(isinstance(m, NoReZeroMLDB) for m in model.modules()),
            "no_local_qkv_attention": sum(isinstance(m, NoLocalQKVAttention) for m in model.modules()),
        }
        expected_key = {
            "no_directional": "no_directional_mldb",
            "no_rezero": "no_rezero_mldb",
            "no_local_qkv": "no_local_qkv_attention",
        }.get(variant)
        for key, count in module_counts.items():
            if (key == expected_key) != (count > 0):
                raise RuntimeError(f"Module identity gate failed for {variant}: {key}={count}")

        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
        scaler = torch.amp.GradScaler("cuda", enabled=True)
        criterion = CharbonnierLoss(1e-3)
        low = torch.rand(2, 1, 256, 256, device="cuda") * 0.2 - 0.1
        high = torch.rand_like(low) * 0.2 - 0.1
        overflows = 0
        successful_updates = 0
        attempts = 0
        torch.cuda.reset_peak_memory_stats()
        while successful_updates < 1 and attempts < 9:
            attempts += 1
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = model(low)
                loss = criterion(prediction, high) + 0.1 * gradient_loss(prediction, high)
            if prediction.shape != high.shape or not torch.isfinite(loss):
                raise FloatingPointError(f"Synthetic forward failed for {variant}")
            before = float(scaler.get_scale())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            overflow = not bool(torch.isfinite(norm).item()) or float(scaler.get_scale()) < before
            if overflow:
                overflows += 1
            else:
                successful_updates += 1
        passed = successful_updates == 1 and overflows <= 8
        results.append({
            "variant": variant,
            "config": str(config_path),
            "config_sha256": sha256(config_path),
            "parameters": parameters,
            "parameter_difference_from_proposed_percent": 100.0 * (parameters - proposed_parameters) / proposed_parameters,
            "module_counts": module_counts,
            "input_shape": list(low.shape),
            "output_shape": list(prediction.shape),
            "loss_finite": bool(torch.isfinite(loss).item()),
            "attempts": attempts,
            "amp_scale_settling_overflows": overflows,
            "successful_updates": successful_updates,
            "peak_vram_gb": torch.cuda.max_memory_allocated() / 1024**3,
            "step120000_learning_rate": calculated_lr,
            "passed": passed,
        })
        del model, optimizer, scaler, low, high, prediction, loss
        torch.cuda.empty_cache()
        if not passed:
            raise SystemExit(f"Synthetic gate failed for {variant}")
    if seen != expected_variants:
        raise RuntimeError(f"Variant set mismatch: {seen}")
    payload = {
        "protocol": "effect-free synthetic gate for five fixed-120k ablations",
        "git_commit": git_commit(),
        "proposed_parameters": proposed_parameters,
        "original_model_sha256": sha256(PROJECT_ROOT / "src/mri_lgformer/model.py"),
        "original_blocks_sha256": sha256(PROJECT_ROOT / "src/mri_lgformer/blocks.py"),
        "train_data_accessed": False,
        "validation_accessed": False,
        "test_accessed": False,
        "variants": results,
        "passed": all(item["passed"] for item in results),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
