#!/usr/bin/env python3
"""Train one independent ablation to fixed step 120k on the 140k LR path."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ablation_models import build_ablation_model
from mri_lgformer.augmentations import PairedTrainingView
from mri_lgformer.checkpoint import save_model_checkpoint, save_recovery_state
from mri_lgformer.dataset import PairedT1Dataset
from mri_lgformer.losses import CharbonnierLoss, gradient_loss
from mri_lgformer.reproducibility import git_commit, infinite_batches, seed_everything, sha256


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def inventory(path: Path) -> tuple[int, int, set[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return len(rows), len({row["subject_id"] for row in rows}), {row["modality"] for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    train_cfg = config["training"]
    freeze = config["freeze"]
    if int(train_cfg["max_steps"]) != 120000:
        raise RuntimeError("Formal ablation endpoint must be fixed at 120000")
    if int(train_cfg["scheduler_t_max"]) != 140000:
        raise RuntimeError("Scheduler T_max must remain the original 140000")
    if int(freeze["formal_comparison_step"]) != 120000:
        raise RuntimeError("Frozen comparison step mismatch")
    if freeze["test_access"] != "forbidden":
        raise RuntimeError("Test must remain forbidden")

    train_csv = resolve(config["paths"]["train_csv"])
    data_root = resolve(config["paths"]["data_root"])
    output_dir = resolve(config["paths"]["output_dir"])
    if sha256(train_csv) != freeze["train_csv_sha256"]:
        raise RuntimeError("Train CSV SHA-256 mismatch")
    rows, subjects, modalities = inventory(train_csv)
    if {"pairs": rows, "subjects": subjects} != config["expected_inventory"]:
        raise RuntimeError("Formal Train inventory mismatch")
    if modalities != {"T1w"}:
        raise RuntimeError("Formal Train must be T1-only")
    if args.resume is None:
        if output_dir.exists():
            raise FileExistsError(f"Refusing existing output directory: {output_dir}")
        output_dir.mkdir(parents=True)
    elif not output_dir.is_dir():
        raise FileNotFoundError(output_dir)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    seed = int(train_cfg["seed"])
    seed_everything(seed)
    torch.backends.cudnn.benchmark = bool(train_cfg["cudnn_benchmark"])
    torch.set_float32_matmul_precision("high")
    base = PairedT1Dataset(train_csv, data_root)
    dataset = PairedTrainingView(base, crop_size=int(train_cfg["crop_size"]), augment=True)
    loader_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=True,
        drop_last=True,
        persistent_workers=int(train_cfg["num_workers"]) > 0,
        generator=loader_generator,
    )
    device = torch.device("cuda")
    model = build_ablation_model(config["model"]).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(train_cfg["scheduler_t_max"]),
        eta_min=float(train_cfg["eta_min"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    start_step = 0
    overflow_total = 0
    overflow_consecutive = 0
    if args.resume is not None:
        payload = torch.load(args.resume.resolve(), map_location="cpu")
        if payload["config"]["config_sha256"] != sha256(config_path):
            raise RuntimeError("Resume config SHA-256 mismatch")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        scheduler.load_state_dict(payload["scheduler"])
        scaler.load_state_dict(payload["scaler"])
        loader_generator.set_state(payload["loader_generator_state"])
        random.setstate(payload["python_rng_state"])
        np.random.set_state(payload["numpy_rng_state"])
        torch.set_rng_state(payload["torch_rng_state"])
        torch.cuda.set_rng_state_all(payload["cuda_rng_state_all"])
        start_step = int(payload["step"])
        overflow_total = int(payload.get("amp_overflow_total", 0))
        overflow_consecutive = int(payload.get("amp_overflow_consecutive", 0))

    frozen_config = {
        **config,
        "protocol": "Independent fixed-120k ablation on original 140k cosine path",
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "train_csv_sha256": sha256(train_csv),
        "git_commit": git_commit(),
        "parameter_count": sum(p.numel() for p in model.parameters()),
        "formal_test_accessed": False,
        "historical_test_accessed": False,
    }
    config_output = output_dir / "config.json"
    if config_output.exists() and args.resume is None:
        raise FileExistsError(config_output)
    config_output.write_text(json.dumps(frozen_config, indent=2), encoding="utf-8")
    metrics_path = output_dir / "train_metrics.jsonl"
    amp_path = output_dir / "amp_overflow_events.jsonl"
    if metrics_path.exists() and args.resume is None:
        raise FileExistsError(metrics_path)

    criterion = CharbonnierLoss(float(train_cfg["charbonnier_epsilon"]))
    iterator = infinite_batches(loader)
    gradient_weight = float(train_cfg["gradient_weight"])
    checkpoint_start = int(train_cfg["checkpoint_start"])
    checkpoint_every = int(train_cfg["checkpoint_every"])
    recovery_every = int(train_cfg["recovery_every"])
    log_every = int(train_cfg["log_every"])
    max_steps = int(train_cfg["max_steps"])
    max_consecutive = int(config["safety"]["max_consecutive_overflows"])
    max_total = int(config["safety"]["max_total_overflows"])
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.time()
    for step in range(start_step + 1, max_steps + 1):
        batch = next(iterator)
        low = batch["low"].to(device, non_blocking=True)
        high = batch["high"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = model(low)
            pixel = criterion(prediction, high)
            grad = gradient_loss(prediction, high)
            loss = pixel + gradient_weight * grad
        if prediction.shape != high.shape or not torch.isfinite(loss):
            raise FloatingPointError(f"Invalid formal ablation step={step}")
        scale_before = float(scaler.get_scale())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        grad_finite = bool(torch.isfinite(grad_norm).item())
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        overflow = (not grad_finite) or scale_after < scale_before
        if overflow:
            overflow_total += 1
            overflow_consecutive += 1
            event = {"step": step, "scale_before": scale_before, "scale_after": scale_after, "total": overflow_total, "consecutive": overflow_consecutive}
            with amp_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
            print(json.dumps({"event": "amp_overflow", **event}), flush=True)
        else:
            overflow_consecutive = 0
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        if overflow_consecutive > max_consecutive or overflow_total > max_total:
            raise FloatingPointError("AMP safety threshold exceeded")
        if step == 1 or step % log_every == 0:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "pixel_loss": float(pixel.detach().cpu()),
                "gradient_loss": float(grad.detach().cpu()),
                "gradient_norm": float(grad_norm.detach().cpu()) if grad_finite else None,
                "amp_overflow": overflow,
                "amp_overflow_total": overflow_total,
                "amp_scale": scale_after,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.time() - started,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        if step >= checkpoint_start and step % checkpoint_every == 0:
            checkpoint = output_dir / f"checkpoint_step_{step:06d}.pth"
            if checkpoint.exists():
                raise FileExistsError(checkpoint)
            save_model_checkpoint(checkpoint, model=model, step=step, config=frozen_config)
        if step % recovery_every == 0:
            save_recovery_state(
                output_dir / "recovery_state.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                loader_generator=loader_generator,
                step=step,
                config=frozen_config,
                amp_overflow_total=overflow_total,
                amp_overflow_consecutive=overflow_consecutive,
            )
    observed_lr = float(optimizer.param_groups[0]["lr"])
    expected_lr = float(freeze["proposed_step120000_learning_rate"])
    if abs(observed_lr - expected_lr) > 1e-15:
        raise RuntimeError(f"Step-120k LR mismatch: {observed_lr} != {expected_lr}")
    save_model_checkpoint(output_dir / "final_model_step_120000.pth", model=model, step=max_steps, config=frozen_config)
    save_recovery_state(
        output_dir / "final_state_step_120000.pth",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        loader_generator=loader_generator,
        step=max_steps,
        config=frozen_config,
        amp_overflow_total=overflow_total,
        amp_overflow_consecutive=overflow_consecutive,
    )


if __name__ == "__main__":
    main()
