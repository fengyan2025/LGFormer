#!/usr/bin/env python3
"""Train MRI-LGFormer-T1 from scratch under a fixed T1-only protocol."""

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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mri_lgformer import build_model
from mri_lgformer.augmentations import PairedTrainingView
from mri_lgformer.checkpoint import save_model_checkpoint, save_recovery_state
from mri_lgformer.dataset import PairedT1Dataset
from mri_lgformer.losses import CharbonnierLoss, gradient_loss
from mri_lgformer.reproducibility import (
    git_commit,
    infinite_batches,
    seed_everything,
    sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def resolve(project_root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (project_root / path).resolve()


def csv_inventory(path: Path) -> tuple[int, int, set[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    return (
        len(rows),
        len({row["subject_id"] for row in rows}),
        {row["modality"] for row in rows},
    )


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    paths = config["paths"]
    train_cfg = config["training"]
    safety_cfg = config["safety"]
    model_cfg = config["model"]

    data_root = resolve(PROJECT_ROOT, paths["data_root"])
    train_csv = resolve(PROJECT_ROOT, paths["train_csv"])
    output_dir = resolve(PROJECT_ROOT, paths["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    final_path = output_dir / "final_model.pth"
    metrics_path = output_dir / "train_metrics.jsonl"
    if final_path.exists():
        raise FileExistsError(f"Refusing to overwrite {final_path}")
    if metrics_path.exists() and args.resume is None:
        raise FileExistsError(f"Metrics exist without --resume: {metrics_path}")

    rows, subjects, modalities = csv_inventory(train_csv)
    expected = config["expected_inventory"]
    observed = {"pairs": rows, "subjects": subjects}
    if observed != expected:
        raise RuntimeError(f"Observed inventory={observed}, expected={expected}")
    if modalities != {"T1w"}:
        raise RuntimeError(f"Training CSV is not T1-only: {modalities}")

    seed = int(train_cfg["seed"])
    seed_everything(seed)
    torch.backends.cudnn.benchmark = bool(train_cfg.get("cudnn_benchmark", True))
    torch.set_float32_matmul_precision("high")

    train_base = PairedT1Dataset(train_csv, data_root)
    train_dataset = PairedTrainingView(
        train_base,
        crop_size=int(train_cfg["crop_size"]),
        augment=True,
    )
    loader_generator = torch.Generator()
    loader_generator.manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=True,
        drop_last=True,
        persistent_workers=int(train_cfg["num_workers"]) > 0,
        generator=loader_generator,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("The formal trainer requires CUDA")
    device = torch.device("cuda")
    model = build_model(model_cfg).to(device)
    criterion = CharbonnierLoss(float(train_cfg.get("charbonnier_epsilon", 1e-3)))
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        betas=(0.9, 0.999),
    )
    max_steps = int(train_cfg["max_steps"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max_steps,
        eta_min=float(train_cfg["eta_min"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    start_step = 0
    amp_overflow_total = 0
    amp_overflow_consecutive = 0
    if args.resume is not None:
        payload = torch.load(args.resume.resolve(), map_location="cpu")
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
        amp_overflow_total = int(payload.get("amp_overflow_total", 0))
        amp_overflow_consecutive = int(
            payload.get("amp_overflow_consecutive", 0)
        )

    frozen_config = {
        **config,
        "protocol": "Single-seed same-grid T1-only model training",
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "train_csv_sha256": sha256(train_csv),
        "git_commit": git_commit(),
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "formal_test_accessed": False,
        "historical_test_accessed": False,
    }
    (output_dir / "config.json").write_text(
        json.dumps(frozen_config, indent=2),
        encoding="utf-8",
    )

    checkpoint_every = int(train_cfg["checkpoint_every"])
    checkpoint_start = int(train_cfg["checkpoint_start"])
    recovery_every = int(train_cfg["recovery_every"])
    log_every = int(train_cfg["log_every"])
    gradient_weight = float(train_cfg["gradient_weight"])
    max_consecutive = int(safety_cfg["max_consecutive_overflows"])
    max_total = int(safety_cfg["max_total_overflows"])
    amp_events_path = output_dir / "amp_overflow_events.jsonl"

    iterator = infinite_batches(loader)
    model.train()
    started = time.time()
    optimizer.zero_grad(set_to_none=True)
    for step in range(start_step + 1, max_steps + 1):
        batch = next(iterator)
        low = batch["low"].to(device, non_blocking=True)
        high = batch["high"].to(device, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            prediction = model(low)
            pixel = criterion(prediction, high)
            grad = gradient_loss(prediction, high)
            loss = pixel + gradient_weight * grad
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at step {step}: {loss.item()}")

        scale_before = float(scaler.get_scale())
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norm_finite = bool(torch.isfinite(grad_norm).item())
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        overflow = (not grad_norm_finite) or scale_after < scale_before
        if overflow:
            amp_overflow_total += 1
            amp_overflow_consecutive += 1
            event = {
                "event": "amp_overflow",
                "step": step,
                "scale_before": scale_before,
                "scale_after": scale_after,
                "total": amp_overflow_total,
                "consecutive": amp_overflow_consecutive,
            }
            with amp_events_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(event) + "\n")
            print(json.dumps(event), flush=True)
        else:
            amp_overflow_consecutive = 0
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        if amp_overflow_consecutive > max_consecutive:
            raise FloatingPointError("Too many consecutive AMP overflows")
        if amp_overflow_total > max_total:
            raise FloatingPointError("Too many total AMP overflows")

        if step == 1 or step % log_every == 0:
            row = {
                "step": step,
                "loss": float(loss.detach().cpu()),
                "pixel_loss": float(pixel.detach().cpu()),
                "gradient_loss": float(grad.detach().cpu()),
                "gradient_norm": (
                    float(grad_norm.detach().cpu()) if grad_norm_finite else None
                ),
                "amp_overflow": overflow,
                "amp_overflow_total": amp_overflow_total,
                "amp_scale": scale_after,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "elapsed_seconds": time.time() - started,
            }
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)

        if (
            checkpoint_every > 0
            and step >= checkpoint_start
            and step % checkpoint_every == 0
        ):
            save_model_checkpoint(
                output_dir / f"checkpoint_step_{step:06d}.pth",
                model=model,
                step=step,
                config=frozen_config,
            )
        if recovery_every > 0 and step % recovery_every == 0:
            save_recovery_state(
                output_dir / "recovery_state.pth",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                loader_generator=loader_generator,
                step=step,
                config=frozen_config,
                amp_overflow_total=amp_overflow_total,
                amp_overflow_consecutive=amp_overflow_consecutive,
            )

    save_model_checkpoint(
        final_path,
        model=model,
        step=max_steps,
        config=frozen_config,
    )
    save_recovery_state(
        output_dir / "final_state.pth",
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        scaler=scaler,
        loader_generator=loader_generator,
        step=max_steps,
        config=frozen_config,
        amp_overflow_total=amp_overflow_total,
        amp_overflow_consecutive=amp_overflow_consecutive,
    )


if __name__ == "__main__":
    main()
