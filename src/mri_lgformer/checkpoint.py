"""Checkpoint serialization with explicit model and protocol metadata."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch


def save_model_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    step: int,
    config: dict,
) -> None:
    torch.save(
        {"model": model.state_dict(), "step": int(step), "config": config},
        Path(path),
    )


def save_recovery_state(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    loader_generator: torch.Generator,
    step: int,
    config: dict,
    amp_overflow_total: int,
    amp_overflow_consecutive: int,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "loader_generator_state": loader_generator.get_state(),
            "python_rng_state": random.getstate(),
            "numpy_rng_state": np.random.get_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
            "step": int(step),
            "config": config,
            "amp_overflow_total": int(amp_overflow_total),
            "amp_overflow_consecutive": int(amp_overflow_consecutive),
        },
        Path(path),
    )
