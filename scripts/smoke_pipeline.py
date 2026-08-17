#!/usr/bin/env python3
"""Two-step engineering smoke test without writing a training checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mri_lgformer import MRILGFormerT1
from mri_lgformer.augmentations import PairedTrainingView
from mri_lgformer.dataset import PairedT1Dataset
from mri_lgformer.losses import CharbonnierLoss, gradient_loss
from mri_lgformer.reproducibility import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    seed_everything(20260730)
    dataset = PairedTrainingView(
        PairedT1Dataset(args.train_csv, args.data_root),
        crop_size=64,
        augment=True,
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model = MRILGFormerT1().cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
    criterion = CharbonnierLoss()
    losses = []
    updated = False
    parameter_before = next(model.parameters()).detach().clone()
    for step, batch in enumerate(loader, start=1):
        low = batch["low"].cuda()
        high = batch["high"].cuda()
        prediction = model(low)
        loss = criterion(prediction, high) + 0.1 * gradient_loss(prediction, high)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite smoke loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        if step == 2:
            break
    updated = not torch.equal(parameter_before, next(model.parameters()).detach())
    result = {
        "status": "pass" if updated and len(losses) == 2 else "fail",
        "steps": len(losses),
        "losses": losses,
        "parameters_updated": updated,
        "input_shape": [1, 1, 64, 64],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
