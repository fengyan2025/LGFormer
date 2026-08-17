#!/usr/bin/env python3
"""Evaluate one ablation checkpoint on Validation only; Test is unsupported."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ablation_models import build_ablation_model
from mri_lgformer.dataset import PairedT1Dataset
from mri_lgformer.metrics import sample_metrics, summarize
from mri_lgformer.reproducibility import git_commit, sha256


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config["freeze"]["test_access"] != "forbidden":
        raise RuntimeError("Test must remain forbidden")
    validation_csv = resolve(config["paths"]["validation_csv"])
    if sha256(validation_csv) != config["freeze"]["validation_csv_sha256"]:
        raise RuntimeError("Validation CSV SHA-256 mismatch")
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(output_dir)
    output_dir.mkdir(parents=True)
    dataset = PairedT1Dataset(validation_csv, resolve(config["paths"]["data_root"]))
    subjects = len({row["subject_id"] for row in dataset.rows})
    if len(dataset) != 2140 or subjects != 107:
        raise RuntimeError("Validation inventory mismatch")
    checkpoint_path = args.checkpoint.resolve()
    payload = torch.load(checkpoint_path, map_location="cpu")
    model = build_ablation_model(payload["config"]["model"]).cuda().eval()
    model.load_state_dict(payload["model"], strict=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, persistent_workers=True)
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            low = batch["low"].cuda(non_blocking=True)
            high = batch["high"].cuda(non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = model(low).clamp(-1.0, 1.0)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Non-finite Validation prediction")
            low_np = low.float().cpu().numpy()[:, 0]
            high_np = high.float().cpu().numpy()[:, 0]
            pred_np = prediction.float().cpu().numpy()[:, 0]
            for index in range(len(pred_np)):
                output_metrics = sample_metrics(pred_np[index], high_np[index])
                input_metrics = sample_metrics(low_np[index], high_np[index])
                rows.append({
                    "pair_id": batch["pair_id"][index],
                    "subject_id": batch["subject_id"][index],
                    "slice_id": int(batch["slice_id"][index]),
                    **{f"output_{key}": value for key, value in output_metrics.items()},
                    **{f"input_{key}": value for key, value in input_metrics.items()},
                })
    with (output_dir / "per_pair_metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "split": "validation",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_step": int(payload["step"]),
        "variant": config["model"]["variant"],
        "validation_csv_sha256": sha256(validation_csv),
        "pairs": len(rows),
        "subjects": subjects,
        "git_commit": git_commit(),
        "output": summarize(rows),
        "test_accessed": False,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
