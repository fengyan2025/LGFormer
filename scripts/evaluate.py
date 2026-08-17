#!/usr/bin/env python3
"""Evaluate one frozen checkpoint on one explicitly named split."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mri_lgformer import build_model
from mri_lgformer.dataset import PairedT1Dataset
from mri_lgformer.metrics import sample_metrics, summarize
from mri_lgformer.reproducibility import git_commit, sha256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--expected-subjects", type=int, required=True)
    parser.add_argument("--expected-pairs", type=int, required=True)
    return parser.parse_args()


def resolve(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    split_csv = resolve(config["paths"][f"{args.split}_csv"])
    data_root = resolve(config["paths"]["data_root"])
    checkpoint_path = args.checkpoint.resolve()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True)

    dataset = PairedT1Dataset(split_csv, data_root)
    observed_subjects = len({row["subject_id"] for row in dataset.rows})
    if len(dataset) != args.expected_pairs:
        raise RuntimeError(
            f"Observed pairs={len(dataset)}, expected={args.expected_pairs}"
        )
    if observed_subjects != args.expected_subjects:
        raise RuntimeError(
            "Observed subjects="
            f"{observed_subjects}, expected={args.expected_subjects}"
        )
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("config", {}).get("model", config["model"])
    model = build_model(model_config).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    if not all(
        torch.isfinite(value).all().item()
        for value in model.state_dict().values()
        if torch.is_tensor(value)
    ):
        raise FloatingPointError("Checkpoint contains non-finite model weights")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=args.num_workers > 0,
    )

    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for batch in loader:
            low = batch["low"].cuda(non_blocking=True)
            high = batch["high"].cuda(non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = model(low).clamp(-1.0, 1.0)
            if not torch.isfinite(prediction).all():
                raise FloatingPointError("Model produced a non-finite prediction")
            low_np = low.float().cpu().numpy()[:, 0]
            high_np = high.float().cpu().numpy()[:, 0]
            pred_np = prediction.float().cpu().numpy()[:, 0]
            for index in range(len(pred_np)):
                output_metrics = sample_metrics(pred_np[index], high_np[index])
                input_metrics = sample_metrics(low_np[index], high_np[index])
                rows.append(
                    {
                        "pair_id": batch["pair_id"][index],
                        "subject_id": batch["subject_id"][index],
                        "slice_id": int(batch["slice_id"][index]),
                        **{f"output_{key}": value for key, value in output_metrics.items()},
                        **{f"input_{key}": value for key, value in input_metrics.items()},
                    }
                )
    per_pair_path = output_dir / "per_pair_metrics.csv"
    with per_pair_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "split": args.split,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256(checkpoint_path),
        "checkpoint_step": int(checkpoint.get("step", -1)),
        "split_csv": str(split_csv),
        "split_csv_sha256": sha256(split_csv),
        "pairs": len(rows),
        "subjects": observed_subjects,
        "git_commit": git_commit(),
        "output": summarize(rows),
        "input": summarize(
            [
                {
                    **row,
                    **{
                        f"output_{key.removeprefix('input_')}": value
                        for key, value in row.items()
                        if key.startswith("input_")
                    },
                }
                for row in rows
            ]
        ),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
