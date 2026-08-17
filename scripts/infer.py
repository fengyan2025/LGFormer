#!/usr/bin/env python3
"""Run MRI-LGFormer-T1 inference on one normalized NPZ image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from mri_lgformer import build_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_image(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as payload:
        if "image" not in payload:
            raise KeyError(f"{path} does not contain an 'image' array")
        image = np.asarray(payload["image"], dtype=np.float32)
    if image.ndim != 2:
        raise ValueError(f"Expected a 2D image, got shape={image.shape}")
    if not np.all(np.isfinite(image)):
        raise ValueError("Input contains non-finite values")
    if image.shape[0] % 8 or image.shape[1] % 8:
        raise ValueError(f"Image dimensions must be divisible by 8, got {image.shape}")
    return image


def main() -> None:
    args = parse_args()
    checkpoint_path = args.checkpoint.expanduser().resolve()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    device = torch.device(args.device)

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model_config = checkpoint.get("config", {}).get("model", {})
    model = build_model(model_config).to(device).eval()
    model.load_state_dict(checkpoint["model"], strict=True)

    image = load_image(input_path)
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0).to(device)
    with torch.inference_mode():
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                prediction = model(tensor)
        else:
            prediction = model(tensor)
    prediction = prediction.clamp(-1.0, 1.0).float().cpu().numpy()[0, 0]
    if not np.all(np.isfinite(prediction)):
        raise FloatingPointError("Model produced non-finite output")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, image=prediction.astype(np.float32))
    print(f"Saved enhanced image to {output_path}")


if __name__ == "__main__":
    main()
