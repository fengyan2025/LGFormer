"""Frozen image-quality metrics for same-grid normalized MRI."""

from __future__ import annotations

import math

import numpy as np
from scipy.ndimage import gaussian_laplace
from skimage.metrics import structural_similarity


def psnr(mse: float, data_range: float = 2.0) -> float:
    return 10.0 * math.log10(data_range**2 / max(mse, 1e-12))


def hfen(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_log = gaussian_laplace(prediction, sigma=1.5)
    target_log = gaussian_laplace(target, sigma=1.5)
    return float(
        np.linalg.norm(pred_log - target_log)
        / max(np.linalg.norm(target_log), 1e-12)
    )


def gradient_mae(prediction: np.ndarray, target: np.ndarray) -> float:
    pred_x = np.diff(prediction, axis=1)
    true_x = np.diff(target, axis=1)
    pred_y = np.diff(prediction, axis=0)
    true_y = np.diff(target, axis=0)
    return float(
        np.mean(np.abs(pred_x - true_x))
        + np.mean(np.abs(pred_y - true_y))
    )


def sample_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    foreground_threshold: float = -0.95,
) -> dict[str, float]:
    difference = prediction - target
    absolute = np.abs(difference)
    foreground = target > foreground_threshold
    if not np.any(foreground):
        foreground = np.ones_like(target, dtype=bool)
    global_ssim, ssim_map = structural_similarity(
        target,
        prediction,
        data_range=2.0,
        full=True,
    )
    foreground_error = absolute[foreground]
    return {
        "psnr": psnr(float(np.mean(difference**2))),
        "foreground_psnr": psnr(float(np.mean(difference[foreground] ** 2))),
        "ssim": float(global_ssim),
        "foreground_ssim": float(np.mean(ssim_map[foreground])),
        "mae": float(np.mean(absolute)),
        "foreground_mae": float(np.mean(foreground_error)),
        "nmse": float(
            np.sum(difference**2) / max(float(np.sum(target**2)), 1e-12)
        ),
        "hfen": hfen(prediction, target),
        "gradient_mae": gradient_mae(prediction, target),
        "foreground_abs_error_p95": float(np.quantile(foreground_error, 0.95)),
        "foreground_abs_error_p99": float(np.quantile(foreground_error, 0.99)),
        "mean_bias": float(np.mean(difference)),
        "foreground_mean_bias": float(np.mean(difference[foreground])),
    }


def summarize(rows: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    metric_names = [
        key.removeprefix("output_")
        for key in rows[0]
        if key.startswith("output_")
    ]
    result: dict[str, dict[str, float]] = {}
    for metric in metric_names:
        values = np.asarray(
            [float(row[f"output_{metric}"]) for row in rows],
            dtype=np.float64,
        )
        result[metric] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "median": float(np.median(values)),
            "p05": float(np.quantile(values, 0.05)),
            "p95": float(np.quantile(values, 0.95)),
        }
    return result
