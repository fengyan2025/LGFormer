import numpy as np

from mri_lgformer.metrics import sample_metrics


def test_perfect_prediction():
    target = np.zeros((32, 32), dtype=np.float32)
    metrics = sample_metrics(target.copy(), target)
    assert metrics["mae"] == 0.0
    assert metrics["foreground_mae"] == 0.0
    assert metrics["nmse"] == 0.0
    assert metrics["hfen"] == 0.0
    assert metrics["psnr"] > 100.0
