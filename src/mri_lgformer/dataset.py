"""T1-only paired HCP dataset for same-grid MRI enhancement."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset


REQUIRED_COLUMNS = {
    "pair_id",
    "subject_id",
    "modality",
    "slice_id",
    "low_path",
    "high_path",
}


class PairedT1Dataset(Dataset):
    """Load normalized 2D low/reference NPZ pairs listed in a CSV manifest."""

    def __init__(
        self,
        split_csv: str | Path,
        data_root: str | Path,
        *,
        validate_paths: bool = False,
    ) -> None:
        self.split_csv = Path(split_csv).resolve()
        self.data_root = Path(data_root).resolve()
        with self.split_csv.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = REQUIRED_COLUMNS.difference(reader.fieldnames or ())
            if missing:
                raise ValueError(f"{self.split_csv} lacks columns: {sorted(missing)}")
            self.rows = [dict(row) for row in reader]
        if not self.rows:
            raise ValueError(f"No rows in {self.split_csv}")
        modalities = {row["modality"] for row in self.rows}
        if modalities != {"T1w"}:
            raise ValueError(
                f"Standalone project accepts T1w-only CSVs, got {modalities}"
            )
        if validate_paths:
            missing_paths = [
                str(path)
                for row in self.rows
                for path in (self._resolve(row["low_path"]), self._resolve(row["high_path"]))
                if not path.is_file()
            ]
            if missing_paths:
                raise FileNotFoundError(
                    f"{len(missing_paths)} files are missing; first={missing_paths[0]}"
                )

    def __len__(self) -> int:
        return len(self.rows)

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else self.data_root / path

    @staticmethod
    def _load(path: Path) -> torch.Tensor:
        with np.load(path, allow_pickle=False) as payload:
            image = np.asarray(payload["image"], dtype=np.float32)
        if image.ndim != 2 or not np.all(np.isfinite(image)):
            raise ValueError(f"Invalid image at {path}: shape={image.shape}")
        return torch.from_numpy(image.copy()).unsqueeze(0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        low_path = self._resolve(row["low_path"])
        high_path = self._resolve(row["high_path"])
        low = self._load(low_path)
        high = self._load(high_path)
        if low.shape != high.shape:
            raise ValueError(
                f"Shape mismatch for {row['pair_id']}: {low.shape} vs {high.shape}"
            )
        return {
            "low": low,
            "high": high,
            "subject_id": row["subject_id"],
            "modality": "T1w",
            "slice_id": int(row["slice_id"]),
            "pair_id": row["pair_id"],
            "low_path": str(low_path),
            "high_path": str(high_path),
        }
