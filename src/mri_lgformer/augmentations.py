"""Paired spatial augmentations."""

from __future__ import annotations

import torch
from torch.utils.data import Dataset


class PairedTrainingView(Dataset):
    def __init__(self, base: Dataset, crop_size: int, augment: bool = True) -> None:
        self.base = base
        self.crop_size = crop_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int):
        item = self.base[index]
        low, high = item["low"], item["high"]
        _, height, width = low.shape
        if self.crop_size > min(height, width):
            raise ValueError(f"crop_size={self.crop_size} exceeds image {low.shape}")
        top = int(torch.randint(0, height - self.crop_size + 1, (1,)).item())
        left = int(torch.randint(0, width - self.crop_size + 1, (1,)).item())
        low = low[:, top : top + self.crop_size, left : left + self.crop_size]
        high = high[:, top : top + self.crop_size, left : left + self.crop_size]
        if self.augment:
            if torch.rand(()) < 0.5:
                low, high = low.flip(-1), high.flip(-1)
            if torch.rand(()) < 0.5:
                low, high = low.flip(-2), high.flip(-2)
            k = int(torch.randint(0, 4, (1,)).item())
            low = torch.rot90(low, k, (-2, -1))
            high = torch.rot90(high, k, (-2, -1))
        return {**item, "low": low.contiguous(), "high": high.contiguous()}
