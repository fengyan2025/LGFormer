"""Training losses used by MRI-LGFormer-T1."""

from __future__ import annotations

import torch
from torch import nn


class CharbonnierLoss(nn.Module):
    def __init__(self, epsilon: float = 1e-3) -> None:
        super().__init__()
        self.epsilon = epsilon

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(
            torch.sqrt((prediction - target) ** 2 + self.epsilon**2)
        )


def gradient_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_x = prediction[..., :, 1:] - prediction[..., :, :-1]
    true_x = target[..., :, 1:] - target[..., :, :-1]
    pred_y = prediction[..., 1:, :] - prediction[..., :-1, :]
    true_y = target[..., 1:, :] - target[..., :-1, :]
    return torch.mean(torch.abs(pred_x - true_x)) + torch.mean(
        torch.abs(pred_y - true_y)
    )
