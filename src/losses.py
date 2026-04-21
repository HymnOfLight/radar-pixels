"""Custom losses used by SS-BiGAN."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def sad_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Spectral Angle Distance averaged over the batch.

    Both inputs are ``(B, num_bands)`` tensors. The result measures how similar
    two spectra are in shape independent of their absolute magnitude.
    """
    y_true = F.normalize(y_true, p=2, dim=1)
    y_pred = F.normalize(y_pred, p=2, dim=1)
    cosine_sim = torch.sum(y_true * y_pred, dim=1)
    cosine_sim = torch.clamp(cosine_sim, -1.0 + 1e-7, 1.0 - 1e-7)
    return torch.mean(torch.acos(cosine_sim))


def rmse(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Root Mean Squared Error between two tensors of the same shape."""
    return torch.sqrt(torch.mean((y_pred - y_true) ** 2))
