"""Network modules for the Spatial-Spectral Bi-directional GAN (SS-BiGAN)."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class SpatialSpectralUnmixer(nn.Module):
    """3D-CNN unmixing generator.

    Input is a ``(Batch, 1, Bands, P, P)`` spatial-spectral patch. The output is
    the abundance vector of the patch's center pixel, with Softmax enforcing the
    non-negativity constraint (ANC) and sum-to-one constraint (ASC).
    """

    def __init__(self, num_bands: int, num_endmembers: int, patch_size: int = 3):
        super().__init__()
        self.num_bands = num_bands
        self.num_endmembers = num_endmembers
        self.patch_size = patch_size

        self.conv1 = nn.Conv3d(
            in_channels=1,
            out_channels=16,
            kernel_size=(7, 3, 3),
            padding=(3, 1, 1),
        )
        self.conv2 = nn.Conv3d(
            in_channels=16,
            out_channels=32,
            kernel_size=(5, 3, 3),
            padding=(2, 0, 0),
        )

        final_spatial_size = patch_size - 2
        if final_spatial_size <= 0:
            raise ValueError(
                "patch_size must be >= 3 so the second conv has a spatial output."
            )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(
            32 * num_bands * final_spatial_size * final_spatial_size, 128
        )
        self.fc2 = nn.Linear(128, num_endmembers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = self.flatten(x)
        x = F.leaky_relu(self.fc1(x), 0.2)
        return F.softmax(self.fc2(x), dim=1)


class NonlinearMixer(nn.Module):
    """Generator G_mix: maps an abundance vector to a reconstructed spectrum.

    The output is composed of a physics-grounded linear term (``A @ E``) plus a
    small nonlinear residual predicted by an MLP. This residual-style
    parameterisation stabilises GAN training and lets the network reduce to the
    linear mixing model when the scene is only weakly nonlinear.
    """

    def __init__(
        self,
        num_bands: int,
        num_endmembers: int,
        endmembers_matrix: np.ndarray,
        nonlinear_scale: float = 0.1,
        trainable_endmembers: bool = False,
    ):
        super().__init__()
        self.num_bands = num_bands
        self.num_endmembers = num_endmembers
        self.nonlinear_scale = nonlinear_scale

        e_tensor = torch.as_tensor(endmembers_matrix, dtype=torch.float32)
        if e_tensor.shape != (num_endmembers, num_bands):
            raise ValueError(
                f"Endmember matrix must have shape ({num_endmembers}, {num_bands}), "
                f"got {tuple(e_tensor.shape)}."
            )
        self.E = nn.Parameter(e_tensor, requires_grad=trainable_endmembers)

        self.nonlinear_net = nn.Sequential(
            nn.Linear(num_endmembers, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, num_bands),
            nn.Tanh(),
        )

    def forward(self, abundance: torch.Tensor) -> torch.Tensor:
        linear_part = torch.matmul(abundance, self.E)
        nonlinear_residual = self.nonlinear_net(abundance)
        return linear_part + self.nonlinear_scale * nonlinear_residual


class SpectralDiscriminator(nn.Module):
    """MLP discriminator that distinguishes real and reconstructed spectra."""

    def __init__(self, num_bands: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_bands, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

    def forward(self, spectra: torch.Tensor) -> torch.Tensor:
        return self.net(spectra)
