"""Data utilities: patch extraction and real .mat hyperspectral dataset loading."""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np
import torch


def create_patches(
    hsi_image: np.ndarray, patch_size: int = 3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert an HSI cube of shape (H, W, B) into 3D patches.

    Returns patches of shape (N, 1, B, P, P) and their center pixel spectra of
    shape (N, B).
    """
    h, w, b = hsi_image.shape
    pad = patch_size // 2
    padded_img = np.pad(
        hsi_image, ((pad, pad), (pad, pad), (0, 0)), mode="reflect"
    )

    patches = []
    center_spectra = []
    for i in range(h):
        for j in range(w):
            patch = padded_img[i : i + patch_size, j : j + patch_size, :]
            patch = np.transpose(patch, (2, 0, 1))  # (B, P, P)
            patches.append(patch)
            center_spectra.append(hsi_image[i, j, :])

    patches_np = np.array(patches, dtype=np.float32)
    patches_np = np.expand_dims(patches_np, axis=1)  # (N, 1, B, P, P)
    center_np = np.array(center_spectra, dtype=np.float32)
    return torch.from_numpy(patches_np), torch.from_numpy(center_np)


def _reshape_hsi(y: np.ndarray, height: int, width: int) -> np.ndarray:
    """Convert a (bands, pixels) or (pixels, bands) matrix into (H, W, B)."""
    if y.ndim == 3:
        if y.shape[-1] > y.shape[0]:
            return y
        return np.transpose(y, (1, 2, 0))
    num_pixels = height * width
    if y.shape[0] == num_pixels:
        bands = y.shape[1]
        return y.reshape((height, width, bands))
    if y.shape[1] == num_pixels:
        bands = y.shape[0]
        return y.T.reshape((height, width, bands))
    raise ValueError(
        f"Cannot reshape data with shape {y.shape} to image of size {height}x{width}."
    )


def load_mat_hsi_dataset(
    data_mat_path: str,
    gt_mat_path: Optional[str] = None,
    height: int = 95,
    width: int = 95,
    data_keys: Tuple[str, ...] = ("Y", "V", "X", "data"),
    endmember_keys: Tuple[str, ...] = ("M", "U", "E", "endmembers"),
    abundance_keys: Tuple[str, ...] = ("A", "S", "abundances"),
) -> dict:
    """Load a hyperspectral .mat dataset.

    Works with the common Samson / Jasper Ridge / Urban formats distributed
    alongside many open-source unmixing projects.
    """
    try:
        import scipy.io as sio
    except ImportError as exc:
        raise ImportError(
            "scipy is required to read .mat files. Install it via 'pip install scipy'."
        ) from exc

    if not os.path.exists(data_mat_path):
        raise FileNotFoundError(f"Data file not found: {data_mat_path}")

    data_dict = sio.loadmat(data_mat_path)
    y = None
    for key in data_keys:
        if key in data_dict:
            y = data_dict[key]
            break
    if y is None:
        # Fall back to the first non-private key that looks like a matrix.
        candidates = [
            (k, v)
            for k, v in data_dict.items()
            if not k.startswith("__") and hasattr(v, "shape") and v.ndim >= 2
        ]
        if not candidates:
            raise ValueError(
                f"Could not find hyperspectral data in {data_mat_path}. "
                f"Keys available: {list(data_dict.keys())}"
            )
        y = sorted(candidates, key=lambda kv: -np.prod(kv[1].shape))[0][1]

    y = np.asarray(y, dtype=np.float32)
    hsi_img = _reshape_hsi(y, height, width)
    max_v = float(hsi_img.max())
    if max_v > 0:
        hsi_img = hsi_img / max_v

    gt_endmembers = None
    gt_abundances = None

    def _search(dd: dict, keys: Tuple[str, ...]):
        for k in keys:
            if k in dd:
                return dd[k]
        return None

    if gt_mat_path and os.path.exists(gt_mat_path):
        gt_dict = sio.loadmat(gt_mat_path)
        gt_endmembers = _search(gt_dict, endmember_keys)
        gt_abundances = _search(gt_dict, abundance_keys)
    else:
        gt_endmembers = _search(data_dict, endmember_keys)
        gt_abundances = _search(data_dict, abundance_keys)

    if gt_endmembers is not None:
        gt_endmembers = np.asarray(gt_endmembers, dtype=np.float32)
        # Ensure shape is (num_endmembers, num_bands)
        num_bands = hsi_img.shape[-1]
        if gt_endmembers.shape[0] == num_bands:
            gt_endmembers = gt_endmembers.T
        if gt_endmembers.max() > 1.0:
            gt_endmembers = gt_endmembers / gt_endmembers.max()

    if gt_abundances is not None:
        gt_abundances = np.asarray(gt_abundances, dtype=np.float32)
        # Ensure shape is (num_endmembers, height, width)
        if gt_abundances.ndim == 2:
            ne = gt_abundances.shape[0]
            if gt_abundances.shape[1] == height * width:
                gt_abundances = gt_abundances.reshape((ne, height, width))
            elif gt_abundances.shape[0] == height * width:
                gt_abundances = gt_abundances.T.reshape((ne, height, width))

    return {
        "hsi": hsi_img,
        "endmembers": gt_endmembers,
        "abundances": gt_abundances,
    }
