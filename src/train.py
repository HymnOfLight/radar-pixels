"""Training entrypoint for the Spatial-Spectral Bi-directional GAN."""
from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .data import create_patches, load_mat_hsi_dataset
from .losses import rmse, sad_loss
from .models import (
    NonlinearMixer,
    SpatialSpectralUnmixer,
    SpectralDiscriminator,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the Spatial-Spectral Bi-directional GAN (SS-BiGAN)"
    )
    parser.add_argument("--dataset", choices=["mock", "samson", "jasper", "urban"], default="mock")
    parser.add_argument("--data-root", default="datasets")
    parser.add_argument("--patch-size", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--d-lr-scale", type=float, default=0.1)
    parser.add_argument("--d-update-interval", type=int, default=2)
    parser.add_argument("--label-real", type=float, default=0.9)
    parser.add_argument("--label-fake", type=float, default=0.1)
    parser.add_argument("--w-forward", type=float, default=10.0)
    parser.add_argument("--w-backward", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--save-dir", default="checkpoints")
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _first_existing(*paths: str) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def _load_dataset(args: argparse.Namespace) -> dict:
    """Prepare the (hsi_image, endmembers, gt_abundances) triple."""
    if args.dataset == "mock":
        bands = 156
        endmembers = 4
        hsi = np.random.rand(40, 40, bands).astype(np.float32)
        em = np.random.rand(endmembers, bands).astype(np.float32)
        return {"hsi": hsi, "endmembers": em, "abundances": None}

    root = args.data_root

    if args.dataset == "samson":
        data_path = _first_existing(
            os.path.join(root, "Samson", "Samson.mat"),
            os.path.join(root, "Samson", "samson_1.mat"),
        )
        gt_path = _first_existing(
            os.path.join(root, "Samson", "Samson_GT.mat"),
            os.path.join(root, "Samson", "end3.mat"),
        )
        if data_path is None:
            raise FileNotFoundError(
                f"Could not find Samson data under {root}/Samson. "
                "Run scripts/download_datasets.sh first."
            )
        return load_mat_hsi_dataset(data_path, gt_path, height=95, width=95)

    if args.dataset == "jasper":
        data_path = _first_existing(
            os.path.join(root, "JasperRidge", "jasperRidge2_R198.mat"),
        )
        gt_path = _first_existing(
            os.path.join(root, "JasperRidge", "Jasper_GT.mat"),
            os.path.join(root, "JasperRidge", "end4.mat"),
        )
        if data_path is None:
            raise FileNotFoundError(
                f"Could not find Jasper Ridge data under {root}/JasperRidge. "
                "Run scripts/download_datasets.sh first."
            )
        return load_mat_hsi_dataset(data_path, gt_path, height=100, width=100)

    if args.dataset == "urban":
        data_path = _first_existing(
            os.path.join(root, "Urban", "Urban.mat"),
            os.path.join(root, "Urban", "Urban_R162.mat"),
        )
        gt_path = _first_existing(
            os.path.join(root, "Urban", "end4_groundTruth.mat"),
            os.path.join(root, "Urban", "end5_groundTruth.mat"),
            os.path.join(root, "Urban", "end6_groundTruth.mat"),
        )
        if data_path is None:
            raise FileNotFoundError(
                f"Could not find Urban data under {root}/Urban. "
                "Run scripts/download_datasets.sh first."
            )
        return load_mat_hsi_dataset(data_path, gt_path, height=307, width=307)
    if args.dataset == "samson":
        data_path = os.path.join(args.data_root, "Samson", "samson_1.mat")
        gt_path = os.path.join(args.data_root, "Samson", "end3.mat")
        return load_mat_hsi_dataset(data_path, gt_path, height=95, width=95)

    if args.dataset == "jasper":
        data_path = os.path.join(
            args.data_root, "JasperRidge", "jasperRidge2_R198.mat"
        )
        gt_path = os.path.join(args.data_root, "JasperRidge", "end4.mat")
        return load_mat_hsi_dataset(data_path, gt_path, height=100, width=100)

    if args.dataset == "urban":
        data_path = os.path.join(args.data_root, "Urban", "Urban_R162.mat")
        gt_path = os.path.join(args.data_root, "Urban", "end6_groundTruth.mat")
        return load_mat_hsi_dataset(
            data_path, gt_path, height=307, width=307
        )

    raise ValueError(f"Unknown dataset: {args.dataset}")


def _evaluate_abundances(
    unmixer: SpatialSpectralUnmixer,
    patches: torch.Tensor,
    gt_abundances: Optional[np.ndarray],
    device: torch.device,
    batch_size: int,
) -> Optional[float]:
    if gt_abundances is None:
        return None
    unmixer.eval()
    preds = []
    with torch.no_grad():
        for start in range(0, patches.size(0), batch_size):
            pb = patches[start : start + batch_size].to(device)
            preds.append(unmixer(pb).cpu())
    unmixer.train()
    predicted = torch.cat(preds, dim=0).numpy()  # (N, E)

    num_endmembers = predicted.shape[1]
    gt = gt_abundances.reshape(num_endmembers, -1).T  # (N, E)
    if gt.shape != predicted.shape:
        return None
    return float(np.sqrt(((predicted - gt) ** 2).mean()))


def train(args: argparse.Namespace) -> None:
    _set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[SS-BiGAN] device: {device}")

    print(f"[SS-BiGAN] Loading dataset: {args.dataset}")
    data = _load_dataset(args)
    hsi = data["hsi"]
    endmembers = data["endmembers"]
    gt_abundances = data.get("abundances")

    if endmembers is None:
        print(
            "[SS-BiGAN] No ground-truth endmembers found; using random endmembers. "
            "For a real experiment, extract them with VCA or N-FINDR first."
        )
        endmembers = np.random.rand(4, hsi.shape[-1]).astype(np.float32)

    num_bands = int(hsi.shape[-1])
    num_endmembers = int(endmembers.shape[0])
    print(
        f"[SS-BiGAN] HSI shape: {hsi.shape} | bands: {num_bands} | endmembers: {num_endmembers}"
    )

    patches_tensor, center_spectra_tensor = create_patches(hsi, patch_size=args.patch_size)
    dataset = torch.utils.data.TensorDataset(patches_tensor, center_spectra_tensor)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True
    )

    g_unmix = SpatialSpectralUnmixer(num_bands, num_endmembers, args.patch_size).to(device)
    g_mix = NonlinearMixer(num_bands, num_endmembers, endmembers).to(device)
    discriminator = SpectralDiscriminator(num_bands).to(device)

    optimizer_g = optim.Adam(
        list(g_unmix.parameters()) + list(g_mix.parameters()),
        lr=args.lr,
        betas=(0.5, 0.999),
    )
    optimizer_d = optim.Adam(
        discriminator.parameters(),
        lr=args.lr * args.d_lr_scale,
        betas=(0.5, 0.999),
    )

    criterion_bce = nn.BCELoss()
    criterion_mse = nn.MSELoss()

    print("[SS-BiGAN] Starting bi-directional GAN training (TTUR + label smoothing + D/G pacing).")
    for epoch in range(args.epochs):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        d_updates = 0

        for i, (patches, real_spectra) in enumerate(dataloader):
            patches = patches.to(device)
            real_spectra = real_spectra.to(device)
            batch_sz = patches.size(0)

            valid = torch.full((batch_sz, 1), args.label_real, device=device)
            fake = torch.full((batch_sz, 1), args.label_fake, device=device)

            if i % args.d_update_interval == 0:
                optimizer_d.zero_grad()

                pred_abundance_d = g_unmix(patches)
                fake_spectra_d = g_mix(pred_abundance_d)

                real_loss = criterion_bce(discriminator(real_spectra), valid)
                fake_loss = criterion_bce(discriminator(fake_spectra_d.detach()), fake)
                d_loss = 0.5 * (real_loss + fake_loss)

                d_loss.backward()
                optimizer_d.step()

                epoch_d_loss += float(d_loss.item())
                d_updates += 1

            optimizer_g.zero_grad()

            pred_abundance_g = g_unmix(patches)
            fake_spectra_g = g_mix(pred_abundance_g)

            adv_loss = criterion_bce(discriminator(fake_spectra_g), valid)
            mse_recon = criterion_mse(fake_spectra_g, real_spectra)
            sad_recon = sad_loss(fake_spectra_g, real_spectra)
            forward_loss = mse_recon + 0.1 * sad_recon

            rand_abundance = torch.rand(batch_sz, num_endmembers, device=device)
            rand_abundance = rand_abundance / rand_abundance.sum(dim=1, keepdim=True)

            fake_spectra_backward = g_mix(rand_abundance)
            fake_patches = (
                fake_spectra_backward.view(batch_sz, 1, num_bands, 1, 1)
                .expand(-1, -1, -1, args.patch_size, args.patch_size)
            )
            recovered_abundance = g_unmix(fake_patches)
            backward_loss = criterion_mse(recovered_abundance, rand_abundance)

            g_loss = (
                adv_loss
                + args.w_forward * forward_loss
                + args.w_backward * backward_loss
            )
            g_loss.backward()
            optimizer_g.step()

            epoch_g_loss += float(g_loss.item())

        if (epoch + 1) % args.log_every == 0 or epoch == 0:
            avg_d = epoch_d_loss / max(d_updates, 1)
            avg_g = epoch_g_loss / len(dataloader)
            rmse_ab = _evaluate_abundances(
                g_unmix, patches_tensor, gt_abundances, device, args.batch_size
            )
            rmse_str = f" | Abundance RMSE: {rmse_ab:.4f}" if rmse_ab is not None else ""
            print(
                f"Epoch [{epoch + 1}/{args.epochs}] | D Loss: {avg_d:.4f} | "
                f"G Loss: {avg_g:.4f}{rmse_str}"
            )

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        ckpt = os.path.join(args.save_dir, f"ss_bigan_{args.dataset}.pt")
        torch.save(
            {
                "g_unmix": g_unmix.state_dict(),
                "g_mix": g_mix.state_dict(),
                "discriminator": discriminator.state_dict(),
                "num_bands": num_bands,
                "num_endmembers": num_endmembers,
                "patch_size": args.patch_size,
            },
            ckpt,
        )
        print(f"[SS-BiGAN] Saved checkpoint to {ckpt}")

    print("[SS-BiGAN] Training complete.")


if __name__ == "__main__":
    train(parse_args())
