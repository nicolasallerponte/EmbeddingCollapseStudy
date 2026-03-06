"""
SimCLR training loop with geometry logging.

Designed to run on Google Colab (T4 GPU) or CPU.
Logs uniformity, alignment, and effective rank at regular intervals.

Usage:
    python src/train.py
    python src/train.py --tau 0.07 --batch_size 512
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm

from metrics import alignment, effective_rank, singular_value_distribution, uniformity


# ---------------------------------------------------------------------------
# Augmentation pipeline (SimCLR-style)
# ---------------------------------------------------------------------------

class SimCLRTransform:
    """Returns two randomly augmented views of the same image."""

    def __init__(self, image_size: int = 32):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010],
            ),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x)


# ---------------------------------------------------------------------------
# Model: ResNet-18 backbone + projection head
# ---------------------------------------------------------------------------

class SimCLRModel(nn.Module):
    def __init__(self, projection_dim: int = 128):
        super().__init__()
        backbone = models.resnet18(weights=None)
        self.encoder = nn.Sequential(*list(backbone.children())[:-1])  # remove fc
        self.projection_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encoder(x).squeeze(-1).squeeze(-1)  # (N, 512)
        z = self.projection_head(h)                  # (N, projection_dim)
        return F.normalize(z, dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Returns backbone features (for linear probe)."""
        with torch.no_grad():
            return self.encoder(x).squeeze(-1).squeeze(-1)


# ---------------------------------------------------------------------------
# InfoNCE loss
# ---------------------------------------------------------------------------

def infonce_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float) -> torch.Tensor:
    """
    NT-Xent (InfoNCE) loss for SimCLR.

    Args:
        z1, z2: (N, D) L2-normalized embeddings
        tau: temperature
    """
    N = z1.size(0)
    z = torch.cat([z1, z2], dim=0)          # (2N, D)
    sim = torch.mm(z, z.T) / tau            # (2N, 2N)

    # Mask out self-similarities
    mask = torch.eye(2 * N, dtype=torch.bool, device=z.device)
    sim.masked_fill_(mask, float("-inf"))

    # Positive pairs: (i, i+N) and (i+N, i)
    labels = torch.cat([
        torch.arange(N, 2 * N),
        torch.arange(0, N),
    ]).to(z.device)

    return F.cross_entropy(sim, labels)


# ---------------------------------------------------------------------------
# Geometry logging
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_geometry(model: SimCLRModel, loader: DataLoader, device: torch.device, n_batches: int = 10) -> dict:
    """
    Computes uniformity, alignment, and effective rank on a subset of the data.
    Uses projection head embeddings (z), consistent with training objective.
    """
    model.eval()
    z1_list, z2_list = [], []

    for i, ((x1, x2), _) in enumerate(loader):
        if i >= n_batches:
            break
        z1_list.append(model(x1.to(device)))
        z2_list.append(model(x2.to(device)))

    z1 = torch.cat(z1_list)
    z2 = torch.cat(z2_list)
    z_all = torch.cat([z1, z2])

    metrics = {
        "uniformity": uniformity(z_all).item(),
        "alignment": alignment(z1, z2).item(),
        "effective_rank": effective_rank(z_all),
    }

    # Top-5 singular values (normalized) for collapse diagnosis
    sv = singular_value_distribution(z_all)
    for i, v in enumerate(sv[:5].tolist()):
        metrics[f"sv_{i}"] = v

    model.train()
    return metrics


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(cfg: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Config: tau={cfg['tau']}, batch_size={cfg['batch_size']}, epochs={cfg['epochs']}")

    # Paths
    run_name = f"tau{cfg['tau']}_bs{cfg['batch_size']}"
    ckpt_dir = Path(cfg["checkpoint_dir"]) / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = ckpt_dir / "geometry_log.jsonl"

    # Dataset
    transform = SimCLRTransform(image_size=32)
    dataset = datasets.CIFAR10(
        root=cfg["data_root"], train=True, download=True, transform=transform
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=cfg["num_workers"],
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )

    # Model, optimizer
    model = SimCLRModel(projection_dim=cfg["projection_dim"]).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"]
    )

    # Training
    log_file = open(log_path, "w")

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        epoch_loss = 0.0

        for (x1, x2), _ in tqdm(loader, desc=f"Epoch {epoch}/{cfg['epochs']}", leave=False):
            x1, x2 = x1.to(device), x2.to(device)
            z1, z2 = model(x1), model(x2)
            loss = infonce_loss(z1, z2, tau=cfg["tau"])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        avg_loss = epoch_loss / len(loader)

        # Log geometry every N epochs
        if epoch % cfg["log_every"] == 0 or epoch == 1:
            geom = compute_geometry(model, loader, device)
            record = {"epoch": epoch, "loss": avg_loss, **geom}
            print(
                f"  Epoch {epoch:3d} | loss={avg_loss:.4f} | "
                f"unif={geom['uniformity']:.4f} | "
                f"align={geom['alignment']:.4f} | "
                f"erank={geom['effective_rank']:.2f}"
            )
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()

        # Save checkpoint
        if epoch % cfg["save_every"] == 0:
            ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": avg_loss,
                "cfg": cfg,
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    log_file.close()
    print(f"\nDone. Geometry log: {log_path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> dict:
    parser = argparse.ArgumentParser(description="SimCLR geometry study")
    parser.add_argument("--tau",            type=float, default=0.1)
    parser.add_argument("--batch_size",     type=int,   default=256)
    parser.add_argument("--epochs",         type=int,   default=200)
    parser.add_argument("--lr",             type=float, default=3e-4)
    parser.add_argument("--weight_decay",   type=float, default=1e-4)
    parser.add_argument("--projection_dim", type=int,   default=128)
    parser.add_argument("--data_root",      type=str,   default="data/")
    parser.add_argument("--checkpoint_dir", type=str,   default="outputs/checkpoints/")
    parser.add_argument("--num_workers",    type=int,   default=2)
    parser.add_argument("--log_every",      type=int,   default=10)
    parser.add_argument("--save_every",     type=int,   default=50)
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    cfg = parse_args()
    train(cfg)
