"""
Geometric metrics for embedding space analysis.

References:
  Wang & Isola (2020) - Uniformity and Alignment
  Hua et al. (2021)   - Dimensional collapse via effective rank
"""
import torch
import torch.nn.functional as F


def uniformity(z: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """
    Uniformity loss from Wang & Isola (2020).
    Measures how uniformly embeddings are distributed on the unit hypersphere.
    Lower is better (more uniform).

    Args:
        z: (N, D) normalized embeddings
        t: kernel bandwidth
    """
    z = F.normalize(z, dim=1)
    sq_dists = torch.pdist(z, p=2).pow(2)
    return sq_dists.mul(-t).exp().mean().log()


def alignment(z1: torch.Tensor, z2: torch.Tensor, alpha: float = 2.0) -> torch.Tensor:
    """
    Alignment loss from Wang & Isola (2020).
    Measures how close positive pairs are in embedding space.
    Lower is better (more aligned).

    Args:
        z1, z2: (N, D) normalized embeddings of two augmented views
        alpha: exponent
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    return (z1 - z2).norm(dim=1).pow(alpha).mean()


def effective_rank(z: torch.Tensor) -> float:
    """
    Effective rank of the embedding matrix via singular value entropy.
    Higher = richer representation, lower = more collapsed.

    Roy & Vetterli (2007): erank = exp(H(p)) where p are normalized singular values.

    Args:
        z: (N, D) embeddings (need not be normalized)
    """
    z_centered = z - z.mean(dim=0, keepdim=True)
    _, S, _ = torch.linalg.svd(z_centered, full_matrices=False)
    p = S / S.sum()
    entropy = -(p * (p + 1e-8).log()).sum()
    return entropy.exp().item()


def singular_value_distribution(z: torch.Tensor) -> torch.Tensor:
    """
    Returns normalized singular values of the embedding matrix.
    Useful for visualizing collapse.

    Args:
        z: (N, D) embeddings
    """
    z_centered = z - z.mean(dim=0, keepdim=True)
    _, S, _ = torch.linalg.svd(z_centered, full_matrices=False)
    return S / S.sum()
