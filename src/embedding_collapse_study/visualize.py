"""
Visualization utilities: t-SNE, UMAP, singular value plots.
"""
import numpy as np
import matplotlib.pyplot as plt


def plot_singular_values(sv_dict: dict, save_path: str | None = None) -> None:
    """
    Plots normalized singular value distributions for multiple configurations.

    Args:
        sv_dict: {label: np.ndarray of singular values}
        save_path: if provided, saves the figure
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, sv in sv_dict.items():
        ax.plot(sv / sv.sum(), label=label)
    ax.set_xlabel("Singular value index")
    ax.set_ylabel("Normalized magnitude")
    ax.set_title("Singular Value Distribution of Embedding Matrix")
    ax.legend()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_uniformity_alignment(
    results: list[dict], save_path: str | None = None
) -> None:
    """
    Scatter plot of uniformity vs alignment across configurations.

    Args:
        results: list of dicts with keys: label, uniformity, alignment
        save_path: if provided, saves the figure
    """
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in results:
        ax.scatter(r["uniformity"], r["alignment"], label=r["label"], s=80)
    ax.set_xlabel("Uniformity ↓")
    ax.set_ylabel("Alignment ↓")
    ax.set_title("Uniformity vs Alignment")
    ax.legend(fontsize=8)
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
