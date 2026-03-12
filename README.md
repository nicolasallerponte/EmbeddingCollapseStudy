# EmbeddingCollapseStudy

> _An empirical investigation into the geometry of contrastive representation learning_

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-orange?style=flat-square)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-In%20Progress-yellow?style=flat-square)]()

---

## Overview

This project empirically studies how training hyperparameters — specifically **InfoNCE temperature** (τ) and **negative batch size** — shape the geometric properties of learned embedding spaces in self-supervised contrastive learning.

Rather than optimizing for downstream accuracy alone, we analyze the _internal geometry_ of the representation space throughout training, measuring phenomena such as **dimensional collapse**, **uniformity**, and **alignment** as defined by Wang & Isola (2020).

The goal is to understand _why_ certain configurations fail, not just _that_ they fail.

---

## Research Questions

1. How does InfoNCE temperature τ affect the uniformity/alignment tradeoff?
2. Under what conditions does **dimensional collapse** emerge, and how early in training can it be detected?
3. Is downstream linear probe accuracy a reliable proxy for geometric quality?

---

## Experimental Setup

| Variable               | Values tested                |
| ---------------------- | ---------------------------- |
| Temperature τ          | 0.07, 0.1, 0.2, 0.5, 1.0     |
| Batch size (negatives) | 128, 256, 512, 1024          |
| Architecture           | ResNet-18 (SimCLR framework) |
| Dataset                | CIFAR-10, STL-10             |
| Training epochs        | 200                          |

**Metrics tracked at each checkpoint:**

- Uniformity loss `L_unif` and alignment loss `L_align` (Wang & Isola)
- Effective rank of the embedding matrix (singular value distribution)
- Linear probe top-1 accuracy on frozen representations
- t-SNE / UMAP projections of learned embeddings

---

## Key Concepts

**InfoNCE Loss**
$$\mathcal{L} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k)/\tau)}$$

**Uniformity** measures how evenly representations are distributed on the unit hypersphere — a proxy for information content.

**Alignment** measures how close embeddings of augmented views of the same image are — a proxy for invariance.

**Dimensional collapse** occurs when the embedding space effectively spans far fewer dimensions than available, reducing expressive capacity.

---

## Stack

- **Training:** PyTorch + [lightly](https://github.com/lightly-ai/lightly)
- **Logging:** Weights & Biases
- **Analysis:** NumPy, scikit-learn (SVD, linear probe)
- **Visualization:** matplotlib, UMAP
- **Package management:** [uv](https://github.com/astral-sh/uv)
- **Report:** LaTeX (Overleaf), ICLR-style short paper format

---

## Repository Structure

```
EmbeddingCollapseStudy/
├── configs/              # Hydra/YAML experiment configs
├── src/
│   ├── train.py          # SimCLR training loop
│   ├── metrics.py        # Uniformity, alignment, effective rank
│   ├── probe.py          # Linear evaluation protocol
│   └── visualize.py      # t-SNE / UMAP / singular value plots
├── notebooks/
│   └── analysis.ipynb    # Exploratory analysis and figures
├── paper/                # LaTeX source for the technical report
└── README.md
```

---

## Results

### Temperature Sweep (batch size N=256)

| τ    | Loss | Uniformity ↓ | Alignment ↓ | Eff. Rank ↑ | Test Acc ↑ |
| ---- | ---- | ------------ | ----------- | ----------- | ---------- |
| 0.07 | 0.51 | -3.85        | 0.56        | 88.1        | 71.3%      |
| 0.1  | 0.85 | -3.86        | 0.48        | 89.1        | 73.3%      |
| 0.2  | 2.62 | -3.77        | 0.37        | 67.1        | **75.3%**  |
| 0.5  | 4.64 | -3.42        | 0.28        | 27.8        | 74.2%      |
| 1.0  | 5.41 | -2.98        | 0.23        | 12.2        | 71.4%      |

### Batch Size Sweep (τ=0.1)

| Batch size | Loss | Uniformity ↓ | Alignment ↓ | Eff. Rank ↑ |
| ---------- | ---- | ------------ | ----------- | ----------- |
| 128        | 0.55 | -3.83        | 0.47        | 75.7        |
| 256        | 0.85 | -3.86        | 0.48        | 89.1        |
| 512        | 1.28 | -3.87        | 0.51        | 96.9        |
| 1024       | 1.85 | -3.87        | 0.55        | **99.2**    |

### Key Findings

- High τ causes severe dimensional collapse (erank 88 → 12)
- Batch size independently controls effective rank with minimal effect on uniformity
- Best downstream accuracy (τ=0.2, 75.3%) does not coincide with best geometry (τ=0.1, erank=89)

---

## References

- Wang, T., & Isola, P. (2020). _Understanding Contrastive Representation Learning through Alignment and Uniformity on the Hypersphere._ ICML 2020.
- Chen, T., et al. (2020). _A Simple Framework for Contrastive Learning of Visual Representations._ ICML 2020.
- Hua, T., et al. (2021). _On Feature Decorrelation in Self-Supervised Learning._ ICCV 2021.

---

## Author

**Nicolás Aller Ponte** — CS student @ Universidade da Coruña  
[GitHub](https://github.com/nicolasallerponte) · [LinkedIn](https://linkedin.com/in/nicolasallerponte)
