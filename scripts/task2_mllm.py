#!/usr/bin/env python3
"""Task 2: MLLM Transfer – CLIP-Adapter & Tip-Adapter (few-shot).

Methods compared:
  - CLIP-Adapter  (learnable bottleneck adapter + linear head, N-shot)
  - Tip-Adapter   (non-parametric RBF key-value cache, N-shot)

The experiment (dataset pair + hyper-parameters) is fully controlled by
configs/experiments.yaml.

Usage:
    python scripts/task2_mllm.py --experiment svhn_to_mnist
    python scripts/task2_mllm.py --experiment office31_amazon_to_dslr
    python scripts/task2_mllm.py --config path/to/custom.yaml --experiment my_exp
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score,
)
from torch.utils.data import DataLoader, Subset

try:
    import clip
except ImportError:
    raise ImportError(
        "CLIP is not installed. Run:\n"
        "  pip install git+https://github.com/openai/CLIP.git"
    )

from dataset_factory import CLIP_MEAN, CLIP_STD, get_transform_clip, load_split

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiments.yaml"

# CLIP ViT-B/32 output dimension
CLIP_DIM = 512


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(config_path: Path, experiment: str) -> dict:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    defaults = cfg.get("defaults", {})
    exp = cfg["experiments"].get(experiment)
    if exp is None:
        available = list(cfg["experiments"].keys())
        raise ValueError(
            f"Experiment '{experiment}' not found.\n"
            f"Available: {available}"
        )

    merged_task2 = {**defaults.get("task2", {}), **exp.get("task2", {})}
    exp["task2"] = merged_task2
    return exp


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class CLIPAdapter(nn.Module):
    """Bottleneck adapter with learnable residual mixing (alpha)."""

    def __init__(self, input_dim: int = CLIP_DIM, ratio: float = 0.25):
        super().__init__()
        hidden = int(input_dim * ratio)
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )
        self.alpha = nn.Parameter(torch.ones(1) * 0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.alpha * self.adapter(x) + (1 - self.alpha) * x


# ---------------------------------------------------------------------------
# Tip-Adapter helpers
# ---------------------------------------------------------------------------

def build_tip_cache(clip_model, loader, device):
    """Build the non-parametric key-value cache from few-shot samples."""
    print("  Building Tip-Adapter cache …")
    cache_keys, cache_values = [], []
    clip_model.eval()
    num_classes = None
    with torch.no_grad():
        for images, labels in loader:
            feats = clip_model.encode_image(images.to(device)).float()
            feats = F.normalize(feats, dim=-1)
            cache_keys.append(feats)
            cache_values.append(labels)
            if num_classes is None:
                num_classes = int(labels.max().item()) + 1  # initial estimate
    all_labels = torch.cat(cache_values)
    num_classes = int(all_labels.max().item()) + 1
    keys   = torch.cat(cache_keys).T                                              # (D, N)
    values = F.one_hot(all_labels, num_classes).float().to(device)                # (N, C)
    return keys, values


def tip_inference(test_features: torch.Tensor, keys: torch.Tensor, values: torch.Tensor, beta: float):
    """RBF-kernel affinity retrieval."""
    test_features = F.normalize(test_features, dim=-1)
    affinity = torch.exp(-beta * (1 - test_features @ keys))  # (B, N)
    return affinity @ values                                   # (B, C)


# ---------------------------------------------------------------------------
# Few-shot sampling
# ---------------------------------------------------------------------------

def get_few_shot_indices(dataset, n_shots: int) -> list:
    """Return indices with at most n_shots examples per class."""
    # Works for both torchvision datasets (.targets list) and ImageFolder (.targets)
    targets = np.array(dataset.targets)
    classes = np.unique(targets)
    indices = []
    for c in classes:
        idx = np.where(targets == c)[0][:n_shots]
        indices.extend(idx.tolist())
    return indices


# ---------------------------------------------------------------------------
# Evaluation / visualisation
# ---------------------------------------------------------------------------

def get_metrics(preds, labels) -> dict:
    return {
        "Acc.":      accuracy_score(labels, preds),
        "Prec.":     precision_score(labels, preds, average="macro", zero_division=0),
        "Rec.":      recall_score(labels, preds, average="macro", zero_division=0),
        "F1":        f1_score(labels, preds, average="macro", zero_division=0),
        "Bal. Acc.": balanced_accuracy_score(labels, preds),
    }


def visualize_results(images: torch.Tensor, preds: list, title: str, save_path=None):
    n = min(8, len(images))
    fig, axes = plt.subplots(1, n, figsize=(12, 3))
    clip_mean = np.array(CLIP_MEAN)
    clip_std  = np.array(CLIP_STD)
    for i, ax in enumerate(axes):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = img * clip_std + clip_mean          # de-normalise
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(f"Pred: {preds[i]}")
        ax.axis("off")
    fig.suptitle(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=100)
        print(f"  Saved → {save_path}")
    else:
        plt.show()
    plt.close(fig)


def print_results_table(results: dict, title: str):
    sep = "=" * 85
    print(f"\n{title}")
    print(sep)
    print(f"{'Method':<15} | {'Acc.':<8} | {'Prec.':<8} | {'Rec.':<8} | {'F1':<8} | {'Bal.Acc':<8}")
    print("-" * 85)
    for method, v in results.items():
        print(
            f"{method:<15} | {v['Acc.']:.4f} | {v['Prec.']:.4f} | "
            f"{v['Rec.']:.4f} | {v['F1']:.4f} | {v['Bal. Acc.']:.4f}"
        )
    print(sep)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    exp = load_config(Path(args.config), args.experiment)
    t2  = exp["task2"]

    print(f"\n{'='*60}")
    print(f"Experiment : {args.experiment}")
    print(f"Description: {exp.get('description', '')}")
    print(f"Few-shot N : {t2['n_shots']} per class")
    print(f"{'='*60}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = REPO_ROOT / "out" / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    transform = get_transform_clip()

    # Load full train and test splits
    print("Loading datasets …")
    train_ds = load_split(exp["target"], transform)   # few-shot pool from *target*
    test_ds  = load_split(exp["test"],   transform)

    # Subsample to N-shot
    few_shot_idx = get_few_shot_indices(train_ds, n_shots=t2["n_shots"])
    fewshot_ds   = Subset(train_ds, few_shot_idx)

    bs = t2["batch_size"]
    train_loader = DataLoader(fewshot_ds, batch_size=32, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,    batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    num_classes = exp["num_classes"]

    # Load frozen CLIP backbone
    print("Loading CLIP ViT-B/32 …")
    clip_model, _ = clip.load("ViT-B/32", device=device)
    clip_model.eval()

    # ------------------------------------------------------------------
    # A.  CLIP-Adapter
    # ------------------------------------------------------------------
    adapter    = CLIPAdapter(input_dim=CLIP_DIM, ratio=t2["adapter_ratio"]).to(device)
    classifier = nn.Linear(CLIP_DIM, num_classes).to(device)
    optimizer  = torch.optim.Adam(
        list(adapter.parameters()) + list(classifier.parameters()), lr=1e-3
    )

    print(f"\nTraining CLIP-Adapter for {t2['adapter_epochs']} epochs …")
    for epoch in range(t2["adapter_epochs"]):
        adapter.train(); classifier.train()
        running = 0.0
        for imgs, lbls in train_loader:
            with torch.no_grad():
                feat = clip_model.encode_image(imgs.to(device)).float()
            optimizer.zero_grad()
            loss = F.cross_entropy(classifier(adapter(feat)), lbls.to(device))
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f"  Epoch {epoch + 1}/{t2['adapter_epochs']}  loss={running / len(train_loader):.4f}")

    # ------------------------------------------------------------------
    # B.  Tip-Adapter (non-parametric)
    # ------------------------------------------------------------------
    print("\nBuilding Tip-Adapter cache …")
    keys, values = build_tip_cache(clip_model, train_loader, device)

    # ------------------------------------------------------------------
    # C.  Evaluation
    # ------------------------------------------------------------------
    adapter.eval(); classifier.eval()
    c_preds, t_preds, all_labels = [], [], []
    sample_imgs = None

    print("Evaluating …")
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs_gpu = imgs.to(device)
            feat = clip_model.encode_image(imgs_gpu).float()

            c_preds.extend(classifier(adapter(feat)).argmax(dim=1).cpu().numpy())
            t_preds.extend(tip_inference(feat, keys, values, beta=t2["tip_beta"]).argmax(dim=1).cpu().numpy())
            all_labels.extend(lbls.numpy())

            if sample_imgs is None:
                sample_imgs = imgs.cpu()

    visualize_results(
        sample_imgs, c_preds[:8],
        f"CLIP-Adapter – {args.experiment}",
        save_path=out_dir / "task2_clip_adapter.png",
    )
    visualize_results(
        sample_imgs, t_preds[:8],
        f"Tip-Adapter – {args.experiment}",
        save_path=out_dir / "task2_tip_adapter.png",
    )

    results = {
        "CLIP-Adapter": get_metrics(c_preds, all_labels),
        "Tip-Adapter":  get_metrics(t_preds, all_labels),
    }
    print_results_table(results, f"Task 2 Results – {args.experiment}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--experiment", "-e", required=True,
        help="Experiment name defined in the config file.",
    )
    parser.add_argument(
        "--config", "-c", default=str(DEFAULT_CONFIG),
        help="Path to the YAML config file.",
    )
    main(parser.parse_args())
