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

from dataset_factory import CLIP_MEAN, CLIP_STD, get_class_names, get_transform_clip, load_split

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


def format_comparison_title(exp: dict) -> str:
    dataset_name = {
        "mnist": "MNIST",
        "usps": "USPS",
        "svhn": "SVHN",
        "office31": "Office31",
        "officehome": "OfficeHome",
        "pacs": "PACS",
    }

    def fmt_dataset(name: str) -> str:
        return dataset_name.get(str(name).lower(), str(name))

    def fmt_domain(domain: str) -> str:
        return str(domain).replace(" ", "-")

    src = exp["source"]
    tgt = exp["target"]
    src_ds = fmt_dataset(src["dataset"])
    tgt_ds = fmt_dataset(tgt["dataset"])
    src_domain = src.get("domain")
    tgt_domain = tgt.get("domain")

    if src["dataset"] == tgt["dataset"] and src_domain and tgt_domain:
        return f"{src_ds} ({fmt_domain(src_domain)} -> {fmt_domain(tgt_domain)})"

    left = f"{src_ds} ({fmt_domain(src_domain)})" if src_domain else src_ds
    right = f"{tgt_ds} ({fmt_domain(tgt_domain)})" if tgt_domain else tgt_ds
    return f"{left} -> {right}"


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


def select_informative_indices(
    preds_by_method: dict,
    labels: np.ndarray,
    method_scores: dict,
    n_samples: int = 6,
    better_quota: int = 5,
) -> np.ndarray:
    rng = np.random.default_rng()
    methods = list(preds_by_method.keys())
    if not methods:
        return np.array([], dtype=int)

    pred_mat = np.stack([np.array(preds_by_method[m]) for m in methods], axis=0)  # (M, N)
    correct = pred_mat == labels[None, :]

    # Rank methods by quantitative performance (accuracy), deterministic on method name.
    ranked_methods = sorted(
        methods,
        key=lambda m: (method_scores.get(m, float("-inf")), m),
        reverse=True,
    )
    better_method = ranked_methods[0]
    worse_method = ranked_methods[-1]
    method_to_idx = {m: i for i, m in enumerate(methods)}
    better_idx = method_to_idx[better_method]
    worse_idx = method_to_idx[worse_method]

    better_only_mask = correct[better_idx] & (~correct[worse_idx])
    worse_only_mask = correct[worse_idx] & (~correct[better_idx])

    # For Task 2 we have two methods, so disagreement-only masks are enough.
    better_candidates = rng.permutation(np.where(better_only_mask)[0]).tolist()
    worse_candidates = rng.permutation(np.where(worse_only_mask)[0]).tolist()

    worse_quota = max(0, n_samples - better_quota)
    take_better = min(better_quota, len(better_candidates))
    take_worse = min(worse_quota, len(worse_candidates))

    selected = better_candidates[:take_better] + worse_candidates[:take_worse]

    # Reallocate unfilled quota from whichever bucket has leftovers.
    remaining = n_samples - len(selected)
    if remaining > 0:
        extra_better = better_candidates[take_better:]
        extra_worse = worse_candidates[take_worse:]
        pool = extra_better + extra_worse
        selected.extend(pool[:remaining])

    # Final fallback to ensure we always return n_samples indices.
    if len(selected) < n_samples:
        for idx in rng.permutation(pred_mat.shape[1]):
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= n_samples:
                break

    return np.array(selected[:n_samples], dtype=int)


def get_images_by_indices(dataset, indices: np.ndarray) -> torch.Tensor:
    images = [dataset[int(i)][0] for i in indices]
    return torch.stack(images)


def visualize_method_comparison(
    images: torch.Tensor,
    labels: np.ndarray,
    preds_by_method: dict,
    title: str,
    class_names: list,
    save_path=None,
):
    n = min(6, len(images))
    fig = plt.figure(figsize=(3.2 * n, 7.0))
    gs = fig.add_gridspec(2, n, height_ratios=[3.8, 2.3])
    axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
    table_ax = fig.add_subplot(gs[1, :])
    table_ax.axis("off")

    methods = list(preds_by_method.keys())
    short_name = {
        "CLIP-Adapter": "CLIP-Adapter",
        "Tip-Adapter": "Tip-Adapter",
    }

    col_labels = ["Method"] + [f"S{i + 1}" for i in range(n)]
    table_text = []
    gt_row = ["GT"]
    for i in range(n):
        gt_idx = int(labels[i])
        gt_row.append(class_names[gt_idx] if 0 <= gt_idx < len(class_names) else str(gt_idx))
    table_text.append(gt_row)

    for method in methods:
        row = [short_name.get(method, method)]
        preds = preds_by_method[method]
        for i in range(n):
            pred_idx = int(preds[i])
            pred_name = class_names[pred_idx] if 0 <= pred_idx < len(class_names) else str(pred_idx)
            row.append(pred_name)
        table_text.append(row)

    clip_mean = np.array(CLIP_MEAN)
    clip_std  = np.array(CLIP_STD)
    for i, ax in enumerate(axes):
        img = images[i].permute(1, 2, 0).cpu().numpy()
        img = img * clip_std + clip_mean          # de-normalise
        ax.imshow(np.clip(img, 0, 1))
        ax.set_title(f"S{i + 1}", fontsize=14, pad=8)
        ax.axis("off")

    table = table_ax.table(
        cellText=table_text,
        colLabels=col_labels,
        cellLoc="center",
        colLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(13)
    table.scale(1.0, 1.65)

    for r in range(1, len(table_text) + 1):
        method_name = "GT" if r == 1 else methods[r - 2]
        for c in range(1, n + 1):
            cell = table[(r, c)]
            if method_name == "GT":
                cell.set_facecolor("#f0f0f0")
                continue
            pred_idx = int(preds_by_method[method_name][c - 1])
            gt_idx = int(labels[c - 1])
            cell.set_facecolor("#d9f2d9" if pred_idx == gt_idx else "#f7d6d6")

    for c in range(0, n + 1):
        table[(0, c)].set_facecolor("#e6e6e6")
    for r in range(1, len(table_text) + 1):
        table[(r, 0)].set_facecolor("#efefef")

    fig.suptitle(title, y=0.97)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
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
    class_names = get_class_names(test_ds)

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

    print("Evaluating …")
    with torch.no_grad():
        for imgs, lbls in test_loader:
            imgs_gpu = imgs.to(device)
            feat = clip_model.encode_image(imgs_gpu).float()

            c_preds.extend(classifier(adapter(feat)).argmax(dim=1).cpu().numpy())
            t_preds.extend(tip_inference(feat, keys, values, beta=t2["tip_beta"]).argmax(dim=1).cpu().numpy())
            all_labels.extend(lbls.numpy())

    c_preds = np.array(c_preds)
    t_preds = np.array(t_preds)
    all_labels = np.array(all_labels)
    method_preds = {
        "CLIP-Adapter": c_preds,
        "Tip-Adapter": t_preds,
    }
    results = {
        "CLIP-Adapter": get_metrics(c_preds, all_labels),
        "Tip-Adapter":  get_metrics(t_preds, all_labels),
    }
    acc_scores = {name: metrics["Acc."] for name, metrics in results.items()}
    sample_idx = select_informative_indices(method_preds, all_labels, acc_scores, n_samples=6, better_quota=4)
    sample_images = get_images_by_indices(test_ds, sample_idx)
    sample_labels = all_labels[sample_idx]
    sample_preds = {k: v[sample_idx] for k, v in method_preds.items()}
    visualize_method_comparison(
        sample_images,
        sample_labels,
        sample_preds,
        format_comparison_title(exp),
        class_names,
        save_path=out_dir / "task2_method_comparison.png",
    )

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
