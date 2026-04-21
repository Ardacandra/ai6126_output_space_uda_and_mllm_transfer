#!/usr/bin/env python3
"""Task 1: Output-Space UDA with Pseudo-Labels.

Methods compared:
  - Source Only (trained on source, evaluated on target)
  - Vanilla pseudo-labelling
  - CBST  (Class-Balanced Self-Training)
  - CRST  (Confidence-Regularised Self-Training)

The experiment (dataset pair + hyper-parameters) is fully controlled by
configs/experiments.yaml.

Usage:
    python scripts/task1_uda.py --experiment svhn_to_mnist
    python scripts/task1_uda.py --experiment office31_amazon_to_dslr
    python scripts/task1_uda.py --config path/to/custom.yaml --experiment my_exp
"""

import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as tv_models
import yaml
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    f1_score, precision_score, recall_score,
)
from torch.utils.data import DataLoader

from dataset_factory import (
    get_transform_clip,
    get_transform_resnet,
    get_transform_simplenet,
    load_split,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "configs" / "experiments.yaml"


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

    # Merge defaults → experiment (experiment values win)
    merged_task1 = {**defaults.get("task1", {}), **exp.get("task1", {})}
    exp["task1"] = merged_task1
    return exp


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class SimpleNet(nn.Module):
    """Lightweight CNN for 28×28 single-channel inputs."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        # 28→26→24→maxpool→12 : 64*12*12 = 9216
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        return self.fc2(F.relu(self.fc1(x)))


def get_model(model_name: str, num_classes: int) -> nn.Module:
    if model_name == "simplenet":
        return SimpleNet(num_classes=num_classes)
    if model_name == "resnet18":
        m = tv_models.resnet18(weights=tv_models.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = nn.Linear(512, num_classes)
        return m
    raise ValueError(f"Unknown model '{model_name}'. Supported: simplenet, resnet18")


def clone_model(model: nn.Module, device) -> nn.Module:
    m = copy.deepcopy(model)
    return m.to(device)


# ---------------------------------------------------------------------------
# Training / evaluation
# ---------------------------------------------------------------------------

def train_source_only(model, loader, device, epochs: int, lr: float = 1e-3):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    for epoch in range(epochs):
        running = 0.0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(data), target)
            loss.backward()
            optimizer.step()
            running += loss.item()
        print(f"  Epoch {epoch + 1}/{epochs}  loss={running / len(loader):.4f}")
    print("  Source-only training complete.")


def evaluate_metrics(model, loader, device) -> dict:
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, target in loader:
            preds = model(data.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(target.numpy())
    return {
        "Acc.":      accuracy_score(all_labels, all_preds),
        "Prec.":     precision_score(all_labels, all_preds, average="macro", zero_division=0),
        "Rec.":      recall_score(all_labels, all_preds, average="macro", zero_division=0),
        "F1":        f1_score(all_labels, all_preds, average="macro", zero_division=0),
        "Bal. Acc.": balanced_accuracy_score(all_labels, all_preds),
    }


# ---------------------------------------------------------------------------
# Pseudo-label generation
# ---------------------------------------------------------------------------

def get_pseudo_labels(model, loader, device, mode: str, threshold: float, cbst_p: float):
    model.eval()
    all_probs, all_images = [], []
    with torch.no_grad():
        for data, _ in loader:
            probs = F.softmax(model(data.to(device)), dim=1)
            all_probs.append(probs.cpu())
            all_images.append(data)
    all_probs  = torch.cat(all_probs)
    all_images = torch.cat(all_images)
    conf, labels = torch.max(all_probs, dim=1)
    num_classes = all_probs.shape[1]

    if mode == "vanilla":
        mask = conf > threshold
        return all_images[mask], labels[mask]

    if mode == "cbst":
        final_mask = torch.zeros(len(labels), dtype=torch.bool)
        for c in range(num_classes):
            class_mask = labels == c
            if class_mask.sum() == 0:
                continue
            k = max(1, int(class_mask.sum() * cbst_p))
            thres_c = torch.topk(conf[class_mask], k)[0][-1]
            final_mask[class_mask & (conf >= thres_c)] = True
        return all_images[final_mask], labels[final_mask]

    if mode == "crst":
        mask = conf > threshold
        return all_images[mask], all_probs[mask]

    raise ValueError(f"Unknown pseudo-label mode: {mode}")


# ---------------------------------------------------------------------------
# Adaptation loop
# ---------------------------------------------------------------------------

def adapt_with_pseudo_labels(
    base_model, pseudo_images, pseudo_targets,
    device, mode: str, epochs: int, batch_size: int,
) -> nn.Module:
    model = clone_model(base_model, device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    model.train()

    for _ in range(epochs):
        idx = torch.randperm(len(pseudo_images))
        for i in range(0, len(pseudo_images), batch_size):
            b = idx[i : i + batch_size]
            data   = pseudo_images[b].to(device)
            target = pseudo_targets[b].to(device)
            optimizer.zero_grad()
            output = model(data)
            if mode == "crst":
                loss = nn.KLDivLoss(reduction="batchmean")(
                    F.log_softmax(output, dim=1), target
                )
            else:
                loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
    return model


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def visualize_results(model, loader, device, title: str, save_path=None):
    model.eval()
    images, _ = next(iter(loader))
    n = min(8, len(images))
    with torch.no_grad():
        preds = model(images[:n].to(device)).argmax(dim=1).cpu().numpy()
    fig, axes = plt.subplots(1, n, figsize=(12, 3))
    for i, ax in enumerate(axes):
        img = images[i]
        if img.shape[0] == 1:
            ax.imshow(img.squeeze(), cmap="gray")
        else:
            ax.imshow(img.permute(1, 2, 0).numpy().clip(0, 1))
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
    t1  = exp["task1"]

    print(f"\n{'='*60}")
    print(f"Experiment : {args.experiment}")
    print(f"Description: {exp.get('description', '')}")
    print(f"Model      : {t1['model']}")
    print(f"{'='*60}\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    out_dir = REPO_ROOT / "out" / args.experiment
    out_dir.mkdir(parents=True, exist_ok=True)

    # Select transform based on model type
    if t1["model"] == "simplenet":
        transform = get_transform_simplenet()
    else:
        transform = get_transform_resnet()

    # Load datasets
    print("Loading datasets …")
    src_ds  = load_split(exp["source"], transform)
    tgt_ds  = load_split(exp["target"], transform)
    test_ds = load_split(exp["test"], transform)

    bs = t1["batch_size"]
    src_loader  = DataLoader(src_ds,  batch_size=bs, shuffle=True,  num_workers=4, pin_memory=True)
    tgt_loader  = DataLoader(tgt_ds,  batch_size=bs, shuffle=True,  num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True)

    num_classes = exp["num_classes"]

    # ------------------------------------------------------------------
    # Step 1 – Source-only baseline
    # ------------------------------------------------------------------
    print("\nStep 1: Training source-only model …")
    model_src = get_model(t1["model"], num_classes).to(device)
    train_source_only(model_src, src_loader, device, epochs=t1["epochs"])

    results = {"Source Only": evaluate_metrics(model_src, test_loader, device)}
    visualize_results(
        model_src, test_loader, device,
        f"Source-Only – {args.experiment}",
        save_path=out_dir / "task1_source_only.png",
    )

    # ------------------------------------------------------------------
    # Step 2 – Pseudo-label adaptation (Vanilla / CBST / CRST)
    # ------------------------------------------------------------------
    for mode in ["vanilla", "cbst", "crst"]:
        print(f"\n--- {mode.upper()} ---")
        pseudo_images, pseudo_targets = get_pseudo_labels(
            model_src, tgt_loader, device,
            mode=mode,
            threshold=t1["pseudo_threshold"],
            cbst_p=t1["cbst_p"],
        )
        print(f"  Pseudo-labelled samples: {len(pseudo_images)}")
        if len(pseudo_images) == 0:
            print("  No samples selected – skipping.")
            continue

        adapted = adapt_with_pseudo_labels(
            model_src, pseudo_images, pseudo_targets,
            device, mode=mode,
            epochs=t1["adapt_epochs"],
            batch_size=bs,
        )
        results[mode.upper()] = evaluate_metrics(adapted, test_loader, device)
        visualize_results(
            adapted, test_loader, device,
            f"Post-{mode.upper()} – {args.experiment}",
            save_path=out_dir / f"task1_{mode}.png",
        )

    print_results_table(results, f"Task 1 Results – {args.experiment}")


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
