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
    get_class_names,
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


def collect_predictions(model, loader, device):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for data, target in loader:
            preds = model(data.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(target.numpy())
    return np.array(all_preds), np.array(all_labels)


def get_metrics(preds, labels) -> dict:
    return {
        "Acc.":      accuracy_score(labels, preds),
        "Prec.":     precision_score(labels, preds, average="macro", zero_division=0),
        "Rec.":      recall_score(labels, preds, average="macro", zero_division=0),
        "F1":        f1_score(labels, preds, average="macro", zero_division=0),
        "Bal. Acc.": balanced_accuracy_score(labels, preds),
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

def select_informative_indices(
    preds_by_method: dict,
    labels: np.ndarray,
    method_scores: dict,
    top_method: str,
    n_samples: int = 8,
) -> np.ndarray:
    methods = list(preds_by_method.keys())
    if not methods:
        return np.array([], dtype=int)

    pred_mat = np.stack([np.array(preds_by_method[m]) for m in methods], axis=0)  # (M, N)
    correct = pred_mat == labels[None, :]

    method_to_idx = {m: i for i, m in enumerate(methods)}
    if top_method not in method_to_idx:
        top_method = max(methods, key=lambda m: method_scores.get(m, float("-inf")))
    top_idx = method_to_idx[top_method]
    other_idx = [i for i, m in enumerate(methods) if m != top_method]

    unique_preds = np.array([len(np.unique(pred_mat[:, i])) for i in range(pred_mat.shape[1])])
    num_correct = correct.sum(axis=0)
    wrong_others = (~correct[other_idx]).sum(axis=0) if other_idx else np.zeros(pred_mat.shape[1], dtype=int)

    top_correct = correct[top_idx]
    others_any_wrong = np.any(~correct[other_idx], axis=0) if other_idx else np.zeros(pred_mat.shape[1], dtype=bool)
    exclusive_top_mask = top_correct & (np.all(~correct[other_idx], axis=0) if other_idx else np.zeros(pred_mat.shape[1], dtype=bool))
    primary_mask = top_correct & others_any_wrong

    def rank(mask: np.ndarray) -> np.ndarray:
        cand = np.where(mask)[0]
        if len(cand) == 0:
            return cand
        # Prioritise strongest evidence: more non-best models wrong, then stronger disagreement.
        score = np.stack([wrong_others[cand], unique_preds[cand], (len(methods) - num_correct[cand])], axis=1)
        order = np.lexsort((-score[:, 2], -score[:, 1], -score[:, 0]))[::-1]
        return cand[order]

    selected = []

    # Force at least one sample where only the top method is correct (if available).
    for idx in rank(exclusive_top_mask):
        selected.append(int(idx))
        break

    for idx in rank(primary_mask):
        if idx not in selected:
            selected.append(int(idx))
        if len(selected) >= n_samples:
            break

    if len(selected) < n_samples and other_idx:
        secondary_mask = top_correct & (~np.all(correct[other_idx], axis=0))
        for idx in rank(secondary_mask):
            if idx not in selected:
                selected.append(int(idx))
            if len(selected) >= n_samples:
                break

    if len(selected) < n_samples:
        fallback_mask = (num_correct > 0) & (num_correct < len(methods))
        for idx in rank(fallback_mask):
            if idx not in selected:
                selected.append(int(idx))
            if len(selected) >= n_samples:
                break

    if len(selected) < n_samples:
        for idx in range(pred_mat.shape[1]):
            if idx not in selected:
                selected.append(idx)
            if len(selected) >= n_samples:
                break

    return np.array(selected[:n_samples], dtype=int)


def get_top_method_name(results: dict) -> str:
    """Pick a single top method with deterministic tie-breakers."""
    return max(
        results.keys(),
        key=lambda m: (
            results[m]["Acc."],
            results[m]["Bal. Acc."],
            results[m]["F1"],
            results[m]["Prec."],
            results[m]["Rec."],
            m,
        ),
    )


def get_images_by_indices(dataset, indices: np.ndarray) -> torch.Tensor:
    images = [dataset[int(i)][0] for i in indices]
    return torch.stack(images)


def visualize_method_comparison(
    images: torch.Tensor,
    labels: np.ndarray,
    preds_by_method: dict,
    class_names: list,
    title: str,
    save_path=None,
):
    n = min(8, len(images))
    fig = plt.figure(figsize=(3.2 * n, 7.2))
    gs = fig.add_gridspec(2, n, height_ratios=[3.8, 2.4])
    axes = [fig.add_subplot(gs[0, i]) for i in range(n)]
    table_ax = fig.add_subplot(gs[1, :])
    table_ax.axis("off")

    methods = list(preds_by_method.keys())
    short_name = {
        "Source Only": "Source",
        "Vanilla": "Vanilla",
        "CBST": "CBST",
        "CRST": "CRST",
    }

    # Row 1 is GT labels, following rows are method predictions.
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

    for i, ax in enumerate(axes):
        img = images[i]
        if img.shape[0] == 1:
            ax.imshow(img.squeeze(), cmap="gray")
        else:
            ax.imshow(img.permute(1, 2, 0).numpy().clip(0, 1))
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

    # Color-code cells for quick comparison: green=correct, red=wrong.
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

    # Emphasize header and method-name column.
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
    class_names = get_class_names(test_ds)

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

    src_preds, all_labels = collect_predictions(model_src, test_loader, device)
    method_preds = {"Source Only": src_preds}
    results = {"Source Only": get_metrics(src_preds, all_labels)}

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
        preds, labels = collect_predictions(adapted, test_loader, device)
        method_name = "Vanilla" if mode == "vanilla" else mode.upper()
        results[method_name] = get_metrics(preds, labels)
        method_preds[method_name] = preds

    acc_scores = {name: metrics["Acc."] for name, metrics in results.items()}
    top_method = get_top_method_name(results)
    sample_idx = select_informative_indices(method_preds, all_labels, acc_scores, top_method, n_samples=8)
    sample_images = get_images_by_indices(test_ds, sample_idx)
    sample_labels = all_labels[sample_idx]
    sample_preds = {k: v[sample_idx] for k, v in method_preds.items()}
    visualize_method_comparison(
        sample_images,
        sample_labels,
        sample_preds,
        class_names,
        "Task 1 Method Comparison",
        save_path=out_dir / "task1_method_comparison.png",
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
