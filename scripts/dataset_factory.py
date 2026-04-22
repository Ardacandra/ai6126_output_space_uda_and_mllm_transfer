#!/usr/bin/env python3
"""Dataset factory – loads any dataset defined in configs/experiments.yaml.

Supported datasets
------------------
Torchvision (auto-downloaded):
  mnist, svhn, usps

Folder-based (images already on disk):
  office31   – domain: amazon | dslr | webcam
  officehome – domain: Art | Clipart | Product | Real World
  pacs       – domain: art_painting | cartoon | photo | sketch

Transforms available
---------------------
  get_transform_simplenet()  – 28×28 greyscale  (Task 1 with SimpleNet)
  get_transform_resnet()     – 224×224 RGB       (Task 1 with ResNet-18)
  get_transform_clip()       – 224×224 RGB CLIP  (Task 2)
"""

from pathlib import Path

import numpy as np
from torchvision import datasets, transforms

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = REPO_ROOT / "dataset"

# CLIP ViT-B/32 constants
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD  = (0.26862954, 0.26130258, 0.27577711)

# ImageNet constants (for ResNet-18)
IN_MEAN = (0.485, 0.456, 0.406)
IN_STD  = (0.229, 0.224, 0.225)


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_transform_simplenet():
    """28×28 single-channel – for SimpleNet (digit datasets)."""
    return transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])


def get_transform_resnet():
    """224×224 RGB with ImageNet normalisation – for ResNet-18."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(IN_MEAN, IN_STD),
    ])


def get_transform_clip():
    """224×224 RGB with CLIP normalisation – for Task 2."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_split(split_cfg: dict, transform):
    """Return a torch Dataset for one split/domain entry from the config.

    split_cfg keys
    --------------
    dataset : str   – dataset name (mnist | svhn | usps | office31 | officehome | pacs)
    split   : str   – 'train' or 'test' (torchvision datasets)
    domain  : str   – domain folder name (folder-based datasets)
    """
    name = split_cfg["dataset"].lower()

    if name == "mnist":
        root = DATASET_ROOT / "MNIST"
        return datasets.MNIST(
            root=str(root),
            train=split_cfg.get("split", "train") == "train",
            download=False,
            transform=transform,
        )

    if name == "svhn":
        root = DATASET_ROOT / "SVHN"
        return datasets.SVHN(
            root=str(root),
            split=split_cfg.get("split", "train"),
            download=False,
            transform=transform,
        )

    if name == "usps":
        root = DATASET_ROOT / "USPS"
        return datasets.USPS(
            root=str(root),
            train=split_cfg.get("split", "train") == "train",
            download=False,
            transform=transform,
        )

    if name == "office31":
        domain = split_cfg["domain"]
        root = DATASET_ROOT / "office31" / domain
        return datasets.ImageFolder(root=str(root), transform=transform)

    if name == "officehome":
        domain = split_cfg["domain"]
        root = DATASET_ROOT / "OfficeHomeDataset" / domain
        return datasets.ImageFolder(root=str(root), transform=transform)

    if name == "pacs":
        domain = split_cfg["domain"]
        root = DATASET_ROOT / "pacs" / domain
        return datasets.ImageFolder(root=str(root), transform=transform)

    raise ValueError(f"Unknown dataset '{name}'. "
                     "Supported: mnist, svhn, usps, office31, officehome, pacs")


def get_class_names(dataset) -> list:
    """Return class names for a dataset, handling Subset and torchvision datasets."""
    if hasattr(dataset, "dataset"):
        return get_class_names(dataset.dataset)

    if hasattr(dataset, "classes"):
        return [str(c) for c in dataset.classes]

    if hasattr(dataset, "targets"):
        targets = np.array(dataset.targets)
        return [str(c) for c in np.unique(targets).tolist()]

    raise ValueError("Dataset does not expose class information (.classes or .targets).")
