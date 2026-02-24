"""
eda.py — Exploratory Data Analysis for the ISIC Skin Cancer Dataset.
Generates visualizations for class distribution, image sizes, and sample images.
"""

import os
from collections import Counter
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import config


def _collect_image_paths() -> Tuple[List[str], List[str]]:
    """Collect all image file paths and their class labels from Train directory."""
    image_paths = []
    labels = []
    for class_name in config.CLASS_NAMES:
        class_dir = os.path.join(config.TRAIN_DIR, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(class_name)
    return image_paths, labels


def plot_class_distribution(labels: List[str]) -> None:
    """Plot bar chart of class distribution and save."""
    counter = Counter(labels)
    classes = sorted(counter.keys())
    counts = [counter[c] for c in classes]

    fig, ax = plt.subplots(figsize=(14, 6))
    bars = ax.bar(classes, counts, color=sns.color_palette("viridis", len(classes)))
    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Class Distribution — ISIC Skin Cancer Dataset (Train)", fontsize=14)
    ax.set_xticklabels(classes, rotation=35, ha="right", fontsize=9)

    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                str(count), ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    save_path = os.path.join(config.EDA_DIR, "class_distribution.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[EDA] Class distribution saved → {save_path}")


def plot_image_size_distribution(image_paths: List[str], sample_size: int = 500) -> None:
    """Plot scatter/histogram of image dimensions from a sample."""
    rng = np.random.RandomState(config.SEED)
    sampled = rng.choice(image_paths, min(sample_size, len(image_paths)), replace=False)

    widths, heights = [], []
    for path in sampled:
        img = cv2.imread(path)
        if img is not None:
            h, w = img.shape[:2]
            widths.append(w)
            heights.append(h)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Scatter
    axes[0].scatter(widths, heights, alpha=0.4, s=10, color="teal")
    axes[0].set_xlabel("Width")
    axes[0].set_ylabel("Height")
    axes[0].set_title("Image Dimensions (Width vs Height)")

    # Width histogram
    axes[1].hist(widths, bins=30, color="steelblue", edgecolor="black")
    axes[1].set_xlabel("Width (px)")
    axes[1].set_title("Width Distribution")

    # Height histogram
    axes[2].hist(heights, bins=30, color="salmon", edgecolor="black")
    axes[2].set_xlabel("Height (px)")
    axes[2].set_title("Height Distribution")

    plt.tight_layout()
    save_path = os.path.join(config.EDA_DIR, "image_size_distribution.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[EDA] Image size distribution saved → {save_path}")


def plot_sample_images_per_class(image_paths: List[str], labels: List[str],
                                  samples_per_class: int = 5) -> None:
    """Display sample images for each class in a grid."""
    class_images: Dict[str, List[str]] = {c: [] for c in config.CLASS_NAMES}
    for path, label in zip(image_paths, labels):
        class_images[label].append(path)

    n_classes = len(config.CLASS_NAMES)
    fig, axes = plt.subplots(n_classes, samples_per_class,
                              figsize=(samples_per_class * 3, n_classes * 3))

    rng = np.random.RandomState(config.SEED)
    for i, class_name in enumerate(config.CLASS_NAMES):
        paths = class_images[class_name]
        chosen = rng.choice(paths, min(samples_per_class, len(paths)), replace=False)
        for j in range(samples_per_class):
            ax = axes[i][j] if n_classes > 1 else axes[j]
            if j < len(chosen):
                img = cv2.imread(chosen[j])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (config.IMG_SIZE, config.IMG_SIZE))
                ax.imshow(img)
            ax.axis("off")
            if j == 0:
                ax.set_ylabel(class_name, fontsize=8, rotation=0, labelpad=80, va="center")

    fig.suptitle("Sample Images per Class", fontsize=14, y=1.01)
    plt.tight_layout()
    save_path = os.path.join(config.EDA_DIR, "sample_images_per_class.png")
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"[EDA] Sample images saved → {save_path}")


def detect_class_imbalance(labels: List[str]) -> Dict[str, int]:
    """Detect and report class imbalance."""
    counter = Counter(labels)
    total = sum(counter.values())
    max_count = max(counter.values())
    min_count = min(counter.values())
    imbalance_ratio = max_count / min_count

    print("\n" + "=" * 60)
    print("CLASS IMBALANCE REPORT")
    print("=" * 60)
    for cls in config.CLASS_NAMES:
        count = counter[cls]
        pct = count / total * 100
        print(f"  {cls:<30s} {count:>5d}  ({pct:5.1f}%)")
    print(f"\n  Total images: {total}")
    print(f"  Max class count: {max_count}")
    print(f"  Min class count: {min_count}")
    print(f"  Imbalance ratio: {imbalance_ratio:.1f}x")
    if imbalance_ratio > 3:
        print("  ⚠️  SIGNIFICANT class imbalance detected — using weighted loss!")
    print("=" * 60 + "\n")

    return dict(counter)


def run_eda() -> Tuple[List[str], List[str]]:
    """Run the full EDA pipeline."""
    print("\n🔍 Running Exploratory Data Analysis...\n")
    image_paths, labels = _collect_image_paths()

    plot_class_distribution(labels)
    plot_image_size_distribution(image_paths)
    plot_sample_images_per_class(image_paths, labels)
    detect_class_imbalance(labels)

    return image_paths, labels


if __name__ == "__main__":
    config.set_seed()
    run_eda()
