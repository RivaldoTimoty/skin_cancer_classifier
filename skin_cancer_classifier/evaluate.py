"""
evaluate.py — Model evaluation with comprehensive metrics and visualizations.
Generates classification report, confusion matrix, and ROC-AUC curves.
"""

import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

import config


# ============================================================================
# Prediction Collection
# ============================================================================

def get_predictions(
    model: nn.Module, loader: DataLoader, device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Collect all predictions, true labels, and probability scores.
    Returns: (true_labels, predicted_labels, probabilities)
    """
    model.eval()
    all_labels = []
    all_preds = []
    all_probs = []

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Evaluating", leave=False):
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


# ============================================================================
# Metrics
# ============================================================================

def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
) -> Dict:
    """Compute comprehensive classification metrics."""
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    precision_weighted = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall_weighted = recall_score(y_true, y_pred, average="weighted", zero_division=0)

    # ROC-AUC (one-vs-rest)
    try:
        roc_auc_macro = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        roc_auc_weighted = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
    except ValueError:
        roc_auc_macro = roc_auc_weighted = 0.0

    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_weighted": precision_weighted,
        "recall_weighted": recall_weighted,
        "roc_auc_macro": roc_auc_macro,
        "roc_auc_weighted": roc_auc_weighted,
    }
    return metrics


# ============================================================================
# Visualizations
# ============================================================================

def plot_confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str],
) -> str:
    """Plot and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14)
    plt.xticks(rotation=40, ha="right", fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    save_path = os.path.join(config.OUTPUT_DIR, "confusion_matrix.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[EVAL] Confusion matrix saved → {save_path}")
    return save_path


def plot_roc_curves(
    y_true: np.ndarray, y_prob: np.ndarray, class_names: List[str],
) -> str:
    """Plot per-class ROC curves and save."""
    n_classes = len(class_names)
    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))

    for i in range(n_classes):
        # Binarize: 1 if class i, 0 otherwise
        y_binary = (y_true == i).astype(int)
        if y_binary.sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_binary, y_prob[:, i])
        auc_val = roc_auc_score(y_binary, y_prob[:, i])
        ax.plot(fpr, tpr, color=colors[i], lw=1.5,
                label=f"{class_names[i]} (AUC={auc_val:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Per Class", fontsize=14)
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()

    save_path = os.path.join(config.OUTPUT_DIR, "roc_curves.png")
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"[EVAL] ROC curves saved → {save_path}")
    return save_path


# ============================================================================
# Report Generation
# ============================================================================

def generate_evaluation_report(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
    kfold_results: Dict = None,
) -> str:
    """Generate and save a comprehensive evaluation report in Markdown."""
    class_names = config.CLASS_NAMES
    metrics = compute_metrics(y_true, y_pred, y_prob)

    # Confusion matrix & ROC plots
    plot_confusion_matrix(y_true, y_pred, class_names)
    plot_roc_curves(y_true, y_prob, class_names)

    # Classification report string
    cls_report = classification_report(y_true, y_pred, target_names=class_names,
                                        zero_division=0)

    # Build markdown report
    lines = [
        "# Skin Cancer Classification — Evaluation Report\n",
        f"**Model**: EfficientNet-B0 (Transfer Learning)  ",
        f"**Dataset**: ISIC 9-Class Skin Cancer  ",
        f"**Date**: Generated automatically  \n",
        "---\n",
        "## Overall Metrics\n",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Accuracy | {metrics['accuracy']:.4f} |",
        f"| Precision (Macro) | {metrics['precision_macro']:.4f} |",
        f"| Recall (Macro) | {metrics['recall_macro']:.4f} |",
        f"| F1-Score (Macro) | {metrics['f1_macro']:.4f} |",
        f"| F1-Score (Weighted) | {metrics['f1_weighted']:.4f} |",
        f"| Precision (Weighted) | {metrics['precision_weighted']:.4f} |",
        f"| Recall (Weighted) | {metrics['recall_weighted']:.4f} |",
        f"| ROC-AUC (Macro) | {metrics['roc_auc_macro']:.4f} |",
        f"| ROC-AUC (Weighted) | {metrics['roc_auc_weighted']:.4f} |",
        "",
    ]

    if kfold_results:
        lines.extend([
            "## K-Fold Cross Validation\n",
            f"| Folds | Mean Accuracy | Std Accuracy |",
            f"|-------|---------------|--------------|",
            f"| {config.NUM_FOLDS} | {kfold_results['mean_acc']:.4f} | {kfold_results['std_acc']:.4f} |",
            f"\nPer-fold accuracies: {kfold_results['fold_accs']}",
            "",
        ])

    lines.extend([
        "## Classification Report\n",
        "```",
        cls_report,
        "```\n",
        "## Confusion Matrix\n",
        "See `outputs/confusion_matrix.png`\n",
        "## ROC Curves\n",
        "See `outputs/roc_curves.png`\n",
    ])

    report_text = "\n".join(lines)
    report_path = os.path.join(config.OUTPUT_DIR, "evaluation_report.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    # Also save metrics as JSON
    metrics_path = os.path.join(config.OUTPUT_DIR, "metrics.json")
    # Convert numpy types to Python types for JSON
    json_metrics = {k: float(v) for k, v in metrics.items()}
    with open(metrics_path, "w") as f:
        json.dump(json_metrics, f, indent=2)

    print(f"[EVAL] Report saved → {report_path}")
    print(f"[EVAL] Metrics JSON → {metrics_path}")

    # Print summary to console
    print("\n" + "=" * 60)
    print("  FINAL PERFORMANCE SUMMARY")
    print("=" * 60)
    for k, v in metrics.items():
        print(f"  {k:<25s}: {v:.4f}")
    print("=" * 60)
    print(f"\n{cls_report}")

    return report_path
