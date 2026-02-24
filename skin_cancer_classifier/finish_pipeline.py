"""
finish_pipeline.py — Completes the remaining pipeline steps.
Runs folds 4-5, final model training (reduced epochs), evaluation, and Grad-CAM.
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import numpy as np

import config
config.set_seed()

from dataset import (
    SkinLesionDataset, collect_data, compute_class_weights,
    create_dataloaders, get_kfold_splits, get_stratified_splits,
    get_train_transforms, get_val_transforms,
)
from model import get_model
from train import train_model
from evaluate import generate_evaluation_report, get_predictions
from gradcam import visualize_gradcam_samples
from torch.utils.data import DataLoader

# Reduce epochs for faster completion
KFOLD_EPOCHS = 10
FINAL_EPOCHS = 15


def main():
    t0 = time.time()

    # Collect data
    all_train_paths, all_train_labels = collect_data(config.TRAIN_DIR)
    class_weights = compute_class_weights(all_train_labels)
    splits = get_stratified_splits(all_train_paths, all_train_labels)

    # ================================================================
    # Run remaining K-Fold folds (4 and 5) — folds 1-3 already done
    # ================================================================
    folds = get_kfold_splits(all_train_paths, all_train_labels)

    # Previous fold accuracies from first run
    fold_accuracies = [0.5759, 0.6027, 0.5491]  # folds 1-3

    for fold_idx in [3, 4]:  # 0-indexed: folds 4 and 5
        fold_num = fold_idx + 1
        print(f"\n--- Fold {fold_num}/{config.NUM_FOLDS} ---")

        train_indices, val_indices = folds[fold_idx]
        fold_train_paths = [all_train_paths[i] for i in train_indices]
        fold_train_labels = [all_train_labels[i] for i in train_indices]
        fold_val_paths = [all_train_paths[i] for i in val_indices]
        fold_val_labels = [all_train_labels[i] for i in val_indices]

        train_ds = SkinLesionDataset(fold_train_paths, fold_train_labels, get_train_transforms())
        val_ds = SkinLesionDataset(fold_val_paths, fold_val_labels, get_val_transforms())

        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                                  num_workers=0, pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                                num_workers=0, pin_memory=True)

        model = get_model(pretrained=True)
        result = train_model(
            model, train_loader, val_loader, class_weights,
            model_save_name=f"fold_{fold_num}_best.pt",
            epochs=KFOLD_EPOCHS, unfreeze_at_epoch=3,
        )
        fold_accuracies.append(result["best_val_acc"])
        print(f"  Fold {fold_num} Best Acc: {result['best_val_acc']:.4f}")

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    kfold_results = {"mean_acc": mean_acc, "std_acc": std_acc, "fold_accs": fold_accuracies}
    print(f"\n{'='*60}")
    print(f"  K-FOLD RESULTS: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Per-fold: {[f'{a:.4f}' for a in fold_accuracies]}")
    print(f"{'='*60}\n")

    # ================================================================
    # Train final model
    # ================================================================
    print("\n🚀 Training Final Model...")
    final_train_paths = splits["train"][0] + splits["val"][0]
    final_train_labels = splits["train"][1] + splits["val"][1]
    final_class_weights = compute_class_weights(final_train_labels)

    final_train_ds = SkinLesionDataset(final_train_paths, final_train_labels, get_train_transforms())
    final_val_ds = SkinLesionDataset(splits["test"][0], splits["test"][1], get_val_transforms())

    final_train_loader = DataLoader(final_train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
                                    num_workers=0, pin_memory=True, drop_last=True)
    final_val_loader = DataLoader(final_val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
                                  num_workers=0, pin_memory=True)

    final_model = get_model(pretrained=True)
    train_result = train_model(
        final_model, final_train_loader, final_val_loader,
        final_class_weights, model_save_name="best_model_final.pt",
        epochs=FINAL_EPOCHS, unfreeze_at_epoch=3,
    )

    # Load best checkpoint
    checkpoint = torch.load(train_result["model_path"], map_location=config.DEVICE, weights_only=True)
    final_model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded best model (acc={checkpoint['val_acc']:.4f})")

    # ================================================================
    # Evaluation
    # ================================================================
    print("\n📈 Evaluation...")
    test_loader = DataLoader(
        SkinLesionDataset(splits["test"][0], splits["test"][1], get_val_transforms()),
        batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True,
    )
    y_true, y_pred, y_prob = get_predictions(final_model, test_loader, config.DEVICE)
    report_path = generate_evaluation_report(y_true, y_pred, y_prob, kfold_results)

    # ================================================================
    # Grad-CAM
    # ================================================================
    print("\n🔍 Grad-CAM...")
    visualize_gradcam_samples(
        final_model, splits["test"][0], splits["test"][1],
        transform=get_val_transforms(), samples_per_class=2,
    )

    elapsed = time.time() - t0
    print(f"\n✅ Pipeline complete in {elapsed/60:.1f} minutes")
    print(f"  K-Fold: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Final model: {train_result['model_path']}")
    print(f"  Report: {report_path}")


if __name__ == "__main__":
    main()
