"""
optimize_pipeline.py — Runs the pipeline to test Phase 1 Optimization (WeightedRandomSampler).
Skips K-Fold to save time and immediately evaluates on the hold-out test set.
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
    SkinLesionDataset, collect_data, create_dataloaders,
    get_stratified_splits, get_train_transforms, get_val_transforms,
)
from model import get_model
from train import train_model
from evaluate import generate_evaluation_report, get_predictions
from gradcam import visualize_gradcam_samples
from torch.utils.data import DataLoader

FINAL_EPOCHS = 50

def main():
    t0 = time.time()

    print("\n[WEB APP DEPLOYMENT] Phase 2: Training Strategy Reset (EfficientNet-B3)...")
    
    # Collect data
    all_train_paths, all_train_labels = collect_data(config.TRAIN_DIR)
    splits = get_stratified_splits(all_train_paths, all_train_labels)

    # ================================================================
    # Get dataloaders
    # ================================================================
    loaders = create_dataloaders(splits)
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]

    # ================================================================
    # Train optimized final model
    # ================================================================
    print(f"\n🚀 Training Optimized Model for {FINAL_EPOCHS} epochs...")
    optimized_model = get_model(pretrained=True)
    
    train_result = train_model(
        optimized_model, train_loader, val_loader,
        model_save_name="best_model_optimized.pt",
        epochs=FINAL_EPOCHS,
    )

    # Load best checkpoint
    checkpoint = torch.load(train_result["model_path"], map_location=config.DEVICE, weights_only=True)
    optimized_model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded optimized model (val_acc={checkpoint['val_acc']:.4f})")

    # ================================================================
    # Evaluation
    # ================================================================
    print("\n📈 Evaluation...")
    y_true, y_pred, y_prob = get_predictions(optimized_model, test_loader, config.DEVICE)
    report_path = generate_evaluation_report(y_true, y_pred, y_prob, None) # Null K-Fold results

    # ================================================================
    # Grad-CAM
    # ================================================================
    print("\n🔍 Grad-CAM...")
    visualize_gradcam_samples(
        optimized_model, splits["test"][0], splits["test"][1],
        transform=get_val_transforms(), samples_per_class=2,
    )

    elapsed = time.time() - t0
    print(f"\n✅ Optimization Pipeline complete in {elapsed/60:.1f} minutes")
    print(f"  Final model: {train_result['model_path']}")
    print(f"  Report: {report_path}")

if __name__ == "__main__":
    main()
