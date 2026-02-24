"""
main.py — Orchestrator for the Skin Cancer Classification Pipeline.
Runs the full end-to-end pipeline: EDA → Data Prep → K-Fold CV → Training → Evaluation → Grad-CAM.
"""

import json
import os
import sys
import time

import torch

# Add project directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from dataset import collect_data, compute_class_weights, create_dataloaders, get_stratified_splits
from eda import run_eda
from evaluate import generate_evaluation_report, get_predictions
from gradcam import visualize_gradcam_samples
from model import get_model
from train import run_kfold_cv, train_model
from dataset import get_val_transforms


def save_class_mapping() -> str:
    """Save class-to-index mapping as JSON."""
    mapping_path = os.path.join(config.OUTPUT_DIR, "class_mapping.json")
    with open(mapping_path, "w") as f:
        json.dump({
            "class_to_idx": config.CLASS_TO_IDX,
            "idx_to_class": config.IDX_TO_CLASS,
            "num_classes": config.NUM_CLASSES,
        }, f, indent=2)
    print(f"[MAIN] Class mapping saved → {mapping_path}")
    return mapping_path


def main():
    """Run the complete pipeline."""
    pipeline_start = time.time()

    # ========================================================================
    # Step 0: Seed & Configuration
    # ========================================================================
    print("\n" + "=" * 70)
    print("  🏥 SKIN CANCER CLASSIFICATION PIPELINE")
    print("  ISIC 9-Class Dataset | EfficientNet-B0 Transfer Learning")
    print("=" * 70 + "\n")

    config.set_seed()
    save_class_mapping()

    # ========================================================================
    # Step 1: Exploratory Data Analysis
    # ========================================================================
    print("\n📊 STEP 1: Exploratory Data Analysis")
    print("-" * 40)
    run_eda()

    # ========================================================================
    # Step 2: Data Preparation
    # ========================================================================
    print("\n📦 STEP 2: Data Preparation")
    print("-" * 40)

    # Collect all training data
    all_train_paths, all_train_labels = collect_data(config.TRAIN_DIR)
    print(f"  Total training images: {len(all_train_paths)}")

    # Also collect test data (from the dataset's own test split)
    test_paths_orig, test_labels_orig = collect_data(config.TEST_DIR)
    print(f"  Original test images: {len(test_paths_orig)}")

    # Stratified split of training data into train/val/test
    splits = get_stratified_splits(all_train_paths, all_train_labels)
    class_weights = compute_class_weights(splits["train"][1])

    # Create DataLoaders
    loaders = create_dataloaders(splits)

    # ========================================================================
    # Step 3: K-Fold Cross Validation
    # ========================================================================
    print("\n🔄 STEP 3: Stratified K-Fold Cross Validation")
    print("-" * 40)
    kfold_results = run_kfold_cv(
        all_train_paths, all_train_labels,
        epochs_per_fold=15,  # Reduced for CV efficiency
    )

    # ========================================================================
    # Step 4: Train Final Model (full train+val set)
    # ========================================================================
    print("\n🚀 STEP 4: Training Final Model")
    print("-" * 40)

    # Combine train + val for final model training
    final_train_paths = splits["train"][0] + splits["val"][0]
    final_train_labels = splits["train"][1] + splits["val"][1]
    final_class_weights = compute_class_weights(final_train_labels)

    # Create final loaders (use original test split for validation monitoring)
    from dataset import SkinLesionDataset, get_train_transforms
    from torch.utils.data import DataLoader

    final_train_ds = SkinLesionDataset(
        final_train_paths, final_train_labels, get_train_transforms()
    )
    final_val_ds = SkinLesionDataset(
        splits["test"][0], splits["test"][1], get_val_transforms()
    )

    final_train_loader = DataLoader(
        final_train_ds, batch_size=config.BATCH_SIZE, shuffle=True,
        num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    final_val_loader = DataLoader(
        final_val_ds, batch_size=config.BATCH_SIZE, shuffle=False,
        num_workers=config.NUM_WORKERS, pin_memory=True,
    )

    final_model = get_model(pretrained=True)
    train_result = train_model(
        final_model, final_train_loader, final_val_loader,
        final_class_weights, model_save_name="best_model_final.pt",
        epochs=config.EPOCHS, unfreeze_at_epoch=5,
    )

    # Load best checkpoint
    checkpoint = torch.load(train_result["model_path"], map_location=config.DEVICE,
                            weights_only=True)
    final_model.load_state_dict(checkpoint["model_state_dict"])
    print(f"  Loaded best model from epoch {checkpoint['epoch']} "
          f"(acc={checkpoint['val_acc']:.4f})")

    # ========================================================================
    # Step 5: Evaluation on Test Set
    # ========================================================================
    print("\n📈 STEP 5: Evaluation")
    print("-" * 40)

    test_loader = loaders["test"]
    y_true, y_pred, y_prob = get_predictions(final_model, test_loader, config.DEVICE)
    report_path = generate_evaluation_report(y_true, y_pred, y_prob, kfold_results)

    # ========================================================================
    # Step 6: Grad-CAM Explainability
    # ========================================================================
    print("\n🔍 STEP 6: Grad-CAM Explainability")
    print("-" * 40)

    test_paths_for_gradcam = splits["test"][0]
    test_labels_for_gradcam = splits["test"][1]
    visualize_gradcam_samples(
        final_model, test_paths_for_gradcam, test_labels_for_gradcam,
        transform=get_val_transforms(), samples_per_class=2,
    )

    # ========================================================================
    # Summary
    # ========================================================================
    elapsed = time.time() - pipeline_start
    print("\n" + "=" * 70)
    print("  ✅ PIPELINE COMPLETE")
    print(f"  Total time: {elapsed / 60:.1f} minutes")
    print(f"  Best model: {train_result['model_path']}")
    print(f"  Report: {report_path}")
    print(f"  K-Fold: {kfold_results['mean_acc']:.4f} ± {kfold_results['std_acc']:.4f}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
