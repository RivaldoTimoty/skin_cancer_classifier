"""
train.py — Training loop with early stopping, LR scheduling, and K-Fold CV.
Supports weighted CrossEntropy for class imbalance handling.
"""

import csv
import os
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import config
from dataset import (
    SkinLesionDataset,
    collect_data,
    compute_class_weights,
    get_kfold_splits,
    get_train_transforms,
    get_val_transforms,
)
from model import SkinCancerModel, get_model


# ============================================================================
# Training Utilities
# ============================================================================

class EarlyStopping:
    """Early stopping to prevent overfitting."""

    def __init__(self, patience: int = config.PATIENCE, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.should_stop = False

    def __call__(self, val_loss: float) -> None:
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
                print(f"  [EARLY STOP] No improvement for {self.patience} epochs")
        else:
            self.best_loss = val_loss
            self.counter = 0





class TrainingLogger:
    """Logs training metrics to a CSV file."""

    def __init__(self, log_path: str):
        self.log_path = log_path
        self.rows: List[Dict] = []
        with open(log_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "val_acc", "lr"])

    def log(self, epoch: int, train_loss: float, val_loss: float,
            val_acc: float, lr: float) -> None:
        self.rows.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_acc": val_acc, "lr": lr,
        })
        with open(self.log_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.5f}", f"{val_loss:.5f}",
                             f"{val_acc:.4f}", f"{lr:.6f}"])


# ============================================================================
# Train / Validate One Epoch
# ============================================================================

def train_one_epoch(
    model: nn.Module, loader: DataLoader, criterion: nn.Module,
    optimizer: torch.optim.Optimizer, device: torch.device,
) -> float:
    """Train for one epoch. Returns average training loss."""
    model.train()
    running_loss = 0.0
    num_batches = 0

    for images, labels in tqdm(loader, desc="  Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        num_batches += 1

    return running_loss / max(num_batches, 1)


def validate(
    model: nn.Module, loader: DataLoader, criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Validate the model. Returns (val_loss, val_accuracy)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="  Validation", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / max(len(loader), 1)
    accuracy = correct / max(total, 1)
    return avg_loss, accuracy


# ============================================================================
# Full Training Loop
# ============================================================================

def train_model(
    model: SkinCancerModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    class_weights: Optional[torch.Tensor] = None,
    model_save_name: str = "best_model.pt",
    epochs: int = config.EPOCHS,
    unfreeze_at_epoch: int = 5,
) -> Dict[str, List]:
    """
    Full training loop with:
      - Phase 1 (epochs 0–unfreeze_at): frozen backbone, train head only
      - Phase 2 (epochs unfreeze_at–end): unfreeze last N layers, fine-tune
      - Early stopping & ReduceLROnPlateau
      - Gradient clipping
      - Best-model checkpointing
    """
    device = config.DEVICE
    
    # Phase 2 Web App strategy: unweighted label-smoothed CrossEntropy
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Full backbone fine-tuning from epoch 1
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    early_stop = EarlyStopping(patience=config.PATIENCE)

    log_path = os.path.join(config.LOG_DIR, "training_log.csv")
    logger = TrainingLogger(log_path)

    best_val_acc = 0.0
    save_path = os.path.join(config.MODEL_DIR, model_save_name)

    print(f"\n🚀 Starting training for {epochs} epochs on {device}...\n")

    for epoch in range(1, epochs + 1):
        t0 = time.time()


        # Train & validate
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_loss)
        elapsed = time.time() - t0

        logger.log(epoch, train_loss, val_loss, val_acc, current_lr)

        print(f"  Epoch {epoch:>3d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f} | "
              f"Time: {elapsed:.1f}s")

        # Checkpoint best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_acc": val_acc,
                "val_loss": val_loss,
            }, save_path)
            print(f"  ✅ Best model saved → {save_path} (acc={val_acc:.4f})")

        early_stop(val_loss)
        if early_stop.should_stop:
            break

    print(f"\n🏁 Training complete. Best Val Accuracy: {best_val_acc:.4f}")
    return {"log_path": log_path, "best_val_acc": best_val_acc, "model_path": save_path}


# ============================================================================
# K-Fold Cross Validation
# ============================================================================

def run_kfold_cv(
    image_paths: List[str],
    labels: List[int],
    epochs_per_fold: int = 20,
) -> Dict[str, float]:
    """
    Run Stratified K-Fold Cross Validation.
    Returns mean ± std accuracy across folds.
    """
    folds = get_kfold_splits(image_paths, labels)
    fold_accuracies = []
    class_weights = compute_class_weights(labels)

    print(f"\n{'='*60}")
    print(f"  STRATIFIED {config.NUM_FOLDS}-FOLD CROSS VALIDATION")
    print(f"{'='*60}\n")

    for fold_idx, (train_indices, val_indices) in enumerate(folds, 1):
        print(f"\n--- Fold {fold_idx}/{config.NUM_FOLDS} ---")

        # Prepare fold data
        fold_train_paths = [image_paths[i] for i in train_indices]
        fold_train_labels = [labels[i] for i in train_indices]
        fold_val_paths = [image_paths[i] for i in val_indices]
        fold_val_labels = [labels[i] for i in val_indices]

        train_ds = SkinLesionDataset(fold_train_paths, fold_train_labels,
                                     get_train_transforms())
        val_ds = SkinLesionDataset(fold_val_paths, fold_val_labels,
                                   get_val_transforms())

        train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE,
                                  shuffle=True, num_workers=config.NUM_WORKERS,
                                  pin_memory=True, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE,
                                shuffle=False, num_workers=config.NUM_WORKERS,
                                pin_memory=True)

        # Fresh model per fold
        model = get_model(pretrained=True)
        result = train_model(
            model, train_loader, val_loader, class_weights,
            model_save_name=f"fold_{fold_idx}_best.pt",
            epochs=epochs_per_fold,
            unfreeze_at_epoch=3,
        )
        fold_accuracies.append(result["best_val_acc"])
        print(f"  Fold {fold_idx} Best Acc: {result['best_val_acc']:.4f}")

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    print(f"\n{'='*60}")
    print(f"  K-FOLD RESULTS: {mean_acc:.4f} ± {std_acc:.4f}")
    print(f"  Per-fold: {[f'{a:.4f}' for a in fold_accuracies]}")
    print(f"{'='*60}\n")

    return {"mean_acc": mean_acc, "std_acc": std_acc, "fold_accs": fold_accuracies}
