"""
dataset.py — Data loading, transforms, and splitting for the ISIC Skin Cancer Dataset.
Uses Albumentations for advanced augmentation and PyTorch DataLoaders.
"""

import os
from typing import Dict, List, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import config


# ============================================================================
# Transforms
# ============================================================================

def get_train_transforms() -> A.Compose:
    """Advanced training augmentation pipeline using Albumentations."""
    return A.Compose([
        A.RandomResizedCrop(height=config.IMG_SIZE, width=config.IMG_SIZE,
                            scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                scale=(0.85, 1.15), rotate=(-30, 30), p=0.5),
        A.OneOf([
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30,
                                  val_shift_limit=20),
        ], p=0.5),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.GaussNoise(noise_scale_factor=0.1),
        ], p=0.3),
        A.CoarseDropout(num_holes_range=(1, 8),
                        hole_height_range=(5, 20),
                        hole_width_range=(5, 20),
                        fill=0, p=0.3),
        A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms() -> A.Compose:
    """Validation/test transforms — deterministic resize, center crop, normalize."""
    return A.Compose([
        A.Resize(height=config.IMG_SIZE + 32, width=config.IMG_SIZE + 32),
        A.CenterCrop(height=config.IMG_SIZE, width=config.IMG_SIZE),
        A.Normalize(mean=config.IMAGENET_MEAN, std=config.IMAGENET_STD),
        ToTensorV2(),
    ])


# ============================================================================
# Dataset
# ============================================================================

class SkinLesionDataset(Dataset):
    """PyTorch Dataset for skin lesion images."""

    def __init__(self, image_paths: List[str], labels: List[int],
                 transform: Optional[A.Compose] = None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Read image in RGB via OpenCV
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label


# ============================================================================
# Data Collection
# ============================================================================

def collect_data(data_dir: str) -> Tuple[List[str], List[int]]:
    """Collect image paths and integer labels from a directory with class subfolders."""
    image_paths = []
    labels = []
    for class_name in config.CLASS_NAMES:
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
                image_paths.append(os.path.join(class_dir, fname))
                labels.append(config.CLASS_TO_IDX[class_name])
    return image_paths, labels


# ============================================================================
# Stratified Split
# ============================================================================

def get_stratified_splits(
    image_paths: List[str],
    labels: List[int],
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> Dict[str, Tuple[List[str], List[int]]]:
    """
    Stratified train/val/test split.
    Returns dict with keys 'train', 'val', 'test' mapping to (paths, labels).
    """
    # First split: train+val vs test
    train_val_paths, test_paths, train_val_labels, test_labels = train_test_split(
        image_paths, labels,
        test_size=test_ratio,
        random_state=config.SEED,
        stratify=labels,
    )
    # Second split: train vs val
    adjusted_val_ratio = val_ratio / (1 - test_ratio)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_val_paths, train_val_labels,
        test_size=adjusted_val_ratio,
        random_state=config.SEED,
        stratify=train_val_labels,
    )

    splits = {
        "train": (train_paths, train_labels),
        "val": (val_paths, val_labels),
        "test": (test_paths, test_labels),
    }

    for name, (paths, lbls) in splits.items():
        print(f"  [{name.upper():>5s}] {len(paths):>5d} images")

    return splits


# ============================================================================
# Class Weights
# ============================================================================

def compute_class_weights(labels: List[int]) -> torch.Tensor:
    """Compute inverse-frequency class weights for weighted CrossEntropy."""
    counts = np.bincount(labels, minlength=config.NUM_CLASSES)
    total = len(labels)
    weights = total / (config.NUM_CLASSES * counts.astype(np.float64) + 1e-6)
    weights_tensor = torch.FloatTensor(weights)
    print(f"  [WEIGHTS] Class weights: {weights_tensor.numpy().round(2)}")
    return weights_tensor

# ============================================================================
# DataLoaders
# ============================================================================

def create_dataloaders(
    splits: Dict[str, Tuple[List[str], List[int]]],
) -> Dict[str, DataLoader]:
    """Create DataLoaders for train, val, and test splits."""
    train_paths, train_labels = splits["train"]
    val_paths, val_labels = splits["val"]
    test_paths, test_labels = splits["test"]

    train_dataset = SkinLesionDataset(train_paths, train_labels, get_train_transforms())
    val_dataset = SkinLesionDataset(val_paths, val_labels, get_val_transforms())
    test_dataset = SkinLesionDataset(test_paths, test_labels, get_val_transforms())

    loaders = {
        "train": DataLoader(
            train_dataset, batch_size=config.BATCH_SIZE, shuffle=True,
            num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True,
        ),
        "val": DataLoader(
            val_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
            num_workers=config.NUM_WORKERS, pin_memory=True,
        ),
        "test": DataLoader(
            test_dataset, batch_size=config.BATCH_SIZE, shuffle=False,
            num_workers=config.NUM_WORKERS, pin_memory=True,
        ),
    }
    return loaders


def get_kfold_splits(
    image_paths: List[str], labels: List[int]
) -> List[Tuple[List[int], List[int]]]:
    """Generate Stratified K-Fold split indices."""
    skf = StratifiedKFold(n_splits=config.NUM_FOLDS, shuffle=True,
                          random_state=config.SEED)
    return list(skf.split(image_paths, labels))
