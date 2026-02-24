"""
config.py — Central configuration for the Skin Cancer Classification Pipeline.
All hyperparameters, paths, and constants are defined here for reproducibility.
"""

import os
import random
import torch
import numpy as np
import kagglehub


# ============================================================================
# Reproducibility
# ============================================================================
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Set random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


# ============================================================================
# Device
# ============================================================================
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================================
# Dataset Paths
# ============================================================================
DATASET_PATH = kagglehub.dataset_download("nodoubttome/skin-cancer9-classesisic")
DATA_ROOT = os.path.join(
    DATASET_PATH,
    "Skin cancer ISIC The International Skin Imaging Collaboration",
)
TRAIN_DIR = os.path.join(DATA_ROOT, "Train")
TEST_DIR = os.path.join(DATA_ROOT, "Test")

# ============================================================================
# Class Names (sorted alphabetically for deterministic ordering)
# ============================================================================
CLASS_NAMES = sorted(os.listdir(TRAIN_DIR))
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {idx: name for name, idx in CLASS_TO_IDX.items()}

# ============================================================================
# Project Output Paths
# ============================================================================
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")
EDA_DIR = os.path.join(OUTPUT_DIR, "eda")
GRADCAM_DIR = os.path.join(OUTPUT_DIR, "gradcam")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

for _dir in [EDA_DIR, GRADCAM_DIR, MODEL_DIR, LOG_DIR]:
    os.makedirs(_dir, exist_ok=True)

# ============================================================================
# Image & Training Hyperparameters
# ============================================================================
IMG_SIZE = 300
BATCH_SIZE = 16
NUM_WORKERS = 0  # Safe default for Windows
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 50
PATIENCE = 7  # Early stopping patience
NUM_FOLDS = 5  # K-Fold CV

# ImageNet normalization stats
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============================================================================
# Model
# ============================================================================
MODEL_NAME = "efficientnet_b3"
DROPOUT_RATE = 0.3
DROPOUT_RATE_HEAD = 0.4
UNFREEZE_LAYERS = 20  # Number of backbone layers to unfreeze for fine-tuning

print(f"[CONFIG] Device: {DEVICE}")
print(f"[CONFIG] Dataset root: {DATA_ROOT}")
print(f"[CONFIG] Classes ({NUM_CLASSES}): {CLASS_NAMES}")
