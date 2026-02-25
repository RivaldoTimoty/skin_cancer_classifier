import argparse
import json
import os
import sys

# Ensure local imports work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn.functional as F
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2

import config
from model import get_model
from gradcam import generate_gradcam_for_inference

def load_inference_model(model_path: str, device: torch.device):
    """Loads the best model checkpoint for inference."""
    model = get_model(pretrained=False)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model

def load_class_mapping():
    """Loads the index-to-class-name mapping."""
    mapping_path = os.path.join(config.OUTPUT_DIR, "class_mapping.json")
    if os.path.exists(mapping_path):
        with open(mapping_path, "r") as f:
            mapping = json.load(f)
        
        # Handle nested dictionary if present
        idx_mapping = mapping.get("idx_to_class", mapping)
        
        idx_to_class = {int(k): v for k, v in idx_mapping.items()}
        return idx_to_class
    # Fallback to config if JSON doesn't exist
    return config.IDX_TO_CLASS

def run_inference(image_path: str, use_tta: bool = False, model_save_name: str = "best_model_optimized.pt", model=None, idx_to_class=None):
    """
    Runs model inference on a single image.
    Supports Test Time Augmentation (TTA).
    """
    device = config.DEVICE
    
    if model is None:
        print(f"[INFERENCE] Loading model on {device}...")
        model_path = os.path.join(config.MODEL_DIR, model_save_name)
        model = load_inference_model(model_path, device)
        
    if idx_to_class is None:
        idx_to_class = load_class_mapping()

    # Read image
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    base_transform = A.Compose([
        A.Resize(config.IMG_SIZE, config.IMG_SIZE),
        A.Normalize(config.IMAGENET_MEAN, config.IMAGENET_STD),
        ToTensorV2(),
    ])

    if not use_tta:
        # Standard inference
        tensor = base_transform(image=img_rgb)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            output = model(tensor)
            probs = F.softmax(output, dim=1)[0]
    else:
        # Test Time Augmentation (Phase 3)
        # We process 4 transformations: Original, Horizontal Flip, Vertical Flip, 90° Rotation
        t0 = base_transform(image=img_rgb)["image"].unsqueeze(0).to(device)
        t1 = base_transform(image=cv2.flip(img_rgb, 1))["image"].unsqueeze(0).to(device) # H flip
        t2 = base_transform(image=cv2.flip(img_rgb, 0))["image"].unsqueeze(0).to(device) # V flip
        t3 = base_transform(image=cv2.rotate(img_rgb, cv2.ROTATE_90_CLOCKWISE))["image"].unsqueeze(0).to(device) # 90 deg rotation
        
        with torch.no_grad():
            o0 = model(t0)
            o1 = model(t1)
            o2 = model(t2)
            o3 = model(t3)
            # Average logits before Softmax
            avg_logits = (o0 + o1 + o2 + o3) / 4.0
            probs = F.softmax(avg_logits, dim=1)[0]

    confidence, pred_idx = torch.max(probs, dim=0)
    pred_idx = pred_idx.item()
    confidence = confidence.item()
    pred_class = idx_to_class[pred_idx]
    
    # All probabilities mapping
    prob_dict = {idx_to_class[i]: probs[i].item() * 100 for i in range(config.NUM_CLASSES)}

    print(f"\n[RESULT] Predicted: {pred_class}")
    print(f"[RESULT] Confidence: {confidence * 100:.2f}%\n")

    print("[INFERENCE] Saving Grad-CAM visualization...")
    gradcam_path = os.path.join(config.GRADCAM_DIR, "inference_gradcam.png")
    generate_gradcam_for_inference(model, image_path, gradcam_path, device, target_class=pred_idx)
    print(f"[RESULT] Grad-CAM saved to {gradcam_path}")
    
    return pred_class, confidence, prob_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Skin Cancer Classification Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--tta", action="store_true", help="Enable Test Time Augmentation")
    parser.add_argument("--model_name", type=str, default="best_model_optimized.pt", help="Name of the model file to use")
    args = parser.parse_args()

    run_inference(args.image, args.tta, args.model_name)
